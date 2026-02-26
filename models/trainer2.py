import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.nn.functional import mse_loss, l1_loss
from types import SimpleNamespace
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

from pytorch_lightning import LightningModule

from models.model_helper2 import create_model, get_model_checkpoint, load_model
from .utils.graphaug import GraphAugmenter
from models.modules.contrastive import ContrastiveLoss
from data.batchclipper import BatchClipper, TupleBatchClipper

from scipy.stats import spearmanr
from scipy.stats import pearsonr

def compute_corr(predictions, targets):
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()

    spearman_corr, p_spearman = spearmanr(predictions, targets)
    rmse = np.sqrt(np.mean((predictions-targets)**2))
    pearson_corr, p_pearson = pearsonr(predictions.flatten(), targets.flatten())
    
    return {
        "rmse": rmse,
        "pearson_corr": pearson_corr,
        "spearman_corr": spearman_corr,
    }

class LTrainer(LightningModule):
    def __init__(self, hparams, prior_model=None, mean=None, std=None):
        super().__init__()
        
        # Convert dict to namespace-like object if needed
        if isinstance(hparams, dict):
            hparams = SimpleNamespace(**hparams)

        self.save_hyperparameters(vars(hparams))
        
        self.pretraining = self.hparams.pretraining
        self.self_cond = self.hparams.self_cond

        self.update_normalizer = True
        self.noise_scale = self.hparams.noise_scale
        self.reg_noise_scale = 0.005  # Noise applied for regularization on embedding structure

        self.ema_model = None
        self.reset_stored_outputs()  # Create object to store outputs
        self.store_outputs = False  # Whether or not to store outputs from each step in self.step_outputs
        if self.hparams.compute_corr:
            self.store_outputs = True
 
        if self.hparams.restart:
            checkpoint_path, _step, _epoch = get_model_checkpoint(self.hparams.log_dir, 
                                                                  epoch=self.hparams.load_epoch)
            if checkpoint_path is None:
                print(f'WARNING: restart is on but no checkpoints found in {self.hparams.log_dir}')

            else:
                ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                ckpt_epoch = ckpt["epoch"]
                ckpt_step = ckpt["global_step"]

                if not ckpt_epoch == _epoch:
                    print(f'WARNING: Checkpoint epoch {ckpt_epoch} does not match requested restart epoch {_epoch}. loaded step {ckpt_step}')
                if not ckpt_step-1 == _step:
                    print(f'WARNING: Checkpoint step {ckpt_step} does not match expected step {_step}')
            print(f"Restarting from checkpoint: {checkpoint_path}, epoch: {_epoch}, step: {_step}")
            
            self.hparams.load_model = checkpoint_path

        if self.hparams.load_model:
            self.model, ema_model, ckpt = load_model(
                self.hparams.load_model,
                args=self.hparams
            )
            self.ema_model = ema_model

        elif self.hparams.pretrained_model:
            self.model, ema_model, ckpt = load_model(
                self.hparams.pretrained_model,
                args=self.hparams,
                mean=mean,
                std=std
            )
            
            if self.hparams.use_ema_model:  # Load pretrained ema model instead
                self.model = ema_model
        else:
            self.model = create_model(self.hparams, prior_model, mean=mean, std=std)

        if self.pretraining:
            # Freeze atom embedding layer
            self.model.pretrain()
            self.hparams.weight_decay_on_head = False
        else:
            # Make sure no layers are frozen
            self.model.finetune()

        # Determine graph creation parameters
        if self.hparams.allow_periodic:
            # Use slower graph generation that handles periodic boundary conditions
            self.model.rep_model.legacy = False
        else:
            if not self.hparams.noise_in_loader:
                # Default to torchnet-md optimized graph creation (ignores existing edges in batch)
                self.model.rep_model.legacy = True

        # Set up droppath parameters
        self.droppath_warmup = self.hparams.droppath_warmup
        self.init_droppath = self.hparams.init_droppath
        self.final_droppath = self.hparams.final_droppath
        self.droppath_warmup_steps = self.hparams.lr_warmup_steps + 2000
        self.update_droppath(step=self.global_step)

        # Check if model has derivative turned on
        if self.hparams.derivative is None:
            self.hparams.derivative = self.model.derivative
            if self.hparams.derivative is None:
                self.hparams.derivative = False

        # Initialize exponential smoothing
        self.ema = None
        self._reset_ema_dict()

        # Initialize loss collection
        self.losses = None
        self._reset_losses_dict()

        self.noise_init = self.noise_scale
        self.noise_final = self.noise_scale
        self.noise_steps = 10000

        self.aug_batch = GraphAugmenter(
            rand_translate=0.0,
            rand_rotate=0.0,
            pos_shear_scale=0.0,
            pos_noise_scale=self.noise_scale,
            pos_noise_ratio=1.0,
            noise_type='gaussian',
            joint_mask=False,
            z_mask_id=0,
            z_mask_ratio=0.0,
            z_id_ignore_mask=2,
            center=True,
            recenter=False,
            device=None
        )
        
        #create a batch clipper
        max_nodes = self.hparams.max_nodes_per_batch
        max_cache_size = self.hparams.batch_clipper_cache_size

        if not self.hparams.noise_in_loader:
            #the all augmentations/noiseing is handled by batch noiser
            print('--- Creating Standard BatchClipper ---')
            self.batch_clipper = BatchClipper(max_nodes=max_nodes, max_cache_samples=max_cache_size)
        else: 
            if not self.self_cond:
                # then either simple noise-pretraining or basic finetuning
                print('--- Creating Standard BatchClipper ---')
                self.batch_clipper = BatchClipper(max_nodes=max_nodes, max_cache_samples=max_cache_size)
            else:
                #could be SCD pretraining, or tuple inputs for supervised task 
                print('--- Creating TupleBatchClipper for self-conditioned model with noise in loader ---')
                self.batch_clipper = TupleBatchClipper(max_nodes=max_nodes, max_cache_samples=max_cache_size)

        if max_nodes is None:
            print('--- Batch clipping disabled ---')
            self.batch_clipper.active = False
        else:
            print(f'--- Batch clipping enabled: max nodes per batch = {max_nodes} ---')
        
        self.clip_on_test = self.hparams.allow_test_clipping
        if self.clip_on_test:
            print(f'--- NOTE: Batch clipping enabled on val/test set')
            if not self.pretraining:
                print(f'!!!!!! WARNING: Batch clipping on test/val may lead to inconsistent results during finetuning!!!!!!!')
        
        self.val_ema_model = self.hparams.val_ema_model # if True use ema model for validation
        self.use_momentum_update = self.hparams.momentum_update
        self.l = self.hparams.model_momentum # 0.996
        self.ema_model = None
        if self.use_momentum_update:
            if self.ema_model is None:
                self.ema_model = deepcopy(self.model)
            for param in self.ema_model.parameters():
                param.requires_grad = False
            self.ema_model.eval()
            print('--- Using momentum encoder with EMA model ---')
            if self.val_ema_model:
                print('--- Using EMA model for validation ---')


        use_contrastive_loss = self.hparams.create_contrastive_loss
        self.ctr_loss_weight = self.hparams.contrastive_loss_weight

        self.ctr_loss = None
        if use_contrastive_loss:
            self.ctr_loss = ContrastiveLoss(dim=self.model.emb_dim,
                                            hidden_dim=2048,
                                            proj_dim=128,
                                            learn_tau=True,
                                            tau=self.hparams.ctr_init_tau if hasattr(self.hparams, 'ctr_init_tau') else 0.1,
                                            )

    def on_load_checkpoint(self, checkpoint):
        if self.hparams.restart:
            print(f'-- Restoring global step: {checkpoint["global_step"]} ---')
            self.update_droppath(step=checkpoint['global_step'], log=False)
            dp_stat = self.model.rep_model.attention_layers[0].joint_droppath.drop_prob
            print(f"----- refreshed droppath rate: {dp_stat} -----")

    def reset_stored_outputs(self):
        self.step_outputs = {
            'train_y_true': [],
            'train_y_pred': [],
            'val_y_true': [],
            'val_y_pred': [],
            'test_y_true': [],
            'test_y_pred': [],
            }

    def log_output_corr(self, stage):
        if len(self.step_outputs[f'{stage}_y_true']) == 0:
            #if no stored outputs, just skip
            # print(f'{stage}: No stored outputs to compute correlation')
            return
        
        # Concatenate local predictions and targets
        y_true_local = torch.cat(self.step_outputs[f'{stage}_y_true'], dim=0)
        y_pred_local = torch.cat(self.step_outputs[f'{stage}_y_pred'], dim=0)
        
        ### TODO verify this is correct for distributed training ###
        # Gather from all processes if using distributed training
        if self.trainer.world_size > 1:
            # all_gather returns a list of tensors, one from each process
            y_true_gathered = self.all_gather(y_true_local)
            y_pred_gathered = self.all_gather(y_pred_local)
            
            # Flatten the gathered tensors
            # all_gather returns shape [world_size, batch_size, ...] 
            # We need to flatten to [world_size * batch_size, ...]
            y_true = y_true_gathered.reshape(-1, *y_true_gathered.shape[2:])
            y_pred = y_pred_gathered.reshape(-1, *y_pred_gathered.shape[2:])
        else:
            # Single GPU/CPU - no gathering needed
            y_true = y_true_local
            y_pred = y_pred_local
        
        # Now compute correlation on the complete dataset
        corr_dict = compute_corr(y_pred, y_true)
        
        # Log the metrics (only on rank 0 to avoid duplicates)
        if self.trainer.is_global_zero:
            for key, value in corr_dict.items():
                self.log(f'{stage}_{key}', value, prog_bar=True, sync_dist=False, rank_zero_only=True)
            
            #log number of samples used in correlation for debugging check
            corr_len = len(y_true)
            self.log(f'{stage}_corr_len', corr_len, prog_bar=True, sync_dist=False, rank_zero_only=True)
        
        return

    def log_allcorr(self):
        # log correlation for all stages - if any outputs are stored
        self.log_output_corr('train')
        self.log_output_corr('val')
        self.log_output_corr('test')


    @torch.no_grad()
    def momentum_update(self):
        """Momentum update of the EMA model"""
        if not self.use_momentum_update or self.ema_model is None:
            return
        
        # momentum update of the key encoder
        for param_q, param_k in zip(self.model.parameters(), self.ema_model.parameters()):
            param_k.data = param_k.data * self.l + param_q.data* (1. - self.l)
  
    def configure_optimizers(self):

        beta1 = self.hparams.beta1 if hasattr(self.hparams, 'beta1') else 0.9
        beta2 = self.hparams.beta2 if hasattr(self.hparams, 'beta2') else 0.999
        betas = (beta1, beta2) # FIXME testing

        all_param_groups = []

        if self.ctr_loss is not None and self.pretraining and self.self_cond:
            all_param_groups += self.ctr_loss.get_parameter_groups(weight_decay=self.hparams.weight_decay)
            print(f'--- Using contrastive loss with proj dim {self.ctr_loss.proj_head.mlp[-1].out_features} ---')

        if not hasattr(self.model, 'get_parameter_groups'):
            print('WARNING: model has no get_parameter_groups function, using all parameters for optimization')
            
            #add model parameters to all param groups
            all_param_groups.append({'params': self.model.parameters(), 'weight_decay': self.hparams.weight_decay})

            optimizer = AdamW(
                all_param_groups,
                lr=self.hparams.lr,
                betas=betas, # keep betas default
            )
            
        else:
            print('-- Using grouped weights for optimization --')
            param_group = self.model.get_parameter_groups(weight_decay=self.hparams.weight_decay, weight_decay_on_head=self.hparams.weight_decay_on_head)
            
            all_param_groups += param_group
            
            optimizer = AdamW(
                # param_group,
                all_param_groups,
                lr=self.hparams.lr,
                betas=betas, # FIXME testing
            )
            # print(f'!!!!!!!! TESTING: altered betas {betas} !!!!!!!!')

        if self.hparams.lr_schedule == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, self.hparams.lr_cosine_length)
            lr_scheduler = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        elif self.hparams.lr_schedule == 'reduce_on_plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                "min",
                factor=self.hparams.lr_factor,
                patience=self.hparams.lr_patience,
                min_lr=self.hparams.lr_min,
            )
            lr_scheduler = {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            }
        else:
            raise ValueError(f"Unknown lr_schedule: {self.hparams.lr_schedule}")
        return [optimizer], [lr_scheduler]
    
    def double_forward(self, batch_emb, batch_pred, stage=None):
        out_emb = self.forward(batch_emb, stage=stage, cond=None)
        out_pred = self.forward(batch_pred, stage=stage, cond=out_emb['mol_emb'])
        return out_emb, out_pred

    def forward(self, batch, stage=None, cond=None):
        z = batch.z
        pos = batch.pos
        batch_idx = batch.batch

        kwargs = {}
        model_inputs = {
            'z': z,
            'pos': pos,
            'batch': batch_idx,
            'graph_batch': None,
            'cond': cond,
            }

        if self.hparams.allow_periodic or self.hparams.noise_in_loader:
            model_inputs['graph_batch'] = batch
       
        if self.hparams.allow_node_mask:
            ## place any masking logic here
            if batch.get('ligand_mask', None) is not None:
                kwargs['atom_mask'] = batch.ligand_mask
            
            # inverse the mask if specified
            if self.hparams.inverse_node_mask:
                kwargs['atom_mask'] = (~kwargs['atom_mask'].bool()).float()
            

        if not self.use_momentum_update or not self.val_ema_model:
            # return self.model(z, pos, batch=batch_idx, graph_batch=graph_batch, **kwargs)
            return self.model(**model_inputs, **kwargs)
        else:
            assert self.ema_model is not None, 'EMA model not initialized!'
            if stage is None or stage == 'train': # default to train
                # return self.model(z, pos, batch=batch_idx, graph_batch=graph_batch, **kwargs)
                return self.model(**model_inputs, **kwargs)
            elif stage == 'val' or stage == 'test':
                with torch.no_grad():
                    # return self.ema_model(z, pos, batch=batch_idx, graph_batch=graph_batch, **kwargs)
                    return self.ema_model(**model_inputs, **kwargs)

    def training_step(self, batch, batch_idx):
        return self.step(batch, mse_loss, "train")

    def validation_step(self, batch, batch_idx, *args):
        #check if batch object is a tuple (from noise_in_loader)
        if isinstance(batch, list):
            batch_0, _ = batch
            batch_size = batch_0.num_graphs if hasattr(batch_0, 'num_graphs') else len(batch_0.batch.unique())
        else:
            batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else len(batch.batch.unique())
            
        if len(args) == 0 or (len(args) > 0 and args[0] == 0):
            # validation step
            loss = self.step(batch, mse_loss, "val")

            ###FIXME
            self.log(
                'val_loss',
                loss,
                on_step=True,     # only aggregate over the epoch
                on_epoch=True,     # produce an epoch-level val_loss
                prog_bar=True,
                sync_dist=True,    # important under DDP
                batch_size= batch_size, #batch.num_graphs if hasattr(batch, 'num_graphs') else len(batch.batch.unique())
                )

            return loss
        # test step
        loss = self.step(batch, l1_loss, "test")
        self.log(
            'test_loss',
            loss,
            on_step=False,     # only aggregate over the epoch
            on_epoch=True,     # produce an epoch-level test_loss
            prog_bar=True,
            sync_dist=True,    # important under DDP
            batch_size=batch_size, #batch.num_graphs if hasattr(batch, 'num_graphs') else len(batch.batch.unique())
        )
        return loss

    def test_step(self, batch, batch_idx):

        if self.hparams.derivative:
            self.model.train()
            torch.set_grad_enabled(True)

        loss = self.step(batch, l1_loss, "test")

        self.log(
            'test_loss',
            loss,
            on_step=False,     # only aggregate over the epoch
            on_epoch=True,     # produce an epoch-level test_loss
            prog_bar=True,
            sync_dist=True,    # important under DDP
        )

        return loss
    
    def force_denoise(self, batch, reg_noise_scale=0.0):
        ### This function is only used for the MD17 benhcmark to match previous work

        #### create two views, clean and corrupted ####
        x1, x1_mask, aug_dict1 = self.aug_batch(batch)

        ### add regularization noise to clean sample ###
        if reg_noise_scale > 0.0:
            x1.pos = x1.pos + torch.randn_like(x1.pos)*reg_noise_scale

        ### first pass to get embedding of clean structure ###
        out_clean = self.model(x1.z, x1.pos, x1.batch, cond=None)
        
        ### pass embedding of clean structure as condition to corrupted structure ###
        # out_corrupted = self.model(x1_mask.z, x1_mask.pos, x1_mask.batch, cond=out_clean['mol_emb'])
        out_corrupted = self.model(x1_mask.z, x1_mask.pos, x1_mask.batch, cond=None)

        return out_clean, out_corrupted, aug_dict1, x1, x1_mask
    

    def scd_step(self, batch, loss_fn, stage, return_outputs=False): # TODO: rename this function to scd_ssl_step
        
        # --------------------- scd forward ---------------------
        #remove noise if already added
        # if hasattr(batch, 'pos_target'):
        #     batch.pos = batch.pos - batch.pos_target
        if self.hparams.noise_in_loader:
            x1, x1_mask = batch
            aug_dict1 = {'pos_noise': x1_mask.noise}

        else:
            x1, x1_mask, aug_dict1 = self.aug_batch(batch)

            ##### add regularization noise to x1
            reg_noise_scale = self.reg_noise_scale #0.005
            x1.pos = x1.pos + torch.randn_like(x1.pos)*reg_noise_scale
            ##### add regularization noise to x1

        if self.hparams.allow_periodic or self.hparams.noise_in_loader:
            g_batch_clean = x1
            g_batch_mask = x1_mask
        else:
            g_batch_clean = None
            g_batch_mask = None

        target_noise = aug_dict1['pos_noise']
        
        target_out = self.model(x1.z, x1.pos, x1.batch, cond=None, graph_batch=g_batch_clean)
        # mol_emb = target_out['mol_emb']

        s_out = self.model(x1_mask.z, x1_mask.pos, x1_mask.batch, 
                            cond=target_out['mol_emb'],
                            graph_batch=g_batch_mask
                            )
        # --------------------- scd forward ---------------------
        ### alternative simpler implementation ###
        # target_out, s_out, aug_dict1, x1, x1_mask = self.scd_forward(batch, reg_noise_scale=self.reg_noise_scale)
        # target_noise = aug_dict1['pos_noise'] 
        ### alternative simpler implementation ###

        pred = s_out['y']
        noise_pred = s_out['noise_pred']

        # "use" both outputs of the model's forward (see comment above).
        noise_pred = noise_pred + pred.sum() * 0 
        normalized_pos_target = self.model.noise_normalizer(target_noise, update=True)
        
        loss_pos = loss_fn(noise_pred, normalized_pos_target)

        pred_pos = x1_mask.pos - self.model.noise_normalizer.inverse(noise_pred)
        true_pos = x1_mask.pos - target_noise
        pos_mae = torch.nn.functional.l1_loss(pred_pos, true_pos)

        self.losses[stage + "_pos"].append(loss_pos.detach())

        loss = loss_pos
        self.losses[stage].append(loss.detach())

        ##### TODO: Add contrastive loss on mol_emb #####
        if self.ctr_loss is not None and stage == 'train' and self.ctr_loss_weight > 0.0:
            clean_emb = target_out['mol_emb']
            corrupted_emb = s_out['mol_emb']

            contrastive_loss = self.ctr_loss(clean_emb, corrupted_emb)
            ctr_temp = self.ctr_loss.get_tau().item()

            #add contrastive loss to total loss
            loss += self.ctr_loss_weight * contrastive_loss


        ##### TODO: Add contrastive loss on mol_emb #####

        # Frequent per-batch logging for training
        if stage == 'train':
            train_metrics = {k + "_per_step": v[-1] for k, v in self.losses.items() if (k.startswith("train") and len(v) > 0)}
            train_metrics['lr_per_step'] = self.trainer.optimizers[0].param_groups[0]["lr"]
            train_metrics['step'] = self.trainer.global_step   
            # train_metrics['batch_pos_mean'] = batch.pos.mean().item()

            train_metrics['loss_denoise'] = loss_pos.item() 
            train_metrics['pos_noise'] = self.aug_batch.pos_noise_scale

            train_metrics['pos_mae'] = pos_mae.item()

            #track batch clipping stats
            batch_clip_stats = self.batch_clipper.get_stats_dict()
            train_metrics['batch_clipper_cache_size'] = batch_clip_stats['current_cache_size']
            train_metrics['batch_clipper_cache_nodes'] = batch_clip_stats['current_cache_nodes']
            train_metrics['batch_avg_nodes'] = batch_clip_stats['avg_nodes_per_batch']
            train_metrics['batch_std_nodes'] = batch_clip_stats['std_nodes_per_batch']
            if isinstance(batch, list):
                batch_0, _ = batch
                batch_size = batch_0.num_graphs if hasattr(batch_0, 'num_graphs') else len(batch_0.batch.unique())
            else:
                batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else len(batch.batch.unique())
            train_metrics['batch_num_graphs'] = batch_size


            if self.ctr_loss is not None:
                #log contrastive loss and tau
                train_metrics['loss_contrastive'] = contrastive_loss.item()
                train_metrics['tau'] = ctr_temp

            self.log_dict(train_metrics, sync_dist=True)
        
        if return_outputs:
            # For plotting/eval: construct output_dict from s_out
            output_dict = s_out
            x1_mask.noise = target_noise
            return output_dict, x1_mask
        
        return loss

    def step(self, batch, loss_fn, stage, return_outputs=False):
        #### Debug check ####
        # if hasattr(batch, 'pos_target'):
        #     # batch.pos = batch.pos - batch.pos_target
        #     assert False, 'pos_target found in batch!!?'

        #batch clipping
        is_train = stage == 'train'
        if is_train or self.clip_on_test: 
            batch = self.batch_clipper(batch, use_cache=is_train)

        denoising_is_on = False
        if self.pretraining: ####  do denoising only pretraining step
            if self.self_cond:
                # do Self conditioned denoising pass
                loss = self.scd_step(batch, loss_fn, stage, return_outputs=return_outputs)
                return loss
            
            else:# Do regular denoising pass
                #### corrupt inputs ####
                assert self.noise_scale > 0.0, 'noise_scale must be > 0 if denoising pretraining!'

                if self.hparams.noise_in_loader:
                    #noise target is already in batch
                    # if self.noise_scale > 0.0:
                    # assert self.noise_scale > 0.0, 'noise_scale must be > 0 if noise_in_loader is set!'
                    assert hasattr(batch, 'noise'), 'pos_noise not found in batch despite noise_in_loader being set!'
                    denoising_is_on = True

                else:
                    _, batch, aug_dict = self.aug_batch(batch)
                    if self.noise_scale > 0.0:
                        batch.noise = aug_dict['pos_noise']
                        denoising_is_on = True
                ### forward pass ###
                # output_dict = self(batch.z, batch.pos, batch.batch)
                output_dict = self(batch, stage=stage)
        
        else: #### Supervised finetuning step
            #TODO: refactor to reduce code duplication/complexity
            
            if stage == "train":
                if self.self_cond: 
                    #### do Self conditioned denoising  to seporate noise prediction ####
                    assert self.noise_scale > 0.0, 'Self conditioned denoising requires noise augmentation!'
                    assert self.hparams.denoising_weight > 0.0, 'Self conditioned denoising requires denoising weight to be non-zero!'
                    
                    reg_noise = 0.0 # self.reg_noise_scale
                    denoising_is_on = True

                    if self.hparams.derivative:
                        # for force denoising only 
                        out_clean, out_corrupted, aug_dict, x_clean, x_corrupted = self.force_denoise(batch, reg_noise_scale=0.0)
                        batch.noise = aug_dict['pos_noise'] 
                        batch.pos = x_corrupted.pos # set pos to noised pos for denoising loss monitoring
                        ### reformulate output using clean and corrupted pass ###
                        output_dict = {
                            'y' : out_clean['y'],
                            'noise_pred' : out_corrupted['noise_pred'],
                            'dy' : out_clean['dy']
                            }
                    else:
                        if self.hparams.noise_in_loader:
                            #throw not implemented error for now
                            # raise NotImplementedError('Self conditioned denoising with noise_in_loader not implemented yet!')
                            x_clean, x_corrupt = batch
                        else:
                            x_clean, x_corrupt, aug_dict = self.aug_batch(batch)
                            x_corrupt.noise = aug_dict['pos_noise']


                        out_emb, out_pred = self.double_forward(x_clean, x_corrupt, stage=stage) 
                        
                        ###experimental - use output from second pass for y prediction 
                        # out_clean['y'] = out_corrupted['y']
                        batch = x_corrupt
                        output_dict = out_pred
                    

                        # out_clean, out_corrupted, aug_dict, x_clean, x_corrupted = self.scd_forward(batch, reg_noise_scale=reg_noise)
                    
                    # target_noise = aug_dict['pos_noise'] 
                    # noised_pos = batch.pos + aug_dict['pos_noise']
                    # batch.pos = noised_pos # set pos to noised pos for denoising loss monitoring
                    # batch.noise = aug_dict['pos_noise'] 
                    # batch.pos = x_corrupted.pos # set pos to noised pos for denoising loss monitoring
                   
                    ### reformulate output using clean and corrupted pass ###
                    # output_dict = {
                    #     'y' : out_clean['y'],
                    #     'noise_pred' : out_corrupted['noise_pred'],
                    #     'dy' : out_clean['dy']
                    #     }
                    
                # elif self.match_ema:
                #     output_dict, node_align_loss = self.ema_matching_fwd(batch, stage)
                #     denoising_is_on = False
                # elif self.forward_forces:
                #     denoising_is_on = False
                #     output_dict = self(batch.z, batch.pos, batch.batch)
                #     ema_output = self.ema_model(batch.z, batch.pos, batch.batch)

                else: # do regular forward pass
                    #### corrupt inputs ####
                    if self.hparams.noise_in_loader:
                        assert hasattr(batch, 'noise'), 'pos_noise not found in batch despite noise_in_loader being set!'
                        denoising_is_on = True

                    else:
                        _, batch, aug_dict = self.aug_batch(batch)
                        if self.noise_scale > 0.0:
                            batch.noise = aug_dict['pos_noise']
                            denoising_is_on = True
                    
                    ### forward pass ###
                    # output_dict = self(batch.z, batch.pos, batch.batch)
                    output_dict = self(batch, stage=stage)
                    pass
            else:
                if self.self_cond:
                    #EXPERIMENTAL: double pass
                    if self.hparams.noise_in_loader:
                        #throw not implemented error for now
                        # raise NotImplementedError('Self conditioned denoising with noise_in_loader not implemented yet!')
                        # x_clean, x_corrupt = batch
                        x_clean = batch
                    else:
                        x_clean = batch

                    out_emb, out_pred = self.double_forward(x_clean, x_clean, stage=stage) 
                    output_dict = out_pred
                
                else:
                    # is in val/test stage, do regular noiseless pass
                    with torch.set_grad_enabled(self.hparams.derivative):
                        # output_dict = self(batch.z, batch.pos, batch.batch)
                        output_dict = self(batch, stage=stage)

        
        ### collect outputs ###
        pred = output_dict['y']
        noise_pred = output_dict.get('noise_pred', None)
        deriv = output_dict.get('dy', None)
        
        if return_outputs:
            #used for plotting and eval purposes
            return output_dict, batch


        loss_y, loss_dy, loss_pos = 0, 0, 0
        loss_dy_direct = 0
        

        # check if dy is present in batch
        has_dy = hasattr(batch, 'dy') and batch.dy is not None
        dy_is_target = self.hparams.derivative or self.hparams.direct_force_pred
        
        if has_dy and dy_is_target:
            if "y" not in batch:
                # "use" both outputs of the model's forward function but discard the first
                # to only use the derivative and avoid 'Expected to have finished reduction
                # in the prior iteration before starting a new one.', which otherwise get's
                # thrown because of setting 'find_unused_parameters=False' in the DDPPlugin
                deriv = deriv + pred.sum() * 0
    
            if self.hparams.derivative:
                # force/derivative loss
                loss_dy = loss_fn(deriv, batch.dy)
                loss_dy_l1 = F.l1_loss(deriv.detach(), batch.dy).item()

                if stage in ["train", "val"] and self.hparams.ema_alpha_dy < 1:
                    if self.ema[stage + "_dy"] is None:
                        self.ema[stage + "_dy"] = loss_dy.detach()
                    # apply exponential smoothing over batches to dy
                    loss_dy = (
                        self.hparams.ema_alpha_dy * loss_dy
                        + (1 - self.hparams.ema_alpha_dy) * self.ema[stage + "_dy"]
                    )
                    self.ema[stage + "_dy"] = loss_dy.detach()

                if self.hparams.force_weight > 0:
                    self.losses[stage + "_dy"].append(loss_dy.detach())
                    self.losses[stage + "_dy_l1"].append(loss_dy_l1)
            
            #TODO: update ema dict
            #TODO: update logging dict
            if self.hparams.direct_force_pred:
                denoising_is_on = False # turn off noise prediction for direct force pred

                # direct force prediction loss
                loss_dy_direct = loss_fn(noise_pred, batch.dy)
                loss_dy_direct_l1 = F.l1_loss(noise_pred.detach(), batch.dy).item()

                if stage in ["train", "val"] and self.hparams.ema_alpha_dy < 1:
                    if self.ema[stage + "_dy_direct"] is None:
                        self.ema[stage + "_dy_direct"] = loss_dy_direct.detach()
                    # apply exponential smoothing over batches to dy
                    loss_dy_direct = (
                        self.hparams.ema_alpha_dy * loss_dy_direct
                        + (1 - self.hparams.ema_alpha_dy) * self.ema[stage + "_dy_direct"]
                    )
                    self.ema[stage + "_dy_direct"] = loss_dy_direct.detach()

                if self.hparams.force_weight > 0:
                    self.losses[stage + "_dy_direct"].append(loss_dy_direct.detach())
                    self.losses[stage + "_dy_direct_l1"].append(loss_dy_direct_l1)


        if "y" in batch:
            if (noise_pred is not None) and not denoising_is_on:
                # "use" both outputs of the model's forward (see comment above).
                pred = pred + noise_pred.sum() * 0

            if batch.y.ndim == 1:
                batch.y = batch.y.unsqueeze(1)

            # track model outputs for correlation computation
            if self.store_outputs:
                self.step_outputs[f'{stage}_y_true'].append(batch.y.detach().cpu())
                self.step_outputs[f'{stage}_y_pred'].append(pred.detach().cpu())

            # energy/prediction loss
            loss_y = loss_fn(pred, batch.y)

            loss_y_l1 = F.l1_loss(pred.detach(), batch.y).item() # for logging only
            self.losses[stage + "_y_l1"].append(loss_y_l1)

            if stage in ["train", "val"] and self.hparams.ema_alpha_y < 1:
                if self.ema[stage + "_y"] is None:
                    self.ema[stage + "_y"] = loss_y.detach()
                # apply exponential smoothing over batches to y
                loss_y = (
                    self.hparams.ema_alpha_y * loss_y
                    + (1 - self.hparams.ema_alpha_y) * self.ema[stage + "_y"]
                )
                self.ema[stage + "_y"] = loss_y.detach()

            if self.hparams.energy_weight > 0:
                self.losses[stage + "_y"].append(loss_y.detach())

        if denoising_is_on:
            if "y" not in batch:
                # "use" both outputs of the model's forward (see comment above).
                noise_pred = noise_pred + pred.sum() * 0
            
            #NOTE: updateing position normalizer may be off in some variants
            # normalized_pos_target = self.model.pos_normalizer(batch.pos_target)
            normalized_noise = self.model.noise_normalizer(batch.noise, update=self.update_normalizer)
            loss_pos = loss_fn(noise_pred, normalized_noise)
            self.losses[stage + "_pos"].append(loss_pos.detach())

            #compute mae with predicted position
            tru_pos = batch.pos - batch.noise
            pred_pos = batch.pos - self.model.noise_normalizer.inverse(noise_pred)
            pos_mae = torch.nn.functional.l1_loss(pred_pos, tru_pos)
            # self.losses[stage + "_pos_mae"].append(pos_mae.detach())

        # total loss
        loss = loss_y * self.hparams.energy_weight + loss_dy * self.hparams.force_weight + loss_pos * self.hparams.denoising_weight
        loss += loss_dy_direct * self.hparams.force_weight

        # if self.match_ema and stage == 'train':
        #     loss = loss + node_align_loss * self.match_ema_weight
            # self.losses[stage + "_ema_align_loss"].append(node_align_loss.detach())

        self.losses[stage].append(loss.detach())


        # Frequent per-batch logging for training
        if stage == 'train':
            train_metrics = {k + "_per_step": v[-1] for k, v in self.losses.items() if (k.startswith("train") and len(v) > 0)}
            train_metrics['lr_per_step'] = self.trainer.optimizers[0].param_groups[0]["lr"]
            train_metrics['step'] = self.trainer.global_step   
            train_metrics['batch_pos_mean'] = batch.pos.mean().item()
            # train_metrics['dimr_inv_mol'] = dimr_loss.item() if isinstance(dimr_loss, torch.Tensor) else dimr_loss
            train_metrics['loss_y'] = loss_y.item() if isinstance(loss_y, torch.Tensor) else loss_y
            train_metrics['loss_dy'] = loss_dy.item() if isinstance(loss_dy, torch.Tensor) else loss_dy
            train_metrics['loss_dy_direct'] = loss_dy_direct.item() if isinstance(loss_dy_direct, torch.Tensor) else loss_dy_direct
            train_metrics['loss_pos'] = loss_pos.item() if isinstance(loss_pos, torch.Tensor) else loss_pos

            #track batch clipping stats
            batch_clip_stats = self.batch_clipper.get_stats_dict()
            train_metrics['batch_clipper_cache_size'] = batch_clip_stats['current_cache_size']
            train_metrics['batch_clipper_cache_nodes'] = batch_clip_stats['current_cache_nodes']
            train_metrics['batch_avg_nodes'] = batch_clip_stats['avg_nodes_per_batch']
            train_metrics['batch_std_nodes'] = batch_clip_stats['std_nodes_per_batch']

            if isinstance(batch, list):
                batch_0, _ = batch
                batch_size = batch_0.num_graphs if hasattr(batch_0, 'num_graphs') else len(batch_0.batch.unique())
                batch_num_nodes = batch_0.num_nodes
            else:
                batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else len(batch.batch.unique())
                batch_num_nodes = batch.num_nodes
            train_metrics['batch_num_graphs'] = batch_size
            train_metrics['batch_num_nodes'] = batch_num_nodes

            # if self.match_ema:
            #     train_metrics['loss_ema_align'] = node_align_loss.item()

            # if "y" in batch:
            #     train_metrics['loss_y_l1'] = loss_y_l1

            if denoising_is_on:
                train_metrics['pos_mae'] = pos_mae.item()

            if self.pretraining:
                if not self.hparams.noise_in_loader:
                    train_metrics['pos_noise'] = self.aug_batch.pos_noise_scale
                else:
                    ## TODO
                    #Logging pos_noise not implemented for noise_in_loader yet!
                    pass
            

            self.log_dict(train_metrics, sync_dist=True)
        
        elif stage == 'val' or stage == 'test':
            
            val_metrics = {k + "_per_step": v[-1] for k, v in self.losses.items() if (k.startswith(stage) and len(v) > 0)}
            val_metrics['step'] = self.trainer.global_step
        
            self.log_dict(val_metrics, sync_dist=True)
            pass

        return loss

    def update_droppath(self, step=None, log=True):
        
        #debugging check
        if log:
            dp_stat = self.model.rep_model.attention_layers[0].joint_droppath.drop_prob
            #track on current drop path prob on wandb
            self.log('droppath_prob', dp_stat, sync_dist=True)
        
        if not self.droppath_warmup:
            # use default set by model 
            return
        
        if step is not None:
            #force refresh at given step
            global_step = step
        elif hasattr(self, 'trainer') and self.trainer is not None: 
            global_step = self.trainer.global_step
        else:
            global_step = self.global_step

        if global_step < self.droppath_warmup_steps:
            new_droppath = self.init_droppath - (self.init_droppath - self.final_droppath) * (global_step / self.droppath_warmup_steps)
            #ensure drop path is within bounds
            new_droppath = max(min(new_droppath, self.init_droppath), self.final_droppath)
            rep_model = self.model.rep_model
            for layer in rep_model.attention_layers:
                layer.set_droppath(new_droppath)

        elif global_step == self.droppath_warmup_steps or step is not None:
            rep_model = self.model.rep_model
            for layer in rep_model.attention_layers:
                layer.set_droppath(self.final_droppath)
    
    def optimizer_step(self, *args, **kwargs):
        optimizer = kwargs["optimizer"] if "optimizer" in kwargs else args[2]
        if self.trainer.global_step < self.hparams.lr_warmup_steps:
            lr_scale = min(
                1.0,
                float(self.trainer.global_step + 1)
                / float(self.hparams.lr_warmup_steps),
            ) 

            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.lr
        
        if self.trainer.global_step == self.hparams.lr_warmup_steps:
            #reset momentum at peak of warmup
            pass
            # #FIXME: experimental
            # for group in optimizer.param_groups:
            #     for param in group['params']:
            #         state = optimizer.state[param]
            #         if 'exp_avg' in state:
            #             state['exp_avg'].zero_()
            #         if 'exp_avg_sq' in state:
            #             state['exp_avg_sq'].zero_()

        super().optimizer_step(*args, **kwargs)

        if self.use_momentum_update:
            self.momentum_update()
        
        if self.droppath_warmup:
            self.update_droppath()

        optimizer.zero_grad()

    def on_train_epoch_end(self):
        # This method is intentionally left empty as reset_val_dataloader is not available in PyTorch Lightning 2.x
        # Lightning automatically handles dataloader management
        if self.hparams.compute_corr:
            self.log_allcorr()
        self.reset_stored_outputs()

        #clear batch clipper cache at epoch end
        self.batch_clipper.reset()


    def save_best_checkpoint(self):
        #check if self has attribute trainer
        if not hasattr(self, "trainer"):
            print("--- WARNING: No trainer found, cannot save best checkpoint.")
            return
        if self.trainer is None:
            print("--- WARNING: Trainer is None, cannot save best checkpoint.")
            return
            
    def on_validation_epoch_end(self):

        if self.hparams.compute_corr:
            self.log_allcorr()
        self.reset_stored_outputs()

        self.batch_clipper.reset()

        # construct dict of logged metrics
        result_dict = {
            "epoch": self.current_epoch,
            "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
        }
        
        # Only log losses if they exist (not during sanity check)
        if len(self.losses["train"]) > 0:
            result_dict["train_loss"] = torch.stack(self.losses["train"]).mean()
        if len(self.losses["val"]) > 0:
            val_loss = torch.stack(self.losses["val"]).mean()
            self.log("val_loss", val_loss, sync_dist=True)
            result_dict["val_loss"] = val_loss

        # add test loss if available
        if len(self.losses["test"]) > 0:
            result_dict["test_loss"] = torch.stack(self.losses["test"]).mean()

        # if prediction and derivative are present, also log them separately
        if len(self.losses["train_y"]) > 0 and len(self.losses["train_dy"]) > 0:
            result_dict["train_loss_y"] = torch.stack(self.losses["train_y"]).mean()
            result_dict["train_loss_dy"] = torch.stack(
                self.losses["train_dy"]
            ).mean()
            result_dict["val_loss_y"] = torch.stack(self.losses["val_y"]).mean()
            result_dict["val_loss_dy"] = torch.stack(self.losses["val_dy"]).mean()

            if len(self.losses["test"]) > 0:
                result_dict["test_loss_y"] = torch.stack(
                    self.losses["test_y"]
                ).mean()
                result_dict["test_loss_dy"] = torch.stack(
                    self.losses["test_dy"]
                ).mean()

        if len(self.losses["train_y"]) > 0:
            result_dict["train_loss_y"] = torch.stack(self.losses["train_y"]).mean()
        if len(self.losses['val_y']) > 0:
          result_dict["val_loss_y"] = torch.stack(self.losses["val_y"]).mean()
        if len(self.losses["test_y"]) > 0:
            result_dict["test_loss_y"] = torch.stack(
                self.losses["test_y"]
            ).mean()

        # if denoising is present, also log it
        if len(self.losses["train_pos"]) > 0:
            result_dict["train_loss_pos"] = torch.stack(
                self.losses["train_pos"]
            ).mean()

        if len(self.losses["val_pos"]) > 0:
            result_dict["val_loss_pos"] = torch.stack(
                self.losses["val_pos"]
            ).mean()

        if len(self.losses["test_pos"]) > 0:
            result_dict["test_loss_pos"] = torch.stack(
                self.losses["test_pos"]
            ).mean()

        # Only log if we have any metrics to log
        if len(result_dict) > 2:  # More than just epoch and lr
            self.log_dict(result_dict, sync_dist=True)

        self._reset_losses_dict()

    def _reset_losses_dict(self):
        self.losses = {
            "train": [],
            "val": [],
            "test": [],

            "train_y": [],
            "val_y": [],
            "test_y": [],

            "train_y_l1": [],
            "val_y_l1": [],
            "test_y_l1": [],

            "train_dy": [],
            "val_dy": [],
            "test_dy": [],

            "train_dy_l1": [],
            "val_dy_l1": [],
            "test_dy_l1": [],

            "train_dy_direct": [],
            "val_dy_direct": [],
            "test_dy_direct": [],

            "train_dy_direct_l1": [],
            "val_dy_direct_l1": [],
            "test_dy_direct_l1": [],

            "train_pos": [],
            "val_pos": [],
            "test_pos": [],
        }

    def _reset_ema_dict(self):
        self.ema = {"train_y": None, 
                    "val_y": None, 
                    "train_dy": None, 
                    "val_dy": None,
                    "train_dy_direct": None,
                    "val_dy_direct": None
                    }
