import numpy as np  # sometimes needed to avoid mkl-service error
import sys

# sys.path.append('/home/tjp/Projects/Model_repos/pre-training-via-denoising')

import os
import argparse
import logging
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

import models
from data import datasets
from data.loaders import DataModule
from data.utils import LoadFromFile, LoadFromCheckpoint, save_argparse, number

from tqdm import tqdm

# from models.trainer import LTrainer
# from models.model_helper import get_model_checkpoint, create_prior_models

from models.trainer2 import LTrainer # new simplified version of models
from models.model_helper2 import get_model_checkpoint, create_prior_models

##TODO: NEW
from models.callbacks import ParityPlot, LimitRun

import torch

from pathlib import Path
import wandb

from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from datetime import timedelta
from datetime import datetime
import time

import shutil

# from models.ET_models.priors import Atomref

# class LimitRun(Callback):
#     """
#     Stop *this run only* after `minutes` from start (no accumulation across restarts).
#     Saves a checkpoint before stopping so the next job can resume.
#     """

#     def __init__(self, 
#                  minutes : int = None, 
#                  max_epochs_per_run: int = None,
#                  out_dir: str = None, 
#                  buffer_sec: int = 30,
#                   ):
#         super().__init__()
#         self.minutes = minutes
#         self.out_dir = out_dir
#         self.buffer_sec = buffer_sec
#         self.max_epochs_per_run = max_epochs_per_run
#         self._t0 = None
#         self._start_epoch = None

#     def on_fit_start(self, trainer, pl_module):
#         # This is called before checkpoint loading
#         self._t0 = time.time()
#         os.makedirs(os.path.join(self.out_dir), exist_ok=True)

#     def on_train_start(self, trainer, pl_module):
#         # This is called AFTER checkpoint loading
#         self._start_epoch = trainer.current_epoch

#         current_steps = trainer.global_step
#         if current_steps < 1:
#             self._start_epoch = -1 

#         # Print what limits are active
#         limits = []
#         if self.minutes is not None:
#             limits.append(f"{self.minutes} minutes")
#         if self.max_epochs_per_run is not None:
#             limits.append(f"{self.max_epochs_per_run} epochs")

#         if len(limits) > 0 and trainer.global_rank == 0:
#             print(f"Will stop this run after: {' OR '.join(limits)} from start epoch {self._start_epoch}")
    
#     def get_filename(self, trainer):
#         step = trainer.global_step
#         epoch = trainer.current_epoch
        
#         filename = f"step={step}-epoch={epoch}.stop_ckpt"
#         return filename

#     def _save_checkpoint_and_stop(self, trainer, reason="time limit"):
#         """Helper method to save checkpoint and stop training"""
        
#         ckpt_dir = os.path.join(self.out_dir)
#         tmp_path = os.path.join(ckpt_dir, self.get_filename(trainer))
#         # trainer.save_checkpoint(tmp_path)

#         # if trainer.global_rank == 0:  # Only rank 0 saves
#             # Optionally update last.ckpt
#             # last_path = os.path.join(ckpt_dir, "last.ckpt")
#             # shutil.copy2(tmp_path, last_path)

#         model_checkpoint_call_found = False
#         for callback in trainer.callbacks:
#             if isinstance(callback, ModelCheckpoint):
#                 # callback._save_last_checkpoint(tmp_path)
                
#                 callback._save_topk_checkpoint(trainer, trainer.logged_metrics)
#                 # Also save last checkpoint if enabled
#                 if callback.save_last:
#                     callback._save_last_checkpoint(trainer, trainer.logged_metrics)

#                 model_checkpoint_call_found = True
#         if not model_checkpoint_call_found:
#             print('WARNING: no model checkpoint callback found!! check point was not saved. no manual save process set up')
        
#         print(f"--- STOPING TRAINING: due to {reason}. Saved checkpoint: {tmp_path}")
        
#         trainer.should_stop = True  # graceful stop at the next safe point
    
#     def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
#         """Check time limit during training"""
#         if self._t0 is None or self.minutes is None:
#             return
            
#         elapsed = time.time() - self._t0
#         limit = max(0, 60 * self.minutes - self.buffer_sec)
#         if elapsed >= limit:
#             self._save_checkpoint_and_stop(trainer, "time limit")
    
#     def on_train_epoch_end(self, trainer, pl_module):
#         """Check epoch limit at the end of each epoch"""
#         if self._start_epoch is None or self.max_epochs_per_run is None:
#             return
            
#         epochs_completed = trainer.current_epoch - self._start_epoch #+ 1
#         # print(f"Completed epochs: {epochs_completed}, current epoch {trainer.current_epoch}, start epoch, {self._start_epoch}")
#         if epochs_completed >= self.max_epochs_per_run:
#             self._save_checkpoint_and_stop(trainer, f"epoch limit ({epochs_completed} epochs completed)")
    
    # def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    #     # Check elapsed wall time from the start of *this* run
    #     if self._t0 is None:
    #         return
    #     elapsed = time.time() - self._t0
    #     limit = max(0, 60 * self.minutes - self.buffer_sec)
    #     if elapsed >= limit:
    #         # Save an emergency checkpoint and request a clean stop
    #         ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    #         # ckpt_dir = os.path.join(self.out_dir, "checkpoints")
    #         ckpt_dir = os.path.join(self.out_dir)
    #         # tmp_path = os.path.join(ckpt_dir, f"last_runstop_{ts}.temp_ckpt")
    #         tmp_path = os.path.join(ckpt_dir, self.get_filename(trainer))
    #         trainer.save_checkpoint(tmp_path)
    #         # also refresh/overwrite a stable "last.ckpt" pointer
    #         last_path = os.path.join(ckpt_dir, "last.ckpt")
    #         # try:
    #         #     if os.path.islink(last_path) or os.path.exists(last_path):
    #         #         os.remove(last_path)
    #         #     os.symlink(os.path.basename(tmp_path), last_path)
    #         # except Exception:
    #         #     # fall back to copy if symlink not allowed
    #         #     import shutil
    #         #     shutil.copy2(tmp_path, last_path)
    #         # shutil.copy2(tmp_path, last_path)

    #         trainer.should_stop = True  # graceful stop at the next safe point

def str2bool(v):
    """Convert string to boolean for argparse"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():

    # fmt: off
    parser = argparse.ArgumentParser(description='Training')
    # parser.add_argument('--load-model', action=LoadFromCheckpoint, help='Restart training using a model checkpoint')  # keep first
    parser.add_argument('--pretrained-model', default=None, type=str, help='Pre-trained weights checkpoint.')
    parser.add_argument('--load-model', help='load model from a given epoch, default to last epoch')
    
    parser.add_argument('--load-epoch','-ep', default=-1, help='load model from a given epoch')
    parser.add_argument('--restart', '-r', type=bool, default=False, help='Restart training the load-epoch, default is last')

    parser.add_argument('--log-dir', '-l', default='logs/', help='where training outputs go')
    parser.add_argument('--model_config', '-m', default='configs/model_configs/torchnet_basic.yaml', help='config file for model set up')

    parser.add_argument('--conf', '-c', type=open, action=LoadFromFile, help='Configuration yaml file')  # keep second
    parser.add_argument('--batch-size', default=32, type=int, help='batch size')
    parser.add_argument('--inference-batch-size', default=None, type=int, help='Batchsize for validation and tests.')

    parser.add_argument('--num-epochs', default=300, type=int, help='number of epochs')
    parser.add_argument('--num-steps', default=None, type=int, help='Maximum number of gradient steps.')
    
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--lr-schedule', default="reduce_on_plateau", type=str, choices=['cosine', 'reduce_on_plateau'], help='Learning rate schedule.')
    parser.add_argument('--lr-patience', type=int, default=10, help='Patience for lr-schedule. Patience per eval-interval of validation')
    parser.add_argument('--lr-min', type=float, default=1e-6, help='Minimum learning rate before early stop')
    parser.add_argument('--lr-factor', type=float, default=0.8, help='Minimum learning rate before early stop')
    parser.add_argument('--lr-warmup-steps', type=int, default=0, help='How many steps to warm-up over. Defaults to 0 for no warm-up')
    parser.add_argument('--lr-cosine-length', type=int, default=400000, help='Cosine length if lr_schedule is cosine.')
    parser.add_argument('--early-stopping-patience', type=int, default=30, help='Stop training after this many epochs without improvement')
    
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay strength')
    parser.add_argument('--ema-alpha-y', type=float, default=1.0, help='The amount of influence of new losses on the exponential moving average of y')
    parser.add_argument('--ema-alpha-dy', type=float, default=1.0, help='The amount of influence of new losses on the exponential moving average of dy')
    
    parser.add_argument('--ngpus', type=int, default=-1, help='Number of GPUs, -1 use all available. Use CUDA_VISIBLE_DEVICES=1, to decide gpus')
    parser.add_argument('--gpu-ids', type=str, default=None, help='Comma-separated list of GPU IDs to use (e.g., "0,1,2"). Overrides CUDA_VISIBLE_DEVICES.')
    parser.add_argument('--num-nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--distributed-backend', default='ddp', help='Distributed backend: dp, ddp, ddp2')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for data prefetch')
    parser.add_argument('--precision', type=int, default=32, choices=[16, 32], help='Floating point precision')

    parser.add_argument('--splits', default=None, help='Npz with splits idx_train, idx_val, idx_test')
    parser.add_argument('--train-size', type=number, default=None, help='Percentage/number of samples in training set (None to use all remaining samples)')
    parser.add_argument('--val-size', type=number, default=0.05, help='Percentage/number of samples in validation set (None to use all remaining samples)')
    
    parser.add_argument('--standardize', type=bool, default=False, help='If true, multiply prediction by dataset std and add mean')


   
    parser.add_argument('--test-size', type=number, default=0.1, help='Percentage/number of samples in test set (None to use all remaining samples)')
    parser.add_argument('--test-interval', type=int, default=10, help='Test interval, one test per n epochs (default: 10)')
    parser.add_argument('--save-interval', type=int, default=10, help='Save interval, one save per n epochs (default: 10)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    
    
    parser.add_argument('--redirect', type=bool, default=False, help='Redirect stdout and stderr to log_dir/log')
    parser.add_argument('--wandb-notes', default="", type=str, help='Notes passed to wandb experiment.')
    parser.add_argument('--job-id', default="auto", type=str, help='Job ID. If auto, pick the next available numeric job id.')
    
    
    # dataset specific
    parser.add_argument('--dataset', default=None, type=str, choices=datasets.__all__, help='Name of the torch_geometric dataset')
    parser.add_argument('--dataset-root', default='data', type=str, help='Data storage directory (not used if dataset is "CG")')
    parser.add_argument('--dataset-arg', default=None, type=str, help='Additional dataset argument, e.g. target property for QM9 or molecule for MD17')
    # parser.add_argument('--coord-files', default=None, type=str, help='Custom coordinate files glob')
    # parser.add_argument('--embed-files', default=None, type=str, help='Custom embedding files glob')
    # parser.add_argument('--energy-files', default=None, type=str, help='Custom energy files glob')
    # parser.add_argument('--force-files', default=None, type=str, help='Custom force files glob')
    
    parser.add_argument('--energy-weight', default=1.0, type=float, help='Weighting factor for energies in the loss function')
    parser.add_argument('--force-weight', default=1.0, type=float, help='Weighting factor for forces in the loss function')
    parser.add_argument('--noise-scale', default=0., type=float, help='Scale of Gaussian noise added to positions.')
    parser.add_argument('--denoising-weight', default=0., type=float, help='Weighting factor for denoising in the loss function.')
    parser.add_argument('--denoising-only', type=bool, default=False, help='If the task is denoising only (then val/test datasets also contain noise).')

    #data augmentations
    parser.add_argument('--random-rotate', type=bool, default=True, help='If true, apply random rotation augmentation')
    parser.add_argument('--p-invert', type=float, default=0.0, help='Probability of inverting the coordinates')
    parser.add_argument('--torsion-angle-std', type=float, default=0.0, help='Standard deviation of torsion angles in degrees')
    parser.add_argument('--max-bonds-rotated', type=int, default=0, help='Maximum number of bonds to rotate in torsion augmentation')

    
    parser.add_argument('--atom-filter', type=int, default=-1, help='Only sum over atoms with Z > atom_filter')
    parser.add_argument('--prior-model', type=str, default=None, help='Which prior model to use')

    #method flags
    parser.add_argument('--pretraining', type=bool, default=False, help='If true, the model is pretrained')
    parser.add_argument('--self_cond', type=bool, default=False, help='use self conditining ssl step')

    ##NEW ARGS FOR JOB SCHEDULING
    parser.add_argument("--max_time_minutes", type=int, default=None, help = 'Maximum wall time in minutes')
    parser.add_argument("--max_epochs_per_run", '-mep', type=int, default=None, help = 'Maximum number of epochs per run')

    ###new arges for finetuning - changes to model arch
    # --reset-head: reseat the scalar head of the model if reloaded from pretrain
    parser.add_argument('--reset-head', type=bool, default=False, help='If true, reset the scalar head of the model when loading from a pretrained model.')
    # --reset_embeddings: reset the embeddings of the model
    parser.add_argument('--reset-embeddings', type=bool, default=False, help='If true, reset the embeddings of the model when loading from a pretrained model.')
    # --reset-norms: reset layer norms of the model. string, either LN, CN, or None
    parser.add_argument('--reset-norms', type=str, default=None, help='If true, reset layer norms of the model. string, either LN, CN, or None')
    # --replace-norms: replace conditional norms with regular layer norms
    # parser.add_argument('--replace-norms', type=bool, default=False, help='If true, replace conditional norms with regular layer norms when loading from a pretrained model.')

    parser.add_argument('--droppath_warmup', type=bool, default=False, help='If true, linearly decrease droppath rate during early training phase.')
    parser.add_argument('--init-droppath', type=float, default=0.15, help='Initial droppath rate if droppath_warmup is true.')
    parser.add_argument('--final-droppath', type=float, default=0.0, help='Final droppath rate if droppath_warmup is true.')

    parser.add_argument('--momentum-update', type=bool, default=False, help='If true, maintain a model copy updated by momentum.')
    parser.add_argument('--model-momentum', type=float, default=0.996, help='Decay rate for model momentum updates.')
    parser.add_argument('--use-ema-model', type=bool, default=False, help='If true, when loading a pretrained checkpoint use the EMA model as the primary model if it exists.')
    parser.add_argument('--val-ema-model', type=bool, default=False, help='If true, use the EMA model for validation if it exists.')
   
    # parser.add_argument('--derivative', type=bool, default=None, help='If true, model returns energy and forces (dy)')
    parser.add_argument('--derivative', type=str2bool, default=None, help='If true, model returns energy and forces (dy)') ### TODO: NEW
    parser.add_argument('--gradient-clipping', type=float, default=1.0, help='Gradient clipping norm')
    parser.add_argument('--val-interval', type=int, default=1, help='Test interval, one test per n epochs (default: 10)')

    parser.add_argument('--add-head-to-pred', type=bool, default=False, help='If true, add embedding head prediction to final y prediction.')

    parser.add_argument('--weight-decay-on-head', type=bool, default=False, help='If true, apply weight decay to the scalar head of the model.')
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 for AdamW optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='Beta2 for AdamW optimizer')

    ## for contrastive loss
    parser.add_argument('--create-contrastive-loss', type=bool, default=False, help='If true, create a contrastive loss module.')
    parser.add_argument('--contrastive-loss-weight', type=float, default=0.0, help='Weight of the contrastive loss.')
    parser.add_argument('--ctr-init-tau', type=float, default=0.1, help='Initial temperature for contrastive loss.')
    
    ## NEW args for periodic materials datasets
    parser.add_argument('--noise_in_loader', type=bool, default=False, help='If true, add noise to positions in the data loader instead of in the model.')
    parser.add_argument('--graph_cutoff', type=float, default=5.0, help='if graph creation in loader, Cutoff distance for neighbor list in periodic datasets.')
    parser.add_argument('--p_cell_repeat', type=float, default=0.15, help='if periodic system, number of times to repeat the unit cell in each direction.')
    parser.add_argument('--cell_repeat_iters', type=int, default=1, help='Number of iterations to probabilistically repeat cell based on p_cell_repeat.')
    parser.add_argument('--allow_periodic', type=bool, default=False, help='If true, allow periodic boundary conditions in graph construction.')
    parser.add_argument('--neighbor_method', type=str, default='brute', help='Method for neighbor search in periodic datasets (default: brute). options: brute, grid')
    parser.add_argument('--max_neighbors', type=int, default=32, help='Maximum number of neighbors in loader graph generation.')

    ## NEW args for batch clipping
    parser.add_argument('--max-nodes-per-batch', type=int, default=None, help='Maximum number of nodes per batch. If set, batches will be clipped to this size.')
    parser.add_argument('--batch-clipper-cache-size', type=int, default=1000, help='Maximum number of samples to cache in the batch clipper.')
    parser.add_argument('--allow-test-clipping', type=bool, default=True, help='If true, allow batch clipping during validation and test.')

    #new args for matbench datasets
    parser.add_argument('--predefined_splits', type=bool, default=False, help='If true, use predefined splits for matbench datasets.')
    parser.add_argument('--rep-min-atoms', type=int, default=4, help='Minimum number of atoms to trigger cell repetition.')

    ## NEW for matbench hyperparameters
    parser.add_argument('--set-head-agg', type=str, default=None, help='Aggregation method for the head of the model (default: sum). options: sum, mean, max')
    
    ### new for plotting and debugging 
    parser.add_argument('--no-wandb-resume', action='store_true', default=False, help='If true, allow wandb to resume previous runs in the same log dir. If false, always start a new run in wandb.')
    parser.add_argument('--parity-plot', type=bool, default=True, help='If true, create parity plots after training.')
    parser.add_argument('--plot-interval', type=int, default=-1, help='Interval (in epochs) to create parity plots during training.')
    parser.add_argument('--compute-corr', type=bool, default=False, help='If true, compute and log correlation metrics after each epoch.')

    ### new for LBA dataset ligand masking
    parser.add_argument('--allow-node-mask', type=bool, default=False, help='If true, mask ligand atoms in the LBA dataset during training.')
    parser.add_argument('--inverse-node-mask', type=bool, default=False, help='If true, mask ligand atoms in the LBA dataset during training.')

    ### New for omol
    parser.add_argument('--direct-force-pred', type=str2bool, default=False, help='If true, model directly predicts forces instead of via energy derivative.')

    args = parser.parse_args()

    if args.job_id == "auto":
        assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) == 1, "Might be problematic with DDP."
        if Path(args.log_dir).exists() and len(os.listdir(args.log_dir)) > 0:        
            next_job_id = str(max([int(x.name) for x in Path(args.log_dir).iterdir() if x.name.isnumeric()])+1)
        else:
            next_job_id = "1"
        args.job_id = next_job_id

    args.log_dir = str(Path(args.log_dir, args.job_id))
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    if args.redirect:
        sys.stdout = open(os.path.join(args.log_dir, "log"), "w")
        sys.stderr = sys.stdout
        logging.getLogger("pytorch_lightning").addHandler(
            logging.StreamHandler(sys.stdout)
        )

    if args.inference_batch_size is None:
        args.inference_batch_size = args.batch_size

    save_argparse(args, os.path.join(args.log_dir, "input.yaml"), exclude=["conf"])

    return args


def main():
    print("---STARTING JOB---")
    print("Torch version:", torch.__version__)
    print("Pytorch Lightning version:", pl.__version__)
    args = get_args()

    # Set GPU selection if specified
    if args.gpu_ids is not None:
        # Validate gpu_ids format - should be comma-separated numbers, not "cuda:X"
        gpu_count = torch.cuda.device_count()
        print(f"Found {gpu_count} CUDA devices")
        if args.gpu_ids.startswith('cuda:'):
            print(f"Warning: gpu-ids should be just numbers (e.g., '0,1,2'), not '{args.gpu_ids}'. Converting...")
            args.gpu_ids = args.gpu_ids.replace('cuda:', '')
        
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
        print(f"Set CUDA_VISIBLE_DEVICES to: {args.gpu_ids}")
    
    # Optimize for Tensor Cores on A100 GPUs
    gpu_name = torch.cuda.get_device_name(0)
    if args.pretraining:
        if 'A100' in gpu_name or 'H100' in gpu_name:
            torch.set_float32_matmul_precision('medium')
            print(f"Enabled Tensor Core optimization for {gpu_name}")
        else:
            print(f"GPU: {gpu_name} - keeping default precision")
    else:
        print(f"Fine-tuning run - keeping default precision")
    
    #detect number of nodes if possible in environment
    args.num_nodes = int(os.environ.get('SLURM_NNODES', 1))
    print(f"Using {args.num_nodes} nodes for training.")

    # torch.set_float32_matmul_precision('medium')  # or 'high' for even more speed
    # print("Set float32 matmul precision to 'medium' for Tensor Core optimization")
    
    pl.seed_everything(args.seed, workers=True)


    # initialize data module
    data = DataModule(args)
    data.prepare_data()
    data.setup("fit")

    prior_models = create_prior_models(vars(args), data.dataset)

    args.prior_model = prior_models

    # torch.serialization.add_safe_globals([Atomref])

    checkpoint_path = None
    if args.restart:   
        checkpoint_path, _step, _epoch = get_model_checkpoint(args.log_dir, 
                                                                epoch=args.load_epoch)
        if checkpoint_path is None:
            print(f'WARNING: restart is on but no checkpoints found in {args.log_dir}')

        else:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            ckpt_epoch = ckpt["epoch"]
            ckpt_step = ckpt["global_step"]

            if ckpt_epoch == _epoch:
                print(f"WARNING: Checkpoint epoch {ckpt_epoch} matches requested restart epoch {_epoch}. loaded step {ckpt_step}")
            # assert ckpt_epoch == _epoch, f"Checkpoint epoch {ckpt_epoch} does not match requested restart epoch {_epoch}. loaded step {ckpt_step}"
            if ckpt_step-1 == _step:
                print(f"WARNING: Checkpoint step {ckpt_step} matches expected step {_step}")
            # assert ckpt_step-1 == _step, f"Checkpoint step {ckpt_step} does not match expected step {_step}"
            print(f"Restarting from checkpoint: {checkpoint_path}, epoch: {_epoch}, step: {_step}")
            args.load_model = checkpoint_path

    #TODO add in huggingface loading

    if checkpoint_path is not None:
        # model = LTrainer.load_from_checkpoint(checkpoint_path, args, mean=data.mean, std=data.std)
        # model = LTrainer.load_from_checkpoint(checkpoint_path, map_location="cpu", hparams=args, mean=data.mean, std=data.std)
        model = LTrainer.load_from_checkpoint(checkpoint_path, 
                                              map_location="cpu", 
                                              hparams=args, 
                                              prior_model=prior_models, 
                                              mean=data.mean, 
                                              std=data.std)
    else:
        # model = LTrainer(args, mean=data.mean, std=data.std)
        model = LTrainer(args, prior_model=prior_models, mean=data.mean, std=data.std)

    callbacks = []
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.log_dir,
        monitor="val_loss",
        save_top_k=10,  # -1 to save all
        every_n_epochs=args.save_interval,
        filename="{step}-{epoch}-{val_loss:.4f}-{test_loss:.4f}-{train_per_step:.4f}",
        save_last=True,
    )
    callbacks.append(checkpoint_callback)
    # early_stopping = EarlyStopping("val_loss", patience=args.early_stopping_patience)
    

    if args.pretraining:
        project_name = 'SCD_pretraining'
    else:
        ds_name = args.dataset
        project_name = f'SCD_bench_{ds_name}'
        # project_name='SCD_benchmarking'
   

    # force_id = args.restart and checkpoint_path is not None
    force_id = args.restart and checkpoint_path is not None
    resume_flag = "must" if force_id else 'allow'
    
    ##TODO: NEW
    if args.no_wandb_resume:
        #force a new wandb run
        resume_flag = 'never'
        datetime_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.job_id = f"{args.job_id}_{datetime_str}"
    
    wandb_logger = WandbLogger(
        name=args.job_id,
        # project='SCD_benchmarking',
        # project='SCD_pretraining',
        project=project_name,
        notes=args.wandb_notes,
        # settings=wandb.Settings(start_method='fork', code_dir="."),

        # resume="allow" if args.restart else None,
        # resume="must" if force_id else 'allow',
        resume=resume_flag,

        id=args.job_id #if args.restart else None,

    )

    ddp_strategy = None
    if "ddp" in args.distributed_backend:
        # ddp_strategy = DDPStrategy(find_unused_parameters=False)
        ddp_strategy = DDPStrategy(find_unused_parameters=True) # to addount for the teacher model

    
    limit_run = None
    if args.max_time_minutes is not None or args.max_epochs_per_run is not None:
        limit_run = LimitRun(minutes=args.max_time_minutes,
                             out_dir=args.log_dir,
                             max_epochs_per_run=args.max_epochs_per_run,
                             buffer_sec=30,
                             )
        print(f"Will stop this run after {args.max_time_minutes} minutes (plus up to 30s buffer).")
        callbacks.append(limit_run)

    use_inference_mode = True
    if args.derivative is not None:
        if args.derivative:
            use_inference_mode = False
            print("====> Model returns forces, turning off inference_mode for training.")

    make_parity_plot = args.parity_plot
    print(f"Make parity plot: {make_parity_plot}, interval: {args.plot_interval}")
    if make_parity_plot:
        parity_callback = ParityPlot(
            output_dir=args.log_dir,
            plot_interval=args.plot_interval,
            stages=['test'], # when to plot during training
            end_stages=['train', 'test'] # what to plot at the end of training
        )
        callbacks.append(parity_callback)

    trainer_kwargs = dict(
        inference_mode= use_inference_mode,
        max_epochs=args.num_epochs,
        max_steps=args.num_steps,
        devices=args.ngpus, #MANUALLY SET GPUS
        num_nodes=args.num_nodes,
        accelerator="gpu",
        default_root_dir=args.log_dir,
        # callbacks=[early_stopping, checkpoint_callback], ###FIXME
        # callbacks=[checkpoint_callback], #no early stopping
        # callbacks=[checkpoint_callback, limit_run], #no early stopping
        callbacks=callbacks,
        logger=wandb_logger,
        precision=args.precision,
        # gradient_clip_val=1.0, #1.8,
        gradient_clip_val=args.gradient_clipping,
        check_val_every_n_epoch= args.val_interval,

        log_every_n_steps=25,# default is 50 # FIXME

        reload_dataloaders_every_n_epochs= args.val_interval, #args.test_interval, # to shuffle data if using small val set

        # max_time=timedelta(minutes=args.max_time_minutes) if args.max_time_minutes is not None else None,

        # resume_from_checkpoint=checkpoint_path if args.restart else None,
    )

    if ddp_strategy is not None:
        trainer_kwargs["strategy"] = ddp_strategy

    trainer = pl.Trainer(**trainer_kwargs)


    trainer.fit(model, data, ckpt_path=checkpoint_path,)
    model.eval()
    # run test set after completing the fit
    print()
    print("============== Running final test set evaluation... =================")
    trainer.test(model=model, datamodule=data)
    # trainer.test(datamodule=data)


if __name__ == "__main__":
    main()

#salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --gpus 4 --account m4866

# python train_job.py --conf configs/scd_pretrain_geom.yaml --job-id geom_debug
# python train_job.py --conf configs/scd_pretrain_geom.yaml --job-id geom_debug_2node
# python train_job.py --conf configs/scd_pretrain_geom.yaml --job-id geom_debug40gb --batch-size 32

# python train_job.py --conf configs/scdB_pretrain.yaml --job-id scdBig_debug

# python train_job.py --conf configs/scdS-m-n04_pretrain.yaml --job-id scdS_batchmem_debug

#### test jobs
# python train_job.py --conf configs/scd-m-n004_pretrain.yaml --job-id scd_debug_mean1
# python train_job.py --conf configs/scd-m-n04_pretrain.yaml --job-id scd_debug_mean_n04
# python train_job.py --conf configs/scd-s-n04_pretrain.yaml --job-id scd_debug_sum_n04

# python train_job.py --conf configs/scd-qm9-ft-energy.yaml -r True --pretrained-model experiments/scd_debug_mean/last.ckpt --job-id scd-ft-u0-debug

###### fine tune command examples
# python train_job.py --conf configs/nersc_qm9-ft-energy.yaml -r True --pretrained-model experiments/cet-pcq_2/last.ckpt --job-id cet-pcq2-ft_u0_debug3

# python train_job.py --conf configs/nersc_qm9-ft-energy.yaml -r True --pretrained-model experiments/cet-pcq_2/last.ckpt --job-id cet-pcq2_ft_u0_b512_m


# python train_job.py --conf configs/nersc_qm9-ft-base.yaml -r True --pretrained-model experiments/cet-pcq_2/last.ckpt --job-id debug_reload_ehomo --max_epochs_per_run 5

##### SL job 
# python train_job.py --conf configs/scd-qm9-sl.yaml --job-id scd-sl-u0-test



##### basic pretraining job
# python train_job.py --conf configs/scd_pretrain.yaml --job-id scd_pcq -r True --max_epochs_per_run 1

# python train_job.py --conf configs/scd_pretrain.yaml --job-id scd_debug
# python train_job.py --conf configs/scd-m-n004_pretrain.yaml --job-id scd_debug_mean
# python train_job.py --conf configs/scd_pretrain.yaml --job-id scd_debug_b256
# python train_job.py --conf configs/scd-m-n004_pretrain.yaml --job-id scd_debug_mean1


# job_id=cet-pcq_2
# conf=configs/nersc_pt.yaml
# max_epochs_per_run=1

# python train_job.py --conf configs/nersc_pt.yaml --job-id cet-pcq_2 -r True --max_epochs_per_run 1



# python train_job.py --conf configs/nersc_pt.yaml --job-id cetr-long -r True --max_epochs_per_run 1



# python train_job.py --conf configs/nersc_pt.yaml --job-id cet-long -r True --max_epochs_per_run 1
# python train_job.py --conf configs/nersc_pt.yaml --job-id test_stop -r True --max_epochs_per_run 1

# python train_job.py --conf configs/nersc_pt.yaml --job-id debug_stop -r True --max_epochs_per_run 1



# python train_job.py --conf configs/nersc_pt.yaml --job-id debug --max_time_minutes 10

# python train_job.py --conf configs/nersc_pt.yaml --job-id debug
# python train_job.py --conf configs/nersc_pt.yaml --job-id debug_r

# python train_job.py --conf configs/nersc_pt.yaml --job-id debug_r -r True -ep 20

# salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --gpus 4 --account m4866

# python train.py --conf configs/et-QM9-SL.yaml --job-id et2-qm9sl-baseline

# python train.py --conf configs/etcls-QM9-SL.yaml --job-id SL_etcls
# python train.py --conf configs/etcls-QM9-SL.yaml --job-id SL_etcls-predy
# python train.py --conf configs/etcls-QM9-SL.yaml --job-id SL_etcls-normcls
# python train.py --conf configs/etcls-QM9-SL.yaml --job-id SL_etcls-normcls-predy



#### Petrianing
# python train.py --conf configs/pretrain_cet.yaml --job-id PT-cet-baseline


### Unskew testing
# python train.py --conf configs/et-QM9-SL.yaml --job-id SL_et-deskew004
# python train.py --conf configs/etcls-QM9-SL.yaml --job-id SL_etcls-deskew004
# python train.py --conf configs/etcls-QM9-SL.yaml --job-id SL_etcls-deskew004_predskew
# python train.py --conf configs/etcls-QM9-SL.yaml --job-id SL_etcls-deskew004_predskew_glob

# python train.py --conf configs/etcls-QM9-SL.yaml --job-id SL_etcls-glob

# python train.py --conf configs/pretrain_cet.yaml --job-id PT-cet-baseline
# python train.py --conf configs/pretrain_cet.yaml --job-id PT-cetcls
# python train.py --conf configs/pretrain_cet.yaml --job-id PT-cet-clsemb

# python train.py --conf configs/pretrain_cet.yaml --job-id PT-cetcls-aug


# python train.py --conf configs/cet-QM9-SL.yaml --job-id SL-cet

# python train.py --conf configs/cet-QM9-FT.yaml --job-id FT-cetcls --pretrained-model experiments/PT-cetcls/last.ckpt
# python train.py --conf configs/cet-QM9-FT.yaml --job-id FT-cetcls-fixed --pretrained-model experiments/PT-cetcls/last.ckpt
# python train.py --conf configs/cet-QM9-FT.yaml --job-id FT-cet-clsemb --pretrained-model experiments/PT-cet-clsemb/last.ckpt