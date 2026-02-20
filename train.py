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

# from models.ET_models.cet_model import create_prior_models

# from models.trainer import LTrainer # older version of models
# from models.model_helper import get_model_checkpoint, create_prior_models

from models.trainer2 import LTrainer # new simplified version of models
from models.model_helper2 import get_model_checkpoint, create_prior_models

##TODO: NEW
from models.callbacks import ParityPlot, LimitRun
from pytorch_lightning.callbacks import ModelCheckpoint, Callback

import torch
from pathlib import Path
import wandb
from datetime import timedelta
from datetime import datetime
import time
import shutil




def get_args():

    # fmt: off
    parser = argparse.ArgumentParser(description='Training')
    # parser.add_argument('--load-model', action=LoadFromCheckpoint, help='Restart training using a model checkpoint')  # keep first
    parser.add_argument('--pretrained-model', default=None, type=str, help='Pre-trained weights checkpoint.')
    parser.add_argument('--load-model', help='load model from a given epoch, default to last epoch')
    parser.add_argument('--load-hf', default=None, help='load model from huggingface, provide the repo name')
    # parser.add_argument('--hf-cache', default='experiments/hf_models', help='load model from a given epoch, default to last epoch')

    parser.add_argument('--load-epoch','-ep', default=-1, help='load model from a given epoch')
    parser.add_argument('--restart', '-r', type=bool, default=False, help='Restart training the load-epoch, default is last')

    parser.add_argument('--log-dir', '-l', default='/exp/_new', help='where training outputs go')
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
     #FIXME: prior-model isi not used but needs to exist for standardize
   
    # parser.add_argument('--val-size', type=number, default=0.95, help='Percentage/number of samples in validation set (None to use all remaining samples)')
    # parser.add_argument('--train-size', type=number, default=0.05, help='Percentage/number of samples in training set (None to use all remaining samples)')

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

    ## Prior model related
    parser.add_argument('--atom-filter', type=int, default=-1, help='Only sum over atoms with Z > atom_filter')
    # parser.add_argument('--remove-ref-energy', action='store_true', help='If true, remove the reference energy from the dataset for delta-learning. Total energy can still be predicted by the model during inference by turning this flag off when loading.  The dataset must be compatible with Atomref for this to be used.')
    parser.add_argument('--remove-ref-energy', type=bool, default=False, help='If true, remove the reference energy from the dataset for delta-learning. Total energy can still be predicted by the model during inference by turning this flag off when loading.  The dataset must be compatible with Atomref for this to be used.')
    parser.add_argument('--prior-model', type=str, default=None, help='Which prior model to use. It can be a string, a dict if you want to add arguments for it or a dicts to add more than one prior. e.g. {"Atomref": {"max_z":100}, "Coulomb":{"max_num_neighs"=100, "lower_switch_distance"=4, "upper_switch_distance"=8}', action="extend", nargs="*")


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


    parser.add_argument('--derivative', type=bool, default=None, help='If true, model returns energy and forces (dy)')
    parser.add_argument('--gradient-clipping', type=float, default=1.0, help='Gradient clipping norm')
    parser.add_argument('--val-interval', type=int, default=1, help='Test interval, one test per n epochs (default: 10)')

    parser.add_argument('--add-head-to-pred', type=bool, default=False, help='If true, add embedding head prediction to final y prediction.')

    parser.add_argument('--weight-decay-on-head', type=bool, default=False, help='If true, apply weight decay to the scalar head of the model.')

    parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 for AdamW optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='Beta2 for AdamW optimizer')

    
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

    ##TODO: NEW
    ### new for plotting and debugging 
    parser.add_argument('--no-wandb-resume', action='store_true', default=False, help='If true, allow wandb to resume previous runs in the same log dir. If false, always start a new run in wandb.')
    parser.add_argument('--parity-plot', type=bool, default=True, help='If true, create parity plots after training.')
    parser.add_argument('--plot-interval', type=int, default=-1, help='Interval (in epochs) to create parity plots during training.')
    parser.add_argument('--compute-corr', type=bool, default=False, help='If true, compute and log correlation metrics after each epoch.')

    ### new for LBA dataset ligand masking
    parser.add_argument('--allow-node-mask', type=bool, default=False, help='If true, mask ligand atoms in the LBA dataset during training.')

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

    # torch.set_float32_matmul_precision('medium')  # or 'high' for even more speed
    # print("Set float32 matmul precision to 'medium' for Tensor Core optimization")
    pl.seed_everything(args.seed, workers=True)

    
    # initialize data module
    data = DataModule(args)
    data.prepare_data()
    data.setup("fit")

    prior_models = create_prior_models(vars(args), data.dataset)

    # prior_models = None

    # print('--- CREATED Model Prior Models ---'
    #       , prior_models)

    args.prior_model = prior_models
   

    checkpoint_path = None
    if args.restart:   
        checkpoint_path, _step, _epoch = get_model_checkpoint(args.log_dir, 
                                                                epoch=args.load_epoch)
        
        if checkpoint_path is None:
            print(f'WARNING: restart is on but no checkpoints found in {args.log_dir}')

        else:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
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

    if args.load_hf is not None:

        if checkpoint_path is not None:
            print(f"WARNING: both --restart and --load-hf were specified. Ignoring --load-hf and using checkpoint {checkpoint_path}")
            # args.load_hf = None
        else:
            from huggingface_hub import hf_hub_download
            print(f"Loading model from Huggingface repo: {args.load_hf}")
            # checkpoint_path = hf_hub_download(repo_id=args.load_hf, filename="last.ckpt", cache_dir=args.hf_cache)
            # checkpoint_path = models.model_helper.load_hf_model(args.load_hf)
            # print(f"Downloaded checkpoint to: {checkpoint_path}")
            cache_dir = "/home/tjp/Projects/SelfConditionedDenoising/experiments/hf_models"
            model_name = args.load_hf
            #check if cache dir exists
            assert os.path.exists(cache_dir), f"Cache dir {cache_dir} does not exist. Please Create it first or change the cache dir variable in this script."
            
            hf_ckpt_path = hf_hub_download(
                repo_id=f"Ty-Perez/{model_name}",
                filename="last.ckpt",
                # cache_dir="./downloaded_models"  # Optional: specify cache directory
                cache_dir=cache_dir  # Optional: specify cache directory
                )
            args.pretrained_model = hf_ckpt_path
            # checkpoint_path = None

    # exit()
    
    ###FIXME
    if checkpoint_path is not None:
        # model = LTrainer.load_from_checkpoint(checkpoint_path, args, mean=data.mean, std=data.std)
        model = LTrainer.load_from_checkpoint(checkpoint_path, 
                                              map_location="cpu", 
                                              hparams=args, 
                                              prior_model=prior_models, 
                                              mean=data.mean, 
                                              std=data.std)
    else:
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
   
    # exit()
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
        project=project_name,
        notes=args.wandb_notes,
        # settings=wandb.Settings(start_method='fork', code_dir="."),
        # resume="allow" if args.restart else None,
        # resume="must" if force_id else 'allow',
        resume = resume_flag, ##TODO: NEW
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

    ##TODO: NEW
    make_parity_plot = args.parity_plot
    print(f"Make parity plot: {make_parity_plot}")
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
        devices=[1], #args.ngpus, #MANUALLY SET GPUS
        num_nodes=args.num_nodes,
        accelerator="gpu",
        default_root_dir=args.log_dir,
        # callbacks=[early_stopping, checkpoint_callback], ###FIXME
        # callbacks=[checkpoint_callback], #no early stopping
        # callbacks=[checkpoint_callback, limit_run], #no early stopping\
        callbacks=callbacks,
        logger=wandb_logger,
        precision=args.precision,
        gradient_clip_val=args.gradient_clipping,
        check_val_every_n_epoch= args.val_interval,

        #NEW 
        reload_dataloaders_every_n_epochs= args.val_interval, #args.test_interval, # to shuffle data if using small val set

        log_every_n_steps=25,# default is 50 # FIXME

        # max_time=timedelta(minutes=args.max_time_minutes) if args.max_time_minutes is not None else None,

        # resume_from_checkpoint=checkpoint_path if args.restart else None,
    )

    if ddp_strategy is not None:
        trainer_kwargs["strategy"] = ddp_strategy


    trainer = pl.Trainer(**trainer_kwargs)

    
    
    
    trainer.fit(model, data, ckpt_path=checkpoint_path,)

    ##TODO: NEW
    #set model to eval mode before testing 
    model.eval()

    # run test set after completing the fit
    trainer.test(model=model, datamodule=data)


if __name__ == "__main__":
    main()


## to run 



## nov 21
# python train.py --conf configs/cet-lba_grimm.yaml --job-id lba_ligmask_debug -r true 
# python train.py --conf configs/cet-lba_grimm.yaml --job-id lba_ligmask_debug-sum --set-head-agg sum -r true

# python train.py --conf configs/cet-lba_grimm.yaml --job-id lba_protmask_mean2 -r true

## nov 20
# python train.py --conf configs/cget-mbgap_grimm.yaml --load-hf cget-amp20 --job-id cget-amp20_b16_hm_lr1e4 --lr 1e-4
# python train.py --conf configs/cet-mbgap_grimm.yaml --job-id et_b16_hm_nonstd --standardize False

# python train.py --conf configs/cet-lba_grimm.yaml --job-id et_b5_emay01
# python train.py --conf configs/cet-lba_grimm.yaml --job-id et_b5_wd1 --weight-decay 0.1



### 

## nov 19
# python train.py --conf configs/cet-mbgap_grimm.yaml --load-hf cet-amp20 --job-id cet-amp20_b16_hm_lr1e4 --lr 1e-4
# python train.py --conf configs/cet-mbgap_grimm.yaml --load-hf cet-amp20 --job-id cet-amp20_b16_hm_lr2e4_nz005 --noise-scale 0.005 --denoising-weight 0.1
# python train.py --conf configs/cet-mbgap_grimm.yaml --job-id etmbgap_grm_b16_hm_lr2e4_plt-dbug -r true --load-model experiments/etmbgap_grm_b16_hm_lr2e4/last.ckpt

# python train.py --conf configs/cet-mbgap_grimm.yaml --job-id etmbgap_grm_b16_hm_lr2e4 -r true -ep 54 --no-wandb-resume

# python train.py --conf configs/cet-mbgap_grimm.yaml --job-id etmbgap_debug_dp -r true --no-wandb-resume

# python train.py --conf configs/_debug.yaml --job-id plot_debug_x --no-wandb-resume

## nov 18
# python train.py --conf configs/cet-mbgap_grimm.yaml --job-id etmbgap_grm_b16_hm_lr1e4_nz-aug --noise-scale 0.005 --denoising-weight 0.1 --p_cell_repeat 0.3 --cell_repeat_iters 2
# python train.py --conf configs/cet-mbgap_grimm.yaml --job-id etmbgap_grm_b16_hm_lr2e4 --lr 2e-4 
# python train.py --conf configs/cet-mbgap_grimm.yaml --job-id etmbgap_grm_b16_hm_lr5e5 --lr 5e-5 
# python train.py --conf configs/cet-mbgap_grimm.yaml --job-id etmbgap_grm_b16_hm_lr5e4 --lr 5e-4
# python train.py --conf configs/cet-mbgap_grimm.yaml --load-hf cet-amp20 --job-id cet-amp20_b16_hm_lr2e4
# python train.py --conf configs/cget-mbgap_grimm.yaml --job-id cget_b16_hm_lr2e4
# python train.py --conf configs/cet-mbgap_grimm.yaml --load-hf scd-s-n004_pcq --job-id cet-pcq_b16_hm_lr2e4

# python train.py --conf configs/cet-mbgap_grimm.yaml --job-id cet_b16_hm_lr2e4_cut8 --graph_cutoff 8.0


# python train.py --conf configs/cet-lba_grimm.yaml --job-id etlba_debug_0

# Nov 16
# python train.py --conf configs/cet-mbgap_grimm.yaml --job-id etmbgap_grm_b16_nz0_dp001_mean_lr1e4_wd1e5
# python train.py --conf configs/cet-mbgap_grimm.yaml --job-id etmbgap_grm_b16_nz0_dp001_mean_lr1e4_prep03 --p_cell_repeat 0.3 --cell_repeat_iters 2
# python train.py --conf configs/cet-mbgap_grimm.yaml --job-id etmbgap_grm_b16_nz0_dp001_headmean_lr1e4
# python train.py --conf configs/cet-mbgap_grimm.yaml --job-id etmbgap_grm_b16nz0dp001_hm_lr1e4_prep03 --p_cell_repeat 0.3 --cell_repeat_iters 2
# python train.py --conf configs/cet-mbgap_grimm.yaml --job-id etmbgap_grm_b16_hm_nz005_lr1e4 --noise-scale 0.005 --denoising-weight 0.1



# python train.py --conf configs/cet-mbgap_grimm.yaml --job-id etmbgap_grm_b16_hm_dp01 --final-droppath 0.1

# python train.py --conf configs/cet-mbgap_grimm.yaml --job-id etmbgap_debug_head --load-hf scd-s-n004_pcq

# Nov 16
# python train.py --conf configs/cet-mbgap_grimm.yaml --job-id etmbgap_grm_b16_nz0_dp001_sum
# python train.py --conf configs/cet-mbgap_grimm.yaml --job-id etmbgap_grm_b16_nz0_dp001_mean
# python train.py --conf configs/cet-mbgap_grimm.yaml --job-id etmbgap_grm_b16_nz0_dp001_mean_cutoff8 --graph_cutoff 8.0
# python train.py --conf configs/cet-mbgap_grimm.yaml --job-id etmbgap_grm_b16_nz0_dp001_mean_lr1e3
# python train.py --conf configs/cet-mbgap_grimm.yaml --job-id etmbgap_grm_b16_nz0_dp001_mean_lr1e4

##
# python train.py --conf configs/cet-mbgap_grimm.yaml --job-id et_mbgap_baseline_grm_b24_prep03
# python train.py --conf configs/cet-mbgap_grimm.yaml --job-id et_mbgap_baseline_grm_b24_prep0 --p_cell_repeat 0.0
# python train.py --conf configs/cet-mbgap_grimm.yaml --job-id et_mbgap_baseline_grm_b24_prep0_nz0 --p_cell_repeat 0.0 --noise-scale 0.000001
# python train.py --conf configs/cet-mbgap_grimm.yaml --job-id et_mbgap_baseline_grm_b24_prep03_beta109 --beta1 0.9

# python train.py --conf configs/scd-all_pretrain.yaml --job-id debug_scdall
# python train.py --conf configs/debug_qm9.yaml --job-id debug_graphgen

# python train.py --conf configs/frad_pretrain.yaml --job-id frad_debug
# python train.py --conf configs/scd-s-n004_pretrain.yaml --job-id debug_pt --batch-size 32

# python train.py --conf configs/scd-s-n004_pretrain.yaml --job-id debug_ctr_b64 --batch-size 64 -r True --train-size 5000
# python train.py --conf configs/scd-n004-ctr_pretrain.yaml --job-id debug_ctr2_b64 --batch-size 64 -r True --train-size 5000


# python train.py --conf configs/scd-qm9-ft-base.yaml --job-id debug_ctr_ft_test --pretrained-model experiments/debug_ctr_b64/last.ckpt

# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id load_test_debug --dataset-arg aspirin


### running oct 16
# python train.py --conf configs/scdL-md17.yaml --load-hf scd-s-n004_pcq --job-id cet-pcq_b8_aspirin_1k_dnz --dataset-arg aspirin
# python train.py --conf configs/scdL-md17.yaml --load-hf CET-GEOM10_0 --job-id cet-geom10_b8_aspirin_1k_dnz --dataset-arg aspirin
# python train.py --conf configs/scdL-md17.yaml --load-hf CET-GEOM01_0 --job-id cet-geom1_b8_aspirin_1k_dnz --dataset-arg aspirin

# python train.py --conf configs/cget-md17.yaml --load-hf CGET-pcq_0 --job-id cget-pcq_b8_aspirin_1k_dnz --dataset-arg aspirin
# python train.py --conf configs/cget-qm9-ft-base.yaml --load-hf CGET-pcq_0 --job-id cget-debug --dataset-arg homo
# python train.py --conf configs/cget-md17.yaml --load-hf CGET-pcq_0 --job-id cget-debug-md17 --dataset-arg aspirin

### running oct 15
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdSn004_b8_aspirin_1k_dnz --dataset-arg aspirin
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdSn004_b8_benzene_1k_dnz --dataset-arg benzene
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdSn004_b8_ethanol_1k_dnz --dataset-arg ethanol
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdSn004_b8_malonaldehyde_1k_dnz --dataset-arg malonaldehyde

# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdSn004_b8_naphthalene_1k_dnz --dataset-arg naphtalene 
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdSn004_b8_salicylic_acid_1k_dnz --dataset-arg "salicylic acid"

# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdSn004_b8_toluene_1k_dnz --dataset-arg "toluene"
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdSn004_b8_uracil_1k_dnz --dataset-arg "uracil"

### running oct 14
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id match_ema_atom_emb --dataset-arg aspirin
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id match_ema_mol_emb --dataset-arg aspirin
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id aps_fwd_dy_0 --dataset-arg aspirin
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id aps_fwd_dy_nonorm --dataset-arg aspirin
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id aps_fwd_dy_normx1k --dataset-arg aspirin

### running oct 13

# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id debug_match_ema --dataset-arg aspirin
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id match_ema_1y2dy_1 --dataset-arg aspirin
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id match_ema_dy --dataset-arg aspirin

#### Running oct 10
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id aspirin_lrg_cutoff_10A --dataset-arg aspirin
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id aspirin_eng_cond --dataset-arg aspirin

##### running oct 9
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id debug_aspirin_scd --dataset-arg aspirin
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id aspirin_scd_dnw03 --dataset-arg aspirin --denoising-weight 0.3
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id aspirin_scd_nocond --dataset-arg aspirin

# python train.py --conf configs/scd-md17-SL.yaml --job-id aspirin_scd_nocond_fresh --dataset-arg aspirin

##### running oct 8
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdSn004_b8_aspirin_nullC --dataset-arg aspirin
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdSn004_b8_aspirin_nullC_wdy1wy01  --dataset-arg aspirin --energy-weight 0.1 --force-weight 1.0
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdSn004_b8_aspirin_nullC_wd0 --dataset-arg aspirin --weight-decay 0.0
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdSn004_b8_aspirin_nullC_lr001 --dataset-arg aspirin --lr 0.001
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdSn004_b8_aspirin_nullC_lr001_RoP --dataset-arg aspirin --lr 0.001 --lr-schedule 'reduce_on_plateau' --num-steps 200000
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSm-n04_pcq --job-id scdSmn04_b8_aspirin --dataset-arg aspirin


# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdSn004_b8_aspirin_s2 --dataset-arg aspirin --seed 2

### running oct 7
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdS_benz_b8_LAref --prior-model LearnableAtomref

# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdSn004_b8_aspirin --dataset-arg aspirin
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdSn004_b8_ethanol --dataset-arg ethanol
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdSn004_b8_malonaldehyde --dataset-arg malonaldehyde
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdSn004_b8_naphthalene --dataset-arg naphtalene # NOTE type in torchgeometric libn
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdSn004_b8_salicylic_acid --dataset-arg "salicylic acid"
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdSn004_b8_toluene --dataset-arg "toluene"
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdSn004_b8_uracil --dataset-arg "uracil"
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdSn004_b8_paracetamol --dataset-arg "paracetamol"
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdSn004_b8_azobenzene --dataset-arg "azobenzene"


# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdS_benz_muval_bs8 --batch-size 8

# paracetamol
# azobenzene

##### RUNNING #### - oct 4
# python train.py --conf configs/scd-qm9-ft-long.yaml --load-hf scd-s-n004_pcq --job-id scd-alpha_long-lr2 --dataset-arg alpha
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdS_benz_b16 --batch-size 16
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdS_benz_b16_n1e4 --batch-size 16 --noise-scale 0.0001
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdS_benz_b16_dp01 --batch-size 16 --final-droppath 0.1

# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdS_benz_b16_n5e4 --batch-size 16 --noise-scale 0.0005 # best 
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdS_benz_b16_n1e3 --batch-size 16 --noise-scale 0.001
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdS_benz_b16_n5e4_dp01 --batch-size 16 --noise-scale 0.0005 --final-droppath 0.1
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdS_benz_b16_n5e4_dn --batch-size 16 --noise-scale 0.0005 --denoising-weight 0.01

# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdS_benz_b8_1k
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdS_benz_b8_RlrOnP
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdS_benz_b8_emay01 --ema-alpha-y 0.1
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdS_benz_b8_emay02 --ema-alpha-y 0.2

# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdS_benz_muval_bs8 --batch-size 8
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdS_benz_muval_bs32 --batch-size 32 --test-interval 50


# python train.py --conf configs/scd-qm9-ft-long.yaml --load-hf scd-s-n004_pcq --job-id scd-alpha_nonorms_lr3dp001 --dataset-arg alpha
# python train.py --conf configs/scd-qm9-ft-long.yaml --load-hf scd-s-n004_pcq --job-id scd-alpha_nonoise --dataset-arg alpha
# python train.py --conf configs/scd-qm9-ft-long.yaml --load-hf scd-s-n004_pcq --job-id scd-alpha_LAref --dataset-arg alpha


# python train.py --conf configs/scd-md17-SL.yaml --job-id scdsm-benz-bs512-cos

# python train.py --conf configs/scd-md17-SL.yaml --job-id scdsm-benz-bs8-cos

# python train.py --conf configs/scd-md17-SL.yaml --job-id scdsm-benz-bs16-add_embhead

# python train.py --conf configs/et-rmd17-SL.yaml --job-id et-debug
# python train.py --conf configs/scd-md17-SL.yaml --job-id scd-benz-baseline
# python train.py --conf configs/scd-md17-SL.yaml --job-id scd-benz-wd001
# python train.py --conf configs/scd-md17-SL.yaml --job-id scd-benz-wd001-dbwu
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scd-s-n004_pcq --job-id scd-sn004_benz_debug


# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdS_benz_beta995 --beta1 0.995

# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdS_benz_beta995_bs32 --beta1 0.995 --batch-size 32

# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdS_benz_beta995_n005 --beta1 0.995 --noise-scale 0.005
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdS_benz_b1995_b29 --beta1 0.995 --beta2 0.9
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdS_benz_b1995_nonorms --beta1 0.995

# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdS_benz_b1995_nonorms_lr5e4 --beta1 0.995 --lr 0.0005
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdS_benz_b1995_vecnorms_lr4e4 --beta1 0.995 --lr 0.0004
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdS_benz_b1995_vecnorms_lr1e4_1 --beta1 0.995 --lr 0.0001 --inference-batch-size 64

##### RUNNING ####
# python train.py --conf configs/scd-qm9-ft-energy.yaml --load-hf scd-s-n004_pcq --job-id scd-sn004_u0_b64 --dataset-arg u0
# python train.py --conf configs/scd-qm9-ft-energy.yaml --load-hf scd-s-n004_pcq --job-id scd-sn004_u0_b32 --dataset-arg u0
# python train.py --conf configs/scd-qm9-ft-energy.yaml --load-hf scd-s-n004_pcq --job-id scd-sn004_u296_b128 --dataset-arg u298
# python train.py --conf configs/scd-qm9-ft-energy.yaml --load-hf scd-s-n004_pcq --job-id scd-sn004_h296_b128 --dataset-arg h298
# python train.py --conf configs/scd-qm9-ft-energy.yaml --load-hf scd-s-n004_pcq --job-id scd-sn004_g296_b128 --dataset-arg g298

# python train.py --conf configs/scd-qm9-ft-energy.yaml --load-hf scd-s-n004_pcq --job-id scd-sn004_g296_b128_b1995 --dataset-arg g298




# python train.py --conf configs/scd-qm9-ft-energy.yaml --load-hf scd-s-n004_pcq --job-id scd-sn004_g296_b128_b1995_bs256 --dataset-arg g298 --batch-size 256 --inference-batch-size 256
# python train.py --conf configs/scd-qm9-ft-energy.yaml --load-hf scd-s-n004_pcq --job-id scd-sn004_g296_bs256_bswp --dataset-arg g298 --batch-size 256 --inference-batch-size 256
# python train.py --conf configs/scd-qm9-ft-energy.yaml --load-hf scd-s-n004_pcq --job-id scd-sn004_g296_bs256_b9999 --dataset-arg g298 --batch-size 256 --inference-batch-size 256

# python train.py --conf configs/scd-qm9-ft-energy.yaml --load-hf scd-s-n004_pcq --job-id scd-sn004_g296_bs256_b98 --dataset-arg g298 --batch-size 256 --inference-batch-size 256
# python train.py --conf configs/scd-qm9-ft-energy.yaml --load-hf scd-s-n004_pcq --job-id scd-sn004_g296_bs256_b98_1 --dataset-arg g298 --batch-size 256 --inference-batch-size 256

# python train.py --conf configs/scd-qm9-ft-energy.yaml --load-hf scd-s-n004_pcq --job-id test_g296_bs256_b995_yema001 --dataset-arg g298 --batch-size 256 --inference-batch-size 256 --ema-alpha-y 0.01
# python train.py --conf configs/scd-qm9-ft-energy.yaml --load-hf scd-s-n004_pcq --job-id test_g296_bs256_b995_yema01 --dataset-arg g298 --batch-size 256 --inference-batch-size 256 --ema-alpha-y 0.1




# python train.py --conf configs/scd-md17-SL.yaml --job-id scdsm-benz-bs512

# python train.py --conf configs/et-md17-SL.yaml --job-id etsm-benz-b8
# python train.py --conf configs/et-md17-SL.yaml --job-id etsm-benz-b32
# python train.py --conf configs/et-md17-SL.yaml --job-id etsm-benz-b128
# python train.py --conf configs/et-md17-SL.yaml --job-id etsm-benz-b512

# python train.py --conf configs/scd-md17-SL.yaml --job-id scd-benz-wd001-dbwu-b32

# python train.py --conf configs/et-QM9-FT.yaml --job-id debug-coord-u0 --pretrained-model experiments/coord-baseline/last.ckpt
# python train.py --conf configs/et-QM9-FT-energy.yaml --job-id debug-coord-u0 --pretrained-model experiments/coord-baseline/last.ckpt

# python train.py --conf configs/et-QM9-SL.yaml --job-id et-SL-u0


#### load pretrained hf model 
# python train.py --conf configs/nersc_qm9-ft-energy.yaml --load-hf cet-pcq_2 --job-id cet-ft-u0-test

# python train.py --conf configs/nersc_qm9-ft-energy.yaml --load-hf cet-pcq_2 --job-id cet-ft-u0_atom_test

# python train.py --conf configs/nersc_qm9-ft-energy.yaml --load-hf cet-pcq_2 --job-id cet-ft-u0Aref_embpregate

# python train.py --conf configs/nersc_qm9-ft-energy.yaml --load-hf cet-pcq_2 --job-id cetft-u0Aref_yema08_dnw08_lr5e5

# python train.py --conf configs/nersc_qm9-ft-energy.yaml --load-hf cet-pcq_2 --job-id u0atom-std_momentum
# python train.py --conf configs/nersc_qm9-ft-energy.yaml --load-hf cet-pcq_2 --job-id u0atom-std_yema-modema
# python train.py --conf configs/nersc_qm9-ft-energy.yaml --load-hf cet-pcq_2 --job-id u0atom-std_no-ema

# python train.py --conf configs/nersc_qm9-ft-energy.yaml --load-hf cet-pcq_2 --job-id u0Aref_newLN

# python train.py --conf configs/nersc_qm9-ft-energy.yaml --load-hf cet-pcq_2 --job-id u0Aref_rsetLN-H-E_lowlr_n0001

# python train.py --conf configs/nersc_qm9-ft-energy.yaml --load-hf cet-pcq_2 --job-id u0Aref_rsetLN-H-E_lowlr_n0

# python train.py --conf configs/nersc_qm9-ft-energy.yaml --load-hf cet-pcq_2 --job-id u0Aref_rsetLNHE_lr1e4n005_rDyT

# python train.py --conf configs/nersc_qm9-ft-energy.yaml --load-hf cet-pcq_2 --job-id u0Aref_rsetLNHE_lr1e4n005_rLN

# python train.py --conf configs/nersc_qm9-ft-long.yaml --load-hf cet-pcq_2 --job-id alpha_rsetLNHE_DyT

# python train.py --conf configs/nersc_qm9-ft-energy.yaml --load-hf cet-pcq_2 --job-id h298_wd03
# python train.py --conf configs/nersc_qm9-ft-long.yaml --load-hf cet-pcq_2 --job-id zpve_cet

# python train.py --conf configs/nersc_qm9-ft-base.yaml --load-hf cet-pcq_2 --job-id ehomo_rsetemb

# python train.py --conf configs/nersc_qm9-ft-base.yaml --load-hf cet-pcq_2 --job-id ehomo_asis

# python train.py --conf configs/nersc_qm9-ft-energy.yaml --load-hf cet-pcq_2 --job-id g298_nw05_b64



##### SL TEST
# python train.py --conf configs/scd-qm9-sl.yaml --job-id u0_newscd_test

# python train.py --conf configs/et-QM9-SL.yaml --job-id u0_et-baseline

# python train.py --conf configs/scd-qm9-sl.yaml --job-id u0_newscd_sig-gate

# python train.py --conf configs/scd-qm9-sl.yaml --job-id u0_scd_detach-x

# python train.py --conf configs/scd-qm9-sl.yaml --job-id u0_scd_detach-invpnorm

# python train.py --conf configs/scd-qm9-sl.yaml --job-id u0_scd_detach-invpnorm-dpth0

# python train.py --conf configs/scd-qm9-sl.yaml --job-id u0_scd_detach-invpnorm-jpth01

# python train.py --conf configs/scd-qm9-sl.yaml --job-id u0_scd_detach-invpnorm-jpth01-vecnorm

# python train.py --conf configs/scd-qm9-sl.yaml --job-id u0-scd_wd001_dpth-warmup 

# python train.py --conf configs/scd-qm9-sl.yaml --job-id u0-scd_vecDyT_dpth-warmup

# python train.py --conf configs/scd-qm9-sl.yaml --job-id ckpt_debug_test


# python train.py --conf configs/scd-qm9-sl.yaml --job-id momup_on