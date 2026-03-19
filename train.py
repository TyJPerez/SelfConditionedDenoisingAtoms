import numpy as np  # sometimes needed to avoid mkl-service error
import sys
import os
import argparse
import logging
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from data import datasets
from data.loaders import DataModule
from data.utils import LoadFromFile, LoadFromCheckpoint, save_argparse, number

from tqdm import tqdm
from models.trainer import LTrainer
from models.model_helper import get_model_checkpoint, create_prior_models
from models.callbacks import ParityPlot, LimitRun
from pytorch_lightning.callbacks import ModelCheckpoint, Callback

import torch
import wandb
from pathlib import Path
from datetime import datetime

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

from huggingface_hub import hf_hub_download

def download_model_checkpoint(model_name, save_dir='./experiments', repo_id_prefix="Ty-Perez/"):

    available_models = ["ct-scd-pcq", "ct-scd-amp"]  # List of available model names in the Huggingface repo

    #check if model_name is in available_models
    if model_name not in available_models:
        raise ValueError(f"Model {model_name} not available. Available models: {available_models}")

    print(f"Loading model from Huggingface repo: {model_name}")
    # cache_dir = "/home/tjp/Projects/SelfConditionedDenoising/experiments/hf_models"
    # model_name = args.load_hf
    
    assert os.path.exists(save_dir), f"Save dir {save_dir} does not exist. Please create it first or change the save dir variable in this script."
    
    hf_ckpt_path = hf_hub_download(
        repo_id=f"{repo_id_prefix}{model_name}",
        filename="last.ckpt",
        cache_dir=save_dir
    )
    return hf_ckpt_path

def get_args():

    # fmt: off
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--pretrained-model', default=None, type=str, help='Pre-trained weights checkpoint.')
    parser.add_argument('--load-model', help='load model from a given epoch, default to last epoch')
    parser.add_argument('--load-hf', default=None, help='load model from huggingface, provide the repo name')
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
    
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay strength')
    parser.add_argument('--ema-alpha-y', type=float, default=1.0, help='The amount of influence of new losses on the exponential moving average of y')
    parser.add_argument('--ema-alpha-dy', type=float, default=1.0, help='The amount of influence of new losses on the exponential moving average of dy')
    
    parser.add_argument('--ngpus', type=int, default=-1, help='Number of GPUs, -1 use all available. Use CUDA_VISIBLE_DEVICES=1, to decide gpus')
    parser.add_argument('--gpu-ids', type=str, default=None, help='Comma-separated list of GPU IDs to use (e.g., "0,1,2"). Overrides CUDA_VISIBLE_DEVICES.')
    parser.add_argument('--use-devices', type=int, nargs='+', default=[0], help='List of GPU device indices to use for local (non-cluster) training (e.g., --use-devices 0 1 2). Default: [0]')
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
    
    # Dataset specific arguments
    parser.add_argument('--dataset', default=None, type=str, choices=datasets.__all__, help='Name of the torch_geometric dataset')
    parser.add_argument('--dataset-root', default='data', type=str, help='Data storage directory (not used if dataset is "CG")')
    parser.add_argument('--dataset-arg', default=None, type=str, help='Additional dataset argument, e.g. target property for QM9 or molecule for MD17')
    
    parser.add_argument('--energy-weight', default=1.0, type=float, help='Weighting factor for energies in the loss function')
    parser.add_argument('--force-weight', default=1.0, type=float, help='Weighting factor for forces in the loss function')
    parser.add_argument('--noise-scale', default=0., type=float, help='Scale of Gaussian noise added to positions.')
    parser.add_argument('--denoising-weight', default=0., type=float, help='Weighting factor for denoising in the loss function.')

    # Data augmentation arguments
    parser.add_argument('--random-rotate', type=bool, default=False, help='If true, apply random rotation augmentation to positions')
    parser.add_argument('--p-invert', type=float, default=0.0, help='Probability of inverting the coordinates')
    parser.add_argument('--torsion-angle-std', type=float, default=0.0, help='Standard deviation of torsion angles in degrees')
    parser.add_argument('--max-bonds-rotated', type=int, default=0, help='Maximum number of bonds to rotate in torsion augmentation')

    # Prior model arguments
    parser.add_argument('--prior-model', type=str, default=None, help='Which prior model to use. It can be a string, a dict if you want to add arguments for it or a dicts to add more than one prior. e.g. {"Atomref": {"max_z":100}, "Coulomb":{"max_num_neighs"=100, "lower_switch_distance"=4, "upper_switch_distance"=8}', action="extend", nargs="*")

    # Training mode arguments
    parser.add_argument('--pretraining', type=bool, default=False, help='If true, the model is pretrained')
    parser.add_argument('--self_cond', type=bool, default=False, help='use self conditining ssl step')

    # Job scheduling arguments
    parser.add_argument("--max_time_minutes", type=int, default=None, help = 'Maximum wall time in minutes')
    parser.add_argument("--max_epochs_per_run", '-mep', type=int, default=None, help = 'Maximum number of epochs per run')

    # Fine-tuning: model architecture modifications
    parser.add_argument('--reset-head', type=bool, default=False, help='If true, reset the scalar head of the model when loading from a pretrained model.')
    parser.add_argument('--reset-embeddings', type=bool, default=False, help='If true, reset the embeddings of the model when loading from a pretrained model.')
    parser.add_argument('--reset-norms', type=str, default=None, help='If true, reset layer norms of the model. string, either LN, CN, or None')

    # Regularization arguments
    parser.add_argument('--droppath_warmup', type=bool, default=False, help='If true, linearly decrease droppath rate during early training phase.')
    parser.add_argument('--init-droppath', type=float, default=0.15, help='Initial droppath rate if droppath_warmup is true.')
    parser.add_argument('--final-droppath', type=float, default=0.0, help='Final droppath rate if droppath_warmup is true.')

    # Model momentum/EMA arguments
    parser.add_argument('--momentum-update', type=bool, default=False, help='If true, maintain a model copy updated by momentum.')
    parser.add_argument('--model-momentum', type=float, default=0.996, help='Decay rate for model momentum updates.')
    parser.add_argument('--use-ema-model', type=bool, default=False, help='If true, when loading a pretrained checkpoint use the EMA model as the primary model if it exists.')
    parser.add_argument('--val-ema-model', type=bool, default=False, help='If true, use the EMA model for validation if it exists.')

    # Training configuration
    parser.add_argument('--derivative', type=str2bool, default=None, help='If true, model returns energy and forces (dy)') ### TODO: NEW
    parser.add_argument('--gradient-clipping', type=float, default=1.0, help='Gradient clipping norm')
    parser.add_argument('--val-interval', type=int, default=1, help='Test interval, one test per n epochs (default: 10)')
    parser.add_argument('--add-head-to-pred', type=bool, default=False, help='If true, add embedding head prediction to final y prediction.')
    parser.add_argument('--weight-decay-on-head', type=bool, default=False, help='If true, apply weight decay to the scalar head of the model.')

    # Optimizer arguments
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 for AdamW optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='Beta2 for AdamW optimizer')
    
    # Contrastive loss arguments
    parser.add_argument('--create-contrastive-loss', type=bool, default=False, help='If true, create a contrastive loss module.')
    parser.add_argument('--contrastive-loss-weight', type=float, default=0.0, help='Weight of the contrastive loss.')
    parser.add_argument('--ctr-init-tau', type=float, default=0.1, help='Initial temperature for contrastive loss.')

    # Periodic materials dataset arguments
    parser.add_argument('--noise_in_loader', type=bool, default=False, help='If true, add noise to positions in the data loader instead of in the model.')
    parser.add_argument('--graph_cutoff', type=float, default=5.0, help='if graph creation in loader, Cutoff distance for neighbor list in periodic datasets.')
    parser.add_argument('--p_cell_repeat', type=float, default=0.15, help='if periodic system, number of times to repeat the unit cell in each direction.')
    parser.add_argument('--cell_repeat_iters', type=int, default=1, help='Number of iterations to probabilistically repeat cell based on p_cell_repeat.')
    parser.add_argument('--allow_periodic', type=bool, default=False, help='If true, allow periodic boundary conditions in graph construction.')
    parser.add_argument('--neighbor_method', type=str, default='brute', help='Method for neighbor search in periodic datasets (default: brute). options: brute, grid')
    parser.add_argument('--max_neighbors', type=int, default=32, help='Maximum number of neighbors in loader graph generation.')

    # Batch clipping arguments
    parser.add_argument('--max-nodes-per-batch', type=int, default=None, help='Maximum number of nodes per batch. If set, batches will be clipped to this size.')
    parser.add_argument('--batch-clipper-cache-size', type=int, default=1000, help='Maximum number of samples to cache in the batch clipper.')
    parser.add_argument('--allow-test-clipping', type=bool, default=True, help='If true, allow batch clipping during validation and test.')

    # Matbench dataset arguments
    parser.add_argument('--predefined_splits', type=bool, default=False, help='If true, use predefined splits for matbench datasets.')
    parser.add_argument('--rep-min-atoms', type=int, default=4, help='Minimum number of atoms to trigger cell repetition.')
    parser.add_argument('--set-head-agg', type=str, default=None, help='Aggregation method for the head of the model (default: sum). options: sum, mean, max')

    # Plotting and debugging arguments
    parser.add_argument('--no-wandb-resume', action='store_true', default=False, help='If true, allow wandb to resume previous runs in the same log dir. If false, always start a new run in wandb.')
    parser.add_argument('--parity-plot', type=bool, default=False, help='If true, create parity plots after training.')
    parser.add_argument('--plot-interval', type=int, default=-1, help='Interval (in epochs) to create parity plots during training.')
    parser.add_argument('--compute-corr', type=bool, default=False, help='If true, compute and log correlation metrics after each epoch.')

    ### for LBA dataset ligand masking
    parser.add_argument('--allow-node-mask', type=bool, default=False, help='If true, mask ligand atoms in the LBA dataset during training.')
    parser.add_argument('--inverse-node-mask', type=bool, default=False, help='If true, mask ligand atoms in the LBA dataset during training.')

    ### for omol only
    parser.add_argument('--direct-force-pred', type=str2bool, default=False, help='If true, model directly predicts forces instead of via energy derivative.')


    args = parser.parse_args()

    if args.job_id == "auto":
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cuda_visible:
            assert len(cuda_visible.split(',')) == 1, "Might be problematic with DDP."
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
   
    #format use-devices as an ordered list of integers, e.g. [0,1,2]
    devs = []
    for d in args.use_devices:
        if isinstance(d, int):
            devs.append(d)
        elif isinstance(d, str) and d.isdigit():
            devs.append(int(d))
        else:
            raise ValueError(f"Invalid device index: {d}. use-devices should be a list of integers or strings that can be converted to integers.")
    #order the devices
    devs = sorted(devs)
    args.use_devices = devs
    print(f"Formatted use-devices: {args.use_devices}")

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
    
    use_devices = args.use_devices  # local: user-specified list, e.g. [0] or [0,1,2]

    #check if SLURM_NNODES is set, if so use it to set num_nodes and switch to ngpus
    if 'SLURM_NNODES' in os.environ:
        args.num_nodes = int(os.environ.get('SLURM_NNODES', 1))
        use_devices = args.ngpus  # cluster: let PL pick all visible GPUs
    print(f"Using {args.num_nodes} nodes, devices={use_devices} for training.")

    pl.seed_everything(args.seed, workers=True)

    
    # initialize data module
    data = DataModule(args)
    data.prepare_data()
    data.setup("fit")

    prior_models = create_prior_models(vars(args), data.dataset)
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
            if ckpt_step-1 == _step:
                print(f"WARNING: Checkpoint step {ckpt_step} matches expected step {_step}")
            print(f"Restarting from checkpoint: {checkpoint_path}, epoch: {_epoch}, step: {_step}")
            args.load_model = checkpoint_path

    if args.load_hf is not None:

        if checkpoint_path is not None:
            print(f"WARNING: both --restart and --load-hf were specified. Ignoring --load-hf and using checkpoint {checkpoint_path}")
        else:

            args.pretrained_model = download_model_checkpoint(args.load_hf, save_dir=args.log_dir)
    
    if checkpoint_path is not None:
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

    if args.pretraining:
        project_name = 'SCD_pretraining'
    else:
        ds_name = args.dataset
        project_name = f'SCD_bench_{ds_name}'
    
    force_id = args.restart and checkpoint_path is not None
    resume_flag = "must" if force_id else 'allow'
    
    if args.no_wandb_resume:
        resume_flag = 'never'
        datetime_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.job_id = f"{args.job_id}_{datetime_str}"
        

    wandb_logger = WandbLogger(
        name=args.job_id,
        project=project_name,
        notes=args.wandb_notes,
        resume=resume_flag,
        id=args.job_id
    )

    ddp_strategy = None
    if "ddp" in args.distributed_backend:
        ddp_strategy = DDPStrategy(find_unused_parameters=True)  # To account for the teacher model

    
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
    print(f"Make parity plot: {make_parity_plot}")
    if make_parity_plot:
        parity_callback = ParityPlot(
            output_dir=args.log_dir,
            plot_interval=args.plot_interval,
            stages=['test'],  # When to plot during training
            end_stages=['train', 'test']  # What to plot at the end of training
        )
        callbacks.append(parity_callback)

    trainer_kwargs = dict(
        inference_mode=use_inference_mode,
        max_epochs=args.num_epochs,
        max_steps=args.num_steps,
        # devices=[1],  # Manually set to 1 GPU
        devices=use_devices,
        num_nodes=args.num_nodes,
        accelerator="gpu",
        default_root_dir=args.log_dir,
        callbacks=callbacks,
        logger=wandb_logger,
        precision=args.precision,
        gradient_clip_val=args.gradient_clipping,
        check_val_every_n_epoch=args.val_interval,
        reload_dataloaders_every_n_epochs=args.val_interval,  # To shuffle data if using small val set
        log_every_n_steps=50,
    )

    if ddp_strategy is not None:
        trainer_kwargs["strategy"] = ddp_strategy

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, data, ckpt_path=checkpoint_path)

    # Set model to eval mode before testing
    model.eval()

    # Run test set after completing the fit
    trainer.test(model=model, datamodule=data)


if __name__ == "__main__":
    main()