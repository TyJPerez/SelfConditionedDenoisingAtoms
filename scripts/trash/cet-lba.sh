#!/bin/bash
#SBATCH -J ct-FEbwd-lba-id30_rsethead   # job name
#SBATCH --account=m5068_g  ### m5068_g ,   m4866
#SBATCH -C gpu                          # request GPU nodes
#SBATCH -q premium                      # 4 GPUs -> regular QoS # regular or premium
#SBATCH -t 6:00:00                     # job time limit hh:mm:ss, note 16h is not enough for longer runs of 500k

####### #SBATCH --constraint 'gpu&hbm80g'   # use hbm80g nodes

#SBATCH --nodes=1                       # node count
#SBATCH --ntasks-per-node=4
#SBATCH -c 32                           # 32 CPUs per task (Perlmutter example)
#SBATCH --gpus-per-node=4               # number of gpus per node
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none                 # all-visible or let Lightning/NCCL map
#SBATCH --dependency=singleton      # only one job with this name at a time
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# sbatch -J debut_ft scripts/finetune_qm9_job.sh -t homo -m cet-pcq_2_bk

mkdir -p logs # create logs directory 

conf=configs/cet-lba_nersc.yaml # for others
target=seq-id-30
# target=seq-id-60


# model_name=null
# suffix=_nb_50kstep

# model_name=scd-s-n004_pcq
# model_name=scd-n004_geom_b48n4
# model_name=cet-sair_b12n2_0
# model_name=cet-sairpocket_b12n2_0 # best so far
# model_name=cet-sairligand_b12n2_0
# model_name=cet-amp20_b142n1_0
# model_name=cet-all_b56n4_2 # 

# model_name=ct-scd-omol25_4n # SCD on OMOL25 4M 

## FE pretrained
# model_name=null_FT_s2ef_n4_800ksteps_fwdF # scratch FE pretrain fwd forces
model_name=null_FT_s2ef_n4_800ksteps_bwdF # scratch FE pretrain bwd forces

suffix=_rsethead

# model_name=null_FT_s2ef_yma-f1e01-fwdF-lr8e4ws2k
# suffix=_FEpt

max_epochs_per_run=300000


############# PARSE INPUTS #############
# Function to show usage
usage() {
    echo "Usage: sbatch $0 [-t target] [-m model_name]"
    echo "  -t: Target property (default: homo)"
    echo "  -m: Model name (default: cet-pcq_2_bk)"
    exit 1
}

# Parse arguments
while getopts "t:m:h" opt; do
    case $opt in
        t)
            target="$OPTARG"
            ;;
        m)
            model_name="$OPTARG"
            ;;
        h)
            usage
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            usage
            ;;
    esac
done

# Validate required parameters
if [[ -z "$target" ]]; then
    echo "Error: Target is required"
    usage
fi

echo "Running fine-tuning with:"
echo "  Target: $target"
echo "  Model: $model_name"

############# PARSE INPUTS #############

load_model="experiments/$model_name/last.ckpt"


#construct a wandb job name
job_id="${model_name}_FT_${target}_${suffix}"
echo "job ID: $job_id"

OUT_DIR="experiments/$job_id"  # choose a PSCRATCH path
LOG_DIR="$OUT_DIR/logs"


#make log dir 
mkdir -p $OUT_DIR
mkdir -p $LOG_DIR

set -euo pipefail
export SLURM_CPU_BIND=cores

# NCCL settings for Perlmutter
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_IB_HCA=mlx5
export NCCL_SOCKET_IFNAME=hsn0
export NCCL_TIMEOUT=1800
export NCCL_ASYNC_ERROR_HANDLING=1

# PyTorch distributed settings
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500


# Debug: Check GPU assignment
echo "Task $SLURM_PROCID on node $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi -L

module load python
conda activate torch


if [[ "$model_name" == "null" ]]; then
### regular SL job
    echo "Running without pretrained model (training from scratch)"
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --dataset-arg "$target"

    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --dataset-arg "$target"
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --dataset-arg "$target" --lr 0.001
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --dataset-arg "$target" --noise-scale 0.01
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --dataset-arg "$target" --batch-size 4
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --dataset-arg "$target" --noise-scale 0.00001 --denoising-weight 0.00001
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --dataset-arg "$target" --init-dropout 0.0 --final-dropout 0.0
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --dataset-arg "$target" --allow-node-mask True --set-head-agg sum  # ligsum
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --dataset-arg "$target" --allow-node-mask True --set-head-agg mean # ligmean
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --dataset-arg "$target" --allow-node-mask True --set-head-agg mean --inverse-node-mask True
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --dataset-arg "$target" --allow-node-mask True --set-head-agg sum --inverse-node-mask True
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --dataset-arg "$target" --momentum-update True --val-ema-model True
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --dataset-arg "$target" --ema-alpha-y 0.05

    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --dataset-arg "$target" --allow-node-mask True --set-head-agg mean --lr 0.0002 # ligmean 
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --dataset-arg "$target" --allow-node-mask True --set-head-agg mean --lr 0.00005 # ligmean
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --dataset-arg "$target" --allow-node-mask True --set-head-agg mean --lr 0.0002 --final-droppath 0.1 # best so far
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --dataset-arg "$target" --allow-node-mask True --set-head-agg mean --lr 0.0002 --ema-alpha-y 0.05 

    ## default switched to lr 0.0002, final droppath 0.1
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --dataset-arg "$target" --allow-node-mask True --set-head-agg mean --beta1 0.9
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --dataset-arg "$target" --allow-node-mask True --set-head-agg mean --denoising-weight 0.3
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --dataset-arg "$target" --allow-node-mask True --set-head-agg mean --weight-decay 0.05  #imporved 
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --dataset-arg "$target" --allow-node-mask True --set-head-agg mean --final-droppath 0.15 #improved
    
    #new baseline and sweep
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --dataset-arg "$target" # combine higher weight decay (0.05) and final droppath 0.15, and use full lig/prot
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --dataset-arg "$target" --init-droppath 0.2 --final-droppath 0.2
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --dataset-arg "$target" --init-droppath 0.3 --final-droppath 0.3
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --dataset-arg "$target" --lr-warmup-steps 10000 # long warmup
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --dataset-arg "$target" --lr-warmup-steps 1000 # short warmup
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --dataset-arg "$target" --lr-cosine-length 50000 --num-steps 50000 --lr-warmup-steps 500 # short run # not actually ligmask

    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --dataset-arg "$target" --lr-cosine-length 60000 --num-steps 60000 --lr-warmup-steps 1000 #--allow-node-mask True --set-head-agg mean # short run

    srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --dataset-arg "$target" --lr-cosine-length 30000 --num-steps 30000 --lr-warmup-steps 500 --allow-node-mask True  ### same params as best pretrianed

else
### finetune job 
    echo "Running with pretrained model: $load_model"
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --pretrained-model "$load_model" --dataset-arg "$target"
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --pretrained-model "$load_model" --dataset-arg "$target" --lr 0.0001 #standard
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --pretrained-model "$load_model" --dataset-arg "$target" --lr 0.00005
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --pretrained-model "$load_model" --dataset-arg "$target" --lr 0.0001 --allow-node-mask True # ligand masking
    
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --pretrained-model "$load_model" --dataset-arg "$target" --lr 0.0001 --allow-node-mask True --lr-cosine-length 50000 --num-steps 50000 --lr-warmup-steps 500
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --pretrained-model "$load_model" --dataset-arg "$target" --lr 0.0001 --allow-node-mask True --lr-cosine-length 60000 --num-steps 60000 --lr-warmup-steps 1000
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --pretrained-model "$load_model" --dataset-arg "$target" --lr 0.0001 --allow-node-mask True --lr-cosine-length 40000 --num-steps 40000 --lr-warmup-steps 1000
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --pretrained-model "$load_model" --dataset-arg "$target" --lr 0.0001 --allow-node-mask True --lr-cosine-length 30000 --num-steps 30000 --lr-warmup-steps 1000

    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --pretrained-model "$load_model" --dataset-arg "$target" --lr 0.0004 --allow-node-mask True --lr-cosine-length 40000 --num-steps 40000 --lr-warmup-steps 100
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --pretrained-model "$load_model" --dataset-arg "$target" --lr 0.0004 --allow-node-mask True --lr-cosine-length 40000 --num-steps 40000 --lr-warmup-steps 10000

    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --pretrained-model "$load_model" --dataset-arg "$target" --lr 0.0001 --lr-cosine-length 40000 --num-steps 40000 --lr-warmup-steps 1000
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --pretrained-model "$load_model" --dataset-arg "$target" --lr 0.0001 --lr-cosine-length 40000 --num-steps 40000 --lr-warmup-steps 1000 --allow-node-mask True
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --pretrained-model "$load_model" --dataset-arg "$target" --lr 0.0001 --lr-cosine-length 20000 --num-steps 20000 --lr-warmup-steps 1000 --allow-node-mask True
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --pretrained-model "$load_model" --dataset-arg "$target" --lr 0.00008 --lr-cosine-length 30000 --num-steps 30000 --lr-warmup-steps 1000 --allow-node-mask True
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --pretrained-model "$load_model" --dataset-arg "$target" --lr 0.0001 --lr-cosine-length 30000 --num-steps 30000 --lr-warmup-steps 100 --allow-node-mask True  # lower warmup is worse

    #normal run
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --pretrained-model "$load_model" --dataset-arg "$target" --lr 0.0001 --lr-cosine-length 30000 --num-steps 30000 --lr-warmup-steps 500 --allow-node-mask True   #best results

    #with head reset
    srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --pretrained-model "$load_model" --dataset-arg "$target" --lr 0.0001 --lr-cosine-length 30000 --num-steps 30000 --lr-warmup-steps 500 --allow-node-mask True --reset-head True 

    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --pretrained-model "$load_model" --dataset-arg "$target" --lr 0.0001 --lr-cosine-length 30000 --num-steps 30000 --lr-warmup-steps 500 --allow-node-mask True --dataset LBAR  #residue labels on pocket atoms

fi

