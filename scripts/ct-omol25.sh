#!/bin/bash
#SBATCH -J omol-ct_800ksteps_n4_bwdF  # job name
#SBATCH --account=m5068_g  ### m5068_g ,   m4866
#SBATCH -C gpu                          # request GPU nodes
#SBATCH -q premium                      # 4 GPUs -> regular QoS # regular or premium
#SBATCH -t 3:00:00                     # job time limit hh:mm:ss, note 16h is not enough for longer runs of 500k

#SBATCH --constraint 'gpu&hbm80g'   # use hbm80g nodes

#SBATCH --nodes=4                       # node count
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

conf=configs/ct-omol25.yaml # for others
target=s2ef #

## OLD: running as 'omol25_base_800ksteps_n4_fwdF'
## running as 'omol-ct_800ksteps_n4_fwdF'
# model_name=null
# suffix=n4_800ksteps_fwdF

## OLD: running as 'omol25_base_1p6Msteps_n4_fwdF'
## running as 'omol-ct_1p6Msteps_n4_fwdF'
# model_name=null
# suffix=n4_1p6Msteps_fwdF


## OLD: running as 'omol25_base_800ksteps_n4_bwdF'
## running as 'omol-ct_800ksteps_n4_bwdF'
model_name=null
suffix=n4_800ksteps_bwdF

# ## running as 'ct-scd-omol25_FE_n4_fwdF'
# model_name=ct-scd-omol25_4n # SCD on OMOL25 4M 
# suffix=n4_800ksteps_fwdF

# running as 'ct-scd-omol25_FE_n4_bwdF'
# model_name=ct-scd-omol25_4n # SCD on OMOL25 4M 
# suffix=n4_800ksteps_bwdF

# model_name=scd-s-n004_pcq
# model_name=scd-n004_geom_b48n4
# model_name=cet-sair_b12n2_0
# model_name=cet-sairpocket_b12n2_0 # best so far
# model_name=cet-sairligand_b12n2_0
# model_name=cet-amp20_b142n1_0

## running as 'omol25_ALL_1Msteps'
# model_name=cet-all_b56n4_2 # 
# suffix=_1Msteps

max_epochs_per_run=6000


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
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --dataset-arg "$target" --lr 0.001
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --dataset-arg "$target" --lr 0.0001
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --dataset-arg "$target" --weight-decay 0.001
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --dataset-arg "$target" --ema-alpha-y 0.05

    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --dataset-arg "$target" --ema-alpha-y 0.05 --force-weight 2.0 --energy-weight 0.01
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --dataset-arg "$target" --ema-alpha-y 0.05 --force-weight 2.0 --energy-weight 0.01 --derivative True --direct-force-pred True
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --dataset-arg "$target" --ema-alpha-y 0.05 --force-weight 2.0 --energy-weight 0.01 --derivative False --direct-force-pred True

    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --dataset-arg "$target" --ema-alpha-y 0.05 --force-weight 1.0 --energy-weight 0.1 --derivative False --direct-force-pred True
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --dataset-arg "$target" --ema-alpha-y 0.05 --force-weight 1.0 --energy-weight 0.1 --derivative False --direct-force-pred True --final-droppath 0.1
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --dataset-arg "$target" --ema-alpha-y 0.05 --force-weight 1.0 --energy-weight 0.1 --derivative False --direct-force-pred True --lr 0.0008 --lr-warmup-steps 2000
    srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --dataset-arg "$target" --ema-alpha-y 0.05 --force-weight 1.0 --energy-weight 0.1 --derivative False --direct-force-pred True --lr 0.0008 --lr-warmup-steps 2000 --num-steps 800000 --lr-cosine-length 800000 # fwdF 800k
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --dataset-arg "$target" --ema-alpha-y 0.05 --force-weight 1.0 --energy-weight 0.1 --derivative False --direct-force-pred True --lr 0.0008 --lr-warmup-steps 2000 --num-steps 1600000 --lr-cosine-length 1600000 # fwdF 1.6M
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --dataset-arg "$target" --ema-alpha-y 0.05 --force-weight 1.0 --energy-weight 0.1 --derivative True --direct-force-pred False --lr 0.0008 --lr-warmup-steps 2000 --num-steps 800000 --lr-cosine-length 800000 # bwdF 800k



else
### finetune job 
    echo "Running with pretrained model: $load_model"
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --pretrained-model "$load_model" --dataset-arg "$target"
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --pretrained-model "$load_model" --dataset-arg "$target" --lr 0.0001
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --pretrained-model "$load_model" --dataset-arg "$target" --lr 0.0001 --ema-alpha-y 0.05 --force-weight 1.0 --energy-weight 0.1 --derivative False --direct-force-pred True
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --pretrained-model "$load_model" --dataset-arg "$target" --lr 0.0005  --lr-warmup-steps 1000 --ema-alpha-y 0.05 --force-weight 1.0 --energy-weight 0.1 --derivative False --direct-force-pred True

    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --pretrained-model "$load_model" --dataset-arg "$target" --lr 0.0003  --lr-warmup-steps 2000 --ema-alpha-y 0.05 --force-weight 1.0 --energy-weight 0.1 --derivative False --direct-force-pred True --num-steps 1000000 --lr-cosine-length 1000000
    srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --pretrained-model "$load_model" --dataset-arg "$target" --ema-alpha-y 0.05 --force-weight 1.0 --energy-weight 0.1 --derivative True --direct-force-pred False --lr 0.0008 --lr-warmup-steps 2000 --num-steps 800000 --lr-cosine-length 800000 # bwdF 800k
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --pretrained-model "$load_model" --dataset-arg "$target" --ema-alpha-y 0.05 --force-weight 1.0 --energy-weight 0.1 --derivative False --direct-force-pred True --lr 0.0008 --lr-warmup-steps 2000 --num-steps 800000 --lr-cosine-length 800000 # fwdF 800k
fi

