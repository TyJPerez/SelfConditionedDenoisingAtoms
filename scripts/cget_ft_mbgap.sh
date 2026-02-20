#!/bin/bash
#SBATCH -J cget-AMP20-mbgap_f0
#SBATCH --account=m5068_g  ### m5068_g ,   m4866
#SBATCH -C gpu                          # request GPU nodes
#SBATCH -q regular                      # 4 GPUs -> regular QoS # regular or premium
#SBATCH -t 32:00:00                     # job time limit hh:mm:ss, note 16h is not enough for longer runs of 500k

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

conf=configs/cget-mbgap.yaml # for others
target=0

### cget-baseline-mbgap_f0_0
# model_name=null
model_name=cget-amp20_b32n2_0

suffix=f0_lr8e5

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
    srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --dataset-arg "$target" --lr 0.0002
else
### finetune job 
    echo "Running with pretrained model: $load_model"
    srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --pretrained-model "$load_model" --dataset-arg "$target" --lr 0.00008
fi


## finetune job 
# srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --pretrained-model "$load_model" --dataset-arg "$target"




# python train_job.py --conf configs/scd-qm9-ft-energy.yaml --job-id scd-mn004-u0-debug --pretrained-model experiments/scd-m-n004_pcq/last.ckpt --dataset-arg u0
# python train_job.py --conf configs/scd-qm9-ft-energy.yaml --job-id scd-sn004-u0-db_b128_yema_learnAref --pretrained-model experiments/scd-s-n004_pcq/last.ckpt
# python train_job.py --conf configs/scd-qm9-ft-energy.yaml --job-id scd-sn004-u0-db_b128_yema_learnAref2 --pretrained-model experiments/scd-s-n004_pcq/last.ckpt

# supervised job
# srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --dataset-arg "$target"

