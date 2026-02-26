#!/bin/bash
#SBATCH -J FT-qm9
#SBATCH --account=m4866          # e.g., m1234_g
#SBATCH -C gpu                          # request GPU nodes
#SBATCH -q regular                      # 4 GPUs -> regular QoS # regular or premium
#SBATCH -t 00:25:00                     # job time limit hh:mm:ss

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

### bash scripts/qm9ft_chain.sh -n 60 -s scripts/finetune_qm9_eng.sh -m cet-pcq_2

mkdir -p logs # create logs directory 

# conf=configs/nersc_qm9-ft.yaml # for homo, lumo, gap
conf=configs/nersc_qm9-ft-energy.yaml # for others
max_epochs_per_run=75

model_name=cet-pcq_2
target=u0

suffix=b128_eng_0

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


srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --pretrained-model "$load_model" --dataset-arg "$target"
