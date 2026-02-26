#!/bin/bash

#SBATCH -J cet-amp20_b142n1_0
#SBATCH --account=m5068_g          # , m5068_g
#SBATCH -C gpu                          # request GPU nodes
#SBATCH -q premium                      # 4 GPUs -> regular QoS # regular or premium
#SBATCH -t 30:00:00                     # job time limit hh:mm:ss
#SBATCH --nodes=1                       # node count

#SBATCH --ntasks-per-node=4
#SBATCH -c 32                           # 32 CPUs per task (Perlmutter example)
#SBATCH --gpus-per-node=4               # number of gpus per node
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none                 # all-visible or let Lightning/NCCL map
#SBATCH --dependency=singleton      # only one job with this name at a time
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# (Optional) send SIGTERM 120s before kill so you can wrap up if needed
#SBATCH --signal=TERM@120

mkdir -p logs # create logs directory 

job_id=cet-amp20_b142n1_0 # lr=0.0005
conf=configs/scd-amp20_pretrain.yaml

# job_id=cet-amp20_b142n4_0 # lr=0.0007
# conf=configs/scd-amp20_pretrain_n4.yaml

max_epochs_per_run=3000000


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

echo "Starting distributed training with srun..."
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_PROCID: $SLURM_PROCID"

# python debug_job.py
srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run"

# mv "logs/train_chain_${SLURM_JOB_ID}.out" "$LOG_DIR/" 2>/dev/null || true
# mv "logs/train_chain_${SLURM_JOB_ID}.err" "$LOG_DIR/" 2>/dev/null || true