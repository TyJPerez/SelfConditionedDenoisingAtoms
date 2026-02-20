#!/bin/bash
#SBATCH -J ct-FEbwd_mbgapf0_rsethead #__s50kw5k
#SBATCH --account=m5068_g  ### m5068_g ,   m4866
#SBATCH -C gpu                          # request GPU nodes
#SBATCH -q premium                      # 4 GPUs -> regular QoS # regular or premium
#SBATCH -t 16:00:00                     # job time limit hh:mm:ss, note 16h is not enough for longer runs of 500k

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

conf=configs/cet-mbgap.yaml # for others
target=0

### cet-amp20-mbgap_f0_2
# model_name=cet-amp20_b142n1_0 # model trained on amp20 materials
# suffix=f0_2

### cet-baseline-mbgap_f0_2
# model_name=null
# suffix=_baseline_b4

# using amp 20 model
# model_name=cet-amp20_b142n1_0

# model_name=scd-s-n004_pcq
# model_name=scd-n004_geom_b48n4
# model_name=cet-sair_b12n2_0
# model_name=cet-sairpocket_b12n2_0 # best so far
# model_name=cet-sairligand_b12n2_0
# model_name=cet-all_b56n4_2 

# model_name=ct-scd-omol25_4n # SCD on OMOL25 4M 

## FE pretrained models
# model_name=null_FT_s2ef_n4_800ksteps_fwdF # scratch FE pretrain fwd forces
model_name=null_FT_s2ef_n4_800ksteps_bwdF # scratch FE pretrain bwd forces

suffix=_lr8e5_f0_rsethead

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
    srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --dataset-arg "$target" --batch-size 8
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --dataset-arg "$target" --batch-size 16
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --dataset-arg "$target" --batch-size 32
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --dataset-arg "$target" --batch-size 4
else
### finetune job 
    echo "Running with pretrained model: $load_model"
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --pretrained-model "$load_model" --dataset-arg "$target"
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --pretrained-model "$load_model" --dataset-arg "$target" --lr 0.0001 # best so far

    
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --pretrained-model "$load_model" --dataset-arg "$target" --lr 0.0001 --lr-cosine-length 100000 --num-steps 100000 --lr-warmup-steps 5000
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --pretrained-model "$load_model" --dataset-arg "$target" --lr 0.0001 --lr-cosine-length 50000 --num-steps 50000 --lr-warmup-steps 5000
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --pretrained-model "$load_model" --dataset-arg "$target" --lr 0.00005
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --pretrained-model "$load_model" --dataset-arg "$target" --lr 0.00008 --lr-cosine-length 500000 --num-steps 500000 --lr-warmup-steps 5000
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --pretrained-model "$load_model" --dataset-arg "$target" --lr 0.00008 --self_cond True

    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --pretrained-model "$load_model" --dataset-arg "$target" --lr 0.00008 # new best

    #with reset head
    srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --pretrained-model "$load_model" --dataset-arg "$target" --lr 0.00008 --reset-head True 

    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --pretrained-model "$load_model" --dataset-arg "$target" --lr 0.00008 --graph_cutoff 8.0
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --pretrained-model "$load_model" --dataset-arg "$target" --lr 0.00008 --weight-decay 0.05
    # srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --pretrained-model "$load_model" --dataset-arg "$target" --lr 0.00008 --init-droppath 0.0

fi


## finetune job 
# srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --pretrained-model "$load_model" --dataset-arg "$target"




# python train_job.py --conf configs/scd-qm9-ft-energy.yaml --job-id scd-mn004-u0-debug --pretrained-model experiments/scd-m-n004_pcq/last.ckpt --dataset-arg u0
# python train_job.py --conf configs/scd-qm9-ft-energy.yaml --job-id scd-sn004-u0-db_b128_yema_learnAref --pretrained-model experiments/scd-s-n004_pcq/last.ckpt
# python train_job.py --conf configs/scd-qm9-ft-energy.yaml --job-id scd-sn004-u0-db_b128_yema_learnAref2 --pretrained-model experiments/scd-s-n004_pcq/last.ckpt

# supervised job
# srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --dataset-arg "$target"

