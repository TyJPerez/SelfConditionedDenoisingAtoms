#!/bin/bash
#SBATCH -J ct-FT-1p6M_homo_rsth_nonz
#SBATCH --account=m5068_g  ### m5068_g ,   m4866
#SBATCH -C gpu                          # request GPU nodes
#SBATCH -q premium                      # 4 GPUs -> regular QoS # regular or premium
#SBATCH -t 18:00:00                     # job time limit hh:mm:ss, note 16h is not enough for longer runs of 500k

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

# conf=configs/nersc_qm9-ft.yaml # for homo, lumo, gap

# conf=configs/scd-qm9-ft-base.yaml # for others
# conf=configs/scdB-qm9-ft-base.yaml # BIG model
# target=homo
# target=lumo
# target=gap
# target=alpha

conf=configs/scd-qm9-ft-base.yaml # for others
target=homo
# target=lumo
# target=gap


# conf=configs/scd-qm9-ft-long.yaml # for othersqs
# target=zpve
# target=alpha
# target=cv

# conf=configs/scd-qm9-ft-test.yaml # for others
# target=alpha

# conf=configs/scd-qm9-ft-energy.yaml # for others
# target=u0
# target=u298
# target=h298
# target=g298


# model_name=scd-m-n004_pcq # name of model to load
# model_name=scd-m-n04_pcq # name of model to load
# model_name=scd-s-n004_pcq # name of model to load
# model_name=scd-s-n04_pcq # name of model to load

# model_name=scd-n004_geom_b48n4 # geom10 pretrained
# model_name=scd-n004_geom01_b48n4 # geom1 pretrained

# model_name=scd-n004-ctr_pcq_b128n2 # scd-ctr pretrained
# model_name=scdB_pcq_b96-lr7 # big model
# model_name=scd-n004_pcq300k # model trianed on 300k samples

# model_name=cet-amp20_b142n1_0 # model trained on amp20 materials
# model_name=scd-s-n004_pcq # best pcq model

# model_name=ct-scd-omol25_4n # SCD on OMOL25 4M steps

####### force energy pretrained on OMOL25
# model_name=cet-all_b56n4_2_FT_s2ef__1Msteps # cet-all, then treained on omol25 4M
# model_name=null_FT_s2ef_n4_800ksteps_fwdF # scratch FE pretrain fwd forces
# model_name=null_FT_s2ef_n4_800ksteps_bwdF # scratch FE pretrain bwd forces
model_name=null_FT_s2ef_n4_1p6Msteps_fwdF

suffix=_rsetH_nonz

# model_name=null
# suffix=_baseline_nonoise

########## CGET ##########
### use about 30 hours for run 
# conf=configs/cget-qm9-ft-base.yaml # for others
# target=homo
# target=lumo
# target=gap

# conf=configs/cget-qm9-ft-energy.yaml # for others
# target=u0

# model_name=scd-frad-n004_pcq_b32n4 # frad pretrained
########## CGET ##########



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



## finetune job 
# srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --pretrained-model "$load_model" --dataset-arg "$target" # base

srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --pretrained-model "$load_model" --dataset-arg "$target" --reset-head True --noise-scale 0.0 --denoising-weight 0.0




#no pretrained model
# srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --dataset-arg "$target" 
# srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --dataset-arg "$target" --noise-scale 0.0 --denoising-weight 0.0


# srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --pretrained-model "$load_model" --dataset-arg "$target" --set-head-agg mean --standardize True


# python train_job.py --conf configs/scd-qm9-ft-energy.yaml --job-id scd-mn004-u0-debug --pretrained-model experiments/scd-m-n004_pcq/last.ckpt --dataset-arg u0
# python train_job.py --conf configs/scd-qm9-ft-energy.yaml --job-id scd-sn004-u0-db_b128_yema_learnAref --pretrained-model experiments/scd-s-n004_pcq/last.ckpt
# python train_job.py --conf configs/scd-qm9-ft-energy.yaml --job-id scd-sn004-u0-db_b128_yema_learnAref2 --pretrained-model experiments/scd-s-n004_pcq/last.ckpt

# supervised job
# srun python train_job.py --conf "$conf" --job-id "$job_id" -r True --max_epochs_per_run "$max_epochs_per_run" --dataset-arg "$target"

