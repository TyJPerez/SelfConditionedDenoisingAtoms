#!/bin/bash

### example command
### bash scripts/qm9ft_chain.sh -n 47 -s scripts/finetune_qm9_job.sh -m cet-pcq_2

# Default values
# num_jobs=47
# sbatch_script="scripts/finetune_qm9_job.sh"
# load_model="cet-pcq_2"

# Define target list
# target_list=("homo" "lumo" "gap" "mu" "alpha" "zpve" "cv" "u0" "u298" "h298" "g298") #all
# target_list=("homo" "lumo" "gap")

#Base targets #("homo" "lumo" "gap", "cv")
# target_list=("homo" "lumo" "gap") #, "cv") #("cv") ("homo" "lumo" "gap", "cv")
# num_jobs=1
# sbatch_script="scripts/finetune_qm9_base.sh"
# load_model="cet-pcq_2"

#Base targets #("zpve" "alpha")
# target_list=("zpve") # "alpha") 
# num_jobs=30
# sbatch_script="scripts/finetune_qm9_long.sh"
# load_model="cet-pcq_2"

#Energy targets ("u0" "u298" "h298" "g298")
# target_list=("g298") # "u298" "h298" "g298")
# num_jobs=75
# sbatch_script="scripts/finetune_qm9_eng.sh"
# load_model="cet-pcq_2"


# Function to show usage
usage() {
    echo "Usage: $0 [-n num_jobs] [-s sbatch_script] [-m load_model] [-h]"
    echo "  -n: Number of jobs to chain per target (default: 1)"
    echo "  -s: SBATCH script to submit (default: scripts/finetune_qm9_job.sh)"
    echo "  -m: Model name to load (default: cet-pcq_2_bk)"
    echo "  -h: Show this help message"
    echo ""
    echo "Available targets: ${target_list[*]}"
    echo ""
    echo "This will submit jobs for each target in the target_list."
    echo "Total jobs submitted = num_jobs × number_of_targets"
    exit 1
}

# Parse command line arguments
while getopts "n:s:m:h" opt; do
    case $opt in
        n)
            num_jobs="$OPTARG"
            if ! [[ "$num_jobs" =~ ^[0-9]+$ ]]; then
                echo "Error: -n must be a positive integer"
                exit 1
            fi
            ;;
        s)
            sbatch_script="$OPTARG"
            if [[ ! -f "$sbatch_script" ]]; then
                echo "Error: SBATCH script '$sbatch_script' not found"
                exit 1
            fi
            ;;
        m)
            load_model="$OPTARG"
            ;;
        h)
            usage
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            usage
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            usage
            ;;
    esac
done

# Validate model exists
model_path="experiments/$load_model/last.ckpt"
if [[ ! -f "$model_path" ]]; then
    echo "Warning: Model checkpoint not found: $model_path"
    echo "Proceeding anyway - job will fail if model doesn't exist"
fi

echo "Submitting fine-tuning jobs:"
echo "  Script: $sbatch_script"
echo "  Model: $load_model"
echo "  Jobs per target: $num_jobs"
echo "  Targets: ${target_list[*]}"
echo "  Total jobs: $((num_jobs * ${#target_list[@]}))"
echo ""

# Track all job IDs for summary
all_job_ids=()

# Loop through each target
for target in "${target_list[@]}"; do
    echo "=== Submitting jobs for target: $target ==="
    
    # Submit jobs for this target (each with same job name for singleton dependency)
    for i in $(seq 1 "$num_jobs"); do
        jid=$(sbatch --job-name="qm9_${target}" "$sbatch_script" -t "$target" -m "$load_model" | awk '{print $4}')
        echo "  Job $i for $target: $jid"
        all_job_ids+=("$jid")
    done
    
    echo "  Completed $num_jobs jobs for target $target"
    echo ""
done

echo "=== Summary ==="
echo "Submitted ${#all_job_ids[@]} total jobs"
echo "Job IDs: ${all_job_ids[*]}"