#!/bin/bash

### example command
### bash scripts/submit_chain.sh -n 122 -s scripts/pretrain_job.sh

# Default values
# num_jobs=122
# sbatch_script="scripts/pretrain_job.sh"

num_jobs=20
sbatch_script="scripts/scd_ft_qm9_base.sh"
# sbatch_script="scripts/scd_sl_qm9.sh"

# Function to show usage
usage() {
    echo "Usage: $0 [-n num_jobs] [-s sbatch_script] [-d dependency_type] [-h]"
    echo "  -n: Number of jobs to chain (default: 120)"
    echo "  -s: SBATCH script to submit (default: train_01h.sbatch)"
    echo "  -d: Dependency type (default: afterok)"
    echo "  -h: Show this help message"
    exit 1
}

# Parse command line arguments
while getopts "n:s:d:h" opt; do
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
        # d)
        #     dependency_type="$OPTARG"
        #     ;;
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

echo "Submitting $num_jobs chained jobs using $sbatch_script with dependency type: $dependency_type"

# Submit first job
jid=$(sbatch "$sbatch_script" | awk '{print $4}')
# jid=$(bash "$sbatch_script" | awk '{print $4}')
echo "Submitted first job: $jid"

# Submit remaining jobs with dependency
for i in $(seq 2 "$num_jobs"); do
    # jid=$(sbatch --dependency="$dependency_type:$jid" "$sbatch_script" | awk '{print $4}')
    # jid=$(bash --dependency="$dependency_type:$jid" "$sbatch_script" | awk '{print $4}')
    # if (( i % 10 == 0 )); then
    #     echo "Submitted job $i: $jid"
    # fi
    jid=$(sbatch "$sbatch_script" | awk '{print $4}') 
    echo "Submitted job $i: $jid"
done

echo "Chained $num_jobs jobs, final job ID: $jid"