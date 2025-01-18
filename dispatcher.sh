#!/bin/bash

# ----------------------
# User Configuration
# ----------------------
CSV_FILE="pbmc_healthy_worker_jobs.csv"
RESULTS_DIR="results"
SCRIPT="worker_script.py"
CHUNK_SIZE=100

# Maximum number of total jobs you want to allow
# in the queue (pending or running) at once
MAX_JOBS_IN_QUEUE=1000

# ----------------------
# Script Logic
# ----------------------

# 1) Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# 2) Count total data rows
TOTAL_ROWS=$(($(wc -l < "$CSV_FILE") - 1))
if [ "$TOTAL_ROWS" -le 0 ]; then
    echo "Error: No data rows in $CSV_FILE."
    exit 1
fi

# 3) Number of chunks needed
NUM_CHUNKS=$(( (TOTAL_ROWS + CHUNK_SIZE - 1) / CHUNK_SIZE ))
echo "Dispatching $NUM_CHUNKS chunks total for $TOTAL_ROWS rows..."

# 4) Function: Return number of jobs in queue for current user
jobs_in_queue() {
    squeue -u "$USER" -h | wc -l
}

# 5) Loop over each chunk
for ((i=0; i<NUM_CHUNKS; i++)); do
    OFFSET=$((i * CHUNK_SIZE))
    END=$((OFFSET + CHUNK_SIZE - 1))
    if [ "$END" -ge "$TOTAL_ROWS" ]; then
        END=$((TOTAL_ROWS - 1))
    fi

    # Throttle by checking we don't exceed MAX_JOBS_IN_QUEUE
    while [ "$(jobs_in_queue)" -ge "$MAX_JOBS_IN_QUEUE" ]; do
        echo "You currently have $(jobs_in_queue) jobs in the queue. Limit: $MAX_JOBS_IN_QUEUE"
        echo "Waiting 5 minutes before checking again..."
        sleep 300
    done

    echo "Submitting chunk $i (OFFSET=$OFFSET)..."

    # 6) Retry sbatch if we see AssocMaxSubmitJobLimit or an empty job_id
    while true; do
        submission_output=$(
            sbatch --parsable <<EOT 2>&1
#!/bin/bash
#SBATCH --array=0-$((CHUNK_SIZE-1))
#SBATCH --time=5:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --output=${RESULTS_DIR}/out.%A_%a
#SBATCH --error=${RESULTS_DIR}/err.%A_%a
#SBATCH --killable
#SBATCH --requeue

source /cs/labs/ravehb/idan724/annotatability/annot_venv/bin/activate

ROW_ID=\$((SLURM_ARRAY_TASK_ID + $OFFSET))

# Only run if ROW_ID <= END
if [ "\$ROW_ID" -le "$END" ]; then
    srun python "$SCRIPT" \
        --csv_file="$CSV_FILE" \
        --row_id="\$ROW_ID" \
        --device="cuda" \
        --epoch_num=8 \
        --batch_size=128 \
        --model_name="mlp" \
        --output_dir="results"
else
    echo "Skipping row \$ROW_ID as it exceeds $END."
fi
EOT
        )

        # If submission succeeded, 'submission_output' should be the numeric job ID
        # If it failed, 'submission_output' might contain an error message
        if [[ "$submission_output" =~ ^[0-9]+$ ]]; then
            # Succeeded
            echo "Chunk $i submitted with job ID $submission_output"
            break
        else
            # Check if it's the specific AssocMaxSubmitJobLimit error
            if echo "$submission_output" | grep -q "AssocMaxSubmitJobLimit"; then
                echo "AssocMaxSubmitJobLimit error encountered. Will wait 5 minutes and retry."
                sleep 300
            else
                # Some other error happened; still wait + retry, or exit
                echo "Warning: submission failed for chunk $i with message:"
                echo "$submission_output"
                echo "Will wait 5 minutes and retry..."
                sleep 300
            fi
        fi
    done

done

echo "All $NUM_CHUNKS chunks submitted (with a queue limit of $MAX_JOBS_IN_QUEUE)."