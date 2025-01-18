#!/bin/bash

# ----------------------
# User Configuration
# ----------------------
CSV_FILE="pbmc_healthy_worker_jobs.csv"
RESULTS_DIR="results"
SCRIPT="worker_script.py"
CHUNK_SIZE=1000

# This is the maximum number of total jobs you want to allow
# in the queue (pending or running) at once. Adjust if needed.
MAX_JOBS_IN_QUEUE=2

# ----------------------
# Script Logic
# ----------------------

# 1) Create results dir if it doesn't exist
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
    # -h omits headers; pipe to wc -l to count lines
    squeue -u "$USER" -h | wc -l
}

# 5) Loop over each chunk
for ((i=0; i<NUM_CHUNKS; i++)); do
    OFFSET=$((i * CHUNK_SIZE))
    END=$((OFFSET + CHUNK_SIZE - 1))
    if [ "$END" -ge "$TOTAL_ROWS" ]; then
        END=$((TOTAL_ROWS - 1))
    fi

    # ----------------------
    # Throttle submissions
    # ----------------------
    # Wait until we have fewer than MAX_JOBS_IN_QUEUE in the queue
    # so we do not hit AssocMaxSubmitJobLimit.
    while [ "$(jobs_in_queue)" -ge "$MAX_JOBS_IN_QUEUE" ]; do
        echo "You currently have $(jobs_in_queue) jobs in the queue. Limit: $MAX_JOBS_IN_QUEUE"
        echo "Waiting 5 min before checking again..."
        sleep 300
    done

    echo "Submitting chunk $i (OFFSET=$OFFSET)..."

    # Actually submit the chunk as an array 0-999, then parse out the job ID.
    job_id=$(
        sbatch <<EOT | awk '{print \$4}'
#!/bin/bash
#SBATCH --array=0-999
#SBATCH --time=5:00:00
#SBATCH --output=${RESULTS_DIR}/out.%A_%a
#SBATCH --error=${RESULTS_DIR}/err.%A_%a
#SBATCH --killable
#SBATCH --requeue

# Activate your Python environment
source /cs/labs/ravehb/idan724/annotatability/annot_venv/bin/activate

ROW_ID=\$((SLURM_ARRAY_TASK_ID + $OFFSET))

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

    if [ -z "$job_id" ]; then
        echo "Warning: Failed to submit chunk $i (OFFSET=$OFFSET)."
    else
        echo "Chunk $i submitted with job ID $job_id"
    fi
done

echo "All $NUM_CHUNKS chunks submitted (up to the queue limit $MAX_JOBS_IN_QUEUE at a time)."