#!/bin/bash

CSV_FILE="pbmc_healthy_worker_jobs.csv"
RESULTS_DIR="results"
SCRIPT="worker_script.py"
CHUNK_SIZE=5000

# Ensure results directory exists
mkdir -p "$RESULTS_DIR"

TOTAL_ROWS=$(($(wc -l < "$CSV_FILE") - 1))

# How many chunked job arrays do we need?
NUM_CHUNKS=$(( (TOTAL_ROWS + CHUNK_SIZE - 1) / CHUNK_SIZE ))

# We'll store the last submitted job ID and chain the next chunk after it finishes.
last_job_id=""

for ((i=0; i<NUM_CHUNKS; i++)); do
    OFFSET=$((i * CHUNK_SIZE))
    END=$((OFFSET + CHUNK_SIZE - 1))
    if [ "$END" -ge "$TOTAL_ROWS" ]; then
        END=$((TOTAL_ROWS - 1))
    fi

    echo "Submitting array for chunk $i (OFFSET=$OFFSET)..."

    # If this is the first chunk, submit with no dependency.
    # Otherwise, submit with --dependency=afterok:<previous_job_id>.
    if [ "$i" -eq 0 ]; then
        # Submit the first array
        job_id=$(
            sbatch <<EOT | awk '{print $4}'
#!/bin/bash
#SBATCH --array=0-4999
#SBATCH --time=2:00:00
#SBATCH --output=$RESULTS_DIR/out.%A_%a
#SBATCH --error=$RESULTS_DIR/err.%A_%a
#SBATCH --killable
#SBATCH --requeue

# Activate your Python environment
source /cs/labs/ravehb/idan724/annotatability/annot_venv/bin/activate

# Calculate the actual CSV row using the offset
ROW_ID=\$((\$SLURM_ARRAY_TASK_ID + $OFFSET))

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
  echo "Skipping row \$ROW_ID as it is beyond the valid range."
fi
EOT
        )
    else
        # Subsequent chunks depend on the previous chunk finishing successfully
        job_id=$(
            sbatch --dependency=afterok:${last_job_id} <<EOT | awk '{print $4}'
#!/bin/bash
#SBATCH --array=0-4999
#SBATCH --time=2:00:00
#SBATCH --output=$RESULTS_DIR/out.%A_%a
#SBATCH --error=$RESULTS_DIR/err.%A_%a
#SBATCH --killable
#SBATCH --requeue

source /cs/labs/ravehb/idan724/annotatability/annot_venv/bin/activate

ROW_ID=\$((\$SLURM_ARRAY_TASK_ID + $OFFSET))

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
  echo "Skipping row \$ROW_ID as it is beyond the valid range."
fi
EOT
        )
    fi

    echo "Submitted chunk $i (OFFSET=$OFFSET) with job ID $job_id"
    last_job_id=$job_id
done

echo "Submitted $NUM_CHUNKS chunked arrays for $TOTAL_ROWS total rows."