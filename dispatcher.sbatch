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

for ((i=0; i<NUM_CHUNKS; i++)); do
    OFFSET=$((i * CHUNK_SIZE))
    END=$((OFFSET + CHUNK_SIZE - 1))
    if [ "$END" -ge "$TOTAL_ROWS" ]; then
        END=$((TOTAL_ROWS - 1))
    fi

    echo "Submitting array for chunk $i (OFFSET=$OFFSET)..."

    sbatch <<EOT
#!/bin/bash
#SBATCH --array=0-4999
#SBATCH --time=2:00:00
#SBATCH --output=$RESULTS_DIR/out.%A_%a
#SBATCH --error=$RESULTS_DIR/err.%A_%a
#SBATCH --killable
#SBATCH --requeue

source /cs/labs/ravehb/idan724/annotatability/annot_venv/bin/activate

# Each task uses SLURM_ARRAY_TASK_ID + OFFSET to pick the actual CSV row
ROW_ID=\$((\$SLURM_ARRAY_TASK_ID + $OFFSET))

# Only process if ROW_ID < TOTAL_ROWS
if [ "\$ROW_ID" -le $END ]; then
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

done

echo "Submitted $NUM_CHUNKS chunked arrays for $TOTAL_ROWS total rows."