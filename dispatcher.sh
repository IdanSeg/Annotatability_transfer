#!/bin/bash

CSV_FILE="yourfile.csv"
RESULTS_DIR="slurm_out"
SCRIPT="worker_script.py"
CHUNK_SIZE=5000
TOTAL_ROWS=$(($(wc -l < "$CSV_FILE") - 1))
NUM_CHUNKS=$(( (TOTAL_ROWS + CHUNK_SIZE - 1) / CHUNK_SIZE ))

last_job_id=""

for ((i=0; i<NUM_CHUNKS; i++)); do
    OFFSET=$((i * CHUNK_SIZE))
    END=$((OFFSET + CHUNK_SIZE - 1))
    [ "$END" -ge "$TOTAL_ROWS" ] && END=$((TOTAL_ROWS - 1))
    
    if [ $i -eq 0 ]; then
        # Submit the first array without dependencies
        job_id=$(
            sbatch <<EOT | awk '{print $4}'
#!/bin/bash
#SBATCH --array=0-4999
#SBATCH --time=2:00:00
#SBATCH --output=$RESULTS_DIR/out.%A_%a
#SBATCH --error=$RESULTS_DIR/err.%A_%a

OFFSET=$OFFSET
END=$END
CSV_FILE=$CSV_FILE
SCRIPT=$SCRIPT

source /cs/labs/ravehb/idan724/annotatability/annot_venv/bin/activate

ROW_ID=\$((\$SLURM_ARRAY_TASK_ID + OFFSET))
if [ "\$ROW_ID" -le "\$END" ]; then
  srun python "\$SCRIPT" \
      --csv_file="\$CSV_FILE" \
      --row_id="\$ROW_ID" \
      --device="cuda" \
      --epoch_num=8 \
      --batch_size=128 \
      --model_name="mlp" \
      --output_dir="results"
fi
EOT
        )
    else
        # Submit subsequent arrays with a dependency on the previous one finishing
        job_id=$(
            sbatch --dependency=afterok:${last_job_id} <<EOT | awk '{print $4}'
#!/bin/bash
#SBATCH --array=0-4999
#SBATCH --time=2:00:00
#SBATCH --output=$RESULTS_DIR/out.%A_%a
#SBATCH --error=$RESULTS_DIR/err.%A_%a

OFFSET=$OFFSET
END=$END
CSV_FILE=$CSV_FILE
SCRIPT=$SCRIPT

source /cs/labs/ravehb/idan724/annotatability/annot_venv/bin/activate

ROW_ID=\$((\$SLURM_ARRAY_TASK_ID + OFFSET))
if [ "\$ROW_ID" -le "\$END" ]; then
  srun python "\$SCRIPT" \
      --csv_file="\$CSV_FILE" \
      --row_id="\$ROW_ID" \
      --device="cuda" \
      --epoch_num=8 \
      --batch_size=128 \
      --model_name="mlp" \
      --output_dir="results"
fi
EOT
        )
    fi

    echo "Submitted chunk $i (OFFSET=$OFFSET) with job ID $job_id"
    last_job_id=$job_id
done