#!/bin/bash
#SBATCH --array=0-61812
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=slurm_out/out.%A_%a
#SBATCH --error=slurm_out/err.%A_%a
#SBATCH --killable
#SBATCH --requeue

source /cs/labs/ravehb/idan724/annotatability/annot_venv/bin/activate

CSV_FILE="pbmc_healthy_worker_jobs.csv"
ROW_ID=$SLURM_ARRAY_TASK_ID

srun python worker_script.py \
    --csv_file="$CSV_FILE" \
    --row_id="$ROW_ID" \
    --device="cuda" \
    --epoch_num=25 \
    --batch_size=64 \
    --model_name="mlp" \
    --output_dir="results"