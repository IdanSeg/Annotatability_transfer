#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:1,vmem:32G
#SBATCH --cpus-per-task=16
#SBATCH --mem=42G
#SBATCH --time=23:00:00
#SBATCH --output=slurm_out/out.%j
#SBATCH --error=slurm_out/err.%j

mkdir -p slurm_out

# Initialize or safely reference PYTHONPATH
export PYTHONPATH=~/lab/scTab:${PYTHONPATH:-}

# Activate virtual environment
source /cs/labs/ravehb/idan724/annotatability/annot_venv/bin/activate
echo "Activated virtual environment: $(which python)"
python --version

# Start your Python script
python3 -u /cs/labs/ravehb/idan724/annotatability/test_subset_to_not.py

