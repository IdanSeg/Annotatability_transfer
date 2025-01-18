#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=2:00:00
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
python3 -u /cs/labs/ravehb/idan724/annotatability/tests.py

