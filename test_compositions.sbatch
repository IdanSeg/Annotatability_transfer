#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:1,vmem:32G
#SBATCH --cpus-per-task=16
#SBATCH --mem=42G
#SBATCH --time=23:00:00
#SBATCH --output=slurm_out/out.%j
#SBATCH --error=slurm_out/err.%j

# If needed, load a module that provides `nvidia-smi`
module load cuda  # or the appropriate module on your system

# (Optional) set -euo pipefail for safer scripting, but watch out for unbound vars
set -euo pipefail

mkdir -p slurm_out

echo "Allocated GPU info:"
# Now that the module is loaded, nvidia-smi should work
nvidia-smi || echo "nvidia-smi not found or GPU not accessible"

# Initialize or safely reference PYTHONPATH
export PYTHONPATH=~/lab/scTab:${PYTHONPATH:-}

# Activate virtual environment
source /cs/labs/ravehb/idan724/annotatability/annot_venv/bin/activate
echo "Activated virtual environment: $(which python)"
python --version

# Start your Python script
python3 -u /cs/labs/ravehb/idan724/annotatability/optimal_compositions.py pbmc

