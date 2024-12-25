#!/bin/bash

#SBATCH --partition=medium            # Or short/long as per cluster policy
#SBATCH --nodes=1
#SBATCH --gres=gpu:1                      # Request 1 V100 GPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=slurm_out/out.%j
#SBATCH --error=slurm_out/err.%j

# Create output directory if it doesn't exist
mkdir -p slurm_out

# Navigate to the project directory
cd ~/lab/scTab || exit
echo "Changed directory to $(pwd)"

# Activate the virtual environment
source ~/lab/scTab/merlin-torch/bin/activate
echo "Activated virtual environment."

# Add scTab to PYTHONPATH
export PYTHONPATH=~/lab/scTab:$PYTHONPATH
echo "Set PYTHONPATH to include ~/lab/scTab."

# Run the training script with desired arguments
echo "Starting TabNet training..."
python3 -u /cs/usr/idan724/lab/scTab/scripts/py_scripts/train_tabnet.py \
    --cluster="jsc" \
    --version='version_1' \
    --data_path="/cs/usr/idan724/lab/merlin_cxg_minimal/merlin_cxg_2023_05_15_sf-log1p_minimal" \
    --epochs=30 \
    --batch_size=4096 \
    --sub_sample_frac=1.0 \
    --lr=0.005 \
    --weight_decay=0.05 \
    --use_class_weights=True \
    --lambda_sparse=1e-5 \
    --n_d=32 \
    --n_a=32 \
    --n_steps=1 \
    --gamma=1.3 \
    --n_independent=1 \
    --n_shared=1 \
    --virtual_batch_size=1024 \
    --mask_type='entmax' \
    --augment_training_data=True \
    --lr_scheduler_step_size=1 \
    --lr_scheduler_gamma=0.9 \
    --check_val_every_n_epoch=1 \
    --checkpoint_interval=1 \
    --seed=1
echo "TabNet training script completed."

