#!/bin/bash --login
#SBATCH --job-name=anydoor_med                           # A descriptive name for your job
#SBATCH --nodes=1                                        # Request a single node
#SBATCH --ntasks=1                                       # Request one task (your main script)
#SBATCH --cpus-per-task=12                               # Request 16 CPU cores for your task
#SBATCH --partition=gpuA                                 # Specify the partition based on CSF3 docs (was smp.pe)
#SBATCH --gpus=1                                         # Request 1 GPU
#SBATCH --time=24:00:00                                  # Set a reasonable time limit (e.g., 4 hours). Max for this partition is 7 days.
#SBATCH --output=anydoor_med_%j.out                      # Standard output file
#SBATCH --error=anydoor_med_%j.err                       # Standard error file

conda activate anydoor_med

python -u main.py \
    --logdir models/AnydoorMed/multiple_seeds/ \
    --pretrained_model checkpoints/epoch=1-step=8687.ckpt \
    --base configs/anydoor.yaml \
    --scale_lr False \
    --save_top_k 3 \
    --gpus "0," \
    --seed 13 \