#!/bin/bash --login
#SBATCH --job-name=anydoor_med_finetune_vae_512          # A descriptive name for your job
#SBATCH --nodes=1                                        # Request a single node
#SBATCH --ntasks=1                                       # Request one task (your main script)
#SBATCH --cpus-per-task=12                               # Request 16 CPU cores for your task
#SBATCH --partition=gpuA                                 # Specify the partition based on CSF3 docs (was smp.pe)
#SBATCH --gpus=1                                         # Request 1 GPU
#SBATCH --time=24:00:00                                  # Set a reasonable time limit (e.g., 4 hours). Max for this partition is 7 days.
#SBATCH --output=anydoor_med_finetune_vae_%j.out         # Standard output file
#SBATCH --error=anydoor_med_finetune_vae_%j.err     

conda activate anydoor_med

python -u main.py \
    --logdir autoencoder/512 \
    --pretrained_model checkpoints/autoencoder/anydoor_image_vae.ckpt \
    --base configs/medical_autoencoder.yaml \
    --scale_lr False \
    --save_top_k 3 \
    --gpus "0," \