#!/bin/bash --login
#$ -cwd
#$ -l a100=1
#$ -pe smp.pe 12

conda activate anydoor_med

python -u main.py \
    --logdir models/AnydoorMed/autoencoder/512 \
    --pretrained_model checkpoints/autoencoder/anydoor_image_vae.ckpt \
    --base configs/medical_autoencoder.yaml \
    --scale_lr False \
    --save_top_k 3 \
    --gpus "0," \