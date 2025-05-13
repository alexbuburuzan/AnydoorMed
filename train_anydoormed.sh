#!/bin/bash --login
#$ -cwd
#$ -l a100=1
#$ -pe smp.pe 12

conda activate anydoor_med

python -u main.py \
    --logdir models/AnydoorMed/erase_mask/ \
    --pretrained_model checkpoints/epoch=1-step=8687.ckpt \
    --base configs/anydoor.yaml \
    --scale_lr False \
    --save_top_k 3 \
    --gpus "0," \
    --seed 13 \