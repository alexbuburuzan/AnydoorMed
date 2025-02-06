#!/bin/bash --login
#$ -cwd
#$ -pe smp.pe 16

conda activate anydoor_med

python vindr-mammo/dicom_to_png.py --num-cores 16