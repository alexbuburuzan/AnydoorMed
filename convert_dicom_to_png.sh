#!/bin/bash --login
#SBATCH --job-name=anydoor_med_processing   # A descriptive name for your job
#SBATCH --nodes=1                           # Request a single node
#SBATCH --ntasks=1                          # Request one task (your main script)
#SBATCH --cpus-per-task=16                  # Request 16 CPU cores for your task
#SBATCH --partition=multicore_small         # Specify the partition based on CSF3 docs (was smp.pe)
#SBATCH --time=24:00:00                     # Set a reasonable time limit (e.g., 4 hours). Max for this partition is 7 days.
#SBATCH --output=anydoor_med_%j.out         # Standard output file
#SBATCH --error=anydoor_med_%j.err          # Standard error file

conda activate anydoor_med

python vindr-mammo/dicom_to_png.py --num-cores 16