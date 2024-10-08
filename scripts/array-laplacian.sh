#!/bin/bash

#SBATCH --job-name=NGMB-Siamese                                                            # create a short name for your job
#SBATCH --nodes=1                                                                          # node count
#SBATCH --partition=gpu                                                                    # Name of the partition
#SBATCH --gres=gpu:1                                                                       # Require GPU
#SBATCH --nodelist=gpu[001,002,003,006,007,008,009,012,013]                                # Only keep the nodes with good GPUs (>24G)
#SBATCH --mem-per-gpu=20G                                                                  # CPU memory per GPU
#SBATCH --cpus-per-gpu=8                                                                   # Number of CPUS
#SBATCH --time=47:00:00                                                                    # Time Limit (HH:MM:SS)
#SBATCH --output="/home/jlagesse/ngmb/sbatch-output/%x   %j.out"                           # Output File
#SBATCH --error="/home/jlagesse/ngmb/sbatch-error/%x   %j.out"                             # Error File

set -x
cd ${SLURM_SUBMIT_DIR}

noises=(0.04 0.06 0.08 0.12 0.15 0.18 0.24 0.3)

nvidia-smi

rye run gm-laplacian-performances \
    --dataset  "/scratch/jlagesse/ngmb-data/ER[100,8,${noises[${SLURM_ARRAY_TASK_ID}]}]" \
    --experiment "ER-small" \
    --run-name "Laplacian ER[100,8,${noises[${SLURM_ARRAY_TASK_ID}]}]" \
    --features 32 \
    --batch-size 100 \
    --cuda \