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

echo "### Running $SLURM_JOB_NAME ###"
set -x
cd ${SLURM_SUBMIT_DIR}
module purge

nvidia-smi

bash scripts/train-siamese-gm.sh