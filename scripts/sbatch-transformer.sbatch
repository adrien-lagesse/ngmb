#!/bin/bash

#SBATCH --job-name=GNNCO                                                                  # create a short name for your job
#SBATCH --nodes=1                                                                          # node count
#SBATCH --partition=gpu                                                                    # Name of the partition
#SBATCH --gres=gpu:1                                                                       # Require GPU
#SBATCH --mem-per-gpu=20G                                                                
#SBATCH --time=47:00:00                                                                    # total run time limit (HH:MM:SS)
#SBATCH --mincpus=8                                                                        # Number of CPUS
#SBATCH --output="/home/jlagesse/gnnco/sbatch-output/%x   %j.out"   # Output File

set -x
cd ${SLURM_SUBMIT_DIR}

rye run python aqsol-pe/train.py