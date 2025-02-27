#!/bin/bash

#SBATCH --time=30:00:00
#SBATCH --partition=gpunodes
#SBATCH --nodelist=gpunode19
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/expires-2024-Dec-23/abdulbasit/projection_layer/output.out
#SBATCH --error=/scratch/expires-2024-Dec-23/abdulbasit/projection_layer/output.err

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export CUDA_PATH=/usr/local/cuda/bin

srun python3 final_1.py