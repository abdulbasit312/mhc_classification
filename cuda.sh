#!/bin/bash

#SBATCH --time=50:00:00
#SBATCH --partition=gpunodes
#SBATCH --nodelist=gpunode23
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/expires-2024-Dec-14/abdulbasit/H1_TE1_15P/training_output_H1_TE1_15P.out
#SBATCH --error=/scratch/expires-2024-Dec-14/abdulbasit/H1_TE1_15P/training_output2_H1_TE1_15P.err

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export CUDA_PATH=/usr/local/cuda/bin

srun python3 symptom_stream_2.py