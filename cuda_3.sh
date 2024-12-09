#!/bin/bash

#SBATCH --time=50:00:00
#SBATCH --partition=gpunodes
#SBATCH --nodelist=gpunode17
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/expires-2024-Dec-14/abdulbasit/left_stream_1e-4/training_output_25_1e-4.out
#SBATCH --error=/scratch/expires-2024-Dec-14/abdulbasit/left_stream_1e-4/training_output2_25_1e-4.err

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export CUDA_PATH=/usr/local/cuda/bin

srun python3 symptom_stream_trial.py