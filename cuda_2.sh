#!/bin/bash

#SBATCH --time=50:00:00
#SBATCH --partition=gpunodes
#SBATCH --nodelist=gpunode19
#SBATCH --gres=gpu:1
#SBATCH --output=training_output_smaller.out
#SBATCH --error=training_output2_smaller.err

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export CUDA_PATH=/usr/local/cuda/bin

srun python3 symptom_stream_trial.py