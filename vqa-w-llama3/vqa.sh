#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 11:58:00
#SBATCH -o ./logs/cluster.%A.%a.%x.log
#SBATCH -a 0-0
#SBATCH --gres gpu:1

trap "trap ' ' TERM INT; kill -TERM 0; wait" TERM INT

# Make conda available:
eval "$(conda shell.bash hook)"
# Activate a conda environment:
conda activate playground

python3 vqa-w-llama3.py
