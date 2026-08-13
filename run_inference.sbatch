#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --time=0:30:00
#SBATCH --mem=8G

cd ~/hmc-dust
module load python/3.12.1
source ~/hmc-dust/.venv/bin/activate
python inference_truncated_power_law.py