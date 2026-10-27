#!/bin/bash
#SBATCH --job-name=hankel_1d
#SBATCH --partition=kipac
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=4G
#SBATCH --output=slurm-%j.out


cd ~/hmc-dust
module load python/3.12.1
source ~/hmc-dust/.venv/bin/activate
export PYTHONPATH="$HOME/hmc-dust/src:$PYTHONPATH"
export HMC_DUST_DATA="$SCRATCH/hmc-dust-data"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
python -u "Post Week 7/1D_hankel_transform_real_data.py"