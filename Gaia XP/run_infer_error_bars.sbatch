#!/bin/bash
#SBATCH --job-name=err_bars
#SBATCH --partition=kipac
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=4G
#SBATCH --output=slurm-%j.out

cd ~/hmc-dust
module load python/3.12.1
source ~/hmc-dust/.venv/bin/activate
export PYTHONPATH="$HOME/hmc-dust/src:$PYTHONPATH"
export HMC_DUST_DATA="$SCRATCH/hmc-dust-data"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1
srun --unbuffered python -u "Gaia XP/actual_data_manually_infer_error_bars.py"
