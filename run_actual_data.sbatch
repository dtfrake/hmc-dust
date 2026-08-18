#!/bin/bash
#SBATCH --job-name=high_mean_var
#SBATCH --partition=kipac
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=4G
#SBATCH --array=0-5
#SBATCH --output=slurm-%A_%a.out

MEANS=(4 16 64 4 4 4)
VARS=(16 256 4096 64 256 1024)
export IG_MEAN=${MEANS[$SLURM_ARRAY_TASK_ID]}
export IG_VAR=${VARS[$SLURM_ARRAY_TASK_ID]}
echo "Running with INV_GAMMA_MEAN=$IG_MEAN INV_GAMMA_VAR=$IG_VAR"

cd ~/hmc-dust
module load python/3.12.1
source ~/hmc-dust/.venv/bin/activate
export PYTHONPATH="$HOME/hmc-dust/src:$PYTHONPATH"
export HMC_DUST_DATA="$SCRATCH/hmc-dust-data"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1
srun --unbuffered python -u "Gaia XP/actual_data_inv_gamma_error_bars.py"