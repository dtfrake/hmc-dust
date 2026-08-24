#!/bin/bash
#SBATCH --job-name=mean_a_b
#SBATCH --partition=kipac
#SBATCH --array=0-4
#SBATCH --time=30:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=4G
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err

set -euo pipefail

cd ~/hmc-dust
module load python/3.12.1
source ~/hmc-dust/.venv/bin/activate

export PYTHONPATH="$HOME/hmc-dust/src:${PYTHONPATH:-}"
export HMC_DUST_DATA="$SCRATCH/hmc-dust-data"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

# Keep every CPU thread pool inside the Slurm allocation
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

# NOTE: do not add --intra_op_parallelism_threads here. XLA's --flag parser
# aborts (SIGABRT -> exit 134) on unrecognized flags. Thread count comes
# from OMP_NUM_THREADS above.
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true"

srun python -u "Gaia XP/actual_data_inv_gamma_error_bars_2.py"