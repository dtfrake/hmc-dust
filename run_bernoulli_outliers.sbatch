#!/bin/bash
#SBATCH --job-name=ber_out
#SBATCH --partition=kipac
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

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

# NOTE: do not add --intra_op_parallelism_threads here. XLA's --flag parser
# aborts (SIGABRT -> exit 134) on unrecognized flags. Thread count comes
# from OMP_NUM_THREADS above.
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true"

srun python -u "Gaia XP/bernoulli_outliers.py"