#!/bin/bash
#SBATCH --job-name=nu_sweep
#SBATCH --partition=kipac
#SBATCH --time=24:00:00
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

# <-- set this to the actual filename
SCRIPT="Gaia XP/fixing_nu.py"

# NU_INDEX indexes NU_VALUES = [0.25, 0.5, 1.0, 1.5, 2.0] in the Python file.
# One failing nu does not kill the rest; the job exits nonzero if any failed.
status=0
for NU_INDEX in 0 1 2 3 4; do
    echo "===== NU_INDEX=$NU_INDEX  start $(date '+%F %T') ====="
    if NU_INDEX="$NU_INDEX" srun python -u "$SCRIPT"; then
        echo "===== NU_INDEX=$NU_INDEX  done  $(date '+%F %T') ====="
    else
        echo "===== NU_INDEX=$NU_INDEX  FAILED (exit $?) ====="
        status=1
    fi
done

exit "$status"
