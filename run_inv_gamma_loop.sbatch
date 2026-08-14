#!/bin/bash
#SBATCH --job-name=inv_gamma
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=2G
#SBATCH --array=0-5
#SBATCH --output=logs/slurm-%A_%a.out

# (mean, var) combos, index-matched
MEANS=(1 2 4 1 2 4)
SECONDS_ARR=(1 4 16 4 16 64)

# Set to "std" if the second number in each pair is a standard deviation
SECOND_IS=var

M=${MEANS[$SLURM_ARRAY_TASK_ID]}
S=${SECONDS_ARR[$SLURM_ARRAY_TASK_ID]}

export INV_GAMMA_MEAN=$M
if [ "$SECOND_IS" = "std" ]; then
  export INV_GAMMA_VAR=$(awk "BEGIN{print $S*$S}")
else
  export INV_GAMMA_VAR=$S
fi

cd ~/hmc-dust
mkdir -p logs runs
module load python/3.12.1
source ~/hmc-dust/.venv/bin/activate
export PYTHONPATH="$HOME/hmc-dust/src:$PYTHONPATH"
export HMC_DUST_DATA="$SCRATCH/hmc-dust-data"
export HMC_DUST_OUT="$HOME/hmc-dust/runs"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$SLURM_CPUS_PER_TASK"

echo "task $SLURM_ARRAY_TASK_ID: mean=$INV_GAMMA_MEAN var=$INV_GAMMA_VAR"
srun python "Gaia XP/actual_data_inv_gamma_error_bars.py"