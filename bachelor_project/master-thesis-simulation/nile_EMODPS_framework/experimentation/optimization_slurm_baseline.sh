#!/bin/sh
#
#SBATCH --job-name="python_reservoir_sim_baseline"
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=1G
#SBATCH --account=education-eemcs-courses-cse3000

module load 2022r2
module load python/3.8.12

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun python3 baseline_optimization.py 2 0.02 0.01 0.004 0.5 0.5