#!/bin/bash

#SBATCH --job-name="RL4Water"
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=4GB
#SBATCH --account=education-eemcs-courses-cse3000

module load 2023r1
module load openmpi
module load python

srun python water_management.py 2 2 1 2 0.02 0.01 0.004 0.5 0.5