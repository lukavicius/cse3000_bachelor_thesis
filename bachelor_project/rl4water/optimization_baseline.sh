#!/bin/bash

#SBATCH --job-name="RL4Water"
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=1G
#SBATCH --account=education-eemcs-courses-cse3000

module load 2023r1
module load openmpi
module load python
module load py-h5py
module load py-torch

srun python water_management.py 800 128 1 2 0.02 0.01 0.004 0.5 0.5