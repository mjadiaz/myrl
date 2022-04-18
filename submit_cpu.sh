#!/bin/bash

#SBATCH --job-name=TestR
#SBATCH --time=2:00:00
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=40

source /home/mjad1g20/.bashrc
source activate rrlib

python run_apex_ddpg.py
