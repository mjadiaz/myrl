#!/bin/bash

#SBATCH --job-name=TestR
#SBATCH --time=1:00:00
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4

source /home/mjad1g20/.bashrc
source activate rrlib

python test_d3pg.py
