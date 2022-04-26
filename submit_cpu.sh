#!/bin/bash

#SBATCH --job-name=TestR
#SBATCH --time=12:00:00
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=40

source /home/mjad1g20/.bashrc
source activate rrlib

python run_apex_ddpg_pheno.py --train True --save_path "tests_pheno/pheno_2_step_ou_rllibhp"
