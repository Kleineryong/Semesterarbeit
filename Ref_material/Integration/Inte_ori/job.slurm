#!/bin/bash
#SBATCH -J wonidie
#SBATCH -o ./%x.%j.%N.out
#SBATCH -D ./
#SBATCH -e wronglog.err
#SBATCH --get-user-env
#SBATCH --clusters=inter   # for the beginning
#SBATCH --partition=cm2_inter
#SBATCH --nodes=1      # for the beginning
#SBATCH --mail-type=start,end 	      # you can change this, but use then a correct mail adress
#SBATCH --mail-user=ruihang.dai@tum.de
#SBATCH --export=NONE          # mandatory
#SBATCH --time=02:00:00        # 1 hour
module load slurm_setup
module load python/3.8.11-base                  # if needed
source  ~/venv/py38/bin/activate
python MSI_main.py
