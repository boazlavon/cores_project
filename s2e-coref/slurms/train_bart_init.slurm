#! /bin/sh
#SBATCH --job-name=bart_init
#SBATCH --output=training/bart_init.out
#SBATCH --error=training/bart_init.err
#SBATCH --partition=studentkillable
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

export PYTHONUNBUFFERED=1 && python cores_train.py init_bart 2>&1
