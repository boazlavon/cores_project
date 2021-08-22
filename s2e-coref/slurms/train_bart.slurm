#! /bin/sh
#SBATCH --job-name=train_bart
#SBATCH --output=training/bart.out
#SBATCH --error=training/bart.err
#SBATCH --partition=studentkillable
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

export PYTHONUNBUFFERED=1 && python cores_train.py bart 2>&1
