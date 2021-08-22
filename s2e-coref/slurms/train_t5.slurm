#! /bin/sh
#SBATCH --job-name=t5_train
#SBATCH --output=training/t5.out
#SBATCH --error=training/t5.err
#SBATCH --partition=studentkillable
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

export PYTHONUNBUFFERED=1 && python cores_train.py t5 2>&1
