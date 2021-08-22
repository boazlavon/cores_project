#! /bin/sh
#SBATCH --job-name=train_bert
#SBATCH --output=training/bert.out
#SBATCH --error=training/bert.err
#SBATCH --partition=studentkillable
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

export PYTHONUNBUFFERED=1 && python cores_train.py bert 2>&1
