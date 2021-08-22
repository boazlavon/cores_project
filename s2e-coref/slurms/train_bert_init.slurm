#! /bin/sh
#SBATCH --job-name=bert_init
#SBATCH --output=training/bert_init.out
#SBATCH --error=training/bert_init.err
#SBATCH --partition=studentkillable
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

export PYTHONUNBUFFERED=1 && python cores_train.py bert init 2>&1
