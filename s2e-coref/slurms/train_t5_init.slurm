#! /bin/sh
#SBATCH --job-name=t5_init
#SBATCH --output=training/t5_init.out
#SBATCH --error=training/t5_init.err
#SBATCH --partition=studentkillable
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

export PYTHONUNBUFFERED=1 && python cores_train.py t5 init 2>&1
