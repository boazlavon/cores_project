#! /bin/sh
#SBATCH --job-name=0.1bert
#SBATCH --output=training_results/bert/0.1/seq.out
#SBATCH --error=training_results/bert/0.1/seq.err
#SBATCH --partition=studentkillable
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

export PYTHONUNBUFFERED=1 && python cores_train.py --model bert --epoch 8 --dropout 0.101 2>&1
