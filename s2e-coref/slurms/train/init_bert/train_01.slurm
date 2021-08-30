#! /bin/sh
#SBATCH --job-name=0.1ibert
#SBATCH --output=training_results/init_bert/0.1/seq.out
#SBATCH --error=training_results/init_bert/0.1/seq.err
#SBATCH --partition=studentkillable
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

mkdir -p ./training_results/init_bert/0.1
export PYTHONUNBUFFERED=1 && python cores_train.py --model init_bert --dropout 0.1 2>&1
