#! /bin/sh
#SBATCH --job-name=0.05bart
#SBATCH --output=training_results/bart/0.05/seq.out
#SBATCH --error=training_results/bart/0.05/seq.err
#SBATCH --partition=studentkillable
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

mkdir -p ./training_results/bart/0.05
export PYTHONUNBUFFERED=1 && python cores_train.py --model bart --epoch 8 --dropout 0.05 2>&1
