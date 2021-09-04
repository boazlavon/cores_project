#! /bin/sh
#SBATCH --job-name=inf4bert
#SBATCH --output=infering/bert/0.1_4.out
#SBATCH --error=infering/bert/0.1_4.err
#SBATCH --partition=studentkillable
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

export PYTHONUNBUFFERED=1 && python cores_dir_inference.py --model bert --builder new_builders/test.english.jsonlines.builder.bert.pkl --dropout 0.1 --beam 4
