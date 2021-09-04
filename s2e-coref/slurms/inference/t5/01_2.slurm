#! /bin/sh
#SBATCH --job-name=inf2t501
#SBATCH --output=infering/t5/0.1_2.out
#SBATCH --error=infering/t5/0.1_2.err
#SBATCH --partition=studentkillable
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

export PYTHONUNBUFFERED=1 && python cores_dir_inference.py --model t5 --builder new_builders/test.english.jsonlines.builder.t5.pkl --dropout 0.1 --beam 2
