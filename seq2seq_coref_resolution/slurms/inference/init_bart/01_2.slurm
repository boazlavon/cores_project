#! /bin/sh
#SBATCH --job-name=inf2ibart01
#SBATCH --output=infering/init_bart/0.1_2.out
#SBATCH --error=infering/init_bart/0.1_2.err
#SBATCH --partition=studentkillable
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

export PYTHONUNBUFFERED=1 && python cores_dir_inference.py --model init_bart --builder new_builders/test.english.jsonlines.builder.bart.pkl --dropout 0.1 --beam 2
