#! /bin/sh
#SBATCH --job-name=clinf1bart01
#SBATCH --output=infering/bart/cl.0.1_1.out
#SBATCH --error=infering/bart/cl.0.1_1.err
#SBATCH --partition=studentkillable
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

export PYTHONUNBUFFERED=1 && python cores_dir_inference.py --model bart --builder new_builders/test.english.jsonlines.builder.bart.pkl --dropout 0.1 --beam 1 --tag_only_clusters True
