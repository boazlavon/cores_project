#! /bin/sh
#SBATCH --job-name=t5_pre
#SBATCH --output=training/pre_t5.out
#SBATCH --error=training/pre_t5.err
#SBATCH --partition=studentkillable
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

export PYTHONUNBUFFERED=1 &&  python cores_tokens.py t5 coref_data/train.english.jsonlines && python cores_tokens.py t5 coref_data/dev.english.jsonlines 2>&1
