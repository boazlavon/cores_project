#! /bin/sh
#SBATCH --job-name=bart_pre
#SBATCH --output=training/pre_bart.out
#SBATCH --error=training/pre_bart.err
#SBATCH --partition=studentkillable
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

export PYTHONUNBUFFERED=1 &&  python cores_tokens.py bart coref_data/train.english.jsonlines && python cores_tokens.py bart coref_data/dev.english.jsonlines &&  python data_preprocess.py bart builders/train.english.jsonlines.builder.bart.pkl builders/dev.english.jsonlines.builder.bart.pkl 2>&1
