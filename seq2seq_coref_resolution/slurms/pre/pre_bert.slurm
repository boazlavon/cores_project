#! /bin/sh
#SBATCH --job-name=bert_pre
#SBATCH --output=training/pre_bert.out
#SBATCH --error=training/pre_bert.err
#SBATCH --partition=studentkillable
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

export PYTHONUNBUFFERED=1 &&  python cores_tokens.py bert coref_data/train.english.jsonlines && python cores_tokens.py bert coref_data/dev.english.jsonlines &&  python data_preprocess.py bert builders/train.english.jsonlines.builder.bert.pkl builders/dev.english.jsonlines.builder.bert.pkl 2>&1
