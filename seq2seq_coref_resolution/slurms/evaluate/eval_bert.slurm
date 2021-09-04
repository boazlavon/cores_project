#! /bin/sh
#SBATCH --job-name=eval_bert
#SBATCH --output=evaling/bert.out
#SBATCH --error=evaling/bert.err
#SBATCH --partition=studentkillable
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

export PYTHONUNBUFFERED=1 && python cores_evaluate.py bert ./bert2bert_coref_data/test.english.jsonlines
