#! /bin/sh
#SBATCH --job-name=inf4_bert
#SBATCH --output=infering/inf4_bert.out
#SBATCH --error=infering/inf4_bert.err
#SBATCH --partition=studentkillable
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

export PYTHONUNBUFFERED=1 && python cores_inference.py bert ./bert2bert_coref_data/test.english.jsonlines 4
