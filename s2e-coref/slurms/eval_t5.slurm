#! /bin/sh
#SBATCH --job-name=eval_t5
#SBATCH --output=evaling/t5.out
#SBATCH --error=evaling/t5.err
#SBATCH --partition=studentkillable
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

export PYTHONUNBUFFERED=1 && python cores_evaluate.py t5 ./bert2bert_coref_data/test.english.jsonlines
