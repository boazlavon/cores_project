#! /bin/sh
#SBATCH --job-name=eval_bart
#SBATCH --output=training_results/bart/0.2/eval.out
#SBATCH --error=training_results/bart/0.2/eval.out
#SBATCH --partition=studentkillable
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

export PYTHONUNBUFFERED=1 && python cores_evaluate.py --model bart --dataset_builder_path builders/dev.english.jsonlines.builder.bart.pkl --dropout 0.2 2>&1
