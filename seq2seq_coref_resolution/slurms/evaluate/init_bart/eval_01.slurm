#! /bin/sh
#SBATCH --job-name=ev01ibart
#SBATCH --output=training_results/init_bart/0.1/eval.out
#SBATCH --error=training_results/init_bart/0.1/eval.out
#SBATCH --partition=studentkillable
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

export PYTHONUNBUFFERED=1 && python cores_evaluate.py --model init_bart --dataset_builder_path builders/dev.english.jsonlines.builder.bart.pkl --dropout 0.1 2>&1
