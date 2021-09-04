#! /bin/sh
#SBATCH --job-name=0.101t5
#SBATCH --output=training_results/t5/0.101/seq.out
#SBATCH --error=training_results/t5/0.101/seq.err
#SBATCH --partition=studentkillable
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

export PYTHONUNBUFFERED=1 && python light_trainer.py --model t5 --train_builder_path builders/train.english.jsonlines.builder.t5.pkl --val_builder_path builders/dev.english.jsonlines.builder.t5.pkl --dropout 0.101 2>&1 
