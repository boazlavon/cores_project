#! /bin/sh
#SBATCH --job-name=0.1it5
#SBATCH --output=training_results/init_t5/0.1/seq.out
#SBATCH --error=training_results/init_t5/0.1/seq.err
#SBATCH --partition=studentkillable
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

export PYTHONUNBUFFERED=1 && python light_trainer.py --model init_t5 --train_builder_path builders/train.english.jsonlines.builder.t5.pkl --val_builder_path builders/dev.english.jsonlines.builder.t5.pkl --dropout 0.1 2>&1 
