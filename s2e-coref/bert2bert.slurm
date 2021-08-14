#! /bin/sh
#SBATCH --job-name=bert2bert
#SBATCH --output=bert2bert.out
#SBATCH --error=bert2bert.err
#SBATCH --partition=studentbatch
#SBATCH --time=1
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

./eval_bert2bert_model.sh 2>&1 
