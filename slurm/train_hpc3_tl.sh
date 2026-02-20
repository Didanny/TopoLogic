#!/bin/sh 
#SBATCH -p gpu
#SBATCH -A dutt_lab_gpu
#SBATCH --gres=gpu:L40S:4
#SBATCH --mem=128G
#SBATCH -t 2-00:00:00

export PYTHONPATH=$(pwd):$PYTHONPATH
bash ./tools/dist_train.sh 4 --no-validate
