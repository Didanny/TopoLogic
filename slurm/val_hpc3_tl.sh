#!/bin/sh 
#SBATCH -p gpu
#SBATCH -A dutt_lab_gpu
#SBATCH --gres=gpu:L40S:4
#SBATCH --mem=128G
#SBATCH -t 0-10:00:00

export PYTHONPATH=$(pwd):$PYTHONPATH
bash ./tools/dist_test.sh 4 work_dirs/topologic_2026-01-29_11-02-46 epoch_12