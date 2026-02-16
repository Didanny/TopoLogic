#!/bin/sh 
#SBATCH -n 9 
#SBATCH -N 1 
#SBATCH -p opengpu.p
#SBATCH -w poison 
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH -t 3-00:00:00
#SBATCH -o slurm_logs/log_poison_topologic.out
#SBATCH -e slurm_logs/err_poison_topologic.out

export PYTHONPATH=$(pwd):$PYTHONPATH
./tools/dist_train.sh 4 --autoscale-lr --no-validate
