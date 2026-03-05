#!/bin/sh 
#SBATCH -n 9 
#SBATCH -N 1 
#SBATCH -p opengpu.p
#SBATCH -w poison 
#SBATCH --gres=gpu:4
#SBATCH -t 1-00:00:00
#SBATCH -o slurm_logs/log_poison_topologic.out
#SBATCH -e slurm_logs/err_poison_topologic.out

export PYTHONPATH=$(pwd):$PYTHONPATH
# ./tools/dist_test.sh 4 work_dirs/topologic_2026-01-29_11-02-46 epoch_24
# ./tools/dist_test.sh 4 work_dirs/topologic_2026-01-29_11-02-46 epoch_20
# ./tools/dist_test.sh 4 work_dirs/topologic_2026-01-29_11-02-46 epoch_16
./tools/dist_test.sh 4 work_dirs/baseline_24 official --show