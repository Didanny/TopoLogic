#!/usr/bin/env bash
set -x

timestamp=`date +"%Y-%m-%d_%H-%M-%S"`

WORK_DIR=work_dirs/topologic_${timestamp}
CONFIG=projects/configs/topologic_r50_8x1_24e_olv2_subset_A.py

# Create the work directory if it doesn't exist
mkdir -p ${WORK_DIR}

GPUS=$1
PORT=${PORT:-28510}

# Check if the number of GPUs is set to 1 for single GPU training
if [ "$GPUS" -eq 1 ]; then
    python tools/train.py $CONFIG --work-dir ${WORK_DIR} --deterministic ${@:2} \
        2>&1 | tee ${WORK_DIR}/train.${timestamp}.log
    exit 0
fi

python -m torch.distributed.run  --nproc_per_node=$GPUS --master_port=$PORT\
    tools/train.py $CONFIG --launcher pytorch --work-dir ${WORK_DIR} --deterministic ${@:3} \
    2>&1 | tee ${WORK_DIR}/train.${timestamp}.log

