#!/bin/bash

SCRIPTS_HOME=$(readlink -f `dirname $0`)
cd ${SCRIPTS_HOME}

BATCH_SIZE=32
ACCUM=2
EPOCH=20
lr_scheduler_type=cosine
warmup_steps=40
warmup_ratio=0.2

MODEL_PATH=/path/to/model
DATA_PATH=./multilingual-sentiments
SAVE_PATH=/path/to/save/dir

cuids=0,1,2,3,4,5,6,7

IFS="," read -ra cuid <<< ${cuids}
cuids_num=${#cuid[@]}

if [ -d ${SAVE_PATH} ]; then
    echo 文件夹已存在，跳过训练
else
    echo 文件夹不存在，开始训练
    mkdir -p ${SAVE_PATH}
    CUDA_VISIBLE_DEVICES=${cuids} nohup torchrun --nproc-per-node=${cuids_num} trainer.py \
        --model_name_or_path=${MODEL_PATH} \
        --dataset_name=${DATA_PATH} \
        --report_to="wandb" \
        --learning_rate=1e-6 \
        --per_device_train_batch_size=${BATCH_SIZE} \
        --gradient_accumulation_steps=${ACCUM} \
        --output_dir=${SAVE_PATH} \
        --logging_steps=50 \
        --num_train_epochs=${EPOCH} \
        --max_steps=-1 \
        --lr_scheduler_type=${lr_scheduler_type} \
        --warmup_steps=${warmup_steps} \
        --gradient_checkpointing > ${SAVE_PATH}/out.log 2>&1 &

        cp ${MODEL_PATH}/token* > ${SAVE_PATH}
        cp ${MODEL_PATH}/vocab.json > ${SAVE_PATH}
        cp ${MODEL_PATH}/generation_config.json > ${SAVE_PATH}
        cp ${MODEL_PATH}/merges.txt > ${SAVE_PATH}
fi