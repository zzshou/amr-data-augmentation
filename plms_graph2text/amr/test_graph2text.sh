#!/bin/bash

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

SAVE_DATA_FILE=$5
DATA_FILE=$4
GPUID=$3
CHECK_POINT=$2
MODEL=$1
FOLDER=outputs

export CURRENT_DIR=${ROOT_DIR}
export OUTPUT_DIR=${CURRENT_DIR}/${FOLDER}

#rm -rf $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

export OMP_NUM_THREADS=10

export CUDA_VISIBLE_DEVICES=${GPUID}

python ${ROOT_DIR}/finetune.py \
--data_dir=${DATA_FILE} \
--task graph2text \
--model_name_or_path=${MODEL} \
--eval_batch_size=4 \
--gpus 1 \
--output_dir=$OUTPUT_DIR \
--checkpoint=$CHECK_POINT \
--max_source_length=512 \
--max_target_length=384 \
--val_max_target_length=384 \
--test_max_target_length=384 \
--eval_max_gen_length=384 \
--do_predict \
--eval_beams 5 \
--save_file_name=${SAVE_DATA_FILE}