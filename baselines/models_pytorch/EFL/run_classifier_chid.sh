#!/usr/bin/env bash
TASK_NAME="chid"
#MODEL_NAME="./chinese_roberta_wwm_ext_pytorch"
MODEL_NAME="/media2/xiaoling/local_models/chinese_roberta_wwm_ext_pytorch/"
CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
export CUDA_VISIBLE_DEVICES="0"

export FewCLUE_DATA_DIR=../../../datasets/

# make output dir
if [ ! -d $CURRENT_DIR/${TASK_NAME}_output ]; then
  mkdir -p $CURRENT_DIR/${TASK_NAME}_output
  echo "makedir $CURRENT_DIR/${TASK_NAME}_output"
fi

train_file_name=$1
dev_file_name=$2
test_file_name=$3

train_flag="$(cut -d'.' -f1 <<<$train_file_name)"
test_flag="$(cut -d'.' -f1 <<<$test_file_name)"

output_dir=$CURRENT_DIR/${TASK_NAME}_output/$test_flag/$train_flag/
if [ ! -d $output_dir ];then
    mkdir -p $output_dir
fi

# run task
cd $CURRENT_DIR
echo "Start running..."
if [ $# == 3 ]; then
    python run_classifier.py \
      --model_type=bert \
      --model_name_or_path=$MODEL_NAME \
      --task_name=$TASK_NAME \
      --do_train \
      --do_eval \
      --do_lower_case \
      --data_dir=$FewCLUE_DATA_DIR/${TASK_NAME}/ \
      --train_file_name=$train_file_name \
      --dev_file_name=$dev_file_name \
      --test_file_name=$test_file_name \
      --ratio=1 \
      --max_seq_length=256 \
      --per_gpu_train_batch_size=4 \
      --per_gpu_eval_batch_size=4 \
      --learning_rate=2e-5 \
      --num_train_epochs=10.0 \
      --logging_steps=3335 \
      --save_steps=3335 \
      --output_dir=$output_dir \
      --overwrite_output_dir \
      --seed=42
elif [ $4 == "predict" ]; then
    echo "Start predict..."
    python run_classifier.py \
      --model_type=bert \
      --model_name_or_path=$MODEL_NAME \
      --task_name=$TASK_NAME \
      --do_predict \
      --do_lower_case \
      --data_dir=$FewCLUE_DATA_DIR/${TASK_NAME}/ \
      --train_file_name=$train_file_name \
      --dev_file_name=$dev_file_name \
      --test_file_name=$test_file_name \
      --max_seq_length=256 \
      --per_gpu_train_batch_size=16 \
      --per_gpu_eval_batch_size=16 \
      --learning_rate=2e-5 \
      --predict_checkpoints=0 \
      --num_train_epochs=3.0 \
      --logging_steps=3335 \
      --save_steps=3335 \
      --output_dir=$output_dir \
      --overwrite_output_dir \
      --seed=42
fi
