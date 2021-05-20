#!/usr/bin/env bash
TASK_NAME="tnews"
#MODEL_NAME="./chinese_roberta_wwm_ext_L-12_H-768_A-12/"
MODEL_NAME="./ocnli_output/bert/"
CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
export CUDA_VISIBLE_DEVICES="0"

export FewCLUE_DATA_DIR=../../../datasets/

# make output dir
if [ ! -d $CURRENT_DIR/${TASK_NAME}_output ]; then
  mkdir -p $CURRENT_DIR/${TASK_NAME}_output
  echo "makedir $CURRENT_DIR/${TASK_NAME}_output"
fi

# run task
cd $CURRENT_DIR
echo "Start running..."
if [ $# == 0 ]; then
    python run_classifier.py \
      --model_type=bert \
      --model_name_or_path=$MODEL_NAME \
      --task_name=$TASK_NAME \
      --do_train \
      --do_eval \
      --do_lower_case \
      --data_dir=$FewCLUE_DATA_DIR/${TASK_NAME}/ \
      --max_seq_length=128 \
      --per_gpu_train_batch_size=4 \
      --per_gpu_eval_batch_size=4 \
      --learning_rate=2e-5 \
      --num_train_epochs=10.0 \
      --logging_steps=3335 \
      --save_steps=3335 \
      --output_dir=$CURRENT_DIR/${TASK_NAME}_output/ \
      --overwrite_output_dir \
      --seed=42
elif [ $1 == "predict" ]; then
    echo "Start predict..."
    python run_classifier.py \
      --model_type=bert \
      --model_name_or_path=$MODEL_NAME \
      --task_name=$TASK_NAME \
      --do_predict \
      --do_lower_case \
      --data_dir=$FewCLUE_DATA_DIR/${TASK_NAME}/ \
      --max_seq_length=128 \
      --per_gpu_train_batch_size=16 \
      --per_gpu_eval_batch_size=16 \
      --learning_rate=2e-5 \
      --predict_checkpoints=0 \
      --num_train_epochs=3.0 \
      --logging_steps=3335 \
      --save_steps=3335 \
      --output_dir=$CURRENT_DIR/${TASK_NAME}_output/ \
      --overwrite_output_dir \
      --seed=42
fi
