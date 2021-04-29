#!/usr/bin/env bash
# @Author: Xuanwei Zhang
# @Date:   2021-04-28
# @Last Modified by:   Xuanwei Zhang
# @Last Modified time: 2021-04-28

MODEL_NAME="chinese_L-12_H-768_A-12"
CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
export CUDA_VISIBLE_DEVICES="0"
export BERT_PRETRAINED_MODELS_DIR=$CURRENT_DIR/prev_trained_model
export BERT_BASE_DIR=$BERT_PRETRAINED_MODELS_DIR/$MODEL_NAME
export CLUE_DATA_DIR=$CURRENT_DIR/../../../../ready_data

# download and unzip dataset
if [ ! -d $CLUE_DATA_DIR ]; then
  mkdir -p $CLUE_DATA_DIR
  echo "makedir $CLUE_DATA_DIR"
fi
cd $CLUE_DATA_DIR
# download model
if [ ! -d $BERT_PRETRAINED_MODELS_DIR ]; then
  mkdir -p $BERT_PRETRAINED_MODELS_DIR
  echo "makedir $BERT_PRETRAINED_MODELS_DIR"
fi
cd $BERT_PRETRAINED_MODELS_DIR
if [ ! -d $MODEL_NAME ]; then
  wget https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
  unzip chinese_L-12_H-768_A-12.zip
  rm chinese_L-12_H-768_A-12.zip
else
  cd $MODEL_NAME
  if [ ! -f "bert_config.json" ] || [ ! -f "vocab.txt" ] || [ ! -f "bert_model.ckpt.index" ] || [ ! -f "bert_model.ckpt.meta" ] || [ ! -f "bert_model.ckpt.data-00000-of-00001" ]; then
    cd ..
    rm -rf $MODEL_NAME
    wget https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
    unzip chinese_L-12_H-768_A-12.zip
    rm chinese_L-12_H-768_A-12.zip
  else
    echo "model exists"
  fi
fi
echo "Finish download model."

# run task
cd $CURRENT_DIR
process() {
  DATA_DIR=$CLUE_DATA_DIR/${TASK_NAME}/$1
  echo "Start running..."
  python run_classifier.py \
    --task_name=$TASK_NAME \
    --do_train=true \
    --do_eval=true \
    --data_dir=$DATA_DIR \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
    --max_seq_length=128 \
    --train_batch_size=32 \
    --learning_rate=2e-5 \
    --num_train_epochs=3.0 \
    --output_dir=$CURRENT_DIR/${TASK_NAME}_output/$1

  python run_classifier.py \
    --task_name=$TASK_NAME \
    --do_train=true \
    --do_eval=true \
    --data_dir=$DATA_DIR/test/ \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
    --max_seq_length=128 \
    --train_batch_size=32 \
    --learning_rate=2e-5 \
    --num_train_epochs=3.0 \
    --output_dir=$CURRENT_DIR/${TASK_NAME}_output/$i

  echo "Start predict..."
  python run_classifier.py \
    --task_name=$TASK_NAME \
    --do_train=false \
    --do_eval=false \
    --do_predict=true \
    --data_dir=$DATA_DIR  \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
    --max_seq_length=128 \
    --train_batch_size=32 \
    --learning_rate=2e-5 \
    --num_train_epochs=3.0 \
    --output_dir=$CURRENT_DIR/${TASK_NAME}_output/$1
}

run() {
  TASK_NAME=$1
  TASK_DIR=$CLUE_DATA_DIR/$TASK_NAME
  for i in `seq 0 4`
  do
    SPLIT_DIR=$TASK_DIR/$i
    if [ ! -d $SPLIT_DIR ];then
      mkdir -p $SPLIT_DIR $SPLIT_DIR/test/
    fi
    cp $TASK_DIR/train_${i}.json $SPLIT_DIR/train.json
    cp $TASK_DIR/dev_${i}.json $SPLIT_DIR/dev.json
    cp $TASK_DIR/test.json $SPLIT_DIR/test.json

    cp $TASK_DIR/train_${i}.json $SPLIT_DIR/test/train.json
    cp $TASK_DIR/test_public.json $SPLIT_DIR/test/dev.json
    cp $TASK_DIR/test.json $SPLIT_DIR/test/test.json

    process $i
  done
  ALL_DIR=$TASK_DIR/all
  if [ ! -d $ALL_DIR ];then
    mkdir -p $ALL_DIR $ALL_DIR/test
  fi
  cp $TASK_DIR/train_few_all.json $ALL_DIR/train.json
  cp $TASK_DIR/dev_few_all.json $ALL_DIR/dev.json
  cp $TASK_DIR/test.json $ALL_DIR/test.json
  
  cp $TASK_DIR/train_few_all.json $ALL_DIR/test/train.json
  cp $TASK_DIR/test_public.json $ALL_DIR/test/dev.json
  cp $TASK_DIR/test.json $ALL_DIR/test/test.json

  process all
}

#TASK_LIST=(csldcp)
#TASK_LIST=(cecmmnt tnews iflytek ocnli csl cluewsc bustm)
TASK_LIST=(cecmmnt tnews iflytek ocnli csl cluewsc bustm csldcp)

for item in ${TASK_LIST[@]}
do
  run $item
done
