#!/usr/bin/env bash
export PET_ELECTRA_ROOT=`pwd`
export PYTHONPATH=$PET_ELECTRA_ROOT:$PYTHONPATH
export PYTHON_EXEC=python
set -exu
export CUDA_VISIBLE_DEVICES="4"
TASKNAME=$1
config_file="config/$TASKNAME.json"

for num in "0" #  "1" "2" "3" "4" "few_all"
do
  echo "start training data_$num...."
  python -u -m src.train -c $config_file -k dataset_num=$num
  exp_dir="exp_out/$TASKNAME/$num"
  python -m src.test -e $exp_dir
done

