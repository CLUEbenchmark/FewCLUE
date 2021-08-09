#!/usr/bin/env bash
export PET_ELECTRA_ROOT=`pwd`
export PYTHONPATH=$PET_ELECTRA_ROOT:$PYTHONPATH
export PYTHON_EXEC=python3
set -exu
export CUDA_VISIBLE_DEVICES="0"
cp -r ../../../datasets ./data
process() {
    TASKNAME=$1
    config_file="config/$TASKNAME.json"

    index_list=("0" "1" "2" "3" "4" "few_all")
    for num in ${index_list[@]}
    do
      echo "start training data_$num...."
      python3 -u -m src.train -c $config_file -k dataset_num=$num
      exp_dir="exp_out/$TASKNAME/$num"
      python3 -m src.test -e $exp_dir
    done
}


#task_list=(EPRSTMT bustm ocnli csldcp tnews cluewsc ifytek csl chid)
task_list=(chid csl ifytek cluewsc tnews)
for task in ${task_list[@]}
do
  echo "task"$task
  process $task
done
