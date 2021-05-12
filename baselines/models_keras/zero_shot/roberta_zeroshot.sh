#!/usr/bin/env bash
# export python=/path/to/python # 如果不使用默认的python解释器
TASK_NAME=$1
CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
PET_FILE=$CURRENT_DIR'/../pet/pet_'$TASK_NAME".py"
python $PET_FILE -tt=zero-shot