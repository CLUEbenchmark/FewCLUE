#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:gelin
# datetime:2021/4/24 11:56 上午
# software: PyCharm

"""
情感分析例子，利用LM+P-tuning
"""

from data_util import get_data
from gpt_model import get_model
from loss_evaluate import get_evaluator
from config import *


task_name = TaskName.CECMMNT
task_name = TaskName.BUSTM
task_name = TaskName.OCNLI
# task_name = TaskName.OCEMOTION
task_name = TaskName.CSLDCP

current_task_name = task_name

# 获取任务数据集
train_generator, valid_generator, test_generator, labels = get_data(task_name)

# 获取任务模型
model = get_model()

# 获取任务evaluator
evaluator = get_evaluator(task_name, valid_generator, test_generator, labels)

# 训练模型
model.fit_generator(
    train_generator.forfit(),
    steps_per_epoch=len(train_generator) * 50,
    epochs=100,
    callbacks=[evaluator]
)
