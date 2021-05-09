#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:gelin
# datetime:2021/4/24 11:56 上午
# software: PyCharm

"""
gpt模型进行文本分类/匹配/判别等
"""

from data_util import DataGen
from gpt_model import get_model
from loss_evaluate import get_evaluator
from config import *
import argparse

# 参数
parser = argparse.ArgumentParser(description="参数控制")
parser.add_argument("-t", "--task_name", help="training set index", type=str, default=TaskName.EPRSTMT)
parser.add_argument("-z", "--zero_shot", help="few-shot or zero-shot", action='store_true', default=False)
args = parser.parse_args()
task_name = args.task_name    # 选择任务
zero_shot = args.zero_shot    # 是否进行零样本学习



# # 任务名称
# task_name = TaskName.EPRSTMT
# task_name = TaskName.BUSTM
# task_name = TaskName.OCNLI
# # task_name = TaskName.OCEMOTION # 任务取消了
# task_name = TaskName.CSLDCP
# task_name = TaskName.TNEWS
# task_name = TaskName.IFLYTEK
# task_name = TaskName.WSC
# task_name = TaskName.CSL
# task_name = TaskName.CHID
#
# zero_shot = True    # 是否进行零样本学习

# 获取任务数据集
data_gen = DataGen(task_name, zero_shot)
train_generator, valid_generator, test_generator, labels = data_gen.get_data()

# 获取任务模型
model = get_model()

# 获取任务evaluator
evaluator = get_evaluator(task_name, valid_generator, test_generator, labels, zero_shot)

print("@"*20, "task name:", task_name, ",zero_shot:", zero_shot, "@"*20)

# 训练模型
model.fit_generator(
    train_generator.forfit(),
    steps_per_epoch=len(train_generator) * 50,
    epochs=100,
    callbacks=[evaluator]
)
