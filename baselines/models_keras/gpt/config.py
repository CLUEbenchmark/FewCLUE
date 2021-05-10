#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:gelin
# datetime:2021/4/24 2:24 下午
# software: PyCharm

"""
参数配置
"""

from label_config import *

# 模型路径
config_path = '../../../pretrained_models/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../../../pretrained_models/models_gpt/cn_gpt'
dict_path = '../../../pretrained_models/models_gpt/vocab.txt'

# 数据路径
data_parent_path = "../../../datasets/"
eprstmt_paths = ['eprstmt/train_0.json', 'eprstmt/dev_0.json', 'eprstmt/test_public.json']
bustm_paths = ['bustm/train_0.json', 'bustm/dev_0.json', 'bustm/test_public.json']
ocnli_paths = ['ocnli/train_0.json', 'ocnli/dev_0.json', 'ocnli/test_public.json']
ocemotion_paths = ['ocemotion/train_few_0.json', 'ocemotion/dev_few_0.json', 'ocemotion/test_public.json']
csldcp_paths = ['csldcp/train_0.json', 'csldcp/dev_0.json', 'csldcp/test_public.json']
tnews_paths = ['tnews/train_0.json', 'tnews/dev_0.json', 'tnews/test_public.json']
iflytek_paths = ['iflytek/train_0.json', 'iflytek/dev_0.json', 'iflytek/test_public.json']
wsc_paths = ['cluewsc/train_0.json', 'cluewsc/dev_0.json', 'cluewsc/test_public.json']
csl_paths = ['csl/train_0.json', 'csl/dev_0.json', 'csl/test_public.json']
chid_paths = ['chid/train_0.json', 'chid/dev_0.json', 'chid/test_public.json']


class TaskName(object):
    EPRSTMT = "eprstmt"
    BUSTM = "bustm"
    OCNLI = "ocnli"
    OCEMOTION = "ocemotion"
    CSLDCP = "csldcp"
    TNEWS = "tnews"
    IFLYTEK = "iflytek"
    WSC = "wsc"
    CSL = "csl"
    CHID = "chid"


# 每个任务下，标签的字数要一致
# 各个任务标签
labels_map = {
    TaskName.EPRSTMT: [u"不满意", u"很满意"],
    TaskName.BUSTM: [u"反", u"正"],
    TaskName.OCNLI: [u"中立", u"包含", u"矛盾"],
    TaskName.OCEMOTION: [u"喜欢", u"开心", u"伤心", u"愤怒", u"厌恶", u"惊讶", u"恐惧"],
    TaskName.CSLDCP: [v for v in csldcp_label_map.values()],
    TaskName.TNEWS: [v for v in tnews_label_map.values()],
    TaskName.IFLYTEK: [v for v in iflytek_label_map.values()],
    TaskName.WSC: [v for v in wsc_label_map.values()],
    TaskName.CSL: [v for v in csl_label_map.values()],
    TaskName.CHID: [v for v in range(7)]
}

# 各个任务标签索引
labels_index_map = {
    TaskName.EPRSTMT: {
        "Negative": labels_map[TaskName.EPRSTMT].index(u"不满意"),
        "Positive": labels_map[TaskName.EPRSTMT].index(u"很满意")
    },
    TaskName.BUSTM: {
        0: labels_map[TaskName.BUSTM].index(u"反"),
        1: labels_map[TaskName.BUSTM].index(u"正")
    },
    TaskName.OCNLI: {
        "neutral": labels_map[TaskName.OCNLI].index(u"中立"),
        "entailment": labels_map[TaskName.OCNLI].index(u"包含"),
        "contradiction": labels_map[TaskName.OCNLI].index(u"矛盾")
    },
    TaskName.OCEMOTION: {
        "like": labels_map[TaskName.OCEMOTION].index(u"喜欢"),
        "happiness": labels_map[TaskName.OCEMOTION].index(u"开心"),
        "sadness": labels_map[TaskName.OCEMOTION].index(u"伤心"),
        "anger": labels_map[TaskName.OCEMOTION].index(u"愤怒"),
        "disgust": labels_map[TaskName.OCEMOTION].index(u"厌恶"),
        "surprise": labels_map[TaskName.OCEMOTION].index(u"惊讶"),
        "fear": labels_map[TaskName.OCEMOTION].index(u"恐惧")
    },
    TaskName.CSLDCP: {
        k: labels_map[TaskName.CSLDCP].index(v) for k, v in csldcp_label_map.items()
    },
    TaskName.TNEWS: {
        k: labels_map[TaskName.TNEWS].index(v) for k, v in tnews_label_map.items()
    },
    TaskName.IFLYTEK: {
        k: labels_map[TaskName.IFLYTEK].index(v) for k, v in iflytek_label_map.items()
    },
    TaskName.WSC: {
        k: labels_map[TaskName.WSC].index(v) for k, v in wsc_label_map.items()
    },
    TaskName.CSL: {
        k: labels_map[TaskName.CSL].index(v) for k, v in csl_label_map.items()
    },
    TaskName.CHID: {
        k: k for k in range(7)
    }
}