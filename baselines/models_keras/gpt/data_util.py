#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:gelin
# datetime:2021/4/24 11:56 上午
# software: PyCharm

"""
读取任务数据集
"""

from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
import os
from config import *
import json


# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 对应的任务描述
desc = ['[unused%s]' % i for i in range(1, 9)]
desc_ids = [tokenizer.token_to_id(t) for t in desc]
mask_id = tokenizer.token_to_id("[MASK]")


current_task_name = TaskName.CECMMNT


def get_labels_ids():
    # print("TASK_NAME:", TASK_NAME, "-------------------")
    label_words = labels_map[current_task_name]
    labels_ids = []
    for words in label_words:
        ids = []
        for w in words:
            ids.append(tokenizer.token_to_id(w))
        labels_ids.append(ids)
    # print("labels ids:", labels_ids, "-------------------")
    return labels_ids


MAX_LEN = 128


def get_data(task_name, parent_path=data_parent_path):
    global current_task_name
    current_task_name = task_name
    if task_name == TaskName.CECMMNT:
        batch_size = 32
        train_data, valid_data, test_data = get_cecmmnt_data(parent_path, cecmmnt_paths)
    elif task_name == TaskName.BUSTM:
        batch_size = 32
        train_data, valid_data, test_data = get_bustm_data(parent_path, bustm_paths)
    elif task_name == TaskName.OCNLI:
        batch_size = 32
        train_data, valid_data, test_data = get_ocnli_data(parent_path, ocnli_paths)
    elif task_name == TaskName.OCEMOTION:
        batch_size = 32
        train_data, valid_data, test_data = get_ocemotion_data(parent_path, ocemotion_paths)
    elif task_name == TaskName.CSLDCP:
        batch_size = 32
        train_data, valid_data, test_data = get_csldcp_data(parent_path, csldcp_paths)
    else:
        raise RuntimeError("task_name参数错误，没有该任务，请核实任务名称。")

    # 模拟标注和非标注数据
    train_frac = 1  # 0.01  # 标注数据的比例
    print("0.length of train_data:", len(train_data))  # 16883
    num_labeled = int(len(train_data) * train_frac)
    # unlabeled_data = [(t, 2) for t, l in train_data[num_labeled:]]
    train_data = train_data[:num_labeled]
    print("1.num_labeled data used:", num_labeled, " ;train_data:", len(train_data))  # 168
    # train_data = train_data + unlabeled_data

    # 转换数据集
    train_generator = data_generator(train_data, batch_size)
    valid_generator = data_generator(valid_data, batch_size, False)
    test_generator = data_generator(test_data, batch_size, False)
    return train_generator, valid_generator, test_generator, get_labels_ids()


class data_generator(DataGenerator):
    """数据生成器
    """
    def __init__(self, data, batch_size=32, is_train=True):
        super(data_generator, self).__init__(data, batch_size)
        self.is_train = is_train

    def __iter__(self, random=False):
        labels_ids = get_labels_ids()
        batch_token_ids, batch_labels = [], []
        batch_token_ids_all_labels = [[] for _ in labels_ids]
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=MAX_LEN)
            if zero_shot:
                # zero shot用的是pet的方式
                token_ids = token_ids[:-1]
            else:
                token_ids = token_ids[:1] + desc_ids[:4] + token_ids[:-1]
                token_ids = token_ids + desc_ids[4:]

            label_index = labels_index_map[current_task_name][label]    # 获取label的索引,因为有些label不是数字
            if self.is_train:
                token_ids += labels_ids[label_index]
            else:
                batch_labels.append(labels_ids[label_index])
                for index, labels in enumerate(labels_ids):
                    tem_token_ids = token_ids + labels
                    batch_token_ids_all_labels[index].append(tem_token_ids)
                for _ in labels_ids[label_index]:
                    token_ids.append(mask_id)

            # if label == 0:
            #     token_ids = token_ids + [neg_id]
            # elif label == 1:
            #     token_ids = token_ids + [pos_id]
            batch_token_ids.append(token_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                if self.is_train:
                    yield batch_token_ids, batch_labels
                else:
                    for index, tem_token_ids in enumerate(batch_token_ids_all_labels):
                        tem_token_ids = sequence_padding(tem_token_ids)
                        batch_token_ids_all_labels[index] = tem_token_ids
                    yield batch_token_ids_all_labels, batch_labels
                batch_token_ids, batch_labels = [], []
                batch_token_ids_all_labels = [[] for _ in labels_ids]


def get_csldcp_data(parent_path, data_paths, maxlen=128):
    MAX_LEN = maxlen

    # 加载数据的方法
    def load_data(filename):
        D = []
        with open(filename, encoding='utf-8') as f:
            for jj, l in enumerate(f):
                json_string = json.loads(l.strip())
                content = json_string['content']
                label = json_string['label']
                if zero_shot:
                    content += "这段描述的学科是"
                D.append((content, label))
        return D

    # 加载数据集
    train_data = load_data(os.path.join(parent_path, data_paths[0]))
    valid_data = load_data(os.path.join(parent_path, data_paths[1]))
    test_data = load_data(os.path.join(parent_path, data_paths[2]))

    return train_data, valid_data, test_data


def get_ocemotion_data(parent_path, data_paths, maxlen=128):
    MAX_LEN = maxlen

    # 加载数据的方法
    def load_data(filename):
        D = []
        with open(filename, encoding='utf-8') as f:
            for jj, l in enumerate(f):
                json_string = json.loads(l.strip())
                content = json_string['content']
                label = json_string['label']
                if label in ['surprise', 'fear']:  # 去掉类别太少的这两个样本
                    continue
                if zero_shot:
                    content += "这句话的感情是"
                D.append((content, label))
        return D

    # 加载数据集
    train_data = load_data(os.path.join(parent_path, data_paths[0]))
    valid_data = load_data(os.path.join(parent_path, data_paths[1]))
    test_data = load_data(os.path.join(parent_path, data_paths[2]))

    return train_data, valid_data, test_data


def get_ocnli_data(parent_path, data_paths, maxlen=128):
    MAX_LEN = maxlen

    # 加载数据的方法
    def load_data(filename):
        D = []
        with open(filename, encoding='utf-8') as f:
            for jj, l in enumerate(f):
                # print("l:",l)
                json_string = json.loads(l.strip())
                # print("json_string:",json_string)
                sentence1 = json_string['sentence1']
                sentence2 = json_string['sentence2']
                label = json_string['label']
                text = "。句子一：" + sentence1 + "；句子二：" + sentence2
                if label not in ['neutral', 'entailment', 'contradiction']:  # # 如果是其他标签的话，跳过这行数据
                    continue
                if zero_shot:
                    text += "这两句话的语义是"
                D.append((text, label))
        return D

    # 加载数据集
    train_data = load_data(os.path.join(parent_path, data_paths[0]))
    valid_data = load_data(os.path.join(parent_path, data_paths[1]))
    test_data = load_data(os.path.join(parent_path, data_paths[2]))

    return train_data, valid_data, test_data


def get_bustm_data(parent_path, data_paths, maxlen=80):
    MAX_LEN = maxlen

    # 加载数据的方法
    def load_data(filename):
        D = []
        with open(filename, encoding='utf-8') as f:
            for jj, l in enumerate(f):
                # print("l:",l)
                json_string = json.loads(l.strip())
                # print("json_string:",json_string)
                sentence1 = json_string['sentence1']
                sentence2 = json_string['sentence2']
                label = json_string['label']
                text = "第一个句子是：" + sentence1 + "；另外一个句子是：" + sentence2
                if zero_shot:
                    text += "这两句话的语义是"

                # text, label = l.strip().split('\t')
                D.append((text, int(label)))
        return D

    # 加载数据集
    train_data = load_data(os.path.join(parent_path, data_paths[0]))
    valid_data = load_data(os.path.join(parent_path, data_paths[1]))
    test_data = load_data(os.path.join(parent_path, data_paths[2]))

    return train_data, valid_data, test_data


def get_cecmmnt_data(parent_path, data_paths, maxlen=128):
    MAX_LEN = maxlen

    def load_data(filename):
        D = []
        with open(filename, encoding='utf-8') as f:
            for l in f:
                json_string = json.loads(l.strip())
                text = json_string["sentence"]
                label = json_string["label"]
                if zero_shot:
                    text += "这句话的情感是"
                # text, label = l.strip().split('\t')
                D.append((text, label))
        return D


    # 加载数据集
    train_data = load_data(os.path.join(parent_path, data_paths[0]))
    valid_data = load_data(os.path.join(parent_path, data_paths[1]))
    test_data = load_data(os.path.join(parent_path, data_paths[2]))

    return train_data, valid_data, test_data

