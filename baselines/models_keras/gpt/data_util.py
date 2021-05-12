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
import pdb

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 对应的任务描述
desc = ['[unused%s]' % i for i in range(1, 9)]
desc_ids = [tokenizer.token_to_id(t) for t in desc]
mask_id = tokenizer.token_to_id("[MASK]")


class data_generator(DataGenerator):
    """数据生成器
    """
    def __init__(self, data, labels_ids, task_name, batch_size=32, max_len=128, is_train=True, zero_shot=False):
        super(data_generator, self).__init__(data, batch_size)
        self.labels_ids = labels_ids
        self.task_name = task_name
        self.batch_size = batch_size
        self.max_len = max_len
        self.is_train = is_train
        self.zero_shot = zero_shot

    def __iter__(self, random=False):
        batch_token_ids, batch_labels = [], []
        batch_token_ids_all_labels = [[] for _ in self.labels_ids]
        for is_end, (text, label) in self.sample(random):
            token_ids = []
            if self.task_name == TaskName.CHID:    # todo chid目前只有零样本学习
                candidates, index, answer = label
                if self.is_train:
                    tem_text = text.replace("#idiom#", candidates[answer])
                    token_ids, segment_ids = tokenizer.encode(tem_text, maxlen=self.max_len)
                else:
                    for ind, cand in enumerate(candidates):
                        tem_text = text.replace("#idiom#", cand)
                        tem_token_ids, segment_ids = tokenizer.encode(tem_text, maxlen=self.max_len)
                        batch_token_ids_all_labels[ind].append(tem_token_ids)
                    batch_labels.append((index, answer))
            else:
                token_ids, segment_ids = tokenizer.encode(text, maxlen=self.max_len)
                if self.zero_shot:
                    # zero shot用的是pet的方式
                    token_ids = token_ids[:-1]
                else:
                    token_ids = token_ids[:1] + desc_ids[:4] + token_ids[:-1]
                    token_ids = token_ids + desc_ids[4:]

                label_index = labels_index_map[self.task_name][label]  # 获取label的索引,因为有些label不是数字
                if self.is_train:
                    token_ids += self.labels_ids[label_index]
                else:
                    batch_labels.append(self.labels_ids[label_index])
                    for index, labels in enumerate(self.labels_ids):
                        tem_token_ids = token_ids + labels
                        batch_token_ids_all_labels[index].append(tem_token_ids)
                    for _ in self.labels_ids[label_index]:
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
                batch_token_ids_all_labels = [[] for _ in self.labels_ids]


class DataGen(object):
    def __init__(self, task_name, zero_shot, batch_size=0, max_len=0):
        self.task_name = task_name
        self.zero_shot = zero_shot
        self.max_len = self.get_max_len(max_len)
        self.batch_size = self.get_batch_size(batch_size)

    def get_batch_size(self, batch_size):
        if batch_size > 0: return batch_size
        if self.task_name == TaskName.IFLYTEK:
            return 4
        return 32

    def get_max_len(self, max_len):
        if max_len > 0:
            return max_len
        if self.task_name == TaskName.IFLYTEK or self.task_name == TaskName.CSL:
            return 384
        elif self.task_name == TaskName.BUSTM:
            return 80
        elif self.task_name == TaskName.WSC:
            return 256
        elif self.task_name == TaskName.CHID:
            return 180
        return 128

    def get_labels_ids(self):
        label_words = labels_map[self.task_name]
        labels_ids = []
        for words in label_words:
            ids = []
            for w in words:
                ids.append(tokenizer.token_to_id(w))
            labels_ids.append(ids)
        return labels_ids

    def get_data(self, parent_path=data_parent_path):
        if self.task_name == TaskName.EPRSTMT:
            train_data, valid_data, test_data = self.get_cecmmnt_data(parent_path, eprstmt_paths)
        elif self.task_name == TaskName.BUSTM:
            train_data, valid_data, test_data = self.get_bustm_data(parent_path, bustm_paths)
        elif self.task_name == TaskName.OCNLI:
            train_data, valid_data, test_data = self.get_ocnli_data(parent_path, ocnli_paths)
        elif self.task_name == TaskName.OCEMOTION:
            train_data, valid_data, test_data = self.get_ocemotion_data(parent_path, ocemotion_paths)
        elif self.task_name == TaskName.CSLDCP:
            train_data, valid_data, test_data = self.get_csldcp_data(parent_path, csldcp_paths)
        elif self.task_name == TaskName.TNEWS:
            train_data, valid_data, test_data = self.get_tnews_data(parent_path, tnews_paths)
        elif self.task_name == TaskName.IFLYTEK:
            train_data, valid_data, test_data = self.get_iflytek_data(parent_path, iflytek_paths)
        elif self.task_name == TaskName.WSC:
            train_data, valid_data, test_data = self.get_wsc_data(parent_path, wsc_paths)
        elif self.task_name == TaskName.CSL:
            train_data, valid_data, test_data = self.get_csl_data(parent_path, csl_paths)
        elif self.task_name == TaskName.CHID:
            train_data, valid_data, test_data = self.get_chid_data(parent_path, chid_paths)
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

        labels_ids = labels_map[TaskName.CHID]    # todo chid与其他几个不一致
        if self.task_name != TaskName.CHID:
            labels_ids = self.get_labels_ids()

        # 转换数据集
        train_generator = data_generator(train_data, labels_ids, self.task_name, batch_size=self.batch_size,
                                         max_len=self.max_len, zero_shot=self.zero_shot)
        valid_generator = data_generator(valid_data, labels_ids, self.task_name, batch_size=self.batch_size,
                                         max_len=self.max_len, is_train=False, zero_shot=self.zero_shot)
        test_generator = data_generator(test_data, labels_ids, self.task_name, batch_size=self.batch_size,
                                        max_len=self.max_len, is_train=False, zero_shot=self.zero_shot)
        return train_generator, valid_generator, test_generator, labels_ids

    def get_chid_data(self, parent_path, data_paths):
        # 加载数据的方法
        def load_data(filename):
            D = []
            with open(filename, encoding='utf-8') as f:
                for jj, l in enumerate(f):
                    json_string = json.loads(l.strip())
                    content = json_string['content']
                    index = content.index("#idiom#")
                    candidates = json_string['candidates']    # list
                    answer = json_string['answer']
                    label = (candidates, index, answer)
                    D.append((content, label))
            return D

        # 加载数据集
        train_data = load_data(os.path.join(parent_path, data_paths[0]))
        valid_data = load_data(os.path.join(parent_path, data_paths[1]))
        test_data = load_data(os.path.join(parent_path, data_paths[2]))

        return train_data, valid_data, test_data

    def get_csl_data(self, parent_path, data_paths):
        # 加载数据的方法
        def load_data(filename):
            D = []
            with open(filename, encoding='utf-8') as f:
                for jj, l in enumerate(f):
                    json_string = json.loads(l.strip())
                    content = json_string['abst'][:300]
                    keyword = json_string['keyword']
                    content = content + "与下面的关键词：" + ",".join(keyword)
                    label = json_string['label']
                    if self.zero_shot:
                        content += "，"
                    D.append((content, label))
            return D

        # 加载数据集
        train_data = load_data(os.path.join(parent_path, data_paths[0]))
        valid_data = load_data(os.path.join(parent_path, data_paths[1]))
        test_data = load_data(os.path.join(parent_path, data_paths[2]))

        return train_data, valid_data, test_data

    def get_wsc_data(self, parent_path, data_paths):
        # 加载数据的方法
        def load_data(filename):
            D = []
            with open(filename, encoding='utf-8') as f:
                for jj, l in enumerate(f):
                    json_string = json.loads(l.strip())
                    content = json_string['text']
                    span1_text = json_string['target']['span1_text']
                    span2_text = json_string['target']['span2_text']
                    content = content + "其中" + span2_text + "是" + span1_text
                    label = json_string['label']
                    if self.zero_shot:
                        content += ",这是"
                    D.append((content, label))
            return D

        # 加载数据集
        train_data = load_data(os.path.join(parent_path, data_paths[0]))
        valid_data = load_data(os.path.join(parent_path, data_paths[1]))
        test_data = load_data(os.path.join(parent_path, data_paths[2]))

        return train_data, valid_data, test_data

    def get_iflytek_data(self, parent_path, data_paths):
        # 加载数据的方法
        def load_data(filename):
            D = []
            with open(filename, encoding='utf-8') as f:
                for jj, l in enumerate(f):
                    json_string = json.loads(l.strip())
                    content = json_string['sentence']
                    label = json_string['label_des']
                    if self.zero_shot:
                        content += "这段内容是关于"
                    D.append((content, label))
            return D

        # 加载数据集
        train_data = load_data(os.path.join(parent_path, data_paths[0]))
        valid_data = load_data(os.path.join(parent_path, data_paths[1]))
        test_data = load_data(os.path.join(parent_path, data_paths[2]))

        return train_data, valid_data, test_data

    def get_tnews_data(self, parent_path, data_paths):
        # 加载数据的方法
        def load_data(filename):
            D = []
            with open(filename, encoding='utf-8') as f:
                for jj, l in enumerate(f):
                    json_string = json.loads(l.strip())
                    content = json_string['sentence']
                    label = json_string['label_desc']
                    if self.zero_shot:
                        content += "这段内容是关于"
                    D.append((content, label))
            return D

        # 加载数据集
        train_data = load_data(os.path.join(parent_path, data_paths[0]))
        valid_data = load_data(os.path.join(parent_path, data_paths[1]))
        test_data = load_data(os.path.join(parent_path, data_paths[2]))

        return train_data, valid_data, test_data

    def get_csldcp_data(self, parent_path, data_paths, maxlen=128):
        self.max_len = maxlen

        # 加载数据的方法
        def load_data(filename):
            D = []
            with open(filename, encoding='utf-8') as f:
                for jj, l in enumerate(f):
                    json_string = json.loads(l.strip())
                    content = json_string['content']
                    label = json_string['label']
                    if self.zero_shot:
                        content += "这段描述的学科是"
                    D.append((content, label))
            return D

        # 加载数据集
        train_data = load_data(os.path.join(parent_path, data_paths[0]))
        valid_data = load_data(os.path.join(parent_path, data_paths[1]))
        test_data = load_data(os.path.join(parent_path, data_paths[2]))

        return train_data, valid_data, test_data

    def get_ocemotion_data(self, parent_path, data_paths):
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
                    if self.zero_shot:
                        content += "这句话的感情是"
                    D.append((content, label))
            return D

        # 加载数据集
        train_data = load_data(os.path.join(parent_path, data_paths[0]))
        valid_data = load_data(os.path.join(parent_path, data_paths[1]))
        test_data = load_data(os.path.join(parent_path, data_paths[2]))

        return train_data, valid_data, test_data

    def get_ocnli_data(self, parent_path, data_paths):
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
                    if self.zero_shot:
                        text += "这两句话的语义是"
                    D.append((text, label))
            return D

        # 加载数据集
        train_data = load_data(os.path.join(parent_path, data_paths[0]))
        valid_data = load_data(os.path.join(parent_path, data_paths[1]))
        test_data = load_data(os.path.join(parent_path, data_paths[2]))

        return train_data, valid_data, test_data

    def get_bustm_data(self, parent_path, data_paths):
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
                    if self.zero_shot:
                        text += "这两句话的语义是"

                    # text, label = l.strip().split('\t')
                    D.append((text, int(label)))
            return D

        # 加载数据集
        train_data = load_data(os.path.join(parent_path, data_paths[0]))
        valid_data = load_data(os.path.join(parent_path, data_paths[1]))
        test_data = load_data(os.path.join(parent_path, data_paths[2]))

        return train_data, valid_data, test_data

    def get_cecmmnt_data(self, parent_path, data_paths):
        def load_data(filename):
            D = []
            with open(filename, encoding='utf-8') as f:
                for l in f:
                    json_string = json.loads(l.strip())
                    text = json_string["sentence"]
                    label = json_string["label"]
                    if self.zero_shot:
                        text += "这句话的情感是"
                    # text, label = l.strip().split('\t')
                    D.append((text, label))
            return D


        # 加载数据集
        train_data = load_data(os.path.join(parent_path, data_paths[0]))
        valid_data = load_data(os.path.join(parent_path, data_paths[1]))
        test_data = load_data(os.path.join(parent_path, data_paths[2]))

        return train_data, valid_data, test_data

