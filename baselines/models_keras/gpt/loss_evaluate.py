#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:gelin
# datetime:2021/4/24 11:56 上午
# software: PyCharm

"""
各个数据集的loss和evaluate
"""
import numpy as np
from bert4keras.backend import keras
from config import *
import sys


class Evaluator(keras.callbacks.Callback):
    def __init__(self, evaluate, valid_data, test_data, task_name, labels=[], zero_shot=False):
        self.best_val_acc = 0.
        self.evaluate = evaluate
        self.vaild_data = valid_data
        self.test_data = test_data
        self.task_name = task_name
        self.lalebls = labels
        self.zero_shot = zero_shot

    def on_train_end(self, logs=None):
        self.model.load_weights(self.task_name + '_best_model_gpt.weights')
        test_acc = self.evaluate(self.test_data, self.model, self.lalebls)
        print(
            u'gpt few shot test_acc: %.5f\n' %
            (test_acc)
        )

    def on_epoch_end(self, epoch, logs=None):
        val_acc = self.evaluate(self.vaild_data, self.model, self.lalebls)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.model.save_weights(self.task_name + '_best_model_gpt.weights')
        print(
            u'val_acc: %.5f, best_val_acc: %.5f\n' %
            (val_acc, self.best_val_acc)
        )

    def on_train_begin(self, logs=None):
        if self.zero_shot:
            test_acc = self.evaluate(self.test_data, self.model, self.lalebls)
            print(
                u'gpt zero shot test_acc: %.5f\n' %
                (test_acc)
            )
            print("零样本预测结束，直接退出程序")
            sys.exit()


def evaluate_sentiment(data, model, labels):
    neg_id, pos_id = labels[0][0], labels[1][0]
    total, right = 0., 0.
    for x_true, _ in data:
        y_pred = model.predict(x_true)
        for x, y in zip(x_true, y_pred):    # [seq_len, voc_size]
            x = np.trim_zeros(x)
            y = y[:len(x)][-2, [neg_id, pos_id]].argmax()
            y = [neg_id, pos_id][y]
            if y == x[-1]:
                right += 1
            total += 1
    return right / total


def evaluate(data, model, labels):
    """
    评估准确率，每个label的字数要一致
    Args:
        data:
        model:
        labels:[[label_1_word1_id, label_1_word2_id,...],[label_2_word1_id, label_2_word2_id, ...],...]
    Returns:

    """
    labels_num, label_word_len = len(labels), len(labels[0])
    labels = np.array(labels)
    total, right = 0., 0.
    for x_true, batch_labels in data:
        y_pred = model.predict(x_true)
        for x, y, labels_id in zip(x_true, y_pred, batch_labels):
            total += 1
            x = np.trim_zeros(x)
            ys = []
            for ty, i in zip(y[len(x)-1-label_word_len:len(x)-1], range(label_word_len)):
                ys.append(ty[labels[:, i]])    # 第i个位置的各个字符的概率
            y_index = np.prod(ys, axis=0).argmax()    # 选择标签的概率最大值的索引
            num = 0     # 计算预测标签与真实标签相同字符的个数
            # for xi, yi in zip(x[-label_word_len:], labels[y_index]):
            for xi, yi in zip(labels_id, labels[y_index]):
                if xi == yi:
                    num += 1
            if num == label_word_len:
                right += 1
            # y = y[:len(x)][-2, [neg_id, pos_id]].argmax()
            # y = [neg_id, pos_id][y]
            # if y == x[-1]:
            #     right += 1

    return right / total


def evaluate_seq2seq(data, model, labels):
    """
    评估准确率，每个label的字数要一致
    Args:
        data:
        model:
        labels:[[label_1_word1_id, label_1_word2_id,...],[label_2_word1_id, label_2_word2_id, ...],...]
    Returns:

    """
    labels_num, label_word_len = len(labels), len(labels[0])
    labels = np.array(labels)
    total, right = 0., 0.
    for x_trues, batch_labels in data:
        y_preds = []
        for x_true in x_trues:
            y_pred = model.predict(x_true)
            y_preds.append(y_pred)
        for index, labels_id in enumerate(batch_labels):
            total += 1
            x_len = len(np.trim_zeros(x_trues[0][index]))
            ys, xs = [], []
            for y_pred, x in zip(y_preds, x_trues):
                y = y_pred[index]
                y = y[:x_len]
                x = x[index][:x_len]
                ys.append(y)
                xs.append(x)
            prob_ys = []
            for x, y in zip(xs, ys):
                prob_y = 1.0
                for word_index in range(label_word_len):
                    word_id = x[word_index - label_word_len]
                    prob_y *= y[word_index - label_word_len - 1][word_id]
                prob_ys.append(prob_y)
            y_index = np.array(prob_ys).argmax()
            num = 0  # 计算预测标签与真实标签相同字符的个数
            for xi, yi in zip(labels_id, labels[y_index]):
                if xi == yi:
                    num += 1
            if num == label_word_len:
                right += 1

    return right / total


def evaluate_gpt(data, model, labels):
    """
    评估准确率，每个label的字数要一致
    Args:
        data:
        model:
        labels:[[label_1_word1_id, label_1_word2_id,...],[label_2_word1_id, label_2_word2_id, ...],...]
    Returns:

    """
    # labels_num, label_word_len = len(labels), len(labels[0])
    # labels = np.array(labels)
    total, right = 0., 0.
    for x_trues, batch_labels in data:
        y_preds = []
        for x_true in x_trues:
            y_pred = model.predict(x_true)
            y_preds.append(y_pred)
        for index, (first_index, labels_id) in enumerate(batch_labels):
            total += 1
            x_len = len(np.trim_zeros(x_trues[0][index]))
            ys, xs = [], []
            for y_pred, x in zip(y_preds, x_trues):
                y = y_pred[index]
                y = y[:x_len]
                x = x[index][:x_len]
                ys.append(y)
                xs.append(x)
            prob_ys = []
            for x, y in zip(xs, ys):
                prob_y = 1.0
                # todo 直接从1开始计算整个句子的概率
                for word_index in range(first_index+1, x_len):
                    word_id = x[word_index]
                    prob_y *= y[word_index - 1][word_id]
                prob_ys.append(prob_y)
            y_index = np.array(prob_ys).argmax()
            if y_index == labels_id:
                right += 1

    return right / total


def get_evaluator(task_name, valid_generator, test_generator, labels, zero_shot):
    if task_name == TaskName.EPRSTMT or task_name == TaskName.CSLDCP:
        return Evaluator(evaluate_seq2seq, valid_generator, test_generator, task_name, labels, zero_shot)
    elif task_name == TaskName.CHID:
        return Evaluator(evaluate_gpt, valid_generator, test_generator, task_name, labels, zero_shot)
    else:
        return Evaluator(evaluate_seq2seq, valid_generator, test_generator, task_name, labels, zero_shot)