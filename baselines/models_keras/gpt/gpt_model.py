#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:gelin
# datetime:2021/4/24 11:56 上午
# software: PyCharm

"""
加载gpt ptuning模型
"""

import numpy as np
from bert4keras.backend import keras, K
from bert4keras.layers import Loss, Embedding
from bert4keras.models import build_transformer_model, BERT
from bert4keras.optimizers import Adam
from config import *


class PtuningEmbedding(Embedding):
    """新定义Embedding层，只优化部分Token
    """
    def call(self, inputs, mode='embedding'):
        embeddings = self.embeddings
        embeddings_sg = K.stop_gradient(embeddings)
        mask = np.zeros((K.int_shape(embeddings)[0], 1))
        mask[1:9] += 1  # 只优化id为1～8的token
        self.embeddings = embeddings * mask + embeddings_sg * (1 - mask)
        return super(PtuningEmbedding, self).call(inputs, mode)


class PtuningBERT(BERT):
    """替换原来的Embedding
    """
    def apply(self, inputs=None, layer=None, arguments=None, **kwargs):
        if layer is Embedding:
            layer = PtuningEmbedding
        return super(PtuningBERT,
                     self).apply(inputs, layer, arguments, **kwargs)


# 加载预训练模型
model = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model=PtuningBERT,
    segment_vocab_size=0,  # 去掉segmeng_ids输入
    application='lm',
)  # 建立模型，加载权重

for layer in model.layers:
    if layer.name != 'Embedding-Token':
        layer.trainable = False


def get_sentiment_model():
    global model

    class CrossEntropy(Loss):
        """交叉熵作为loss，并mask掉padding部分
        """
        def compute_loss(self, inputs, mask=None):
            y_true, y_pred = inputs
            if mask[1] is None:
                y_mask = 1.0
            else:
                y_mask = K.cast(mask[1], K.floatx())[:, 1:]
            y_true = y_true[:, 1:]  # 目标token_ids
            y_pred = y_pred[:, :-1]  # 预测序列，错开一位
            accuracy = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
            accuracy = K.sum(accuracy * y_mask) / K.sum(y_mask)
            self.add_metric(accuracy, name='accuracy')
            loss = K.sparse_categorical_crossentropy(y_true, y_pred)
            loss = K.sum(loss * y_mask) / K.sum(y_mask)
            return loss

    output = CrossEntropy(1)([model.input, model.output])

    model = keras.models.Model(model.input, output)
    model.compile(optimizer=Adam(6e-4))
    model.summary()

    return model


def get_model():
    global model
    return get_sentiment_model()

