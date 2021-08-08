#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:gelin
# datetime:2021/7/30 11:50 下午
# software: PyCharm

"""
ptuning 模型，要设置环境变量TF_KERAS=1，某些操作tf.keras可用，keras不可用
"""

import numpy as np
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model, BERT
from bert4keras.optimizers import Adam
from bert4keras.tokenizers import Tokenizer


base_model_path = '../../../pretrained_models/chinese_roberta_wwm_ext_L-12_H-768_A-12/'
config_path = base_model_path+'bert_config.json'
checkpoint_path = base_model_path+'bert_model.ckpt'
dict_path = base_model_path+'vocab.txt'
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """
    def compute_loss(self, inputs, mask=None):
        y_true, y_pred = inputs
        y_mask = K.cast(K.not_equal(y_true, 0), K.floatx())
        accuracy = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        accuracy = K.sum(accuracy * y_mask) / K.sum(y_mask)
        #self.add_metric(accuracy, name='accuracy')   # tf.keras 不能用
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


class PtuningEmbedding(keras.layers.Layer):
    """新定义Embedding层，只优化部分Token
    """
    def __init__(self,
                    pattern_len,
                    emb_dim,
                    **kwargs # 其余参数
                 ):
        super(PtuningEmbedding, self).__init__(**kwargs)
        self.pattern_len = pattern_len
        self.emb_dim = emb_dim
        self.pattern_inputs = np.array([[i for i in range(pattern_len)]])
        self.pattern_embedding = keras.layers.Embedding(self.pattern_len, self.emb_dim, name="ptuning_emb")
        self.lstm1 = keras.layers.Bidirectional(
            keras.layers.LSTM(self.emb_dim // 2, dropout=0, return_sequences=True, name="ptuning_lstm1"))
        self.lstm2 = keras.layers.Bidirectional(
            keras.layers.LSTM(self.emb_dim // 2, dropout=0, return_sequences=True, name="ptuning_lstm2"))
        self.mlp1 = keras.layers.Dense(self.emb_dim, activation="relu", name="ptuning_mlp1")
        self.mlp2 = keras.layers.Dense(self.emb_dim, name="ptuning_mlp2")

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, inputs, **kwargs):
        # emb = self.pattern_embedding(inputs)  # [1, pattern_len, emd_dim]
        emb = self.pattern_embedding(self.pattern_inputs)  # [1, pattern_len, emd_dim]
        y = self.lstm1(emb)  # [1, pattern_len, emd_dim]
        y = self.lstm2(y)  # [1, pattern_len, emd_dim]
        y = self.mlp1(y)  # [1, pattern_len, emd_dim]
        result = self.mlp2(y)  # [1, pattern_len, emd_dim]
        result = keras.backend.squeeze(result, axis=0)  # [pattern_len, emd_dim]
        return result


class PtuningBERT(BERT):
    """
    利用lstm+mlp替换掉pattern中的embedding
    """
    def __init__(self,
                    pattern_len,
                    emb_dim,
                    **kwargs # 其余参数
                ):
        super(PtuningBERT, self).__init__(**kwargs)
        self.pattern_len = pattern_len
        self.emb_dim = emb_dim
        # # embedding
        # self.embedding = torch.nn.Embedding(len(self.cloze_mask[0]), self.hidden_size).to(self.device)
        # # LSTM
        # self.lstm_head = torch.nn.LSTM(input_size=self.hidden_size,
        #                                hidden_size=self.hidden_size // 2,
        #                                num_layers=2,
        #                                dropout=self.args.lstm_dropout,
        #                                bidirectional=True,
        #                                batch_first=True)
        # self.mlp_head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
        #                               nn.ReLU(),
        #                               nn.Linear(self.hidden_size, self.hidden_size))

        self.pattern_inputs = np.array([[i for i in range(pattern_len)]])

        # self.ptuning_emb_layer = PtuningEmbedding(self.pattern_len, self.emb_dim)

        # self.pattern_embedding = keras.layers.Embedding(self.pattern_len, self.emb_dim, name="ptuning_emb")
        # self.lstm1 = keras.layers.Bidirectional(keras.layers.LSTM(self.emb_dim // 2, dropout=0, return_sequences=True, name="ptuning_lstm1"))
        # self.lstm2 = keras.layers.Bidirectional(keras.layers.LSTM(self.emb_dim // 2, dropout=0, return_sequences=True, name="ptuning_lstm2"))
        # self.mlp1 = keras.layers.Dense(self.emb_dim, activation="relu", name="ptuning_mlp1")
        # self.mlp2 = keras.layers.Dense(self.emb_dim, name="ptuning_mlp2")

    def pattern_emb(self):
        # 废弃方法
        emb = self.pattern_embedding(self.pattern_inputs)    # [1, pattern_len, emd_dim]
        y = self.lstm1(emb)    # [1, pattern_len, emd_dim]
        y = self.lstm2(y)    # [1, pattern_len, emd_dim]
        y = self.mlp1(y)    # [1, pattern_len, emd_dim]
        result = self.mlp2(y)    # [1, pattern_len, emd_dim]
        result = keras.backend.squeeze(result, axis=0)    # [pattern_len, emd_dim]
        return result

    def call(self, inputs):
        """定义模型的执行流程"""
        # Embedding
        bert_emb = self.apply_embeddings(inputs)
        # pattern_emb = self.pattern_emb()
        # pattern_emb = self.ptuning_emb_layer(self.pattern_inputs)
        pattern_emb = self.apply(inputs=bert_emb, layer=PtuningEmbedding, pattern_len=self.pattern_len, emb_dim=self.emb_dim)

        inputs_ids, segment_ids = inputs    # [[0, 1, 2, 3, 4, 5, 0]], [[1, 1, 1, 1, 1, 1, 0]]

        pattern_inputs = inputs_ids <= self.pattern_len    # [[1, 1, 1, 1, 0, 0, 1]], pattern_len=3
        pattern_inputs = keras.backend.cast(pattern_inputs, "float32")
        pattern_inputs = pattern_inputs * inputs_ids    # [[0, 1, 2, 3, 0, 0,0]]
        pattern_inputs_bool = keras.backend.cast(pattern_inputs > 0, "float32")    # [[0, 1, 1, 1, 0, 0, 0]]
        pattern_inputs = pattern_inputs - pattern_inputs_bool    # [[0, 0, 1, 2, 0, 0,0]]
        pattern_inputs = keras.backend.cast(pattern_inputs, "int32")    # [[0, 0, 1, 2, 0, 0,0]]
        pattern_emb = keras.backend.gather(pattern_emb, pattern_inputs)    # [batch, seq, emb_dim]

        bert_inputs_bool = 1 - pattern_inputs_bool    # [[1, 0, 0, 0, 1, 1, 1]]
        ptuning_emd = bert_emb * keras.backend.expand_dims(bert_inputs_bool, axis=2) + pattern_emb * keras.backend.expand_dims(pattern_inputs_bool, axis=2)

        # return bert_emb, pattern_emb, ptuning_emd
        # Embedding
        # outputs = self.apply_embeddings(inputs)
        outputs = ptuning_emd
        # Main
        for i in range(self.num_hidden_layers):
            outputs = self.apply_main_layers(outputs, i)
        # Final
        outputs = self.apply_final_layers(outputs)
        return outputs


def get_model(pattern_len: int, emb_dim=768, trainable=False, lr=6e-4):
    # 加载预训练模型
    model = build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        model=PtuningBERT, #PtuningBERT, bert
        with_mlm=True,
        pattern_len=pattern_len,
        emb_dim=emb_dim
    )
    # 102,290,312
    # 110,557,832  ptuning_embedding 8,267,520
    # model.summary()

    for layer in model.layers:
        if layer.name != 'ptuning_embedding': # 如果不是ptuning_embedding层，那么不要训练
            layer.trainable = trainable

    # a = PtuningEmbedding(3, 768)
    # model = keras.models.Sequential([
    #             PtuningEmbedding(3, 768, input_shape=(3,))
    #         ])


    # 训练用模型
    y_in = keras.layers.Input(shape=(None,))
    # output = keras.layers.Lambda(lambda x: x[:, :unused_length+1])(model.output) # TODO TODO TODO
    outputs = CrossEntropy(1)([y_in, model.output])

    train_model = keras.models.Model(model.inputs + [y_in], outputs)
    #lr = 10e-4 # eprstmt
    #lr = 6e-4
    train_model.compile(optimizer=Adam(lr)) # 默认：6e-4. 3e-5学习率太小了
    train_model.summary()

    # 预测模型
    # model = keras.models.Model(model.inputs, output)
    return model, train_model


def test():
    inputs = [
            np.array([[0, 1, 2, 3, 4, 5, 0], [0, 1, 7, 3, 4, 5, 0]]),
            np.array([[1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 0]])
    ]

    bert_emb, pattern_emb, ptuning_emd = model.predict(inputs)

    print("bert_emb\n", bert_emb.shape)
    for e in bert_emb:
        print(e[:, 0])
    print("pattern_emb\n", pattern_emb.shape)
    for e in pattern_emb:
        print(e[:, 0])
    print("ptuning_emd\n", ptuning_emd.shape)
    for e in ptuning_emd:
        print(e[:, 0])
