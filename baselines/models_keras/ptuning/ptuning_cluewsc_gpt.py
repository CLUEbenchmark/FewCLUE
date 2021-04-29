#! -*- coding:utf-8 -*-
# 情感分析例子，利用LM+P-tuning

import numpy as np
from bert4keras.backend import keras, K
from bert4keras.layers import Loss, Embedding
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model, BERT
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Lambda, Dense
import json

maxlen = 128
batch_size = 32
config_path = '/path/language_model/chinese_roberta_wwm_ext_L-12_H-768_A-12/config.json'
checkpoint_path = '/path/language_model/nezha-gpt/cn_gpt'
dict_path = '/path/language_model/nezha-gpt/vocab.txt'


def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            sample = json.loads(l.strip())
            text = sample["text"]
            arg1 = sample["target"]["span1_text"]
            arg2 = sample["target"]["span2_text"]
            label = (sample["label"] == "true")
            # text_format = "{}中，{}和{}"
            # D.append((text_format.format(text, arg1, arg2), int(label)))
            D.append(((text, arg1, arg2), int(label)))
    return D


# 加载数据集
train_data = load_data('./ready_data/cluewsc/train_0.json')
valid_data = load_data('./ready_data/cluewsc/dev_0.json')
test_data = load_data('./ready_data/cluewsc/test_public.json')

# 模拟标注和非标注数据
train_frac = 1  # 标注数据的比例
num_labeled = int(len(train_data) * train_frac)
unlabeled_data = [(t, 2) for t, l in train_data[num_labeled:]]
train_data = train_data[:num_labeled]
# train_data = train_data + unlabeled_data

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 对应的任务描述
desc = ['[unused%s]' % i for i in range(1, 9)]
desc_ids = [tokenizer.token_to_id(t) for t in desc]
pos_id = tokenizer.token_to_id(u'是')
neg_id = tokenizer.token_to_id(u'不')


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids = []
        for is_end, (text, label) in self.sample(random):
            text_token_ids, segment_ids = tokenizer.encode(text[0], maxlen=maxlen)
            arg1_token_ids, segment_ids = tokenizer.encode(text[1], maxlen=maxlen)
            arg2_token_ids, segment_ids = tokenizer.encode(text[2], maxlen=maxlen)
            token_ids = text_token_ids[:1] + desc_ids[:3] + text_token_ids[:-1]
            token_ids = token_ids + arg1_token_ids[1:-1] + desc_ids[3:5]+ arg2_token_ids[1:-1]+desc_ids[5:]
            if label == 0:
                token_ids = token_ids + [neg_id]
            elif label == 1:
                token_ids = token_ids + [pos_id]
            batch_token_ids.append(token_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                yield batch_token_ids, None
                batch_token_ids = []


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

output = CrossEntropy(1)([model.input, model.output])

model = keras.models.Model(model.input, output)
model.compile(optimizer=Adam(6e-4))
model.summary()

# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('best_model_gpt.weights')
        test_acc = evaluate(test_generator)
        print(
            u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n' %
            (val_acc, self.best_val_acc, test_acc)
        )


def evaluate(data):
    total, right = 0., 0.
    for x_true, _ in data:
        y_pred = model.predict(x_true)
        for x, y in zip(x_true, y_pred):
            x = np.trim_zeros(x)
            y = y[:len(x)][-2, [neg_id, pos_id]].argmax()
            y = [neg_id, pos_id][y]
            if y == x[-1]:
                right += 1
            total += 1
    return right / total


if __name__ == '__main__':

    evaluator = Evaluator()

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator) * 50,
        epochs=1000,
        callbacks=[evaluator]
    )

else:

    model.load_weights('best_model_cluewsc_gpt.weights')