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
import argparse
import random

parser = argparse.ArgumentParser(description="training set index")
parser.add_argument("--train_set_index", "-t", help="training set index", type=str, default="0")
args = parser.parse_args()
train_set_index = args.train_set_index
assert train_set_index in {"0", "1", "2", "3", "4", "all"}, 'train_set_index must in {"0", "1", "2", "3", "4", "all"}'
from tqdm import tqdm

config_path = '/path/language_model/chinese_roberta_wwm_ext_L-12_H-768_A-12/config.json'
checkpoint_path = '/path/language_model/nezha-gpt/cn_gpt'
dict_path = '/path/language_model/nezha-gpt/vocab.txt'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)
label_dict = {"1": "一", "2": "二", "3": "三", "4": "四", "5": "五", "6": "六", "7": "七"}
labels = [v for k, v in label_dict.items()]
labels_ids = [tokenizer.token_to_id(v) for v in labels]
maxlen = 256
batch_size = 16
num_per_val_file = 42
acc_list  = []
# 加载数据的方法
def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        for idx,l in enumerate(f):
            #print("l:",l)
            sample=json.loads(l.strip())
            # print("json_string:",json_string)
            answer = sample["answer"]
            sentence = sample["content"]
            candidates = sample["candidates"]
            candidates_str = [label_dict[str(i+1)]+"：" + can +"，" for i, can in enumerate(candidates)]
            sentence = sentence.replace("#idiom#", "".join(candidates_str))
            D.append((sentence, label_dict[str(int(answer)+1)]))
            # 如果使用全量数据训练
            # for can_idx,can in enumerate(sample["candidates"]):
            #     sentence1=can
            #     sentence2 = sample["content"].replace("#idiom#", sentence1)
            #     label=int(can_idx == sample["answer"])
            #     D.append((sentence2, int(label)))
            if idx < 5:
                print(D[idx])
    random.shuffle(D)
    return D


# 加载数据集
train_data = load_data('ready_data/chid/train_{}.json'.format(train_set_index))
valid_data = []
for i in range(5):
    valid_data += load_data('ready_data/chid/dev_{}.json'.format(i))
test_data = load_data('ready_data/chid/test_public.json')
val_label = [l[1] for l in valid_data]
test_label = [l[1] for l in test_data]

# 模拟标注和非标注数据
train_frac = 1  # 标注数据的比例
num_labeled = int(len(train_data) * train_frac)
unlabeled_data = [(t, 2) for t, l in train_data[num_labeled:]]
train_data = train_data[:num_labeled]
# train_data = train_data + unlabeled_data



# 对应的任务描述
desc = ['[unused%s]' % i for i in range(1, 9)]
desc_ids = [tokenizer.token_to_id(t) for t in desc]


def random_masking(token_ids):
    """对输入进行mask
    在BERT中，mask比例为15%，相比auto-encoder，BERT只预测mask的token，而不是重构整个输入token。
    mask过程如下：80%机会使用[MASK]，10%机会使用原词,10%机会使用随机词。
    """
    rands = np.random.random(len(token_ids)) # rands: array([-0.34792592,  0.13826393,  0.8567176 ,  0.32175848, -1.29532141, -0.98499201, -1.11829718,  1.18344819,  1.53478554,  0.24134646])
    source, target = [], []
    for r, t in zip(rands, token_ids):
        if r < 0.15 * 0.8:   # 80%机会使用[MASK]
            source.append(tokenizer._token_mask_id)
            target.append(t)
        elif r < 0.15 * 0.9: # 10%机会使用原词
            source.append(t)
            target.append(t)
        elif r < 0.15:       # 10%机会使用随机词
            source.append(np.random.choice(tokenizer._vocab_size - 1) + 1)
            target.append(t)
        else: # 不进行mask
            source.append(t)
            target.append(0)
    return source, target

class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids = []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            token_ids = token_ids[:1] + desc_ids[:4] + token_ids[:-1]
            token_ids = token_ids + desc_ids[4:]
            token_ids = token_ids + [tokenizer.token_to_id(label)]
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
        model.save_weights('pet_tnews_model.weights')
        val_pred_result = evaluate(valid_generator)
        val_pred_result = np.array(val_pred_result, dtype="int32")
        total_acc = val_pred_result.sum()/val_pred_result.shape[0]
        val_pred_result = val_pred_result.reshape(5, num_per_val_file).sum(1)/num_per_val_file
        # val_acc_mean = val_pred_result.mean() 准确率均值和total准确率相等
        if total_acc > self.best_val_acc:
            self.best_val_acc = total_acc
            model.save_weights('pet_tnews_best_model.weights')
        test_pred_result = np.array(evaluate(test_generator))
        test_acc = test_pred_result.sum()/test_pred_result.shape[0]
        acc_tuple = tuple(val_pred_result.tolist()+[total_acc, self.best_val_acc, test_acc])
        acc_list.append(list(acc_tuple))
        draw_acc(acc_list) # 如果需要对照每个验证集准确率
        print(
            u'val_acc_0: %.5f, val_acc_1: %.5f, val_acc_2: %.5f, val_acc_3: %.5f, val_acc_4: %.5f, val_acc_total: %.5f, best_val_acc: %.5f, test_acc: %.5f\n' %
            acc_tuple
        )

def evaluate(data):
    total, right = 0., 0.
    pred_list = []
    for x_true, _ in data:
        y_pred = model.predict(x_true)
        for x, y in zip(x_true, y_pred):
            x = np.trim_zeros(x)
            y = y[:len(x)][-2, labels_ids].argmax()
            # y = labels_ids[y]
            pred_list.append(y == labels.index(tokenizer.id_to_token(x[-1])))
    return pred_list

def draw_acc(acc_list):
    import matplotlib.pyplot as plt
    epoch = len(acc_list)
    x = np.linspace(0, epoch, epoch)

    fig, ax = plt.subplots()
    label_list = ["val_0", "val_1", "val_2", "val_3", "val_4", "val_total", "val_best", "test"]
    acc_arr = np.array(acc_list).T
    # Using set_dashes() to modify dashing of an existing line
    for idx, y in enumerate(acc_arr):
        ax.plot(x, y, label=label_list[idx])
    ax.legend()
    plt.savefig("ptuning_csl_gpt.svg") # 保存为svg格式图片，如果预览不了svg图片可以把文件后缀修改为'.png'

if __name__ == '__main__':

    evaluator = Evaluator()

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator) * 50,
        epochs=1000,
        callbacks=[evaluator]
    )

else:

    model.load_weights('best_model_tnews_gpt.weights')