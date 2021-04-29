#! -*- coding:utf-8 -*-
# 情感分析例子，利用MLM做 Zero-Shot/Few-Shot/Semi-Supervised Learning
import os

import numpy as np
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Lambda, Dense
import json
import random

# num_classes = 2
maxlen = 128
batch_size = 8

base_model_path='../pretrained_models/chinese_roberta_wwm_ext_L-12_H-768_A-12/'
config_path = base_model_path+'bert_config.json'
checkpoint_path =  base_model_path+'bert_model.ckpt'
dict_path = base_model_path+'vocab.txt'

def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        for jj,l in enumerate(f):
            #print("l:",l)
            json_string=json.loads(l.strip())
            # print("json_string:",json_string)
            sentence1=json_string['sentence1']
            sentence2=json_string['sentence2']
            label=json_string['label']
            text="第一个句子是："+sentence1+"；第二个句子是："+sentence2
            #text, label = l.strip().split('\t')
            if label=='neutral':
                label=0
            elif label=='entailment':
                label=1
            elif label=='contradiction':
                label=2
            else:
                continue # 如果是其他标签的话，跳过这行数据
            D.append((text, int(label)))
    return D

path = '../data/FewCLUEDatasets-master/ready_data/ocnli/'
save_path = '../output/ocnli/'
if not os.path.exists(save_path):
    os.mkdir(save_path)

data_num = '4'
# 加载数据集
train_data = load_data('{}/train_{}.json'.format(path,data_num))
valid_datas = []
for i in range(5):
    valid_data = load_data('{}/dev_{}.json'.format(path,str(i)))
    valid_datas.append(valid_data)
test_data = load_data('{}/test_public.json'.format(path))

# 模拟标注和非标注数据
train_frac = 1 # TODO 0.01  # 标注数据的比例
num_labeled = int(len(train_data) * train_frac)
unlabeled_data = [(t, 2) for t, l in train_data[num_labeled:]]
print("length of unlabeled_data0:",len(unlabeled_data))
train_data = train_data[:num_labeled]
train_data = train_data + unlabeled_data
print("length of train_data1:",len(train_data))


# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 对应的任务描述
prefix =u'相近的两个句子。' # u'相近的两个句子的意思。' #  u'满意。'
mask_idx = 1
# 0: neutral, 1: entailment, 2:contradiction
neutral_id=tokenizer.token_to_id(u'无')
pos_id = tokenizer.token_to_id(u'很')
neg_id = tokenizer.token_to_id(u'不')

label_list=['neutral','entailment','contradiction']
# 0: neutral, 1: entailment, 2:contradiction
label2tokenid_dict={'neutral':neutral_id,'entailment':pos_id,'contradiction':neg_id}
label_tokenid_list=[label2tokenid_dict[x] for x in label_list]

def random_masking(token_ids):
    """对输入进行随机mask
    在BERT中，mask比例为15%，相比auto-encoder，BERT只预测mask的token，而不是重构整个输入token。
    mask过程如下：80%机会使用[MASK]，10%机会使用原词,10%机会使用随机词。
    """
    rands = np.random.random(len(token_ids))
    source, target = [], []
    for r, t in zip(rands, token_ids):
        if r < 0.15 * 0.8:
            source.append(tokenizer._token_mask_id)
            target.append(t)
        elif r < 0.15 * 0.9:
            source.append(t)
            target.append(t)
        elif r < 0.15:
            source.append(np.random.choice(tokenizer._vocab_size - 1) + 1)
            target.append(t)
        else:
            source.append(t)
            target.append(0)
    return source, target


class data_generator(DataGenerator):
    """数据生成器
    目前看只是将原始文本转换为token id
    负向样本（输入是一个[MASK]字符，输出是特定的字符。对于负样本，采用"不"，正样本，采用“很”）
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []
        for is_end, (text, label) in self.sample(random):
            if label != 3:
                text = prefix + text
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            if random:
                source_ids, target_ids = random_masking(token_ids)
            else:
                source_ids, target_ids = token_ids[:], token_ids[:]

            # 0: neutral, 1: entailment, 2:contradiction
            if label == 0:
                source_ids[mask_idx] = tokenizer._token_mask_id
                target_ids[mask_idx] = neutral_id
            elif label == 1:
                source_ids[mask_idx] = tokenizer._token_mask_id
                target_ids[mask_idx] = pos_id
            elif label == 2:
                source_ids[mask_idx] = tokenizer._token_mask_id
                target_ids[mask_idx] = neg_id
            batch_token_ids.append(source_ids)
            batch_segment_ids.append(segment_ids)
            batch_output_ids.append(target_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_output_ids = sequence_padding(batch_output_ids)
                yield [
                    batch_token_ids, batch_segment_ids, batch_output_ids
                ], None
                batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """
    def compute_loss(self, inputs, mask=None):
        y_true, y_pred = inputs
        y_mask = K.cast(K.not_equal(y_true, 0), K.floatx())
        accuracy = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        accuracy = K.sum(accuracy * y_mask) / K.sum(y_mask)
        self.add_metric(accuracy, name='accuracy')
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


# 加载预训练模型
model = build_transformer_model(
    config_path=config_path, checkpoint_path=checkpoint_path, with_mlm=True
)

# 训练用模型
y_in = keras.layers.Input(shape=(None,))
outputs = CrossEntropy(1)([y_in, model.output])

train_model = keras.models.Model(model.inputs + [y_in], outputs)
train_model.compile(optimizer=Adam(3e-5)) # 默认：1e-5；其他候选学习率：6e-4
train_model.summary()

# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generators = []
for valid_data in valid_datas:
    valid_generator = data_generator(valid_data, batch_size)
    valid_generators.append(valid_generator)
test_generator = data_generator(test_data, batch_size)


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        model.save_weights('mlm_model_pet_sentencepair.weights')
        val_accs = 0
        for valid_generator in valid_generators:
            val_accs += evaluate(valid_generator)
        val_acc = val_accs / 5
        if val_acc > self.best_val_acc: # #  保存最好的模型，并记录最好的准确率
            self.best_val_acc = val_acc
            model.save_weights('best_model_pet_sentencepair.weights')
        test_acc = evaluate(test_generator)
        print(
            u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n' %
            (val_acc, self.best_val_acc, test_acc)
        )


def check_two_list(list_true, list_predict):
    """
    计算这两个列表，在相应位置上有多少个相同的值。
    :param list_true:
    :param list_predict:
    :return:
    """
    num_right_=0
    for index, v in enumerate(list_true):
        if v==list_predict[index]:
            num_right_+=1
    return num_right_

# 对验证集进行验证
def evaluate(data):
    total, right = 0., 0.
    for x_true, _ in data:
        x_true, y_true = x_true[:2], x_true[2]
        y_pred = model.predict(x_true)
        y_pred = y_pred[:, mask_idx, label_tokenid_list].argmax(axis=1) # TODO [neg_id, pos_id]
        y_pred_right=[label2tokenid_dict[label_list[index]] for index in y_pred] # e.g. label_list[index]='like'. label2tokenid_dict[...]= token of label.
        y_true_ = y_true[:, mask_idx]
        total += len(y_true)
        # if random.randint(0, 100) == 1:
        #     print("0:y_pred_original:", y_pred.shape, ";y_pred:",y_pred)  # 0:y_pred : (16,) ;y_pred: [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
        #     print("1:y_true_:", y_true_.shape, ";y_true_:",y_true_)  # 1:y_true_: (16,) ;y_true_: [ 839  839  839 2626 2458 2458 2458  839 2458 2626 2458 2458 2584 2626 2458 2584]
        #     print("2:y_pred_right:",y_pred_right)  # 1:y_true_: (16,) ;y_true_: [ 839  839  839 2626 2458 2458 2458  839 2458 2626 2458 2458 2584 2626 2458 2584]
        num_right = check_two_list(y_true_, y_pred_right)
        right += num_right  # (y_true == y_pred).sum()
    return right / total
        # y_true = (y_true[:, mask_idx] == pos_id).astype(int)
        # total += len(y_true)
        # right += (y_true == y_pred).sum()
    # return right / total


if __name__ == '__main__':

    evaluator = Evaluator()

    train_model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=20, # TODO 1000,
        callbacks=[evaluator]
    )

else:
    model.load_weights('best_model_pet_sentencepair.weights')