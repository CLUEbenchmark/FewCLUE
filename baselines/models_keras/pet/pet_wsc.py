#! -*- coding:utf-8 -*-
# 情感分析例子，利用MLM做 Zero-Shot/Few-Shot/Semi-Supervised Learning

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
import re
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
parser = argparse.ArgumentParser(description="training set index")
parser.add_argument("--train_set_index", "-ti", help="training set index", type=str, default="0")
parser.add_argument("--training_type", "-tt", help="few-shot or zero-shot", type=str, default="few-shot")

args = parser.parse_args()
train_set_index = args.train_set_index
training_type = args.training_type

# num_classes = 2
maxlen = 256
batch_size = 8

base_model_path='../../pretrained_models/chinese_roberta_wwm_ext_L-12_H-768_A-12/'
config_path = base_model_path+'bert_config.json'
checkpoint_path =  base_model_path+'bert_model.ckpt'
dict_path = base_model_path+'vocab.txt'
# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)
def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        for jj,l in enumerate(f):
            # print("l:",l)
            json_string=json.loads(l.strip())
            print("json_string:",json_string)
            sentence1=json_string['text']
            span2=json_string["target"]['span2_text']
            span1=json_string["target"]['span1_text']
            label=json_string["label"]
            text=span2 + "锟" +span1 +"，" +sentence1 
            _mask = get_mask_idx(text, "锟")
            #text, label = l.strip().split('\t')
   
            if label=='true':
                label="即"
            elif label=='false':
                label="非"
            else:
                print(label)
            D.append((text, _mask, label))
    return D

def get_mask_idx(text, mask_words):
    """获取每个样本被mask掉的成语的位置"""
    span = re.search(mask_words, text)
    _mask = list(range(span.start()+1, span.end()+1))
    tokens = tokenizer.tokenize(text)
    tokens_rematch = tokenizer.rematch(text, tokens)[1:-1]
    offset = 0
    for t in tokens_rematch:
        if t[0] == _mask[0]:
            break
        else:
            offset += (len(t)-1)
    return [m-offset for m in _mask]

# 加载数据集
train_data = load_data('../../../datasets/cluewsc/train_0.json')
valid_data = load_data('../../../datasets/cluewsc/dev_few_all.json')
test_data = load_data('../../../datasets/cluewsc/test_public.json')

# 模拟标注和非标注数据
train_frac = 1 # TODO 0.01  # 标注数据的比例
num_labeled = int(len(train_data) * train_frac)
unlabeled_data = [(t, 2) for t, l in train_data[num_labeled:]]
print("length of unlabeled_data0:",len(unlabeled_data))
train_data = train_data[:num_labeled]
train_data = train_data + unlabeled_data
print("length of train_data1:",len(train_data))




# 对应的任务描述
# prefix =u'' # u'相近的两个句子的意思。' #  u'满意。'
# 0: neutral, 1: entailment, 2:contradiction
# neutral_id=tokenizer.token_to_id(u'并且')
# pos_id = tokenizer.token_to_id(u'所以')
# neg_id = tokenizer.token_to_id(u'但是')

# label_list=['neutral','entailment','contradiction']
# # 0: neutral, 1: entailment, 2:contradiction
# label2tokenid_dict={'neutral':neutral_id,'entailment':pos_id,'contradiction':neg_id}
label_list = ["即", "非"]
label_tokenid_list=[tuple(tokenizer.tokens_to_ids(x)) for x in label_list]

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
        for is_end, (text, mask_idxs, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            if random:
                source_ids, target_ids = random_masking(token_ids)
            else:
                source_ids, target_ids = token_ids[:], token_ids[:]

            # 0: neutral, 1: entailment, 2:contradiction
            if len(label) >= 1: # label是两个字的文本
                label_ids = tokenizer.encode(label)[0][1:-1] # label_ids: [1093, 689]。 e.g. [101, 1093, 689, 102] =[CLS,农,业,SEP]. tokenizer.encode(label): ([101, 1093, 689, 102], [0, 0, 0, 0])
                for i, label_id_ in zip(mask_idxs, label_ids):
                    source_ids[i] = tokenizer._token_mask_id # i: 7(mask1的index) ;j: 1093(农); i:8 (mask2的index) ;j: 689(业)
                    target_ids[i] = label_id_
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
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        model.save_weights('mlm_model_pet_sentencepair.weights')
        val_acc = evaluate(valid_generator)
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
        mask_idx = np.where(x_true[0]==tokenizer._token_mask_id)[1].reshape(x_true[0].shape[0],1)
        y_pred = [pred[mask, label_tokenid_list] for pred, mask in zip(y_pred, mask_idx)]
        y_pred = [(pred[:,0]*1).argmax() for pred in y_pred]
        y_true = [label_tokenid_list.index(tuple(y[mask])) for mask, y in zip(mask_idx, y_true)]
        total += len(y_true)
        right += np.where(np.array(y_pred)==np.array(y_true))[0].shape[0]  # (y_true == y_pred).sum()
    return right / total


if __name__ == '__main__':

    if training_type == "few-shot":
        evaluator = Evaluator()

        train_model.fit_generator(
            train_generator.forfit(),
            steps_per_epoch=len(train_generator),
            epochs=20,
            callbacks=[evaluator]
        )
    elif training_type == "zero-shot":
        test_acc = evaluate(test_generator)
        print("zero-shot结果: {}".format(test_acc))
    else:
        print("未知的训练类型")

else:
    model.load_weights('best_model_pet_sentencepair.weights')
