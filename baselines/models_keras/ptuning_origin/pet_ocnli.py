#! -*- coding:utf-8 -*-

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
import json
import re
import sys
from modeling import tokenizer


maxlen = 256
batch_size = 16
unused_length=2


def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        for jj,l in enumerate(f):
            #print("l:",l)
            json_string=json.loads(l.strip())
            # print("json_string:",json_string)
            sentence1=json_string['sentence1']
            sentence2=json_string['sentence2']
            if 'label' in json_string:
                label=json_string['label']
            else:
                label='neutral'
            text=sentence1+"？"+"锟斤"+"#"*unused_length+","+sentence2
            _mask = get_mask_idx(text, "锟斤"+"#"*unused_length)
            #text, label = l.strip().split('\t')
            if label=='neutral':
                label="并且"+"#"*unused_length
            elif label=='entailment':
                label="是的"+"#"*unused_length
            elif label=='contradiction':
                label="不是"+"#"*unused_length
            else:
                continue # 如果是其他标签的话，跳过这行数据
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


path = '../../../datasets/ocnli'
data_num = sys.argv[1]

# 加载数据集
train_data = load_data('{}/train_{}.json'.format(path,data_num))
valid_data = load_data('{}/dev_{}.json'.format(path,data_num))
test_data = load_data('{}/test_public.json'.format(path))





# 对应的任务描述
# prefix =u'' # u'相近的两个句子的意思。' #  u'满意。'
# 0: neutral, 1: entailment, 2:contradiction
# neutral_id=tokenizer.token_to_id(u'并且')
# pos_id = tokenizer.token_to_id(u'所以')
# neg_id = tokenizer.token_to_id(u'但是')

# label_list=['neutral','entailment','contradiction']
# # 0: neutral, 1: entailment, 2:contradiction
# label2tokenid_dict={'neutral':neutral_id,'entailment':pos_id,'contradiction':neg_id}
label_list = ["并且", "是的", "不是"]
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
            if len(label) > 1: # label是两个字的文本
                label_ids = tokenizer.encode(label)[0][1:-1] # label_ids: [1093, 689]。 e.g. [101, 1093, 689, 102] =[CLS,农,业,SEP]. tokenizer.encode(label): ([101, 1093, 689, 102], [0, 0, 0, 0])
                for i, ind in enumerate(mask_idxs):
                    if i < 2:
                        source_ids[ind] = tokenizer._token_mask_id
                        target_ids[ind] = label_ids[i]
                    else:
                        source_ids[ind] = i
                        #target_ids[ind] = i
                # for i, label_id_ in zip(mask_idxs, label_ids):
                #     source_ids[i] = tokenizer._token_mask_id # i: 7(mask1的index) ;j: 1093(农); i:8 (mask2的index) ;j: 689(业)
                #     target_ids[i] = label_id_
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


from modeling import get_model
model, train_model = get_model(pattern_len=unused_length, trainable=True, lr=3e-5)


# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size * 8)
test_generator = data_generator(test_data, batch_size * 8)


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        # model.save_weights('mlm_model_pet_sentencepair.weights')
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc: # #  保存最好的模型，并记录最好的准确率
            self.best_val_acc = val_acc
            # model.save_weights('best_model_pet_sentencepair.weights')
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
        mask_idx = np.where(x_true[0]==tokenizer._token_mask_id)[1].reshape(x_true[0].shape[0],2)
        y_pred = [pred[mask, label_tokenid_list] for pred, mask in zip(y_pred, mask_idx)]
        y_pred = [(pred[:,0]*pred[:,1]).argmax() for pred in y_pred]
        y_true = [label_tokenid_list.index(tuple(y[mask[:2]])) for mask, y in zip(mask_idx, y_true)]
        total += len(y_true)
        right += np.where(np.array(y_pred)==np.array(y_true))[0].shape[0]  # (y_true == y_pred).sum()
    return right / total


if __name__ == '__main__':
    evaluator = Evaluator()

    train_model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator) * 2,
        epochs=20,
        callbacks=[evaluator]
    )
