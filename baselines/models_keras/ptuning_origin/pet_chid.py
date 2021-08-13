#! -*- coding:utf-8 -*-

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
import re
import sys
from modeling import tokenizer

maxlen = 256
batch_size = 8
unused_length=2


def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        for idx,l in enumerate(f):
            sample=json.loads(l.strip().replace(" ", "").replace("\t", ""))
            if "answer" in sample:
                answer = int(sample["answer"])
            else:
                answer = 0
            sentence1 = sample["content"].replace("#idiom#", "锟斤烤烫"+ "#" * unused_length) # 替换为生僻词确保输入中没有答案
            _mask = get_mask_idx(sentence1, "锟斤烤烫" + "#" * unused_length)
            D.append((sentence1, sample["candidates"][answer] + "#" * unused_length, _mask,  sample["candidates"]))
            if idx < 5:
                print("************************数据集******************************")
                print(D[idx])
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




path = '../../../datasets/chid'
data_num = sys.argv[1]

# 加载数据集
train_data = load_data('{}/train_{}.json'.format(path,data_num))
valid_data = load_data('{}/dev_{}.json'.format(path,data_num))
test_data = load_data('{}/test_public.json'.format(path))


def encode_candidates(candidates):
    """把候选成语转化为id，防止UNK词出现导致在候选成语中找不到预测成语"""
    ids_list = []
    for can in candidates:
        ids_list.append(tuple(tokenizer.tokens_to_ids(can)))
    return ids_list

# 对应的任务描述和每个样本的mask的位置以及答案
prefix = u'' # chid数据集不需要额外描述
val_mask_idxs = [d[2] for d in valid_data]
test_mask_idxs = [d[2] for d in test_data]
val_labels_list = [encode_candidates(d[3]) for d in valid_data]
test_labels_list = [encode_candidates(d[3]) for d in test_data]

def random_masking(token_ids):
    """对输入进行随机mask
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
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []
        for is_end, (text, label, mask_idxs, _) in self.sample(random):
            if len(label) > 1: # label是两个字的文本
                text = prefix + text # 拼接文本
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            if random:
                source_ids, target_ids = random_masking(token_ids)
            else:
                source_ids, target_ids = token_ids[:], token_ids[:]
            if len(label) > 1: # label是两个字的文本
                label_ids = tokenizer.encode(label)[0][1:-1] # label_ids: [1093, 689]。 e.g. [101, 1093, 689, 102] =[CLS,农,业,SEP]. tokenizer.encode(label): ([101, 1093, 689, 102], [0, 0, 0, 0])
                for i, ind in enumerate(mask_idxs):
                    if i < 4:
                        source_ids[ind] = tokenizer._token_mask_id
                        target_ids[ind] = label_ids[i]
                    else:
                        source_ids[ind] = i - 3
                        target_ids[ind] = i - 3
                # for i, label_id_ in zip(mask_idxs, label_ids):
                #     source_ids[i] = tokenizer._token_mask_id # i: 7(mask1的index) ;j: 1093(农); i:8 (mask2的index) ;j: 689(业)
                #     target_ids[i] = label_id_

            batch_token_ids.append(source_ids)
            batch_segment_ids.append(segment_ids)
            batch_output_ids.append(target_ids)

            if len(batch_token_ids) == self.batch_size or is_end: # 分批padding和生成
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
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)

class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        # model.save_weights('pet_tnews_model.weights')
        val_acc = evaluate(valid_generator, val_type="val")
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            # model.save_weights('pet_tnews_best_model.weights')
        test_acc = evaluate(test_generator, val_type="test")
        print(
            u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n' %
            (val_acc, self.best_val_acc, test_acc)
        )

def evaluate(data, val_type="val"):
    """
    计算候选成语列表中每一个成语（如'狐假虎威'）的联合概率，并与正确的标签做对比。每一个样本存在不同的候选成语的列表。
    :param data:
    val_type: 数据集是开发集的数据集还是测试集的数据
    :return:
    """
    total, right = 0., 0.
    pred_result_list = []
    if val_type == "val":
        labels_list = val_labels_list
        mask_idxs = val_mask_idxs
    elif val_type == "test":
        labels_list = test_labels_list
        mask_idxs = test_mask_idxs
    else:
        raise ValueError('选择正确的数据集类型')
    for idx, X in enumerate(data):
        label_ids = np.array([[np.array([l for l in label])] for label in labels_list[batch_size*idx: batch_size*(idx+1)]])
        tmp_size = label_ids.shape[0]
        label_ids = label_ids.reshape(tmp_size, 7, 4)
        x_true = X[0]
        x_true, y_true = x_true[:2], x_true[2]
        y_pred = model.predict(x_true)
        y_pred = np.array([y[mask_idxs[idx*batch_size+i]].tolist() for i, y in enumerate(y_pred)])
        # 计算候选集中各成语的概率：p(idiom) = p(idiom_1)*p(idiom_2)*p(idiom_3)*p(idiom_4)
        y_pred = [y_pred[i, 0, label_ids[i, :, 0]] * y_pred[i, 1, label_ids[i, :, 1]]* y_pred[i, 2, label_ids[i, :, 2]]* y_pred[i, 3, label_ids[i, :, 3]] for i in range(tmp_size)]
        y_pred = np.array(y_pred)
        y_pred = y_pred.argmax(axis=1)
        true_list = [labels_list[idx*batch_size+i].index(tuple(y[mask_idxs[idx*batch_size+i][:4]])) for i, y in enumerate(y_true)]
        y_true = np.array(true_list)
        total += len(y_true)
        right += np.where(np.array(y_pred) == np.array(y_true))[0].shape[0]  # (y_true == y_pred).sum()
    return right / total
        # pred_result_list += (y_true == y_pred).tolist()
    # return pred_result_list


if __name__ == '__main__':
    evaluator = Evaluator()

    train_model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator)*5,
        epochs=5,
        callbacks=[evaluator]
    )
