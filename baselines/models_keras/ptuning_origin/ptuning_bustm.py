#! -*- coding:utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
from bert4keras.backend import keras, K
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
import json
import sys
from modeling import tokenizer

maxlen = 80
batch_size = 32


# 加载数据的方法
def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        for jj,l in enumerate(f):
            #print("l:",l)
            json_string=json.loads(l.strip())
            #print("json_string:",json_string)
            sentence1=json_string['sentence1']
            sentence2=json_string['sentence2']
            label=json_string['label']
            text="第一个句子是："+sentence1+"；另外一个句子是："+sentence2

            #text, label = l.strip().split('\t')
            D.append((text, int(label)))
    return D


path = '../../../datasets/bustm'
data_num = sys.argv[1]
# 加载数据集
train_data = load_data('{}/train_{}.json'.format(path, data_num))
valid_data = load_data('{}/dev_{}.json'.format(path, data_num))
test_data = load_data('{}/test_public.json'.format(path))


# 对应的任务描述
mask_idx = 1 #5
unused_length=2 # 9
desc = ['[unused%s]' % i for i in range(1, unused_length+1)] # desc: ['[unused1]', '[unused2]', '[unused3]', '[unused4]', '[unused5]', '[unused6]', '[unused7]', '[unused8]', '[unused9]', '[unused10]']
desc.insert(mask_idx - 1, '[MASK]')            # desc: ['[MASK]', '[unused1]', '[unused2]', '[unused3]', '[unused4]', '[unused5]', '[unused6]', '[unused7]', '[unused8]', '[unused9]', '[unused10]
desc.insert(mask_idx, '同')            # desc: ['[MASK]', '[unused1]', '[unused2]', '[unused3]', '[unused4]', '[unused5]', '[unused6]', '[unused7]', '[unused8]', '[unused9]', '[unused10]
desc_ids = [tokenizer.token_to_id(t) for t in desc] # 将token转化为id
print(desc)
print(desc_ids)

pos_id = tokenizer.token_to_id(u'很') # e.g. '[unused9]'. 将正向的token转化为id. 默认值：u'很'
neg_id = tokenizer.token_to_id(u'不') # e.g. '[unused10]. 将负向的token转化为id. 默认值：u'不'


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
        #elif r < 0.15:       # 10%机会使用随机词
        #    source.append(np.random.choice(tokenizer._vocab_size - 1) + 1)
        #    target.append(t)
        else: # 不进行mask
            source.append(t)
            target.append(0)
    return source, target


class data_generator(DataGenerator):
    """数据生成器
    # TODO TODO TODO 这里面的每一行代码，，，
    目前看只是将原始文本转换为token id
    负向样本（输入是一个[MASK]字符，输出是特定的字符。对于负样本，采用"不"，正样本，采用“很”）
    """
    def __iter__(self, random=False): # TODO 这里的random是指否需要对原始文本进行mask
        batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            if label != 2:
                token_ids = token_ids[:1] + desc_ids + token_ids[1:]
                segment_ids = [0] * len(desc_ids) + segment_ids
            if random: # 暂时没有用呢
                source_ids, target_ids = random_masking(token_ids)
            else:
                source_ids, target_ids = token_ids[:], token_ids[:]
            if label == 0: # 负样本
                source_ids[mask_idx] = tokenizer._token_mask_id
                target_ids[mask_idx] = neg_id
            elif label == 1: # 正向样本
                source_ids[mask_idx] = tokenizer._token_mask_id
                target_ids[mask_idx] = pos_id
            batch_token_ids.append(source_ids)
            batch_segment_ids.append(segment_ids)
            batch_output_ids.append(target_ids)
            if len(batch_token_ids) == self.batch_size or is_end: # padding操作
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
valid_generator = data_generator(valid_data, batch_size*10)
test_generator = data_generator(test_data, batch_size*10)


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc, val_auc = evaluate(valid_generator)
        if val_acc > self.best_val_acc: #  保存最好的模型，并记录最好的准确率
            self.best_val_acc = val_acc
            #model.save_weights('best_model_bert_ptuning.weights')
        test_acc, test_auc = evaluate(test_generator)
        print( # 打印准确率
            u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f, test_auc:%.5f\n' %
            (val_acc, self.best_val_acc, test_acc, test_auc)
        )



def softmax(x):
    y = np.exp(x)
    s = np.sum(y, axis=1)
    s = np.expand_dims(s, axis=1)
    y = y/s
    return y


# 对验证集进行验证
def evaluate(data):
    trues, pred = [], []
    result = []
    total, right = 0., 0.
    for x_true, _ in data:
        x_true, y_true = x_true[:2], x_true[2]
        y_pred = model.predict(x_true)
        y_pred = y_pred[:, mask_idx, [neg_id, pos_id]]
        #y_pred = keras.backend.squeeze(y_pred, axis=1)
        #print(y_pred)
        #print("type:", type(y_pred))
        y_pred = softmax(y_pred)
        pred.extend([p for p in y_pred[:, 1]])
        #y_pred = y_pred.argmax(axis=1)
        y_pred = (y_pred[:,1] > 0.5).astype(int)
        y_true = (y_true[:, mask_idx] == pos_id).astype(int)
        trues.extend(y_true)
        result.extend(y_pred)
        total += len(y_true)
        right += (y_true == y_pred).sum()
    from sklearn.metrics import roc_auc_score,confusion_matrix
    auc = roc_auc_score(trues, pred)
    print(confusion_matrix(trues, result))
    return right / total, auc


if __name__ == '__main__':

    evaluator = Evaluator()

    train_model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator) * 20,
        epochs=5,
        callbacks=[evaluator]
    )

else:

    model.load_weights('best_model_bert_ptuning.weights')
