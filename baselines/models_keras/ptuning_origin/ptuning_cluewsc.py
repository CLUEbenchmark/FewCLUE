#! -*- coding:utf-8 -*-

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
import sys
from modeling import tokenizer

label_list=['正确','错误']
label_en2zh={'false':'正确','true':'错误'}
labels_en=[label_en for label_en,label_zh in label_en2zh.items()]

maxlen = 128
batch_size = 16


# 加载数据的方法
def load_data(filename): # 加载数据
    D = []
    with open(filename, encoding='utf-8') as f:
        for i, l in enumerate(f):
            l = json.loads(l)
            #print(i,"l:",l)
            target=l['target']
            label_en = l['label']
            label_zh = label_en2zh[label_en]
            span1_index=target['span1_index'] # 0
            span1_text=target['span1_text'] # 乔
            span2_index=target['span2_index'] # 21
            span2_text=target['span2_text'] # 他
            text=l['text']
            text_list=[x for x in text] # 转化为单个字的list
            text_list.insert(span2_index+len(span2_text),"（这是代词）") # text_new=乔的叔叔仍然可以在网球上击败他（代词），即便他年长他30岁。
            text_list.insert(span1_index+len(span1_text),"（这是实体）") # text_new=乔（这是实体）的叔叔仍然可以在网球上击败他（代词），即便他年长他30岁。
            text_new="".join(text_list)
            #print("text_new:",text_new)
            D.append((text_new, label_zh))
    return D


path = '../../../datasets/cluewsc'
data_num = sys.argv[1]
# 加载数据集
train_data = load_data('{}/train_{}.json'.format(path, data_num))
valid_data = load_data('{}/dev_{}.json'.format(path, data_num))
test_data = load_data('{}/test_public.json'.format(path))


# 对应的任务描述
mask_idxs = [1, 2] # [7, 8] # mask_idx = 1 #5
unused_length=9 # 6,13没有效果提升
desc = ['[unused%s]' % i for i in range(1, unused_length)] # desc: ['[unused1]', '[unused2]', '[unused3]', '[unused4]', '[unused5]', '[unused6]', '[unused7]', '[unused8]']
for mask_id in mask_idxs:
    desc.insert(mask_id - 1, '[MASK]')            # desc: ['[unused1]', '[unused2]', '[unused3]', '[unused4]', '[unused5]', '[unused6]', '[MASK]', '[MASK]', '[unused7]', '[unused8]']
desc_ids = [tokenizer.token_to_id(t) for t in desc] # 将token转化为id

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
    目前看只是将原始文本转换为token id
    负向样本（输入是一个[MASK]字符，输出是特定的字符。对于负样本，采用"不"，正样本，采用“很”）
    """
    def __iter__(self, random=False): #
        batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            if label != 2:
                token_ids = token_ids[:1] + desc_ids + token_ids[1:] # # token_ids[:1] = [CLS]
                segment_ids = [0] * len(desc_ids) + segment_ids
            if random: # 暂时没有用呢
                source_ids, target_ids = random_masking(token_ids)
            else:
                source_ids, target_ids = token_ids[:], token_ids[:]
            #label_ids = tokenizer.encode(label)[0][1:-1] # label_ids: [1092, 752] ;tokenizer.token_to_id(label[0]): 1092. 得到标签（如"财经"）对应的词汇表的编码ID。label_ids: [1093, 689]。 e.g. [101, 1093, 689, 102] =[CLS,农,业,SEP]. tokenizer.encode(label): ([101, 1093, 689, 102], [0, 0, 0, 0])
            # print("label_ids:",label_ids,";tokenizer.token_to_id(label[0]):",tokenizer.token_to_id(label[0]))
            for i,mask_id in enumerate(mask_idxs):
                source_ids[mask_id] = tokenizer._token_mask_id
                target_ids[mask_id] = tokenizer.token_to_id(label[i]) # token_to_id与tokenizer.encode可以实现类似的效果。

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
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc: #  保存最好的模型，并记录最好的准确率
            self.best_val_acc = val_acc
            # model.save_weights('best_model_bert_ptuning.weights')
        test_acc = evaluate(test_generator)
        print( # 打印准确率
            u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n' %
            (val_acc, self.best_val_acc, test_acc)
        )

def evaluate(data):
    """
    计算候选标签列表中每一个标签（如'科技'）的联合概率，并与正确的标签做对比。候选标签的列表：['科技','娱乐','汽车',..,'农业']
    y_pred=(32, 2, 21128)=--->(32, 1, 14) = (batch_size, 1, label_size)---argmax--> (batch_size, 1, 1)=(batch_size, 1, index in the label)
    :param data:
    :return:
    """
    label_ids = np.array([tokenizer.encode(l)[0][1:-1] for l in label_list]) # 获得两个字的标签对应的词汇表的id列表，如: label_id=[1093, 689]。label_ids=[[1093, 689],[],[],..[]]tokenizer.encode('农业') = ([101, 1093, 689, 102], [0, 0, 0, 0])
    total, right = 0., 0.
    for x_true, _ in data:
        x_true, y_true = x_true[:2], x_true[2] # x_true = [batch_token_ids, batch_segment_ids]; y_true = batch_output_ids
        y_pred = model.predict(x_true)[:, mask_idxs]
        y_pred = y_pred[:, 0, label_ids[:, 0]] * y_pred[:, 1, label_ids[:, 1]]
        y_pred = y_pred.argmax(axis=1)
        y_true = np.array([label_list.index(tokenizer.decode(y)) for y in y_true[:, mask_idxs]])
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total

if __name__ == '__main__':

    evaluator = Evaluator()

    train_model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator) * 5,
        epochs=20,
        callbacks=[evaluator]
    )

else:

    model.load_weights('best_model_bert_ptuning.weights')
