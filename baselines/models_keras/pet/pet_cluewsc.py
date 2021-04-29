#! -*- coding:utf-8 -*-
# CLUEWSC上的例子，利用MLM做 Zero-Shot/Few-Shot/Semi-Supervised Learning

import json
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
import pdb
import argparse
parser = argparse.ArgumentParser(description="training set index")
parser.add_argument("--training_type", "-tt", help="few-shot or zero-shot", type=str, default="few-shot")
args = parser.parse_args()
training_type = args.training_type
label_list=['正确','错误']
label_en2zh={'false':'正确','true':'错误',None:'无'}
labels_en=[label_en for label_en,label_zh in label_en2zh.items()]

maxlen = 128
batch_size = 16
config_path = '/path/language_model/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/path/language_model/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/path/language_model/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'

# {"target": {"span1_index": 0, "span1_text": "乔", "span2_index": 21, "span2_text": "他"}, "idx": 32, "text": "乔的叔叔仍然可以在网球上击败他，即便他年长他30岁。", "label": "true"}
def load_data(filename): # 加载数据
    D = []
    with open(filename, encoding='utf-8') as f:
        for i, l in enumerate(f):
            l = json.loads(l)
            # print(i,"l:",l)
            target=l['target']
            label_en = l['label'] if 'label' in l else None
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
            print("text_new:",text_new)
            D.append((text_new, label_zh))
    return D

# 加载数据集，只截取一部分，模拟小数据集
train_data = load_data('./ready_data/cluewsc/train_0.json')
valid_data = load_data('./ready_data/cluewsc/dev_0.json')
test_data = load_data('./ready_data/cluewsc/test_public.json')

# 模拟标注和非标注数据
train_frac = 1 # 标注数据的比例
num_labeled = int(len(train_data) * train_frac)
unlabeled_data = [(t, 2) for t, l in train_data[num_labeled:]]
print("length of unlabeled_data0:",len(unlabeled_data))
train_data = train_data[:num_labeled]
train_data = train_data + unlabeled_data
print("length of train_data1:",len(train_data))

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 对应的任务描述
# prefix = u'下面报导一则体育新闻。' # 完整的pattern: prefix+ mask +sentence. e.g. '下面报导一则体育新闻。[mask][mask]今天新冠疫苗开大'
prefix = u'下面句子的指代关系正确吗？正确。' # 完成的例子：u'指代关系正确吗？[mask][mask] 。乔（这是实体）的叔叔仍然可以在网球上击败他（代词），即便他年长他30岁。'
mask_idxs = [14,15] # [7, 8]


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
        for is_end, (text, label) in self.sample(random):
            if len(label) == 2: # label是两个字的文本
                text = prefix + text # 拼接文本
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            if random:
                source_ids, target_ids = random_masking(token_ids)
            else:
                source_ids, target_ids = token_ids[:], token_ids[:]
            if len(label) == 2: # label是两个字的文本
                label_ids = tokenizer.encode(label)[0][1:-1] # label_ids: [1093, 689]。 e.g. [101, 1093, 689, 102] =[CLS,农,业,SEP]. tokenizer.encode(label): ([101, 1093, 689, 102], [0, 0, 0, 0])
                for i, label_id_ in zip(mask_idxs, label_ids):
                    source_ids[i] = tokenizer._token_mask_id # i: 7(mask1的index) ;j: 1093(农); i:8 (mask2的index) ;j: 689(业)
                    target_ids[i] = label_id_

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

class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分。作用就是只计算目标位置的loss，忽略其他位置的loss。
    """
    def compute_loss(self, inputs, mask=None):
        y_true, y_pred = inputs # y_true:[batch_size, sequence_length]。应该是one-hot的表示，有一个地方为1，其他地方为0：[0,0,1,...0]
        y_mask = K.cast(K.not_equal(y_true, 0), K.floatx()) # y_mask是一个和y_true一致的shape. 1的值还为1.0，0的值还为0.0.即[0.0,0.0,1.0,...0.0]。
        # sparse_categorical_accuracy的例子。y_true = 2; y_pred = (0.02, 0.05, 0.83, 0.1); acc = sparse_categorical_accuracy(y_true, y_pred)
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
train_model.compile(optimizer=Adam(8e-5))
train_model.summary()

# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        model.save_weights('pet_cluewsc_model.weights')
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('pet_cluewsc_best_model.weights')
        test_acc = evaluate(test_generator)
        print(
            u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n' %
            (val_acc, self.best_val_acc, test_acc)
        )

def evaluate(data):
    """
    计算候选标签列表中每一个标签（如'是的'）的联合概率，并与正确的标签做对比。候选标签的列表：['是的','不是']
    y_pred=(32, 2, 21128)=--->(32, 1, 14) = (batch_size, 1, label_size)---argmax--> (batch_size, 1, 1)=(batch_size, 1, index in the label)
    :param data:
    :return:
    """
    label_ids = np.array([tokenizer.encode(l)[0][1:-1] for l in label_list]) # 获得两个字的标签对应的词汇表的id列表，如: label_id=[1093, 689]。label_ids=[[1093, 689],[],[],..[]]tokenizer.encode('农业') = ([101, 1093, 689, 102], [0, 0, 0, 0])
    total, right = 0., 0.
    for x_true, _ in data:
        x_true, y_true = x_true[:2], x_true[2] # x_true = [batch_token_ids, batch_segment_ids]; y_true: batch_output_ids
        y_pred = model.predict(x_true)[:, mask_idxs] # 取出特定位置上的索引下的预测值。y_pred=[batch_size, 2, vocab_size]。mask_idxs = [9, 10]
        # print("y_pred:",y_pred.shape,";y_pred:",y_pred) # (32, 2, 21128)
        # print("label_ids",label_ids) # [[4906 2825],[2031  727],[3749 6756],[3180 3952],[6568 5307],[3136 5509],[1744 7354],[2791  772],[4510 4993],[1092  752],[3125  752],[3152 1265],[ 860 5509],[1093  689]]
        y_pred = y_pred[:, 0, label_ids[:, 0]] * y_pred[:, 1, label_ids[:, 1]] # y_pred=[batch_size,1,label_size]=[32,1,14]。联合概率分布。 y_pred[:, 0, label_ids[:, 0]]的维度为：[32,1,21128]
        # print("y_pred[:, 0, label_ids[:, 0]]:",y_pred[:, 0, label_ids[:, 0]])
        y_pred = y_pred.argmax(axis=1) # 找到概率最大的那个label(词,如“财经”)所在的索引。
        # print("y_pred:",y_pred.shape,";y_pred:",y_pred) # O.K. y_pred: (16,) ;y_pred: [4 0 4 1 1 4 5 3 9 1 0 9]
        # print("y_true.shape:",y_true.shape,";y_true:",y_true) # y_true: (16, 128)
        y_true = np.array([label_list.index(tokenizer.decode(y)) for y in y_true[:, mask_idxs]]) # 正确标签在标签列表中的索引，如0，1.
        total += len(y_true)
        right += (y_true == y_pred).sum()
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
        pred_result = evaluate(test_generator)
        print("zero-shot结果: {}".format(pred_result))
    else:
        print("未知的训练类型")
else:

    model.load_weights('pet_cluewsc_best_model.weights')
