#! -*- coding:utf-8 -*-
# 新闻分类例子，利用MLM做 Zero-Shot/Few-Shot/Semi-Supervised Learning

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
import re
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
parser = argparse.ArgumentParser(description="training set index")
parser.add_argument("--train_set_index", "-ti", help="training set index", type=str, default="0")
parser.add_argument("--training_type", "-tt", help="few-shot or zero-shot", type=str, default="few-shot")

args = parser.parse_args()
train_set_index = args.train_set_index
training_type = args.training_type

maxlen = 128
batch_size = 16
num_per_val_file = 42
acc_list = []

config_path = '/path/language_model/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/path/language_model/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/path/language_model/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'
# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        for idx,l in enumerate(f):
            sample=json.loads(l.strip().replace(" ", "").replace("\t", ""))
            answer = int(sample["answer"])
            sentence1 = sample["content"].replace("#idiom#", "锟斤烤烫") # 替换为生僻词确保输入中没有答案
            _mask = get_mask_idx(sentence1, "锟斤烤烫")
            D.append((sentence1, sample["candidates"][answer], _mask,  sample["candidates"]))
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


# 加载数据集
train_data = load_data('ready_data/chid/train_{}.json'.format(train_set_index))
valid_data = []
for i in range(5):
    valid_data += load_data('ready_data/chid/dev_{}.json'.format(i))
test_data = load_data('ready_data/chid/test_public.json')

# 模拟标注和非标注数据
train_frac = 1 # 标注数据的比例
num_labeled = int(len(train_data) * train_frac)
unlabeled_data = [(t, 2) for t, l in train_data[num_labeled:]]
print("length of unlabeled_data0:",len(unlabeled_data))
train_data = train_data[:num_labeled]
train_data = train_data + unlabeled_data
print("length of train_data1:",len(train_data))


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
train_model.compile(optimizer=Adam(1e-5))
train_model.summary()

# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)

class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        model.save_weights('pet_tnews_model.weights')
        val_pred_result = evaluate(valid_generator, val_type="val")
        val_pred_result = np.array(val_pred_result, dtype="int32")
        total_acc = val_pred_result.sum()/val_pred_result.shape[0]
        val_pred_result = val_pred_result.reshape(5, num_per_val_file).sum(1)/num_per_val_file
        # val_acc_mean = val_pred_result.mean() 准确率均值和total准确率相等
        if total_acc > self.best_val_acc:
            self.best_val_acc = total_acc
            model.save_weights('pet_tnews_best_model.weights')
        test_pred_result = np.array(evaluate(test_generator, val_type="test"))
        test_acc = test_pred_result.sum()/test_pred_result.shape[0]
        acc_tuple = tuple(val_pred_result.tolist()+[total_acc, self.best_val_acc, test_acc])
        acc_list.append(list(acc_tuple))
        draw_acc(acc_list) # 如果需要对照每个验证集准确率
        print(
            u'val_acc_0: %.5f, val_acc_1: %.5f, val_acc_2: %.5f, val_acc_3: %.5f, val_acc_4: %.5f, val_acc_total: %.5f, best_val_acc: %.5f, test_acc: %.5f\n' %
            acc_tuple
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
    for idx, X in tqdm(enumerate(data), desc="{}数据集验证中".format(val_type)):
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
        true_list = [labels_list[idx*batch_size+i].index(tuple(y[mask_idxs[idx*batch_size+i]])) for i, y in enumerate(y_true)]
        y_true = np.array(true_list)
        total += len(y_true)
        pred_result_list += (y_true == y_pred).tolist()
    return pred_result_list

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
    plt.savefig("pet_chid.svg") # 保存为svg格式图片，如果预览不了svg图片可以把文件后缀修改为'.png'


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
        pred_result = evaluate(test_generator, val_type="test")
        pred_result = np.array(pred_result, dtype="int32")
        test_acc = pred_result.sum()/pred_result.shape[0]
        print("zero-shot结果: {}".format(test_acc))
    else:
        print("未知的训练类型")
else:

    model.load_weights('pet_chid_best_model.weights')
