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
import json

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
parser = argparse.ArgumentParser(description="training set index")
parser.add_argument("--train_set_index", "-ti", help="training set index", type=str, default="0")
parser.add_argument("--training_type", "-tt", help="few-shot or zero-shot", type=str, default="few-shot")

args = parser.parse_args()
train_set_index = args.train_set_index
training_type = args.training_type

# 模板
# input_str_format = "{}，黴鹹{}几点内容" # 黴鹹：生僻字组合会被替换为 强调 or 提到，方便寻找mask index [7957, 7919]
input_str_format = "黴鹹用{}概括{}" # 黴鹹：生僻字组合会被替换为 不能 or 可以，方便寻找mask index [7957, 7919]
labels = ["不能", "可以"]
label2words = {"0": "不能", "1":"可以"}

num_classes = 2
maxlen = 256
batch_size = 16
num_per_val_file = 45
acc_list = []

config_path = '/path/language_model/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/path/language_model/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/path/language_model/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'
# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)
def load_data(filename): # 加载数据
    D = []
    with open(filename, encoding='utf-8') as f:
        for i, l in enumerate(f):
            l = json.loads(l.strip())
            keyword = "，".join(l["keyword"])
            abst = l['abst']
            content = input_str_format.format(keyword, abst)
            content_ids, segment_ids = tokenizer.encode(content)
            while len(content_ids) > 256:
                content_ids.pop(-2) # 截断abst内容保证max_seq_length==256
                segment_ids.pop(-2)
            # abst_ids = tokenizer.encode(abst)[0]
            # keyword_ids = tokenizer.encode(keyword)[0]
            # abst_ids_len = min(256-7-2-(len(keyword_ids)-2), len(abst_ids)-2) # seq_length-promopt_length-keyword_length
            # abst = tokenizer.decode(abst_ids[1:1+abst_ids_len])
            
            mask_idxs = [idx for idx, c in enumerate(content_ids) if c ==  7957 and content_ids[idx+1] == 7919]
            mask_idxs.append(mask_idxs[0]+1)
            label = l["label"]
            D.append(((content, content_ids, segment_ids), label2words[label], mask_idxs))
    return D

# 加载数据集，只截取一部分，模拟小数据集
train_data = load_data('ready_data/csl/train_{}.json'.format(train_set_index))
valid_data = []
for i in range(5):
    valid_data += load_data('ready_data/csl/dev_{}.json'.format(i))
test_data = load_data('ready_data/csl/test_public.json')

# 模拟标注和非标注数据
train_frac = 1 # 标注数据的比例
num_labeled = int(len(train_data) * train_frac)
unlabeled_data = [(t, 2) for t, l in train_data[num_labeled:]]
print("length of unlabeled_data0:",len(unlabeled_data))
train_data = train_data[:num_labeled]
train_data = train_data + unlabeled_data
print("length of train_data1:",len(train_data))





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
        for is_end, (content_ids, label, mask_idx) in self.sample(random):
            # if len(label) == 2: # label是两个字的文本
            #     text = text # 拼接文本
            # token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            content, token_ids, segment_ids = content_ids[0], content_ids[1], content_ids[2]
            if random:
                source_ids, target_ids = random_masking(token_ids)
            else:
                source_ids, target_ids = token_ids[:], token_ids[:]
            if len(label) == 2: # label是两个字的文本
                label_ids = tokenizer.encode(label)[0][1:-1] # label_ids: [1093, 689]。 e.g. [101, 1093, 689, 102] =[CLS,农,业,SEP]. tokenizer.encode(label): ([101, 1093, 689, 102], [0, 0, 0, 0])
                for i, label_id_ in zip(mask_idx, label_ids):
                    if tokenizer.id_to_token(source_ids[i]) not in ["黴", "鹹", "[MASK]"]:
                        print(content, tokenizer.id_to_token(source_ids[i]), mask_idx) # 确保mask掉了正确的token
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
    """
    计算候选标签列表中每一个标签（如'科技'）的联合概率，并与正确的标签做对比。候选标签的列表：['科技','娱乐','汽车',..,'农业']
    y_pred=(32, 2, 21128)=--->(32, 1, 14) = (batch_size, 1, label_size)---argmax--> (batch_size, 1, 1)=(batch_size, 1, index in the label)，批量得到联合概率分布最大的标签词语
    :param data:
    :return:
    """
    pred_result_list = []
    label_ids = np.array([tokenizer.encode(l)[0][1:-1] for l in labels]) # 获得两个字的标签对应的词汇表的id列表，如: label_id=[1093, 689]。label_ids=[[1093, 689],[],[],..[]]tokenizer.encode('农业') = ([101, 1093, 689, 102], [0, 0, 0, 0])
    total, right = 0., 0.
    for x_true, _ in data:
        x_true, y_true = x_true[:2], x_true[2] # x_true = [batch_token_ids, batch_segment_ids]; y_true: batch_output_ids
        mask_idxs = np.where(x_true[0] == tokenizer._token_mask_id)[1].reshape(y_true.shape[0], 2)

        y_pred = model.predict(x_true)
        y_pred = np.array([y_pred[i][mask_idx] for i, mask_idx in enumerate(mask_idxs)]) # 取出每个样本特定位置上的索引下的预测值。y_pred=[batch_size, 2, vocab_size]。mask_idxs = [7, 8]
        
        y_true = np.array([y_true[i][mask_idx] for i, mask_idx in enumerate(mask_idxs)])
        # print("y_pred:",y_pred.shape,";y_pred:",y_pred) # (32, 2, 21128)
        # print("label_ids",label_ids) # [[4906 2825],[2031  727],[3749 6756],[3180 3952],[6568 5307],[3136 5509],[1744 7354],[2791  772],[4510 4993],[1092  752],[3125  752],[3152 1265],[ 860 5509],[1093  689]]
        y_pred = y_pred[:, 0, label_ids[:, 0]] * y_pred[:, 1, label_ids[:, 1]] # y_pred=[batch_size,1,label_size]=[32,1,14]。联合概率分布。 y_pred[:, 0, label_ids[:, 0]]的维度为：[32,1,21128]
        y_pred = y_pred.argmax(axis=1) # 找到概率最大的那个label(词)。如“财经”
        # print("y_pred:",y_pred.shape,";y_pred:",y_pred) # O.K. y_pred: (16,) ;y_pred: [4 0 4 1 1 4 5 3 9 1 0 9]
        # print("y_true.shape:",y_true.shape,";y_true:",y_true) # y_true: (16, 128)
        y_true = np.array([labels.index(tokenizer.decode(y)) for y in y_true])
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
    plt.savefig("./baseline/tnews/pet_csl.svg") # 保存为svg格式图片，如果预览不了svg图片可以把文件后缀修改为'.png'


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
        pred_result = np.array(pred_result, dtype="int32")
        test_acc = pred_result.sum()/pred_result.shape[0]
        print("zero-shot结果: {}".format(test_acc))
    else:
        print("未知的训练类型")
else:

    model.load_weights('pet_tnews_best_model.weights')