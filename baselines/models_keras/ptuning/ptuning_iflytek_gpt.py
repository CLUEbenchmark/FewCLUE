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
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser(description="training set index")
parser.add_argument("--train_set_index", "-t", help="training set index", type=str, default="0")
args = parser.parse_args()
train_set_index = args.train_set_index
assert train_set_index in {"0", "1", "2", "3", "4", "all"}, 'train_set_index must in {"0", "1", "2", "3", "4", "all"}'
from tqdm import tqdm
config_path = 'path/language_model/chinese_roberta_wwm_ext_L-12_H-768_A-12/config.json'
checkpoint_path = 'path/language_model/nezha-gpt/cn_gpt'
dict_path = 'path/language_model/nezha-gpt/vocab.txt'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

label_en2zh ={'银行': '银行',
 '社区服务': '社区',
 '电商': '电商',
 '支付': '支付',
 '经营养成': '经营',
 '卡牌': '卡牌',
 '借贷': '借贷',
 '驾校': '驾校',
 '理财': '理财',
 '职考': '职考',
 '新闻': '新闻',
 '旅游资讯': '旅游',
 '公共交通': '交通',
 '魔幻': '魔幻',
 '医疗服务': '医疗',
 '影像剪辑': '影像',
 '动作类': '动作',
 '工具': '工具',
 '体育竞技': '体育',
 '小说': '小说',
 '运动健身': '运动',
 '相机': '相机',
 '辅助工具': '工具',
 '快递物流': '快递',
 '高等教育': '教育',
 '股票': '股票',
 '菜谱': '菜谱',
 '行车辅助': '行车',
 '仙侠': '仙侠',
 '亲子儿童': '亲子',
 '购物咨询': '购物',
 '射击游戏': '射击',
 '漫画': '漫画',
 '中小学': '小学',
 '同城服务': '同城',
 '成人教育': '成人',
 '求职': '求职',
 '电子产品': '电子',
 '艺术': '艺术',
 '薅羊毛': '赚钱',
 '约会社交': '约会',
 '经营': '经营',
 '兼职': '兼职',
 '短视频': '视频',
 '音乐': '音乐',
 '英语': '英语',
 '棋牌中心': '棋牌',
 '摄影修图': '摄影',
 '养生保健': '养生',
 '办公': '办公',
 '政务': '政务',
 '视频': '视频',
 '论坛圈子': '论坛',
 '彩票': '彩票',
 '直播': '直播',
 '其他': '其他',
 '休闲益智': '休闲',
 '策略': '策略',
 '即时通讯': '通讯',
 '汽车交易': '买车',
 '违章': '违章',
 '地图导航':'地图',
 '民航':'民航',
 '电台':'电台',
 '语言(非英语)':'语言',
 '搞笑': '搞笑',
 '婚恋社交': '婚恋',
 '社区超市':'超市',
 '日常养车':'养车',
 '杂志': '杂志',
 '视频教育':'网课',
 '家政': '家政',
 '影视娱乐':'影视',
 '装修家居':'装修',
 '体育咨讯':'资讯',
  '社交工具':'社交',
'餐饮店':'餐饮',
'美颜':'美颜',
'问诊挂号':'挂号',
'飞行空战':'飞行',
'综合预定':'预定',
'电影票务':'票务',
'笔记':'笔记',
'买房':'买房',
'外卖':'外卖',
'母婴':'母婴',
'打车':'打车',
'情侣社交':'情侣',
'日程管理':'日程',
'租车':'租车',
'微博博客':'博客',
'百科':'百科',
'绘画':'绘画', '铁路':'铁路',
'生活社交':'生活',
'租房':'租房',
'酒店':'酒店',
'保险':'保险',
'问答交流':'问答',
'收款':'收款',
'MOBA':'竞技',
'K歌':'唱歌',
'技术':'技术',
'减肥瘦身':'减肥',
'工作社交':'工作',
'团购':'团购',
'记账':'记账',
'女性':'女性',
'公务员':'公务',
'二手':'二手',
'美妆美业':'美妆',
'汽车咨询':'汽车', '行程管理':'行程',
'免费WIFI':'免费', '教辅':'教辅', '成人':'两性'}
labels=[label_zh for label_en,label_zh in label_en2zh.items()]
labels_en=[label_en for label_en,label_zh in label_en2zh.items()]
label_ids = [tuple(tokenizer.tokens_to_ids(label)) for label in labels]
num_labels = len(label_en2zh)
maxlen = 256
batch_size = 16
num_per_val_file = 148
acc_list = []
def load_data(filename, set_type): # 加载数据
    D = []
    with open(filename, encoding='utf-8') as f:
        for i, l in enumerate(f):
            l = json.loads(l)
            label_en=l['label_des']
            if label_en not in labels_en:
                continue
            label_zh=label_en2zh[label_en] # 将英文转化为中文
            if set_type == "train":
                D.append((l['sentence'], label_zh, ""))
            else:
                # 每个样本由样本本身和样本加每个标签的第一个字构成，即一个样本被重构为15个样本。方便一次计算标签第一个字和第二个字的联合概率
                D.append((l['sentence'], "", label_zh))
            if i < 5:
                print(D[i])
        print("*"*30)
    return D


# 加载数据集
train_data = load_data('ready_data/iflytek/iflytek_8/train_few_{}.json'.format(train_set_index), set_type="train")
valid_data = []
for i in range(5):
    valid_data += load_data('ready_data/iflytek/iflytek_8/dev_few_{}.json'.format(i), set_type="val")
test_data = load_data('ready_data/iflytek/iflytek_8/test_public.json', set_type="test")
valid_data = valid_data
test_data = test_data
val_label = [labels.index(l[2]) for l in valid_data]
test_label = [labels.index(l[2]) for l in test_data]

# 模拟标注和非标注数据
train_frac = 1  # 标注数据的比例
num_labeled = int(len(train_data) * train_frac)
unlabeled_data = [(t, 2) for t, l in train_data[num_labeled:]]
random.shuffle(train_data)
train_data = train_data[:num_labeled]
# train_data = train_data[:1]

# train_data = train_data + unlabeled_data



# 对应的任务描述
unused_length=9 # 6,13没有效果提升
desc = ['[unused%s]' % i for i in range(1, unused_length)] # desc: ['[unused1]', '[unused2]', '[unused3]', '[unused4]', '[unused5]', '[unused6]', '[unused7]', '[unused8]']
desc_ids = [tokenizer.token_to_id(t) for t in desc] # 将token转化为id


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids = []
        batch_label_ids = []
        for is_end, (text, label, val_label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            token_ids = token_ids[:1] + desc_ids[:4] + token_ids[:-1]
            token_ids = token_ids + desc_ids[4:]
            if len(label) == 2: # train set, no label prompt
                token_ids = token_ids + tokenizer.tokens_to_ids(label)
            elif len(label) == 1 and val_label=="": # train set, one prompt
                token_ids = token_ids + [tokenizer.token_to_id(label)]
            elif len(label) == 1 and val_label!="":
                # 添加[UNK]字补齐句子，预测阶段模型并不会使用该字，让模型预测标签第二个字
                token_ids = token_ids + [tokenizer._token_unk_id, tokenizer._token_unk_id]
                batch_label_ids.append(labels.index(val_label))
            else:
                # 添加[UNK]字补齐句子，预测阶段模型并不会使用该字，让模型预测标签第一个字
                token_ids = token_ids+ [tokenizer._token_unk_id]
                batch_label_ids.append(labels.index(val_label))                
            batch_token_ids.append(token_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                if val_label == "":
                    yield batch_token_ids, None
                else:
                    yield batch_token_ids, batch_label_ids
                batch_token_ids = []
                batch_label_ids = []

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
    prob_list = []
    pred_list = []
    truth_list = []
    pred_result_list = []
    for idx, d in tqdm(enumerate(data), desc="准确率计算"):
        x_true = d[0]
        y_true = d[1]
        y_pred = model.predict(x_true)
        for b_idx, xy in enumerate(zip(x_true, y_pred, y_true)):
            x, y, _y = xy[0], xy[1], xy[2]
            x = np.trim_zeros(x)
            pointer = idx*batch_size+b_idx
            y = y[:len(x)][-2, [l[0] for l in label_ids]] # 标签第一个字的概率
            pred = beam_search(x, model, y, beamsize=4)
            pred_list.append(pred)
            truth_list.append(_y)
    total = len(truth_list)
    right = np.where(np.array(pred_list) == np.array(truth_list))[0].shape[0]
    pred_result_list += (np.array(pred_list) == np.array(truth_list)).tolist()
    # return right / total
    return pred_result_list

def beam_search(input_ids, model, label_0_prob, beamsize):
    # 模拟一次beam search, 和beam search 不同的是这里只在有限空间内搜索
    if beamsize == 0 or beamsize==1:
        return label_0_prob.argmax()
    input_ids = input_ids[0:-1]
    candidate = label_0_prob.argsort()[-beamsize:]
    input_ids = [input_ids.tolist()+[label_ids[can][0], tokenizer._token_unk_id] for can in candidate]
    outputs = model.predict(np.array(input_ids))
    label_1_prob = [output[:len(input_ids[0])][-2, label_ids[candidate[idx]][1]] for idx, output in enumerate(outputs)]
    label_0_prob.sort()
    p = label_0_prob[-beamsize:] * label_1_prob
    return candidate[p.argmax()]

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
    plt.savefig("ptuning_tnews.svg") # 保存为svg格式图片，如果预览不了svg图片可以把文件后缀修改为'.png'

if __name__ == '__main__':

    evaluator = Evaluator()

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator) * 30,
        epochs=20,
        callbacks=[evaluator]
    )

else:

    model.load_weights('best_model_iflytek_gpt.weights')