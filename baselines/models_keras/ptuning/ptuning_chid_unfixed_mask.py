#! -*- coding:utf-8 -*-
# 情感分析例子，利用MLM+P-tuning

# 英文上面的实验参数（使用超参数搜索的方法）：
# we take learning
# rates from 1e-5, 2e-5, 3e-5 and batch sizes from 16, 32.
# For small datasets (COPA, WSC, CB, RTE), we fine-tune
# pretrained models for 20 epochs
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
from tqdm import tqdm
import tensorflow as tf
from tensorflow.python.ops import array_ops
import re

import random
import argparse
parser = argparse.ArgumentParser(description="training set index")
parser.add_argument("--train_set_index", "-t", help="training set index", type=str, default="0")
args = parser.parse_args()
train_set_index = args.train_set_index

maxlen = 256
batch_size = 32
num_per_val_file = 42
acc_list = []
# 加载预训练模型
base_model_path='/path/language_model/chinese_roberta_wwm_ext_L-12_H-768_A-12/'
config_path = base_model_path+'bert_config.json'
checkpoint_path =  base_model_path+'bert_model.ckpt'
dict_path = base_model_path+'vocab.txt'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)
# 加载数据的方法
def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        for idx,l in enumerate(f):
            sample=json.loads(l.strip().replace(" ", "").replace("\t", ""))
            answer = int(sample["answer"])
            sentence1 = sample["content"].replace("#idiom#", "锟斤烤烫") # 替换为生僻词确保输入中没有答案
            _mask = get_mask_idx(sentence1, "锟斤烤烫")
            assert len(_mask) == 4
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

# def load_valid_data(filename):
#     D = []
#     with open(filename, encoding='utf-8') as f:
#         for idx,l in enumerate(f):
#             #print("l:",l)
#             sample=json.loads(l.strip())
#             # print("json_string:",json_string)
#             for can_idx, can in enumerate(sample["candidates"]):
#                 sentence1=can
#                 sentence2=sample["content"].replace("#idiom#", sentence1)
#                 label=1 if can_idx == sample["answer"] else 0

#                 #text, label = l.strip().split('\t')
#                 D.append((sentence2, int(label)))
#             if idx < 5:
#                 print(D[idx])
#     return D

# 加载数据集
train_data = load_data('ready_data/chid/train_{}.json'.format(train_set_index))
valid_data = []
for i in range(5):
    valid_data += load_data('ready_data/chid/dev_{}.json'.format(i))
test_data = load_data('ready_data/chid/test_public.json')

# 模拟标注和非标注数据
train_frac = 1 # 0.01  # 标注数据的比例
print("0.length of train_data:",len(train_data)) # 16883
num_labeled = int(len(train_data) * train_frac)
# unlabeled_data = [(t, 2) for t, l in train_data[num_labeled:]]
train_data = train_data[:num_labeled]
print("1.num_labeled data used:",num_labeled," ;train_data:",len(train_data)) # 168
def encode_candidates(candidates):
    """把候选成语转化为id，防止UNK词出现导致在候选成语中找不到预测成语"""
    ids_list = []
    for can in candidates:
        ids_list.append(tuple(tokenizer.tokens_to_ids(can)))
    return ids_list
    
val_mask_idxs = [d[2] for d in valid_data]
test_mask_idxs = [d[2] for d in test_data]
val_labels_list = [encode_candidates(d[3]) for d in valid_data]
test_labels_list = [encode_candidates(d[3]) for d in test_data]
# train_data = train_data + unlabeled_data



# 对应的任务描述
# mask_idxs = [5,6,7,8] # [7, 8] # mask_idx = 1 #5
unused_length=1 # 不要[unused]标签，每加一个，准确率下降几个点
desc = ['[unused%s]' % i for i in range(1, unused_length)] # desc: ['[unused1]', '[unused2]', '[unused3]', '[unused4]', '[unused5]', '[unused6]', '[unused7]', '[unused8]']
# for mask_id in mask_idxs:
#     desc.insert(mask_id - 1, '[MASK]')            # desc: ['[unused1]', '[unused2]', '[unused3]', '[unused4]', '[unused5]', '[unused6]', '[MASK]', '[MASK]', '[unused7]', '[unused8]']
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
    # TODO TODO TODO 这里面的每一行代码，，，
    目前看只是将原始文本转换为token id
    负向样本（输入是一个[MASK]字符，输出是特定的字符。对于负样本，采用"不"，正样本，采用“很”）
    """
    def __iter__(self, random=False): # TODO 这里的random是指否需要对原始文本进行mask
        batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []
        for is_end, (text, answer, mask_idxs, _) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            token_ids = token_ids[:1] + desc_ids +token_ids[1:] # # token_ids[:1] = [CLS]
            segment_ids = [0] * len(desc_ids) + segment_ids
            if random: # 暂时没有用呢
                source_ids, target_ids = random_masking(token_ids)
            else:
                source_ids, target_ids = token_ids[:], token_ids[:]
            #label_ids = tokenizer.encode(label)[0][1:-1] # label_ids: [1092, 752] ;tokenizer.token_to_id(label[0]): 1092. 得到标签（如"财经"）对应的词汇表的编码ID。label_ids: [1093, 689]。 e.g. [101, 1093, 689, 102] =[CLS,农,业,SEP]. tokenizer.encode(label): ([101, 1093, 689, 102], [0, 0, 0, 0])
            # print("label_ids:",label_ids,";tokenizer.token_to_id(label[0]):",tokenizer.token_to_id(label[0]))
            for mask_id, an in zip(mask_idxs, answer):
                source_ids[mask_id] = tokenizer._token_mask_id
                target_ids[mask_id] = tokenizer.token_to_id(an) # token_to_id与tokenizer.encode可以实现类似的效果。
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


class PtuningEmbedding(Embedding):
    """新定义Embedding层，只优化部分Token
    如果不考虑计算量，也可以先进行梯度求导，再多数位置加上一个极大的负数(如-10000），再求exp(x)，使得多数位置上的梯度为0.
    """
    def call(self, inputs, mode='embedding'):
        embeddings = self.embeddings
        embeddings_sg = K.stop_gradient(embeddings) # 在tf.gradients()参数中存在stop_gradients，这是一个List，list中的元素是tensorflow graph中的op，一旦进入这个list，将不会被计算梯度，更重要的是，在该op之后的BP计算都不会运行。
        mask = np.zeros((K.int_shape(embeddings)[0], 1)) #e.g. mask = array([[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.]])
        tf.print(mask)
        mask[1:unused_length] += 1  # 只优化id为1～8的token. 注：embedding第一位是[PAD]，跳过。          e.g. mask = array([[0.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.]]); 1-mask = array([[1.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[1.]])
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
    model=PtuningBERT, #PtuningBERT, bert
    with_mlm=True
)

for layer in model.layers:
    if layer.name != 'Embedding-Token': # 如果不是embedding层，那么不要训练
        layer.trainable = False

# 训练用模型
y_in = keras.layers.Input(shape=(None,))
output = keras.layers.Lambda(lambda x: x[:, :])(model.output) # TODO TODO TODO
outputs = CrossEntropy(1)([y_in, model.output])

train_model = keras.models.Model(model.inputs + [y_in], outputs)
train_model.compile(optimizer=Adam(6e-4)) # 可能是稍大好一点。模型：6e-4；finetuing常规学习率：2e-5.
train_model.summary()

# 预测模型
model = keras.models.Model(model.inputs, output)

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
    plt.savefig("ptuning_chid.svg") # 保存为svg格式图片，如果预览不了svg图片可以把文件后缀修改为'.png'

# # 对验证集进行验证
# def chid_evaluate(data, candidates_num=7):
#     total, right = 0., 0.
#     pred_list = []
#     true_list = []
#     for x_true, _ in tqdm(data, desc="chid准确率验证"):
#         x_true, y_true = x_true[:2], x_true[2]
#         y_pred = model.predict(x_true)
#         y_pred = y_pred[:, mask_idx, [neg_id, pos_id]]
#         pred_list += y_pred[:,1].tolist()
#         y_true = (y_true[:, mask_idx] == pos_id).astype(int)
#         true_list += y_true.tolist()
#     assert len(pred_list) % candidates_num == 0, len(pred_list)
#     assert len(true_list) % candidates_num == 0, len(true_list)
#     pred_arr = np.array(pred_list).reshape(int(len(pred_list)/candidates_num), candidates_num)
#     true_arr = np.array(true_list).reshape(int(len(pred_list)/candidates_num), candidates_num)
#     pred_answers = pred_arr.argmax(-1)
#     true_answer = true_arr.argmax(-1)
#     return np.where(pred_answers==true_answer)[0].shape[0] / true_arr.shape[0]


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
    elif val_type == "test":
        labels_list = test_labels_list
    else:
        raise ValueError('选择正确的数据集类型')
    for idx, X in tqdm(enumerate(data), desc="{}数据集验证中".format(val_type)):
        label_ids = np.array([[np.array([l for l in label])] for label in labels_list[batch_size*idx: batch_size*(idx+1)]])
        tmp_size = label_ids.shape[0]
        label_ids = label_ids.reshape(tmp_size, 7, 4)
        x_true = X[0]
        x_true, y_true = x_true[:2], x_true[2]
        mask_idxs = np.where(x_true[0] == tokenizer._token_mask_id)[1].reshape(y_true.shape[0], 4)
        y_pred = model.predict(x_true)
        y_pred = np.array([y[mask_idxs[i]].tolist() for i, y in enumerate(y_pred)])
        # 计算候选集中各成语的概率：p(idiom) = p(idiom_1)*p(idiom_2)*p(idiom_3)*p(idiom_4)
        y_pred = [y_pred[i, 0, label_ids[i, :, 0]] * y_pred[i, 1, label_ids[i, :, 1]]* y_pred[i, 2, label_ids[i, :, 2]]* y_pred[i, 3, label_ids[i, :, 3]] for i in range(tmp_size)]
        y_pred = np.array(y_pred)
        y_pred = y_pred.argmax(axis=1)
        true_list = [labels_list[idx*batch_size+i].index(tuple(y[mask_idxs[i]])) for i, y in enumerate(y_true)]
        y_true = np.array(true_list)
        total += len(y_true)
        pred_result_list += (y_true == y_pred).tolist()
    return pred_result_list
# def evaluate(data):
#     """
#     计算候选标签列表中每一个标签（如'科技'）的联合概率，并与正确的标签做对比。候选标签的列表：['科技','娱乐','汽车',..,'农业']
#     y_pred=(32, 2, 21128)=--->(32, 1, 14) = (batch_size, 1, label_size)---argmax--> (batch_size, 1, 1)=(batch_size, 1, index in the label)，批量得到联合概率分布最大的标签词语
#     :param data:
#     :return:
#     """
#     label_ids = np.array([tokenizer.encode(l)[0][1:-1] for l in labels]) # 获得两个字的标签对应的词汇表的id列表，如: label_id=[1093, 689]。label_ids=[[1093, 689],[],[],..[]]tokenizer.encode('农业') = ([101, 1093, 689, 102], [0, 0, 0, 0])
#     total, right = 0., 0.
#     pred_result_list = []
#     for x_true, _ in data:
#         x_true, y_true = x_true[:2], x_true[2] # x_true = [batch_token_ids, batch_segment_ids]; y_true: batch_output_ids
#         y_pred = model.predict(x_true)[:, mask_idxs] # 取出特定位置上的索引下的预测值。y_pred=[batch_size, 2, vocab_size]。mask_idxs = [7, 8]
#         # print("y_pred:",y_pred.shape,";y_pred:",y_pred) # (32, 2, 21128)
#         # print("label_ids",label_ids) # [[4906 2825],[2031  727],[3749 6756],[3180 3952],[6568 5307],[3136 5509],[1744 7354],[2791  772],[4510 4993],[1092  752],[3125  752],[3152 1265],[ 860 5509],[1093  689]]
#         y_pred = y_pred[:, 0, label_ids[:, 0]] * y_pred[:, 1, label_ids[:, 1]]* y_pred[:, 2, label_ids[:, 2]]* y_pred[:, 3, label_ids[:, 3]] # y_pred=[batch_size,1,label_size]=[32,1,14]。联合概率分布。 y_pred[:, 0, label_ids[:, 0]]的维度为：[32,1,21128]
#         y_pred = y_pred.argmax(axis=1) # 找到概率最大的那个label(词)。如“财经”
#         # print("y_pred:",y_pred.shape,";y_pred:",y_pred) # O.K. y_pred: (16,) ;y_pred: [4 0 4 1 1 4 5 3 9 1 0 9]
#         # print("y_true.shape:",y_true.shape,";y_true:",y_true) # y_true: (16, 128)
#         y_true = np.array([labels.index(tokenizer.decode(y)) for y in y_true[:, mask_idxs]])
#         total += len(y_true)
#         # right += (y_true == y_pred).sum()
#         pred_result_list += (y_true == y_pred).tolist()
#     # return right / total
#     return pred_result_list


if __name__ == '__main__':

    evaluator = Evaluator()

    train_model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator) * 50,
        epochs=20,
        callbacks=[evaluator]
    )

else:

    model.load_weights('best_model_bert_ptuning.weights')