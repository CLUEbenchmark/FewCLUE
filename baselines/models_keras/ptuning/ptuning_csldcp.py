#! -*- coding:utf-8 -*-
# 情感分析例子，利用MLM+P-tuning
# P-tuning: 模板(patter)可以自己学习到，不需要是自然语言；可以只学习少量的（如10个）的embedding，而不用学习所有的embedding.
import os

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

label_en2zh ={'材料科学与工程': '材料',
                             '作物学': '作物',
                             '口腔医学': '口腔',
                             '药学': '药学',
                             '教育学': '教育',
                             '水利工程': '水利',
                             '理论经济学': '理经',
                             '食品科学与工程': '食品',
                             '畜牧学/兽医学': '畜牧',
                             '体育学': '体育',
                             '核科学与技术': '核科',
                             '力学': '力学',
                             '园艺学': '园艺',
                             '水产': '水产',
                             '法学': '法学',
                             '地质学/地质资源与地质工程': '地质',
                             '石油与天然气工程': '石油',
                             '农林经济管理': '农林',
                             '信息与通信工程': '通信',
                             '图书馆、情报与档案管理': '图书',
                             '政治学': '政治',
                             '电气工程': '电气',
                             '海洋科学': '海洋',
                             '民族学': '民族',
                             '航空宇航科学与技术': '航空',
                             '化学/化学工程与技术': '化学',
                             '哲学': '哲学',
                             '公共卫生与预防医学': '卫生',
                             '艺术学': '艺术',
                             '农业工程': '农工',
                             '船舶与海洋工程': '船舶',
                             '计算机科学与技术': '计科',
                             '冶金工程': '冶金',
                             '交通运输工程': '交通',
                             '动力工程及工程热物理': '动力',
                             '纺织科学与工程': '纺织',
                             '建筑学': '建筑',
                             '环境科学与工程': '环境',
                             '公共管理': '公管',
                             '数学': '数学',
                             '物理学': '物理',
                             '林学/林业工程': '林学',
                             '心理学': '心理',
                             '历史学': '历史',
                             '工商管理': '工管',
                             '应用经济学': '应经',
                             '中医学/中药学': '中医',
                             '天文学': '天文',
                             '机械工程': '机械',
                             '土木工程': '土木',
                             '光学工程': '光学',
                             '地理学': '地理',
                             '农业资源利用': '农业',
                             '生物学/生物科学与工程': '生物',
                             '兵器科学与技术': '兵器',
                             '矿业工程': '矿业',
                             '大气科学': '大气',
                             '基础医学/临床医学': '基础',
                             '电子科学与技术': '电子',
                             '测绘科学与技术': '测绘',
                             '控制科学与工程': '控制',
                             '军事学': '军事',
                             '中国语言文学': '中文',
                             '新闻传播学': '新闻', '社会学': '社会',
                             '地球物理学':'地球',
                             '植物保护':'植保'}
labels=[label_zh for label_en,label_zh in label_en2zh.items()]
labels_en=[label_en for label_en,label_zh in label_en2zh.items()]

maxlen = 256
batch_size = 16

# 加载预训练模型
base_model_path = '../pretrained_models/chinese_roberta_wwm_ext_L-12_H-768_A-12/'
config_path = base_model_path+'bert_config.json'
checkpoint_path =  base_model_path+'bert_model.ckpt'
dict_path = base_model_path+'vocab.txt'


# 加载数据的方法
def load_data(filename): # 加载数据
    D = []
    with open(filename, encoding='utf-8') as f:
        for i, l in enumerate(f):
            l = json.loads(l)
            label_en=l['label']
            if label_en not in labels_en:
                continue
            label_zh=label_en2zh[label_en] # 将英文转化为中文
            D.append((l['content'], label_zh))
    return D


path = '../data/FewCLUEDatasets-master/ready_data/csldcp'
save_path = '../output/csldcp/'
if not os.path.exists(save_path):
    os.mkdir(save_path)
# 加载数据集，只截取一部分，模拟小数据集
data_num = '0'
# 加载数据集
train_data = load_data('{}/train_{}.json'.format(path,data_num))
valid_datas = []
for i in range(5):
    valid_data = load_data('{}/dev_{}.json'.format(path,str(i)))
    valid_datas.append(valid_data)
test_data = load_data('{}/test_public.json'.format(path))

# 模拟标注和非标注数据
train_frac = 1# 0.01  # 标注数据的比例
print("0.length of train_data:",len(train_data)) # 16883
num_labeled = int(len(train_data) * train_frac)
train_data = train_data[:num_labeled]
print("1.num_labeled data used:",num_labeled," ;train_data:",len(train_data)) # 168
# train_data = train_data + unlabeled_data

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 对应的任务描述
mask_idxs = [1, 2] # [7, 8] # mask_idx = 1 #5
unused_length=9 # 6,13没有效果提升
desc = ['[unused%s]' % i for i in range(1, unused_length)] # desc: ['[unused1]', '[unused2]', '[unused3]', '[unused4]', '[unused5]', '[unused6]', '[unused7]', '[unused8]']
for mask_id in mask_idxs:
    desc.insert(mask_id - 1, '[MASK]')            # desc: ['[unused1]', '[unused2]', '[unused3]', '[unused4]', '[unused5]', '[unused6]', '[MASK]', '[MASK]', '[unused7]', '[unused8]']
desc_ids = [tokenizer.token_to_id(t) for t in desc] # 将token转化为id

# pos_id = tokenizer.token_to_id(u'很') # e.g. '[unused9]'. 将正向的token转化为id. 默认值：u'很'
# neg_id = tokenizer.token_to_id(u'不') # e.g. '[unused10]. 将负向的token转化为id. 默认值：u'不'


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
        mask[0:unused_length] += 1  # 只优化id为1～8的token. 注：embedding第一位是[PAD]，跳过。  额外加上几个可学习的，因为中间插入的两个目标词的mask。+len(mask_idxs)。        e.g. mask = array([[0.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.]]); 1-mask = array([[1.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[1.]])
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
output = keras.layers.Lambda(lambda x: x[:, :unused_length+1])(model.output) # TODO TODO TODO
outputs = CrossEntropy(1)([y_in, model.output])

train_model = keras.models.Model(model.inputs + [y_in], outputs)
train_model.compile(optimizer=Adam(6e-4)) # 6e-4
train_model.summary()

# 预测模型
model = keras.models.Model(model.inputs, output)

# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generators = []
for valid_data in valid_datas:
    valid_generator = data_generator(valid_data, batch_size)
    valid_generators.append(valid_generator)
test_generator = data_generator(test_data, batch_size)


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_accs = 0
        for valid_generator in valid_generators:
            val_accs += evaluate(valid_generator)
        val_acc = val_accs / 5
        if val_acc > self.best_val_acc: #  保存最好的模型，并记录最好的准确率
            self.best_val_acc = val_acc
            model.save_weights('best_model_bert_ptuning.weights')
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
    label_ids = np.array([tokenizer.encode(l)[0][1:-1] for l in labels]) # 获得两个字的标签对应的词汇表的id列表，如: label_id=[1093, 689]。label_ids=[[1093, 689],[],[],..[]]tokenizer.encode('农业') = ([101, 1093, 689, 102], [0, 0, 0, 0])
    total, right = 0., 0.
    for x_true, _ in data:
        x_true, y_true = x_true[:2], x_true[2] # x_true = [batch_token_ids, batch_segment_ids]; y_true: batch_output_ids
        y_pred = model.predict(x_true)[:, mask_idxs] # 取出特定位置上的索引下的预测值。y_pred=[batch_size, 2, vocab_size]。mask_idxs = [7, 8]
        # print("y_pred:",y_pred.shape,";y_pred:",y_pred) # (32, 2, 21128)
        # print("label_ids",label_ids) # [[4906 2825],[2031  727],[3749 6756],[3180 3952],[6568 5307],[3136 5509],[1744 7354],[2791  772],[4510 4993],[1092  752],[3125  752],[3152 1265],[ 860 5509],[1093  689]]
        y_pred = y_pred[:, 0, label_ids[:, 0]] * y_pred[:, 1, label_ids[:, 1]] # y_pred=[batch_size,1,label_size]=[32,1,14]。联合概率分布。 y_pred[:, 0, label_ids[:, 0]]的维度为：[32,1,21128]
        # print("y_pred[:, 0, label_ids[:, 0]]:",y_pred[:, 0, label_ids[:, 0]])
        y_pred = y_pred.argmax(axis=1) # 找到概率最大的那个label(词,如“财经”)所在的索引(index)。
        # print("y_pred:",y_pred.shape,";y_pred:",y_pred) # O.K. y_pred: (16,) ;y_pred: [4 0 4 1 1 4 5 3 9 1 0 9]
        # print("y_true.shape:",y_true.shape,";y_true:",y_true) # y_true: (16, 128)
        # y_true=[batch_size,sequence_length]=(16,128); y_true[:, mask_idxs]=[batch_size,2]
        y_true = np.array([labels.index(tokenizer.decode(y)) for y in y_true[:, mask_idxs]]) # 找到标签对应的ID,对应的文本，对应的标签列表中所在的顺序。 labels=['科技','娱乐',...,'汽车']
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


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