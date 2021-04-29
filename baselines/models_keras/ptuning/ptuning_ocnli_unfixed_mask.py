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
import re

maxlen = 80
batch_size = 8

# 加载预训练模型
base_model_path='/path/language_model/chinese_roberta_wwm_ext_L-12_H-768_A-12/'
config_path = base_model_path+'bert_config.json'
checkpoint_path =  base_model_path+'bert_model.ckpt'
dict_path = base_model_path+'vocab.txt'
tokenizer = Tokenizer(dict_path, do_lower_case=True)


# 加载数据的方法
def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        for jj,l in enumerate(f):
            #print("l:",l)
            json_string=json.loads(l.strip())
            # print("json_string:",json_string)
            sentence1=json_string['sentence1']
            sentence2=json_string['sentence2']
            label=json_string['label']
            text=sentence1+"锟斤"+sentence2
            _mask = get_mask_idx(text, "锟斤")
            #text, label = l.strip().split('\t')
            if label=='neutral':
                label="并且"
            elif label=='entailment':
                label="所以"
            elif label=='contradiction':
                label="但是"
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
# 加载数据集
train_data = load_data('ready_data/ocnli/train_0.json')
valid_data = load_data('ready_data/ocnli/dev_0.json')
test_data = load_data('ready_data/ocnli/test_public.json')

# 模拟标注和非标注数据
train_frac = 1 # 0.01  # 标注数据的比例
print("0.length of train_data:",len(train_data)) # 16883
num_labeled = int(len(train_data) * train_frac)
# unlabeled_data = [(t, 2) for t, l in train_data[num_labeled:]]
train_data = train_data[:num_labeled]
print("1.num_labeled data used:",num_labeled," ;train_data:",len(train_data)) # 168

# train_data = train_data + unlabeled_data

# 建立分词器
unused_length=9 # 9
desc = ['[unused%s]' % i for i in range(1, unused_length)] # desc: ['[unused1]', '[unused2]', '[unused3]', '[unused4]', '[unused5]', '[unused6]', '[unused7]', '[unused8]', '[unused9]', '[unused10]']
desc_ids = [tokenizer.token_to_id(t) for t in desc] # 将token转化为id
label_list = ["并且", "所以", "但是"]
label_tokenid_list=[tuple(tokenizer.tokens_to_ids(x)) for x in label_list]
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
        for is_end, (text, _mask, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            target_ids = token_ids.copy()
            for _m, _l in zip(_mask, label):
                token_ids[_m] = tokenizer._token_mask_id
                target_ids[_m] = tokenizer.token_to_id(_l)
            token_ids = token_ids[:1] + desc_ids + token_ids[1:]
            target_ids = target_ids[:1] + desc_ids + target_ids[1:]
            segment_ids = [0] * len(desc_ids) + segment_ids
            source_ids = token_ids[:]
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
        mask[0:unused_length] += 1  #.1:unused_length 只优化id为1～8的token. 注：embedding第一位是[PAD]，跳过。          e.g. mask = array([[0.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.]]); 1-mask = array([[1.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[1.]])
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
train_model.compile(optimizer=Adam(3e-5)) # last lr:6e-4.可能是稍大好一点。模型：6e-4；finetuing常规学习率：2e-5.
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
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc: #  保存最好的模型，并记录最好的准确率
            self.best_val_acc = val_acc
            model.save_weights('best_model_bert_ptuning.weights')
        test_acc = evaluate(test_generator)
        print( # 打印准确率
            u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n' %
            (val_acc, self.best_val_acc, test_acc)
        )

# 对验证集进行验证
def evaluate(data):
    total, right = 0., 0.
    for x_true, _ in data:
        x_true, y_true = x_true[:2], x_true[2]
        y_pred = model.predict(x_true)
        mask_idx = np.where(x_true[0]==tokenizer._token_mask_id)[1].reshape(x_true[0].shape[0],2)
        y_pred = [pred[mask, label_tokenid_list] for pred, mask in zip(y_pred, mask_idx)]
        y_pred = [(pred[:,0]*pred[:,1]).argmax() for pred in y_pred]
        y_true = [label_tokenid_list.index(tuple(y[mask])) for mask, y in zip(mask_idx, y_true)]
        total += len(y_true)
        right += np.where(np.array(y_pred)==np.array(y_true))[0].shape[0]  # (y_true == y_pred).sum()
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