#! -*- coding:utf-8 -*-
# 细粒度情感分析例子（5个类别，使用字作为标签。具体是什么字是自己学的。），利用MLM+P-tuning

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
import random
import os
import sys

maxlen = 128
batch_size = 8

taskname=sys.argv[1]
dataset_dir=sys.argv[2]
base_model_path=sys.argv[3]
output_model_path=sys.argv[4]
mode=sys.argv[5]

config_path = os.path.join(base_model_path, 'bert_config.json')
checkpoint_path =  os.path.join(base_model_path,'bert_model.ckpt')
dict_path = os.path.join(base_model_path,'vocab.txt')

# 加载数据的方法
# {"id": 16, "content": "你也不用说对不起，只是，，，，若相惜", "label": "sadness"}
label_list=['like','happiness','sadness','anger','disgust'] ####
label2index={label:i for i,label in enumerate(label_list)} ####
# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

like_id = tokenizer.token_to_id(u'[unused10]') # e.g. '[unused9]'. 将正向的token转化为id. 默认值：u'很'
happiness_id = tokenizer.token_to_id(u'[unused11]') # e.g. '[unused10]. 将负向的token转化为id. 默认值：u'不'
sadness_id = tokenizer.token_to_id(u'[unused12]') # e.g. '[unused9]'. 将正向的token转化为id. 默认值：u'很'
anger_id = tokenizer.token_to_id(u'[unused13]') # e.g. '[unused9]'. 将正向的token转化为id. 默认值：u'很'
disgust_id = tokenizer.token_to_id(u'[unused14]') # e.g. '[unused9]'. 将正向的token转化为id. 默认值：u'很'

label2tokenid_dict={'like':like_id,'happiness':happiness_id,'sadness':sadness_id,'anger':anger_id,'disgust':disgust_id} ####
label_tokenid_list=[label2tokenid_dict[x] for x in label_list] # label_tokenid_list=[token_to_id(u'[unused10]'),(u'[unused12]') ,.....]


# 对应的任务描述
mask_idx = 1 #5
unused_length=9 # 9
desc = ['[unused%s]' % i for i in range(1, unused_length)] # desc: ['[unused1]', '[unused2]', '[unused3]', '[unused4]', '[unused5]', '[unused6]', '[unused7]', '[unused8]', '[unused9]', '[unused10]']
desc.insert(mask_idx - 1, '[MASK]')            # desc: ['[MASK]', '[unused1]', '[unused2]', '[unused3]', '[unused4]', '[unused5]', '[unused6]', '[unused7]', '[unused8]', '[unused9]', '[unused10]
desc_ids = [tokenizer.token_to_id(t) for t in desc] # 将token转化为id
# neg_id = tokenizer.token_to_id(u'不')

def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        for jj,l in enumerate(f):
            json_string=json.loads(l.strip())
            content=json_string['content']
            label=json_string['label']
            if label in ['surprise','fear']: # 去掉类别太少的这两个样本
                continue
            D.append((content, label))
    return D


# 加载数据集
train_data = load_data(os.path.join(dataset_dir, 'train_32.json'))
valid_data = load_data(os.path.join(dataset_dir, 'dev_32.json'))
test_data = load_data(os.path.join(dataset_dir, 'test_public.json'))

# 模拟标注和非标注数据
train_frac = 1# 0.01  # 标注数据的比例
print("0.length of train_data:",len(train_data)) # 16883
num_labeled = int(len(train_data) * train_frac)
# unlabeled_data = [(t, 2) for t, l in train_data[num_labeled:]]
train_data = train_data[:num_labeled]
print("1.num_labeled data used:",num_labeled," ;train_data:",len(train_data)) # 168

# train_data = train_data + unlabeled_data


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
    def __iter__(self, random=False):
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
            # if label == 0: # 负样本
            #    source_ids[mask_idx] = tokenizer._token_mask_id
            #    target_ids[mask_idx] = neg_id
            # elif label == 1: # 正向样本
            #     source_ids[mask_idx] = tokenizer._token_mask_id
            #    target_ids[mask_idx] = pos_id
            ############################################################
            source_ids[mask_idx] = tokenizer._token_mask_id
            # print("label2tokenid_dict:,label2tokenid_dict,label:",label). e.g. {'like':like_id,'happiness':happiness_id,'sadness':sadness_id,'anger':anger_id,'disgust':disgust_id}
            target_id=label2tokenid_dict[label] # label2tokenid_dict:
            target_ids[mask_idx] = target_id
            ############################################################
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
        mask[1:unused_length+5] += 1  # TODO  unused_length+5. 只优化id为1～8的token. 注：embedding第一位是[PAD]，跳过。加上多个label的学习 e.g. mask = array([[0.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.]]); 1-mask = array([[1.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[1.]])
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
train_model.compile(optimizer=Adam(10e-4)) # 默认：6e-4. 3e-5学习率太小了
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
            model.save_weights(os.path.join(output_model_path, 'best_model_bert_ptuning.weights'))
        test_acc = evaluate(test_generator)
        with open(os.path.join(output_model_path, "eval_accuracy.txt"), "a") as val_res:
          val_res.write(json.dumps({"eval_accuracy": val_acc}) + "\n")
        with open(os.path.join(output_model_path, "test_accuracy.txt"), "a") as test_res:
          test_res.write(json.dumps({"test_accuracy": test_acc}) + "\n")
        print( # 打印准确率
            u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n' %
            (val_acc, self.best_val_acc, test_acc)
        )

def check_two_list(list_true, list_predict):
    num_right_=0
    for index, v in enumerate(list_true):
        if v==list_predict[index]:
            num_right_+=1
    return num_right_

# 对验证集进行验证
def evaluate(data):
    total, right = 0., 0.
    for x_true, _ in data: # x_true: batch_token_ids, batch_segment_ids, batch_output_ids
        x_true, y_true = x_true[:2], x_true[2] # x_true: batch_token_ids, batch_segment_ids; y_true: batch_output_ids
        # print("evaluate.y_true:",y_true.shape,";y_true:",y_true,";\nx_true.shape:",x_true.shape,";x_true:",x_true) # y_true：(8, 137+9) =(batch_size, batch_size+ length_learnable)。 128是序列长度；9是可学习参数的数量。
        y_pred = model.predict(x_true) # 给定一段文本对应的表示，做预测
        # print("0.y_pred.shape:",y_pred.shape,";y_pred:",y_pred) # (8, 10, 21128) = (batch_size, learnable_parameter_length). [[[2.22174251e-07 5.57889454e-02 1.27458700e-03 ... 2.99343867e-07 ...]]]
        # label_tokenid_list=[token_to_id(u'[unused10]'),token_to_id(u'[unused12]') ,.....]
        y_pred = y_pred[:, mask_idx, label_tokenid_list].argmax(axis=1) # O.K.只关心特定位置（如：[MASK]所在位置）的输出。[neg_id, pos_id]
        # print("1.y_pred.shape:", y_pred.shape, ";y_pred:", y_pred) # y_pred.shape: (8,)      ;y_pred: [1 2 2 1 2 2 2 2]
        transform_index = unused_length + 1
        y_true_=y_true[:, mask_idx]
        y_true_=y_true_-transform_index
        num_right=check_two_list(y_true_,y_pred)

        if random.randint(0,100)==1:
            print("y_true_:",y_true_) # y_true[:, mask_idx]: [11 12 11 10 12 13 12 11]
            print("num_right:", num_right)
        #y_true = (y_true[:, mask_idx] == right_id).astype(int) # pos_id

        total +=len(y_true)
        right += num_right # (y_true == y_pred).sum()
    return right / total


if __name__ == '__main__':

  if mode == "train":
    evaluator = Evaluator()

    train_model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator) * 50,
        epochs=20,
        callbacks=[evaluator]
    )

  elif mode == "eval":
    model.load_weights(os.path.join(output_model_path, 'best_model_bert_ptuning.weights'))
    val_acc = evaluate(valid_generator)
    test_acc = evaluate(test_generator)
    print(
        u'val_acc: %.5f, test_acc: %.5f\n' %
        (val_acc, test_acc)
    )
