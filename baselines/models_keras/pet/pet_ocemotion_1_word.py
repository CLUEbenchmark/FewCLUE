#! -*- coding:utf-8 -*-
# 细粒度情感分析例子，利用MLM+PET 做 Zero-Shot/Few-Shot/Semi-Supervised Learning

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
import random
import os
import sys

# num_classes = 2
maxlen = 64
batch_size = 8 #32

taskname=sys.argv[1]
dataset_dir=sys.argv[2]
base_model_path=sys.argv[3]
output_model_path=sys.argv[4]
mode=sys.argv[5]

config_path = os.path.join(base_model_path, 'bert_config.json')
checkpoint_path =  os.path.join(base_model_path,'bert_model.ckpt')
dict_path = os.path.join(base_model_path,'vocab.txt')

label_list=['like','happiness','sadness','anger','disgust']
label2index={label:i for i,label in enumerate(label_list)} # label2index={'like':0,'happiness':1,'sadness':2,'anger':3,'disgust':4}

tokenizer = Tokenizer(dict_path, do_lower_case=True)

like_id = tokenizer.token_to_id(u'爱') # e.g. '[unused9]'. 将正向的token转化为id. 默认值：u'很'
happiness_id = tokenizer.token_to_id(u'开') # e.g. '[unused10]. 将负向的token转化为id. 默认值：u'不'
sadness_id = tokenizer.token_to_id(u'伤') # e.g. '[unused9]'. 将正向的token转化为id. 默认值：u'很'
anger_id = tokenizer.token_to_id(u'怒') # e.g. '[unused9]'. 将正向的token转化为id. 默认值：u'很'
disgust_id = tokenizer.token_to_id(u'恶') # e.g. '[unused9]'. 将正向的token转化为id. 默认值：u'很'

label2tokenid_dict={'like':like_id,'happiness':happiness_id,'sadness':sadness_id,'anger':anger_id,'disgust':disgust_id}
label_tokenid_list=[label2tokenid_dict[x] for x in label_list] # label_tokenid_list=[token_to_id(u'[unused10]'),(u'[unused12]') ,.....]


# 对应的任务描述
prefix = u'心的句子。' # u'很满意。'
mask_idx = 1

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
train_frac = 1 #
num_labeled = int(len(train_data) * train_frac)
unlabeled_data = [(t, 2) for t, l in train_data[num_labeled:]]
print("length of unlabeled_data0:",len(unlabeled_data))
train_data = train_data[:num_labeled]
train_data = train_data + unlabeled_data
print("length of train_data1:",len(train_data))

def random_masking(token_ids):
    """对输入进行随机mask
    在BERT中，mask比例为15%，相比auto-encoder，BERT只预测mask的token，而不是重构整个输入token。
    mask过程如下：80%机会使用[MASK]，10%机会使用原词,10%机会使用随机词。
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
    目前看只是将原始文本转换为token id
    负向样本（输入是一个[MASK]字符，输出是特定的字符。对于负样本，采用"不"，正样本，采用“很”）
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []
        for is_end, (text, label) in self.sample(random):
            if label != 2:
                text = prefix + text
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            if random:
                source_ids, target_ids = random_masking(token_ids)
            else:
                source_ids, target_ids = token_ids[:], token_ids[:]
            source_ids[mask_idx] = tokenizer._token_mask_id # 对输入进行mask
            target_id=label2tokenid_dict[label] # 得到目标值。 label2tokenid_dict  ={'like':like_id,'happiness':happiness_id,'sadness':sadness_id,'anger':anger_id,'disgust':disgust_id}
            target_ids[mask_idx] = target_id # 设置mask位置要预测的目标值
            batch_token_ids.append(source_ids)
            batch_segment_ids.append(segment_ids)
            batch_output_ids.append(target_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
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


# 加载预训练模型
model = build_transformer_model(
    config_path=config_path, checkpoint_path=checkpoint_path, with_mlm=True
)

# 训练用模型
y_in = keras.layers.Input(shape=(None,))
outputs = CrossEntropy(1)([y_in, model.output])

train_model = keras.models.Model(model.inputs + [y_in], outputs)
train_model.compile(optimizer=Adam(6e-4)) # 默认：1e-5；其他候选学习率：6e-4
train_model.summary()

# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        model.save_weights(os.path.join(output_model_path, 'mlm_model_pet.weights'))
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc: # #  保存最好的模型，并记录最好的准确率
            self.best_val_acc = val_acc
            model.save_weights(os.path.join(output_model_path, 'best_model_pet.weights'))
        test_acc = evaluate(test_generator)
        with open(os.path.join(output_model_path, "eval_accuracy.txt"), "a") as val_res:
          val_res.write(json.dumps({"eval_accuracy": val_acc}) + "\n")
        with open(os.path.join(output_model_path, "test_accuracy.txt"), "a") as test_res:
          test_res.write(json.dumps({"test_accuracy": test_acc}) + "\n")
        print(
            u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n' %
            (val_acc, self.best_val_acc, test_acc)
        )

def check_two_list(list_true, list_predict):
    """
    计算这两个列表，在相应位置上有多少个相同的值。
    :param list_true:
    :param list_predict:
    :return:
    """
    num_right_=0
    for index, v in enumerate(list_true):
        if v==list_predict[index]:
            num_right_+=1
    return num_right_

# 对验证集进行验证
def evaluate(data):
    total, right = 0., 0.
    for x_true, _ in data:
        x_true, y_true = x_true[:2], x_true[2]
        y_pred = model.predict(x_true)
        y_pred = y_pred[:, mask_idx,label_tokenid_list].argmax(axis=1) # O.K. 2分类时候：[neg_id, pos_id]. 多分类时，找到概率最大的字的所在的索引。
        y_true_=y_true[:, mask_idx]
        total += len(y_true)
        y_pred_right=[label2tokenid_dict[label_list[index]] for index in y_pred] # e.g. label_list[index]='like'. label2tokenid_dict[...]= token of label.
        if random.randint(0, 100) == 1:
            print("0:y_pred_original:",y_pred.shape,";y_pred:",y_pred)     # 0:y_pred : (16,) ;y_pred: [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
            print("1:y_true_:",y_true_.shape,";y_true_:",y_true_) # 1:y_true_: (16,) ;y_true_: [ 839  839  839 2626 2458 2458 2458  839 2458 2626 2458 2458 2584 2626 2458 2584]
            print("2:y_pred_right:",y_pred_right) # 1:y_true_: (16,) ;y_true_: [ 839  839  839 2626 2458 2458 2458  839 2458 2626 2458 2458 2584 2626 2458 2584]
        num_right=check_two_list(y_true_,y_pred_right)
        right += num_right #(y_true == y_pred).sum()
    return right / total


if __name__ == '__main__':
  if mode == "train":
    evaluator = Evaluator()

    train_model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=20, # TODO 1000,
        callbacks=[evaluator]
    )
  elif mode == "eval":
    model.load_weights(output_model_path + '/best_model_pet.weights')
    val_acc = evaluate(valid_generator)
    test_acc = evaluate(test_generator)
    print(
        u'val_acc: %.5f, test_acc: %.5f\n' %
        (val_acc, test_acc)
    )
