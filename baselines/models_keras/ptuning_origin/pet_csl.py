#! -*- coding:utf-8 -*-

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
import json
import sys
from modeling import tokenizer


maxlen = 256
batch_size = 16
unused_length=2



# 模板
# input_str_format = "{}，黴鹹{}几点内容" # 黴鹹：生僻字组合会被替换为 强调 or 提到，方便寻找mask index [7957, 7919]
input_str_format = "#"*unused_length+"黴鹹用{}概括{}" # 黴鹹：生僻字组合会被替换为 不能 or 可以，方便寻找mask index [7957, 7919]
labels = ["不能", "可以"]
label2words = {"0": "不能", "1":"可以"}

num_classes = 2
acc_list = []


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
            
            mask_idxs = [idx for idx, c in enumerate(content_ids) if c == 7957 and content_ids[idx+1] == 7919]
            mask_idxs.append(mask_idxs[0]+1)
            if "label" in l:
                label = l["label"]
            else:
                label = "0"
            D.append(((content, content_ids, segment_ids), label2words[label], mask_idxs))
    return D


path = '../../../datasets/csl'
data_num = sys.argv[1]

# 加载数据集
train_data = load_data('{}/train_{}.json'.format(path,data_num))
valid_data = load_data('{}/dev_{}.json'.format(path,data_num))
test_data = load_data('{}/test_public.json'.format(path))


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
                    #if tokenizer.id_to_token(source_ids[i]) not in ["黴", "鹹", "[MASK]"]:
                    #    print(content, tokenizer.id_to_token(source_ids[i]), mask_idx) # 确保mask掉了正确的token
                    source_ids[i] = tokenizer._token_mask_id # i: 7(mask1的index) ;j: 1093(农); i:8 (mask2的index) ;j: 689(业)
                    target_ids[i] = label_id_
                for i in range(1, unused_length+1):
                    source_ids[i] = i
                    target_ids[i] = i
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


from modeling import get_model
model, train_model = get_model(pattern_len=unused_length, trainable=True, lr=3e-5)


# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        # model.save_weights('pet_tnews_model.weights')
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:  # #  保存最好的模型，并记录最好的准确率
            self.best_val_acc = val_acc
            # model.save_weights('best_model_pet_sentencepair.weights')
        test_acc = evaluate(test_generator)
        print(
            u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n' %
            (val_acc, self.best_val_acc, test_acc)
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
        right += np.where(np.array(y_pred) == np.array(y_true))[0].shape[0]  # (y_true == y_pred).sum()
    return right / total
    #     pred_result_list += (y_true == y_pred).tolist()
    # return pred_result_list


if __name__ == '__main__':
    evaluator = Evaluator()

    train_model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator) * 5,
        epochs=10,
        callbacks=[evaluator]
    )
