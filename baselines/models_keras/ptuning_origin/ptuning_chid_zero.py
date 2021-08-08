#! -*- coding:utf-8 -*-
"""
chid  只有zero shot
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
import numpy as np
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from modeling import tokenizer


def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        for jj, l in enumerate(f):
            json_string = json.loads(l.strip())
            content = json_string['content']
            index = content.index("#idiom#")
            candidates = json_string['candidates']  # list
            answer = json_string['answer']
            label = (candidates, index, answer)
            D.append((content, label))
    return D


path = '../../../datasets/chid'
# 加载数据集
test_data = load_data('../../../datasets/chid/test_public.json')


class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        batch_token_ids, batch_labels = [], []
        batch_token_ids_all_labels = [[] for _ in range(7)]
        batch_segment_ids_all_labels = [[] for _ in range(7)]
        for is_end, (text, label) in self.sample(random):
            token_ids = []
            candidates, index, answer = label
            for ind, cand in enumerate(candidates):
                tem_text = text.replace("#idiom#", cand)
                tem_token_ids, segment_ids = tokenizer.encode(tem_text, maxlen=512)
                batch_token_ids_all_labels[ind].append(tem_token_ids)
                batch_segment_ids_all_labels[ind].append(segment_ids)
                index = len(tem_token_ids)
            batch_labels.append((index, answer))

            batch_token_ids.append(token_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                for index, tem_token_ids in enumerate(batch_token_ids_all_labels):
                    tem_token_ids = sequence_padding(tem_token_ids)
                    segment_ids = sequence_padding(batch_segment_ids_all_labels[index])
                    batch_token_ids_all_labels[index] = tem_token_ids
                    batch_segment_ids_all_labels[index] = segment_ids
                yield (batch_token_ids_all_labels, batch_segment_ids_all_labels), batch_labels

                batch_token_ids, batch_labels = [], []
                batch_token_ids_all_labels = [[] for _ in range(7)]
                batch_segment_ids_all_labels = [[] for _ in range(7)]


from modeling import get_model
model, train_model = get_model(pattern_len=1)

# 转换数据集
batch_size=64
test_generator = data_generator(test_data, batch_size)


def evaluate_gpt(data, model):
    """
    评估准确率，每个label的字数要一致
    Args:
        data:
        model:
        labels:[[label_1_word1_id, label_1_word2_id,...],[label_2_word1_id, label_2_word2_id, ...],...]
    Returns:

    """
    # labels_num, label_word_len = len(labels), len(labels[0])
    # labels = np.array(labels)
    total, right = 0., 0.
    for x_trues, batch_labels in data:
        y_preds = []
        for x_true in zip(x_trues[0], x_trues[1]):
            y_pred = model.predict(x_true)
            y_preds.append(y_pred)
        x_trues = x_trues[0]
        for index, (first_index, labels_id) in enumerate(batch_labels):
            total += 1
            x_len = len(np.trim_zeros(x_trues[0][index]))
            x_len = first_index - 1
            ys, xs = [], []
            for y_pred, x in zip(y_preds, x_trues):
                y = y_pred[index]
                y = y[:x_len]
                x = x[index][:x_len]
                ys.append(y)
                xs.append(x)
            prob_ys = []
            for x, y in zip(xs, ys):
                prob_y = 1.0
                # todo 直接从1开始计算整个句子的概率
                for word_index in range(1, x_len):
                    word_id = x[word_index]
                    #prob_y += np.log(y[word_index - 1][word_id])
                    prob_y *= y[word_index][word_id]
                prob_ys.append(prob_y)
            print(prob_ys, labels_id)
            y_index = np.array(prob_ys).argmax()
            if y_index == labels_id:
                right += 1

    return right / total


if __name__ == '__main__':

    test_acc = evaluate_gpt(test_generator, model)
    print("test_acc:", test_acc)
