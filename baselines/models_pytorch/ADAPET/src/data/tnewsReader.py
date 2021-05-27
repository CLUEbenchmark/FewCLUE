import os
import json
import random
import itertools
import numpy as np
import torch
from collections import defaultdict

from src.data.tokenize import tokenize_pet_txt, tokenize_pet_mlm_txt
from src.utils.util import device

class tnewsReader(object):
    '''
    TnewsReader reads tnews dataset
    '''


    def __init__(self, config, tokenizer,dataset_num=None):
        self.config = config
        self.tokenizer = tokenizer
        self.dataset_num = dataset_num
        self.num_lbl = 15 # number of labels，即有多少个标签

        # modify
        # self.pet_labels = [["很", "不"]]
        self.pet_labels = [["故事", "文化","娱乐","体育","财经","房产","汽车","教育","科技","军事","旅游","国际","股票","农业","电竞"]]
        self.pet_patterns = [["[SENTENCE]","？这是一个关于{}的新闻[SEP]".format(self.tokenizer.mask_token), ""],
                             ["这是一个{}的新闻？","[SENTENCE][SEP]".format(self.tokenizer.mask_token), ""],
                             ["下面这个是{}新闻：", "[SENTENCE][SEP]".format(self.tokenizer.mask_token), ""]]
        #返回两个集合a,b中元素的笛卡尔乘积
        self.pet_pvps = list(itertools.product(self.pet_patterns, self.pet_labels))
        self._num_pets = len(self.pet_pvps)
        self._pet_names = ["PET{}".format(i+1) for i in range(self._num_pets)]

        self.list_true_lbl = []
        # line = {"label": 104, "label_desc": "news_finance", "sentence": "怎么问别人借钱？", "keywords": "", "id": 27956}

        # self.dict_lbl_2_idx = {'Positive': 0, 'Negative': 1}
        self.dict_lbl_2_idx= {"100": 0, "101":1, "102": 2, "103": 3, "104": 4, "106": 5,"107":6,
                              "108": 7,"109":8, "110": 9, "112": 10, "113": 11, "114": 12,"115":13, "116": 14}

        print("self.dict_lbl_2_idx:",self.dict_lbl_2_idx)

    def _get_file(self, split):
        '''
        设置训练/验证/测试集的位置。Get filename of split

        :param split:
        :return:
        '''
        file=''
        if split.lower() == "train":
            file = os.path.join("../../../datasets", "tnews", "train_{}.json".format(self.dataset_num))
        elif split.lower() == "dev":
            file = os.path.join("../../../datasets", "tnews", "dev_{}.json".format(self.dataset_num))
        elif split.lower() == "test":
            file = os.path.join("../../../datasets", "tnews", "test_public.json")
        return file

    def get_num_lbl_tok(self): #
        """
        标签字符数量。如tnews就是2；eprstmt就是1。
        :return:
        """
        return 2

    def read_dataset(self, split=None, is_eval=False):
        '''
        读取数据集。Read the dataset
        :param split: partition of the dataset
        :param is_eval:

        :return:
        '''
        file = self._get_file(split)

        data = []

        with open(file, 'r') as f_in:
            for i, line in enumerate(f_in.readlines()):
                print('1:',i,line)
                # line = {"label": 104, "label_desc": "news_finance", "sentence": "怎么问别人借钱？", "keywords": "", "id": 27956}
                json_string = json.loads(line)

                dict_input = {}
                dict_input["sentence"] = json_string["sentence"]
                dict_input["id"] = json_string["id"]

                dict_output = {}
                if "label" in json_string:
                    # print('json_string["label"]:',json_string["label"])
                    dict_output["lbl"] = self.dict_lbl_2_idx[str(json_string["label"])]
                    print('2:',"read_dataset.lbl:",dict_output["lbl"])
                else:
                    1/0
                    # print("【ERROR】.获取标签失败。原始数据：",line)
                    # dict_output["lbl"] = -1

                dict_input_output = {"input": dict_input, "output": dict_output}
                data.append(dict_input_output)
        return data

    @property
    def pets(self):
        return self._pet_names

    def prepare_pet_batch(self, batch, mode="PET1"):
        '''
        Prepare for train

        :param batch:
        :return:
        '''
        list_sentence = batch["input"]["sentence"]

        list_input_ids = []
        # modify
        bs = len(batch["input"]["sentence"])
        list_mask_idx = np.ones((bs, self.get_num_lbl_tok())) * self.config.max_text_length

        pattern, label = self.pet_pvps[self._pet_names.index(mode)]

        for b_idx, sent in enumerate(list_sentence):
            mask_txt_split_tuple = []

            txt_trim = -1

            for idx, txt_split in enumerate(pattern):
                mask_txt_split_inp = txt_split.replace("[SENTENCE]", sent)
                mask_txt_split_tuple.append(mask_txt_split_inp)

                # Trim the paragraph
                if "[SENTENCE]" in txt_split:
                    txt_trim = idx
            input_ids, mask_idx = tokenize_pet_txt(self.tokenizer, self.config, mask_txt_split_tuple[0], mask_txt_split_tuple[1],
                                                   mask_txt_split_tuple[2], mask_txt_split_tuple[0], mask_txt_split_tuple[1], mask_txt_split_tuple[2], txt_trim)
            list_input_ids.append(input_ids)
            list_mask_idx[b_idx,:self.get_num_lbl_tok()] = range(mask_idx, mask_idx+self.get_num_lbl_tok())

        return torch.tensor(list_input_ids).to(device), torch.tensor(list_mask_idx).to(device), label

    def prepare_pet_mlm_batch(self, batch, mode="PET1"):

        '''
        Prepare for train

        :param batch:
        :return:
        '''

        list_sentence = batch["input"]["sentence"]

        bs = len(batch["input"]["sentence"])

        prep_lbl = np.random.randint(self.num_lbl, size=bs)
        tgt = torch.from_numpy(prep_lbl).long() == batch["output"]["lbl"]

        pattern, label = self.pet_pvps[self._pet_names.index(mode)]

        list_orig_input_ids = []
        list_masked_input_ids = []

        for b_idx, (p, lbl) in enumerate(zip(list_sentence, prep_lbl)):
            txt_split_tuple = []

            txt_trim = -1

            for idx, txt_split in enumerate(pattern):
                txt_split_inp = txt_split.replace("[SENTENCE]", p).replace("[MASK]", label[lbl])
                txt_split_tuple.append(txt_split_inp)

                # Trim the paragraph
                if "[SENTENCE]" in txt_split:
                    txt_trim = idx

            orig_input_ids, masked_input_ids, mask_idx = tokenize_pet_mlm_txt(self.tokenizer, self.config, txt_split_tuple[0], txt_split_tuple[1], txt_split_tuple[2], txt_trim)
            list_orig_input_ids.append(orig_input_ids)
            list_masked_input_ids.append(masked_input_ids)

        return torch.tensor(list_orig_input_ids).to(device),  torch.tensor(list_masked_input_ids).to(device), prep_lbl, tgt.to(device)

    def prepare_eval_pet_batch(self, batch, mode="PET1"):
        return self.prepare_pet_batch(batch, mode)

    def store_test_lbl(self, list_idx, pred_lbl, true_lbl, logits):
        self.list_true_lbl.append(pred_lbl)

    def flush_file(self, write_file):
        self.list_true_lbl = torch.cat(self.list_true_lbl, dim=0).cpu().int().numpy().tolist()

        read_file = self._get_file("test")

        with open(read_file, 'r') as f_in:
            for ctr, line in enumerate(f_in.readlines()):
                answer_dict = {}
                answer_dict["idx"] = ctr
                pred_lbl = self.list_true_lbl[ctr]

                if pred_lbl == 0:
                    answer = "true"
                else:
                    answer = "false"
                answer_dict["label"] = answer

                write_file.write(json.dumps(answer_dict) + "\n")
