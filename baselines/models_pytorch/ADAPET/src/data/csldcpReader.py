import os
import json
import random
import itertools
import numpy as np
import torch
from collections import defaultdict

from src.data.tokenize import tokenize_pet_txt, tokenize_pet_mlm_txt
from src.utils.util import device

class csldcpReader(object):
    '''
    iflytekReader reads iflytek dataset
    '''

    def __init__(self, config, tokenizer,dataset_num=None):
        self.config = config
        self.tokenizer = tokenizer
        self.dataset_num = dataset_num

        # modify
        # self.pet_labels = [["很", "不"]]
        self.pet_labels = [
            ['材料', '作物', '口腔', '药学', '教育', '水利', '理经', '食品', '兽医', '体育', '核能', '力学', '园艺', '水产', '法学', '地质', '能源', '农林',
             '通信', '情报', '政治', '电气', '海洋', '民族', '航空', '化工', '哲学', '卫生', '艺术', '农工', '船舶', '计科', '冶金', '交通', '动力', '纺织',
             '建筑', '环境', '公管', '数学', '物理', '林业', '心理', '历史', '工商', '应经', '中医', '天文', '机械', '土木', '光学', '地理', '农资', '生物',
             '兵器', '矿业', '大气', '医学', '电子', '测绘', '控制', '军事', '语言', '新闻', '社会', '地球', '植物']
        ]
        self.num_lbl = len(self.pet_labels[0]) # number of labels，即有多少个标签


        self.pet_patterns = [["[SENTENCE]","？ 这是一个关于{}的文章[SEP]".format(self.tokenizer.mask_token), ""],
                              ["[SENTENCE]","？ 所以这是一个关于{}的文章。[SEP]".format(self.tokenizer.mask_token), ""],
                              ["这是一个关于{}的文章？因为描述是这样的：","[SENTENCE][SEP]".format(self.tokenizer.mask_token), ""]]

        #返回两个集合a,b中元素的笛卡尔乘积
        self.pet_pvps = list(itertools.product(self.pet_patterns, self.pet_labels))
        self._num_pets = len(self.pet_pvps)
        self._pet_names = ["PET{}".format(i+1) for i in range(self._num_pets)]

        self.list_true_lbl = []
        # {"label": 92, "label_des": "支付", "sentence": "为中小商铺打造的手机支付缴费助手。", "id": 7311}

        #  # self.dict_lbl_2_idx = {'Positive': 0, 'Negative': 1}
        # self.dict_lbl_2_idx= {"100": 0, "101":1, "102": 2, "103": 3, "104": 4, "106": 5,"107":6,
        #                      "108": 7,"109":8, "110": 9, "112": 10, "113": 11, "114": 12,"115":13, "116": 14}
        self.dict_lbl_2_idx= {'材料科学与工程': 0, '作物学': 1, '口腔医学': 2, '药学': 3, '教育学': 4, '水利工程': 5, '理论经济学': 6, '食品科学与工程': 7, '畜牧学/兽医学': 8, '体育学': 9, '核科学与技术': 10, '力学': 11, '园艺学': 12, '水产': 13, '法学': 14, '地质学/地质资源与地质工程': 15, '石油与天然气工程': 16, '农林经济管理': 17, '信息与通信工程': 18, '图书馆、情报与档案管理': 19, '政治学': 20, '电气工程': 21, '海洋科学': 22, '民族学': 23, '航空宇航科学与技术': 24, '化学/化学工程与技术': 25, '哲学': 26, '公共卫生与预防医学': 27, '艺术学': 28, '农业工程': 29, '船舶与海洋工程': 30, '计算机科学与技术': 31, '冶金工程': 32, '交通运输工程': 33, '动力工程及工程热物理': 34, '纺织科学与工程': 35, '建筑学': 36, '环境科学与工程': 37, '公共管理': 38, '数学': 39, '物理学': 40, '林学/林业工程': 41, '心理学': 42, '历史学': 43, '工商管理': 44, '应用经济学': 45, '中医学/中药学': 46, '天文学': 47, '机械工程': 48, '土木工程': 49, '光学工程': 50, '地理学': 51, '农业资源利用': 52, '生物学/生物科学与工程': 53, '兵器科学与技术': 54, '矿业工程': 55, '大气科学': 56, '基础医学/临床医学': 57, '电子科学与技术': 58, '测绘科学与技术': 59, '控制科学与工程': 60, '军事学': 61, '中国语言文学': 62, '新闻传播学': 63, '社会学': 64, '地球物理学': 65, '植物保护': 66}
        # print("self.dict_lbl_2_idx:",self.dict_lbl_2_idx)

    def _get_file(self, split):
        '''
        设置训练/验证/测试集的位置。Get filename of split

        :param split:
        :return:
        '''
        file=''
        if split.lower() == "train":
            file = os.path.join("../../../datasets", "csldcp", "train_{}.json".format(self.dataset_num))
        elif split.lower() == "dev":
            file = os.path.join("../../../datasets", "csldcp", "dev_{}.json".format(self.dataset_num))
        elif split.lower() == "test":
            file = os.path.join("../../../datasets", "csldcp", "test_public.json")
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
                # print('1:',i,line)
                # {"content": "介绍可视化模拟技术与空间信息技术的应用领域、集成模式及其在矿山生产与管理中的应用现状与前景.", "label": "矿业工程", "id": 587}
                json_string = json.loads(line)

                dict_input = {}
                dict_input["sentence"] = json_string["content"]
                dict_input["id"] = str(json_string["id"])

                dict_output = {}
                # print("sentence1:",json_string["sentence"])
                if len(json_string["content"])<3: continue
                if "label" in json_string:
                    # print('json_string["label"]:',json_string["label"])
                    dict_output["lbl"] = self.dict_lbl_2_idx[json_string["label"]]
                    # print('2:',"read_dataset.lbl:",dict_output["lbl"])
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
                # print("1.txt_split:",txt_split,"==;2.p:",str(p),"==;3.label:",label,"==;4.lbl:",lbl)
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