import os
import json
import random
import itertools
import numpy as np
import torch
from collections import defaultdict

from src.data.tokenize import tokenize_pet_txt, tokenize_pet_mlm_txt
from src.utils.util import device

class iflytekReader(object):
    '''
    iflytekReader reads iflytek dataset
    '''

    def __init__(self, config, tokenizer,dataset_num=None):
        self.config = config
        self.tokenizer = tokenizer
        self.dataset_num = dataset_num
        self.num_lbl = 119 # number of labels，即有多少个标签

        # modify
        # self.pet_labels = [["很", "不"]]
        self.pet_labels = [ ['银行', '社区', '电商', '支付', '经营', '卡牌', '借贷', '驾校', '理财', '职考', '新闻', '旅游', '交通', '魔幻', '医疗', '影像', '动作', ' 工具', '体育', '小说', '运动', '相机', '工具', '快递', '教育', '股票', '菜谱', '行车', '仙侠', '亲子', '购物', '射击', '漫画', '小学', '同城', '成人', '求职', '电子', '艺术', ' 赚钱', '约会', '经营', '兼职', '视频', '音乐', '英语', '棋牌', '摄影', '养生', '办公', '政务', '视频', '论坛', '彩票', '直播', '其他', '休闲', '策略', '通讯', '买车', '违章', ' 地图', '民航', '电台', '语言', '搞笑', '婚恋', '超市', '养车', '杂志', '在线', '家政', '影视', '装修', '资讯', '社交', '餐饮', '美颜', '挂号', '飞行', '预定', '票务', '笔记', ' 买房', '外卖', '母婴', '打车', '情侣', '日程', '租车', '博客', '百科', '绘画', '铁路', '生活', '租房', '酒店', '保险', '问答', '收款', '竞技', '唱歌', '技术', '减肥', '工作', ' 团购', '记账', '女性', '公务', '二手', '美妆', '汽车', '行程', '免费', '教辅', '两性', '出国', '婚庆', '民宿']]


        self.pet_patterns = [["[SENTENCE]","？ 这是一个关于{}的应用[SEP]".format(self.tokenizer.mask_token), ""],
                              ["[SENTENCE]","？ 所以这是一个关于{}的应用。[SEP]".format(self.tokenizer.mask_token), ""],
                              ["这是一个关于{}的应用 ？ 因为描述是这样的：","[SENTENCE][SEP]".format(self.tokenizer.mask_token), ""]]

        #返回两个集合a,b中元素的笛卡尔乘积
        self.pet_pvps = list(itertools.product(self.pet_patterns, self.pet_labels))
        self._num_pets = len(self.pet_pvps)
        self._pet_names = ["PET{}".format(i+1) for i in range(self._num_pets)]

        self.list_true_lbl = []
        # {"label": 92, "label_des": "支付", "sentence": "为中小商铺打造的手机支付缴费助手。", "id": 7311}

        #  # self.dict_lbl_2_idx = {'Positive': 0, 'Negative': 1}
        # self.dict_lbl_2_idx= {"100": 0, "101":1, "102": 2, "103": 3, "104": 4, "106": 5,"107":6,
        #                      "108": 7,"109":8, "110": 9, "112": 10, "113": 11, "114": 12,"115":13, "116": 14}
        # self.dict_lbl_2_idx={0: 0,100: 1,101: 2,102: 3,103: 4,104: 5,105: 6,106: 7,107: 8,108: 9,109: 10,10: 11,110: 12,
        #           111: 13,112: 14,113: 15,114: 16,115: 17,116: 18,117: 19,118: 20,11: 21,12: 22,13: 23,14: 24,15: 25, 16: 26,17: 27,18: 28,
        #          19: 29,1: 30,20: 31,21: 32,22: 33,23: 34,24: 35,25: 36,26: 37,27: 38,28: 39,29: 40, 2: 41, 30: 42, 31: 43, 32: 44, 33: 45, 34: 46,
        #         35: 47,36: 48,37: 49,38: 50,39: 51,3: 52,40: 53,41: 54,42: 55,43: 56,44: 57,45: 58,46: 59,47: 60,48: 61,49: 62,4: 63,50: 64,51: 65,
        #        52: 66,53: 67,54: 68,55: 69,56: 70, 57: 71,58: 72,59: 73,5: 74,60: 75,61: 76,62: 77,63: 78,64: 79,65: 80,66: 81,67: 82,68: 83,69: 84,
        #       6: 85,70: 86,71: 87,72: 88,73: 89,74: 90,75: 91,76: 92,77: 93,78: 94,79: 95,7: 96,80: 97,81: 98,82: 99,83: 100,84: 101,85: 102,86: 103,
        #       87: 104,88: 105,89: 106,8: 107,90: 108,91: 109,92: 110,93: 111,94: 112,95: 113,96: 114,97: 115,98: 116, 99: 117,9: 118}
        self.dict_lbl_2_idx= {0: 86, 100: 77, 101: 15, 102: 47, 103: 21, 104: 92, 105: 109, 106: 2, 107: 105, 108: 84, 109: 81, 10: 1, 110: 67, 111: 30, 112: 82, 113: 49, 114: 88, 115: 107, 116: 4, 117: 99, 118: 55, 11: 39, 12: 13, 13: 28, 14: 5, 15: 79, 16: 31, 17: 56, 18: 16, 19: 18, 1: 61, 20: 46, 21: 4, 22: 57, 23: 100, 24: 17, 25: 40, 26: 58, 27: 104, 28: 52, 29: 66, 2: 113, 30: 87, 31: 75, 32: 94, 33: 90, 34: 10, 35: 32, 36: 19, 37: 102, 38: 114, 39: 98, 3: 89, 40: 65, 41: 69, 42: 91, 43: 72, 44: 36, 45: 42, 46: 43, 47: 43, 48: 44, 49: 54, 4: 34, 50: 63, 51: 101, 52: 115, 53: 33, 54: 9, 55: 108, 56: 45, 57: 70, 58: 24, 59: 35, 5: 23, 60: 38, 61: 64, 62: 11, 63: 80, 64: 62, 65: 93, 66: 96, 67: 112, 68: 118, 69: 116, 6: 117, 70: 17, 71: 29, 72: 85, 73: 7, 74: 60, 75: 111, 76: 59, 77: 68, 78: 27, 79: 95, 7: 71, 80: 83, 81: 73, 82: 37, 83: 78, 84: 48, 85: 14, 86: 103, 87: 110, 88: 26, 89: 76, 8: 12, 90: 74, 91: 20, 92: 3, 93: 97, 94: 25, 95: 6, 96: 8, 97: 53, 98: 106, 99: 0, 9: 50}
        # print("self.dict_lbl_2_idx:",self.dict_lbl_2_idx)

    def _get_file(self, split):
        '''
        设置训练/验证/测试集的位置。Get filename of split

        :param split:
        :return:
        '''
        file=''
        if split.lower() == "train":
            file = os.path.join("../../../datasets", "iflytek", "train_{}.json".format(self.dataset_num))
        elif split.lower() == "dev":
            file = os.path.join("../../../datasets", "iflytek", "dev_{}.json".format(self.dataset_num))
        elif split.lower() == "test":
            file = os.path.join("../../../datasets", "iflytek", "test_public.json")
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
                # {"label": 92, "label_des": "支付", "sentence": "为中小商铺打造的手机支付缴费助手。", "id": 7311}
                json_string = json.loads(line)

                dict_input = {}
                dict_input["sentence"] = json_string["sentence"]
                dict_input["id"] = str(json_string["id"])

                dict_output = {}
                # print("sentence1:",json_string["sentence"])
                if len(json_string["sentence"])<3: continue
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