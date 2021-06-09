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
        self.dict_lbl_2_idx = {'银行': 0, '社区服务': 1, '电商': 2, '支付': 3, '经营养成': 4, '卡牌': 5, '借贷': 6, '驾校': 7, '理财': 8, '职考': 9, '新闻': 10,
         '旅游资讯': 11, '公共交通': 12, '魔幻': 13, '医疗服务': 14, '影像剪辑': 15, '动作类': 16, '工具': 17, '体育竞技': 18, '小说': 19,
         '运动健身': 20, '相机': 21, '辅助工具': 22, '快递物流': 23, '高等教育': 24, '股票': 25, '菜谱': 26, '行车辅助': 27, '仙侠': 28, '亲子儿童': 29,
         '购物咨询': 30, '射击游戏': 31, '漫画': 32, '中小学': 33, '同城服务': 34, '成人教育': 35, '求职': 36, '电子产品': 37, '艺术': 38, '薅羊毛': 39,
         '约会社交': 40, '经营': 41, '兼职': 42, '短视频': 43, '音乐': 44, '英语': 45, '棋牌中心': 46, '摄影修图': 47, '养生保健': 48, '办公': 49,
         '政务': 50, '视频': 51, '论坛圈子': 52, '彩票': 53, '直播': 54, '其他': 55, '休闲益智': 56, '策略': 57, '即时通讯': 58, '汽车交易': 59,
         '违章': 60, '地图导航': 61, '民航': 62, '电台': 63, '语言(非英语)': 64, '搞笑': 65, '婚恋社交': 66, '社区超市': 67, '日常养车': 68,
         '杂志': 69, '视频教育': 70, '家政': 71, '影视娱乐': 72, '装修家居': 73, '体育咨讯': 74, '社交工具': 75, '餐饮店': 76, '美颜': 77,
         '问诊挂号': 78, '飞行空战': 79, '综合预定': 80, '电影票务': 81, '笔记': 82, '买房': 83, '外卖': 84, '母婴': 85, '打车': 86, '情侣社交': 87,
         '日程管理': 88, '租车': 89, '微博博客': 90, '百科': 91, '绘画': 92, '铁路': 93, '生活社交': 94, '租房': 95, '酒店': 96, '保险': 97,
         '问答交流': 98, '收款': 99, 'MOBA': 100, 'K歌': 101, '技术': 102, '减肥瘦身': 103, '工作社交': 104, '团购': 105, '记账': 106,
         '女性': 107, '公务员': 108, '二手': 109, '美妆美业': 110, '汽车咨询': 111, '行程管理': 112, '免费WIFI': 113, '教辅': 114, '成人': 115,
         '出国': 116, '婚庆': 117, '民宿短租': 118}
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
                dict_input["id"] =json_string["id"] #  str(

                dict_output = {}
                # print("sentence1:",json_string["sentence"])
                if len(json_string["sentence"])<3: continue
                if "label" in json_string:
                    # print('json_string["label"]:',json_string["label"])
                    dict_output["lbl"] = self.dict_lbl_2_idx[json_string["label_des"]]
                    print('2:',"read_dataset.lbl:",dict_output["lbl"])
                    assert (dict_output["lbl"] < 119 and dict_output["lbl"]>=0)
                else:
                    1/0
                    break
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