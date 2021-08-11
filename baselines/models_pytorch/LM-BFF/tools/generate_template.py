import transformers
from transformers import T5ForConditionalGeneration, T5Tokenizer
import argparse
import torch
import os
from tqdm import tqdm
import json
import argparse
import pandas as pd

def get_text(template, input_text_tuple, label, tokenizer, mapping):
    def enc(text):
        return tokenizer.encode(text, add_special_tokens=False)
    special_token_mapping = {'cls': tokenizer.cls_token_id, 'mask': tokenizer.mask_token_id, 'sep': tokenizer.sep_token_id, 'sep+': tokenizer.sep_token_id}
    for i in range(10):
        # NOTE: 改成 Chinese T5 的形式
        # special_token_mapping["<extra_id_%d>" % (i)] = tokenizer._convert_token_to_id("<extra_id_%d>" % (i))
        special_token_mapping["extra%d" % (i)] = tokenizer._convert_token_to_id("extra%d" % (i))
    template_list = template.split('*')
    input_ids = []
    for part in template_list:
        new_tokens = []
        if part in special_token_mapping:
            if part == 'cls' and 'T5' in type(tokenizer).__name__:
                # T5 does not have cls token
                continue
            new_tokens.append(special_token_mapping[part])
        elif part[:5] == 'label':
            new_tokens += enc(' ' + mapping[label])
        elif part[:5] == 'sent_':
            sent_id = int(part.split('_')[1])
            new_tokens += enc(input_text_tuple[sent_id])
        elif part[:6] == '+sent_':
            sent_id = int(part.split('_')[1])
            new_tokens += enc(' ' + input_text_tuple[sent_id]) # add space
        elif part[:6] == 'sent-_':
            # Delete the last token
            sent_id = int(part.split('_')[1])
            new_tokens += enc(input_text_tuple[sent_id][:-1])
        elif part[:7] == '+sentl_':
            # Lower case the first token
            sent_id = int(part.split('_')[1])
            text = input_text_tuple[sent_id]
            text = text[:1].lower() + text[1:]
            new_tokens += enc(' ' + text)
        elif part[:7] == '+sentu_':
            # Upper case the first token
            sent_id = int(part.split('_')[1])
            text = input_text_tuple[sent_id]
            text = text[:1].upper() + text[1:]
            new_tokens += enc(' ' + text)
        elif part[:6] == 'sentl_':
            # Lower case the first token
            sent_id = int(part.split('_')[1])
            text = input_text_tuple[sent_id]
            text = text[:1].lower() + text[1:]
            new_tokens += enc(text)
        elif part[:6] == 'sentu_':
            # Lower case the first token
            sent_id = int(part.split('_')[1])
            text = input_text_tuple[sent_id]
            text = text[:1].upper() + text[1:]
            new_tokens += enc(text)
        elif part[:7] == 'sentl-_':
            # Lower case the first token
            sent_id = int(part.split('_')[1])
            text = input_text_tuple[sent_id]
            text = text[:1].lower() + text[1:]
            new_tokens += enc(text[:-1])
        else:
            part = part.replace('_', ' ') # there cannot be space in command, so use '_' to replace space
            # handle special case when t5 tokenizer might add an extra space
            if len(part) == 1:
                new_tokens.append(tokenizer._convert_token_to_id(part))
            else:
                new_tokens += enc(part)

        input_ids += new_tokens
    return input_ids

def generate(dataset, template, model, tokenizer, target_number, mapping, beam, label=None, length_limit=None, truncate=None):
    """
    Generate templates based on given inputs

    label: Only use instances with this label (deprecated)
    length_limit: At least generate content as long as length_limit (deprecated)
    """
    input_texts = []
    input_tensors = []
    max_length = 128

    # Process the inputs
    for item in dataset:
        if label is None or item['label'] == label:
            input_text = get_text(template, item['text'], item['label'], tokenizer, mapping)
            if truncate is not None:
                if truncate == 'head':
                    input_text = input_text[-256:]
                elif truncate == 'tail':
                    input_text = input_text[:256]
                else:
                    raise NotImplementedError
            input_ids = torch.tensor(input_text).long()
            max_length = max(max_length, input_ids.size(-1))
            input_tensors.append(input_ids)

    # Concatenate inputs as a batch
    input_ids = torch.zeros((len(input_tensors), max_length)).long()
    attention_mask = torch.zeros((len(input_tensors), max_length)).long()
    for i in range(len(input_tensors)):
        input_ids[i, :input_tensors[i].size(-1)] = input_tensors[i]
        attention_mask[i, :input_tensors[i].size(-1)] = 1

    # Print some examples
    print('####### example #######')
    print(tokenizer.decode(input_ids[0]))
    print(tokenizer.decode(input_ids[1]))
    print(tokenizer.decode(input_ids[2]))
    print('####### example #######\n')

    input_ids = input_ids.cuda()
    attention_mask = attention_mask.cuda()
    assert len(input_tensors) > 0

    # Maximum generate content length
    max_length = 20
    # NOTE:
    # start_mask = tokenizer._convert_token_to_id('<extra_id_0>')
    start_mask = tokenizer._convert_token_to_id('extra0')
    ori_decoder_input_ids = torch.zeros((input_ids.size(0), max_length)).long()
    ori_decoder_input_ids[..., 0] = model.config.decoder_start_token_id

    # decoder_input_ids: decoder inputs for next regressive generation
    # ll: log likelihood
    # output_id: which part of generated contents we are at
    # output: generated content so far
    # last_length (deprecated): how long we have generated for this part
    current_output = [{'decoder_input_ids': ori_decoder_input_ids, 'll': 0, 'output_id': 1, 'output': [], 'last_length': -1}]
    for i in tqdm(range(max_length - 2)):
        new_current_output = []
        for item in current_output:
            if item['output_id'] > target_number:
                # Enough contents
                new_current_output.append(item)
                continue
            decoder_input_ids = item['decoder_input_ids']

            # Forward
            batch_size = 64
            turn = input_ids.size(0) // batch_size
            if input_ids.size(0) % batch_size != 0:
                turn += 1
            aggr_output = []
            for t in range(turn):
                start = t * batch_size
                end = min((t + 1) * batch_size, input_ids.size(0))

                with torch.no_grad():
                    aggr_output.append(model(input_ids[start:end], attention_mask=attention_mask[start:end], decoder_input_ids=decoder_input_ids.cuda()[start:end])[0])
            aggr_output = torch.cat(aggr_output, 0)

            # Gather results across all input sentences, and sort generated tokens by log likelihood
            aggr_output = aggr_output.mean(0)
            log_denominator = torch.logsumexp(aggr_output[i], -1).item()
            ids = list(range(model.config.vocab_size))
            ids.sort(key=lambda x: aggr_output[i][x].item(), reverse=True)
            ids = ids[:beam+3]
            
            for word_id in ids:
                output_id = item['output_id']
                # NOTE
                # if word_id == start_mask - output_id or word_id == tokenizer._convert_token_to_id('</s>'):
                if word_id == start_mask - output_id or word_id == tokenizer._convert_token_to_id('[SEP]'):
                    # Finish one part
                    if length_limit is not None and item['last_length'] < length_limit[output_id - 1]:
                        check = False
                    else:
                        check = True
                    output_id += 1
                    last_length = 0
                else:
                    last_length = item['last_length'] + 1
                    check = True

                output_text = item['output'] + [word_id]
                ll = item['ll'] + aggr_output[i][word_id] - log_denominator
                new_decoder_input_ids = decoder_input_ids.new_zeros(decoder_input_ids.size())
                new_decoder_input_ids[:] = decoder_input_ids
                new_decoder_input_ids[..., i + 1] = word_id

                # Forbid single space token, "....", and ".........."
                if word_id in [3, 19794, 22354]:
                    check = False

                # Forbid continuous "."
                if len(output_text) > 1 and output_text[-2] == 5 and output_text[-1] == 5:
                    check = False

                if check:
                    # Add new results to beam search pool
                    new_item = {'decoder_input_ids': new_decoder_input_ids, 'll': ll, 'output_id': output_id, 'output': output_text, 'last_length': last_length}
                    new_current_output.append(new_item)

        if len(new_current_output) == 0:
            break

        new_current_output.sort(key=lambda x: x['ll'], reverse=True)
        new_current_output = new_current_output[:beam]
        current_output = new_current_output

    result = []
    print("####### generated results #######")
    for item in current_output:
        generate_text = ''
        for token in item['output']:
            generate_text += tokenizer._convert_id_to_token(token)
        print('--------------')
        print('score:', item['ll'].item())
        print('generated ids', item['output'])
        print('generated text', generate_text)
        result.append(generate_text)
    print("####### generated results #######\n")

    return result

def load_dataset(task, data_dir):
    if task in ["MNLI", "MRPC", "QNLI", "QQP", "RTE", "SNLI", "SST-2", "STS-B", "WNLI", "CoLA"]:
        lines = open(os.path.join(data_dir, 'train.tsv')).readlines()
        if task != 'CoLA':
            lines = lines[1:]

        dataset = []
        for line in lines:
            line = line.strip().split('\t')
            if task == 'CoLA':
                dataset.append({'label': line[1], 'text': [line[-1]]})
            elif task == 'MNLI':
                dataset.append({'label': line[-1], 'text': [line[8], line[9]]})
            elif task == 'MRPC':
                dataset.append({'label': line[0], 'text': [line[-2], line[-1]]})
            elif task == 'QNLI':
                dataset.append({'label': line[-1], 'text': [line[1], line[2]]})
            elif task == 'QQP':
                dataset.append({'label': line[-1], 'text': [line[3], line[4]]})
            elif task == 'RTE':
                dataset.append({'label': line[-1], 'text': [line[1], line[2]]})
            elif task == 'SNLI':
                dataset.append({'label': line[-1], 'text': [line[7], line[8]]})
            elif task == 'SST-2':
                dataset.append({'label': line[-1], 'text': [line[0]]})
            elif task == 'STS-B':
                dataset.append({'label': '0' if float(line[-1]) < 2.5 else '1', 'text': [line[-3], line[-2]]})
            elif task == 'WNLI':
                dataset.append({'label': line[-1], 'text': [line[1], line[2]]})
            else:
                raise NotImplementedError
    # TODO: 添加自定义任务的格式
    elif task in ["eprstmt"]:
        lines = pd.read_csv(os.path.join(data_dir, 'train.csv'), header=None).values.tolist()
        dataset = []
        for line in lines:
            dataset.append({'label': line[-1], 'text': [line[1]]})
    elif task in ["iflytek"]:
        lines = pd.read_csv(os.path.join(data_dir, 'train.csv'), header=None).values.tolist()
        dataset = []
        for line in lines:
            dataset.append({'label': line[0], 'text': [line[2]]})
    elif task in ["tnews"]:
        lines = pd.read_csv(os.path.join(data_dir, 'train.csv'), header=None).values.tolist()
        dataset = []
        for line in lines:
            dataset.append({'label': line[0], 'text': [line[2]]})
    elif task in ["ocnli"]:
        lines = pd.read_csv(os.path.join(data_dir, 'train.csv'), header=None).values.tolist()
        dataset = []
        for line in lines:
            dataset.append({'label': line[3], 'text': [line[1], line[2]]})
    elif task in ["bustm"]:
        lines = pd.read_csv(os.path.join(data_dir, 'train.csv'), header=None).values.tolist()
        dataset = []
        for line in lines:
            dataset.append({'label': line[3], 'text': [line[1], line[2]]})
    elif task in ["csldcp"]:
        lines = pd.read_csv(os.path.join(data_dir, 'train.csv'), header=None).values.tolist()
        dataset = []
        for line in lines:
            dataset.append({'label': line[1], 'text': [line[0]]})
    elif task in ["csl"]:
        lines = pd.read_csv(os.path.join(data_dir, 'train.csv'), header=None).values.tolist()
        dataset = []
        for line in lines:
            dataset.append({'label': line[3], 'text': [line[1], ','.join(eval(line[2]))]})
    elif task in ["cluewsc"]:
        lines = pd.read_csv(os.path.join(data_dir, 'train.csv'), header=None).values.tolist()
        dataset = []
        for line in lines:
            span_dict = eval(line[0])
            span1, span2 = span_dict['span1_text'], span_dict['span2_text']
            dataset.append({'label': line[2], 'text': [line[-1], span1, span2]})

    else:
        lines = pd.read_csv(os.path.join(data_dir, 'train.csv')).values.tolist()
        dataset = []
        for line in lines:
            dataset.append({'label': line[0], 'text': [line[1]]})

    return dataset

def search_template(model, tokenizer, task_name, k, seed, beam, output_dir, data_dir):
    print('#', task_name, k, seed, beam)
    dataset_path = os.path.join(data_dir, task_name, "{}-{}".format(k, seed))
    dataset = load_dataset(task_name, dataset_path)
    print('|', 'dataset examples')
    print('|', dataset[0])
    print('|', dataset[-1])
    print()
    
    # Manual label word mappings
    # TODO: 在此添加 label word mappings
    map_of_mapping = {
        'SST-2': {'0':'terrible','1':'great'},
        'sst-5': {0:'terrible',1:'bad',2:'okay',3:'good',4:'great'},
        'mr': {0:'terrible',1:'great'},
        'cr': {0:'terrible',1:'great'},
        'subj': {0:'subjective',1:'objective'},
        'trec': {0:'Description',1:'Entity',2:'Expression',3:'Human',4:'Location',5:'Number'},
        'mpqa': {0:'terrible',1:'great'},
        'CoLA': {'0':'incorrect','1':'correct'},
        'MRPC': {'0':'No','1':'Yes'},
        'QQP': {'0':'No','1':'Yes'},
        'STS-B': {'0':'No','1':'Yes'},
        'MNLI': {'contradiction':'No','entailment':'Yes','neutral':'Maybe'},
        'SNLI': {'contradiction':'No','entailment':'Yes','neutral':'Maybe'},
        'QNLI': {'not_entailment':'No','entailment':'Yes'},
        'RTE': {'not_entailment':'No','entailment':'Yes'},
        "eprstmt": {"Negative": "差", "Positive": "好"},
        "iflytek": {0:'打车',100:'美颜',101:'影像',102:'摄影',103:'相机',104:'绘画',105:'二手',106:'电商',107:'团购',108:'外卖',109:'票务',10:'社区',110:'超市',111:'购物',112:'笔记',113:'办公',114:'日程',115:'女性',116:'经营',117:'收款',118:'其他',11:'赚钱',12:'魔幻',13:'仙侠',14:'卡牌',15:'飞行',16:'射击',17:'休闲',18:'动作',19:'体育',1:'地图',20:'棋牌',21:'养成',22:'策略',23:'竞技',24:'辅助',25:'约会',26:'通讯',27:'工作',28:'论坛',29:'婚恋',2:'免费',30:'情侣',31:'社交',32:'生活',33:'博客',34:'新闻',35:'漫画',36:'小说',37:'技术',38:'教辅',39:'问答',3:'租车',40:'搞笑',41:'杂志',42:'百科',43:'影视',44:'求职',45:'兼职',46:'视频',47:'短视',48:'音乐',49:'直播',4:'同城',50:'电台',51:'唱歌',52:'两性',53:'小学',54:'职考',55:'公务',56:'英语',57:'在线',58:'教育',59:'成人',5:'快递',60:'艺术',61:'语言',62:'旅游',63:'预定',64:'民航',65:'铁路',66:'酒店',67:'行程',68:'民宿',69:'出国',6:'婚庆',70:'工具',71:'亲子',72:'母婴',73:'驾校',74:'违章',75:'汽车',76:'买车',77:'养车',78:'行车',79:'租房',7:'家政',80:'买房',81:'装修',82:'电子',83:'挂号',84:'养生',85:'医疗',86:'减肥',87:'美妆',88:'菜谱',89:'餐饮',8:'交通',90:'资讯',91:'运动',92:'支付',93:'保险',94:'股票',95:'借贷',96:'理财',97:'彩票',98:'记账',99:'银行',9:'政务'},
        # "iflytek": {8: '公共交通', 30: '情侣社交', 102: '摄影修图', 58: '高等教育', 42: '百科', 99: '银行', 47: '短视频', 72: '母婴', 46: '视频', 80: '买房', 82: '电子产品', 116: '经营', 38: '教辅', 21: '经营养成', 92: '支付', 1: '地图导航', 87: '美妆美业', 66: '酒店', 83: '问诊挂号', 73: '驾校', 41: '杂志', 51: 'K歌', 23: 'MOBA', 105: '二手', 61: '语言(非英语)', 67: '行程管理', 111: '购物咨询', 26: '即时通讯', 74: '违章', 96: '理财', 20: '棋牌中心', 103: '相机', 84: '养生保健', 53: '中小学', 65: '铁路', 19: '体育竞技', 52: '成人', 75: '汽车咨询', 10: '社区服务', 55: '公务员', 112: '笔记', 24: '辅助工具', 81: '装修家居', 101: '影像剪辑', 63: '综合预定', 62: '旅游资讯', 29: '婚恋社交', 2: '免费WIFI', 115: '女性', 85: '医疗服务', 100: '美颜', 108: '外卖', 43: '影视娱乐', 44: '求职', 17: '休闲益智', 106: '电商', 118: '其他', 13: '仙侠', 28: '论坛圈子', 77: '日常养车', 25: '约会社交', 71: '亲子儿童', 94: '股票', 37: '技术', 90: '体育咨讯', 54: '职考', 91: '运动健身', 11: '薅羊毛', 39: '问答交流', 27: '工作社交', 36: '小说', 49: '直播', 59: '成人教育', 5: '快递物流', 56: '英语', 93: '保险', 104: '绘画', 64: '民航', 86: '减肥瘦身', 97: '彩票', 109: '电影票务', 107: '团购', 45: '兼职', 60: '艺术', 70: '工具', 79: '租房', 48: '音乐', 95: '借贷', 110: '社区超市', 7: '家政', 32: '生活社交', 113: '办公', 76: '汽车交易', 78: '行车辅助', 16: '射击游戏', 15: '飞行空战', 98: '记账', 114: '日程管理', 40: '搞笑', 9: '政务', 0: '打车', 22: '策略', 18: '动作类', 117: '收款', 68: '民宿短租', 3: '租车', 57: '视频教育', 34: '新闻', 35: '漫画', 31: '社交工具', 89: '餐饮店', 6: '婚庆', 50: '电台', 4: '同城服务', 14: '卡牌', 88: '菜谱', 33: '微博博客', 69: '出国', 12: '魔幻'},
        # "tnews": {100: '事', 101: '文', 102: '娱', 103: '体', 104: '财', 106: '房', 107: '车', 108: '教', 109: '科', 110: '军', 112: '旅', 113: '国', 114: '股', 115: '农', 116: '游'},
        "tnews":{100:'故事',101:'文化',102:'娱乐',103:'体育',104:'财经',106:'房产',107:'汽车',108:'教育',109:'科技',110:'军事',112:'旅游',113:'国际',114:'股票',115:'农业',116:'电竞'},
        "ocnli": {'contradiction':'不','neutral':'或','entailment':'是'},
        "bustm": {0: '否', 1: '是'},
        "csldcp": {'材料科学与工程':'材料','作物学':'作物','口腔医学':'口腔','药学':'药学','教育学':'教育','水利工程':'水利','理论经济学':'理经','食品科学与工程':'食品','畜牧学/兽医学':'兽医','体育学':'体育','核科学与技术':'核能','力学':'力学','园艺学':'园艺','水产':'水产','法学':'法学','地质学/地质资源与地质工程':'地质','石油与天然气工程':'能源','农林经济管理':'农林','信息与通信工程':'通信','图书馆、情报与档案管理':'情报','政治学':'政治','电气工程':'电气','海洋科学':'海洋','民族学':'民族','航空宇航科学与技术':'航空','化学/化学工程与技术':'化工','哲学':'哲学','公共卫生与预防医学':'卫生','艺术学':'艺术','农业工程':'农工','船舶与海洋工程':'船舶','计算机科学与技术':'计科','冶金工程':'冶金','交通运输工程':'交通','动力工程及工程热物理':'动力','纺织科学与工程':'纺织','建筑学':'建筑','环境科学与工程':'环境','公共管理':'公管','数学':'数学','物理学':'物理','林学/林业工程':'林业','心理学':'心理','历史学':'历史','工商管理':'工商','应用经济学':'应经','中医学/中药学':'中医','天文学':'天文','机械工程':'机械','土木工程':'土木','光学工程':'光学','地理学':'地理','农业资源利用':'农资','生物学/生物科学与工程':'生物','兵器科学与技术':'兵器','矿业工程':'矿业','大气科学':'大气','基础医学/临床医学':'医学','电子科学与技术':'电子','测绘科学与技术':'测绘','控制科学与工程':'控制','军事学':'军事','中国语言文学':'语言','新闻传播学':'新闻','社会学':'社会','地球物理学':'地球','植物保护':'植物'},
        "csl": {0:'否',1:'是'},
        "cluewsc": {False:'否',True:'是'},
    }

    mapping = map_of_mapping[task_name]
    print('|', 'mapping')
    print('|', mapping)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, task_name), exist_ok=True)
    f = open(os.path.join(output_dir, task_name, "{}-{}.txt".format(k, seed)), 'w')

    # TODO: 添加相应的 single sentence tasks
    # if task_name in ['SST-2', 'sst-5', 'mr', 'cr', 'subj', 'trec', 'CoLA', 'mpqa']:
    if task_name in ['SST-2', 'sst-5', 'mr', 'cr', 'subj', 'trec', 'CoLA', 'mpqa', 'eprstmt', 'iflytek', "tnews", "csldcp", 'cluewsc']:
        # Single sentence tasks
        # We take two kinds of templates: put [MASK] at the beginning or the end
        # NOTE: 转成 Chinese T5 的格式
        # template = "*cls**sentu_0**<extra_id_0>**label**<extra_id_1>**sep+*"
        template = "*cls**sentu_0**extra0**label**extra1**sep+*"
        if task_name == 'cluewsc':
            template = "*cls**sentu_0**extra0**sent_1**extra1**sent_2**extra2**label**sep+*"
        generate_text = generate(dataset, template, model, tokenizer, target_number=2, mapping=mapping, beam=beam, label=None, truncate='head')[:beam//2]

        print("####### generated templates #######")
        for text in generate_text:
            # Transform T5 outputs to our template format
            # text = text.replace('<extra_id_0>', '*cls**sent_0*')
            # text = text.replace('<extra_id_1>', '*mask*')
            # text = text.replace('<extra_id_2>', '*sep+*')
            # text = text.replace('</s>', '*sep+*')
            if task_name == 'cluewsc':
                text = text.replace('extra0', '*cls**sent_0*')
                text = text.replace('extra1', '*sent_1*')
                text = text.replace('extra2', '*sent_2*')
                text = text.replace('extra3', '*mask*')
                text = text.replace('[SEP]', '*sep+*')
                text = text.replace('▁', '_')
            else:
                text = text.replace('extra0', '*cls**sent_0*')
                text = text.replace('extra1', '*mask*')
                text = text.replace('extra2', '*sep+*')
                text = text.replace('[SEP]', '*sep+*')
                text = text.replace('▁', '_')
            print(text)
            f.write(text + '\n')
        print("####### generated templates #######\n")

        # template = "*cls*.*<extra_id_0>**label**<extra_id_1>**+sentu_0**sep+*"
        template = "*cls*.*extra0**label**extra1**+sentu_0**sep+*"
        if task_name == 'cluewsc':
            template = "*cls*.*extra0**label**extra1**sent_0**extra2**sent_1**extra3**sent_2**sep+*"
            # template = "*cls**sentu_0**extra0**sent_1**extra1**sent_2**extra2**label**sep+*"
        generate_text = generate(dataset, template, model, tokenizer, target_number=2, mapping=mapping, beam=beam, label=None, truncate='tail')[:beam//2]
        print("####### generated templates #######")
        for text in generate_text:
            # Transform T5 outputs to our template format
            # text = text.replace('<extra_id_0>', '*cls*')
            # text = text.replace('<extra_id_1>', '*mask*')
            # text = text.replace('<extra_id_2>', '*+sent_0**sep+*')
            # text = text.replace('</s>', '*+sent_0**sep+*')
            if task_name == 'cluewsc':
                text = text.replace('extra0', '*cls*')
                text = text.replace('extra1', '*mask*')
                text = text.replace('extra2', '*sent_0*')
                text = text.replace('extra3', '*sent_1*')
                text = text.replace('extra4', '*sent_2*')
                text = text.replace('[SEP]', '*sep+*')
                text = text.replace('▁', '_')
            else:
                text = text.replace('extra0', '*cls*')
                text = text.replace('extra1', '*mask*')
                text = text.replace('extra2', '*+sent_0**sep+*')
                text = text.replace('[SEP]', '*+sent_0**sep+*')
                text = text.replace('▁', '_')
            print(text)
            f.write(text + '\n')
        print("####### generated templates #######\n")

    # TODO: 在此添加 sentence pair 任务，转成 Chinese T5 的格式
    elif task_name in ['MRPC', 'QQP', 'STS-B', 'MNLI', 'SNLI', 'QNLI', 'RTE', 'ocnli', 'bustm', 'csl']:
        # Sentence pair tasks
        # We always put [MASK] between the two sentences
        # template = "*cls**sent-_0**<extra_id_0>**label**<extra_id_1>**+sentl_1**sep+*"
        template = "*cls**sent-_0**extra0**label**extra1**+sentl_1**sep+*"
        generate_text = generate(dataset, template, model, tokenizer, target_number=2, mapping=mapping, beam=beam, label=None)
        print("####### generated templates #######")
        for text in generate_text:
            # Transform T5 outputs to our template format
            # text = text.replace('<extra_id_0>', '*cls**sent-_0*')
            # text = text.replace('<extra_id_1>', '*mask*')
            # text = text.replace('<extra_id_2>', '*+sentl_1**sep+*')
            # text = text.replace('</s>', '*+sentl_1**sep+*')
            text = text.replace('extra0', '*cls**sent-_0*')
            text = text.replace('extra1', '*mask*')
            text = text.replace('extra2', '*+sentl_1**sep+*')
            text = text.replace('[SEP]]', '*+sentl_1**sep+*')
            text = text.replace('▁', '_')
            print(text)
            f.write(text + '\n')
        print("####### generated templates #######\n")
    else:
        raise NotImplementedError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--t5_model', type=str, default='t5-3b', help='T5 pre-trained model')
    parser.add_argument('--seed', type=int, nargs='+', default=[42, 13, 21, 100, 87], help="Data split seeds")
    parser.add_argument('--task_name', type=str, nargs='+', default=['SST-2', 'sst-5', 'mr', 'cr', 'subj', 'trec', 'CoLA', 'MRPC', 'QQP', 'STS-B', 'MNLI', 'SNLI', 'QNLI', 'RTE'], help="Task names")
    parser.add_argument('--output_dir', type=str, default='Output directory')

    parser.add_argument('--data_dir', type=str, default="data/k-shot", help="Data directory")
    parser.add_argument('--beam', type=int, default=100, help="Beam search width")
    parser.add_argument('--k', type=int, default=16, help="Number of training instances per label")
 
    args = parser.parse_args()

    model = T5ForConditionalGeneration.from_pretrained(args.t5_model)
    # tokenizer = T5Tokenizer.from_pretrained(args.t5_model)
    # tokenizer.sep_token = '</s>'
    # 使用中文 T5 模型
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(args.t5_model)

    model = model.cuda()
    model.eval()

    for task_name in args.task_name:
        for seed in args.seed:
            search_template(model=model, tokenizer=tokenizer, task_name=task_name, k=args.k, seed=seed, beam=args.beam, output_dir=args.output_dir, data_dir=args.data_dir)

if __name__ == '__main__':
    main()