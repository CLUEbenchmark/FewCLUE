# FewCLUE

中文小样本学习测评基准

FewCLUE: Few-shot learning for Chinese Language Understanding Evaluation

## 简介 Intorudction 
 预训练语言模型，包括用于语言理解(BERT类)或文本生成模型（GPT类），通过海量文本语料上做语言模型的预训练的方式，极大提升了NLP领域上多种任务上的表现并扩展了NLP的应用。使用预训练语言模型结合成数千或上万的标注样本，在下游任务上做微调，通常可以取得在特定任务上较好的效果；但相对于机器需要的大量样本，人类可以通过极少数量的样本上的学习来学会特定的物体的识别或概念理解。

 小样本学习（Few-shot Learning）正是解决这类在极少数据情况下的机器学习问题。结合预训练语言模型通用和强大的泛化能力基础上，探索小样本学习最佳模型和中文上的实践，是本课题的目标。FewCLUE：中文小样本学习测评基准，基于CLUE的积累和经验，并结合少样本学习的特点和近期的发展趋势，精心设计了该测评，希望可以促进中文领域上少样本学习领域更多的研究、应用和发展。

## 任务描述和统计 Task Descriptions and Statistics
| Corpus   | Train     | Dev  |Test Public| Test Private | Num Labels| Unlabeled| Task | Metric | Source |
| :----:| :----:  |:----:  |:----:  |:----:  |:----:  |:----:  |:----:  |:----:  |:----:  |
|   | Single |Sentence | Tasks  |
|   EPRSTMT    | 32 | 32 | 610 | 753 | 2 | 19565 | SntmntAnalysis | Acc | E-CommrceReview |
|   CSLDCP    | 536 |536   | 1784| 2999 | 67 | 18111 | LongTextClassify | Acc |AcademicCNKI |
|   TNEWS    | 240 | 240 |2010| 1500 | 15 |20000| ShortTextClassify | Acc |NewsTitle |
|    IFLYTEK   | 928 | 690  | 1749  | 2279 | 119  | 7558 | LongTextClassify| Acc |AppDesc |
|     | Sentence | Pair | Tasks |
|    OCNLI   | 32  | 32  |  2520 |  3000 | 3  | 20000 | NLI  |  Acc | 5Genres |
|    BUSTM   | 32 | 32  | 1772 | 2000  | 2 | 4251|SemanticSmlarty | Acc | AIVirtualAssistant |
|   |Reading |Comprhnsn |Tasks |
|     CHID  | 42 |  42 | 2002 | 2000  | 7？ | 7585 |  MultipleChoice,idiom | Acc  | Novel,EssayNews |
|     CSL  | 32 |  32 | 2828 | 3000 | 2? | 19841 | KeywordRecogntn| Acc | AcademicCNKI|
|     CLUEWSC  | 32 | 32  |  976 | 290  | 2 | 0| CorefResolution  | Acc | ChineseFictionBooks 

    EPRSTMT:电商评论情感分析；CSLDCP：科学文献学科分类；TNEWS:新闻分类；IFLYTEK:APP应用描述主题分类；
    OCNLI: 自然语言推理；BUSTM: 对话短文本匹配；CHID:成语阅读理解；CSL:摘要判断关键词判别；CLUEWSC: 代词消歧
    EPRSTMT,CSLDCP,BUSTM 为新任务；其他任务（TNEWS,CHID,IFLYTEK,OCNLI,CSL,CLUEWSC）来自于CLUE benchmark，部分数据集做了新的标注。


## 实验结果 Experiments
实验设置：训练集和验证集使用32个样本，或采样16个，测试集正常规模。基础模型使用RoBERT12层chinese_roberta_wwm_ext（GPT系列除外）。

| 模型   | score     | eprstmt  | bustm  | ocnli   | csldcp   | tnews | wsc | ifytek| csl | chid  |
| :----:| :----:  | :----: |:----: |:----: |:----: |:----: |:----: |:----: |:----: |:----: |
| <a href="#">Human</a>        | 82.49 |90.0N  | 88.0N    |  90.3N  | 68.0N |71.0N | 98.0N | 66.0N |  84.0N|  87.1N|
| <a href="https://github.com/ymcui/Chinese-BERT-wwm">FineTuningB</a>        | 46.37 |73.7N   | 61.1N    | 34.0N  | 28.6N |43.6N | 50.2N |29.6N | 50.1N| 15.0N|
| <a href="#">FineTuningR</a>        | | 64.6N |60.0N   | 33.3N    | 36.2N  | 50.6N |49.9N | 34.5N |50.0N | |
| <a href="#">PET</a>      | 57.01 | 87.2N | 64.0  | 43.9N | 56.9N |51.2N  | 59.2N| 35.1N | 55.0N | 61.3N |
| <a href="#">PtuningB</a>      | 51.81| 88.5N | 65.4  | 35.0N | 44.4N |  48.2N  | 51.0N | 32.0N| 50.0N | 57.6N |
| <a href="#">PtuningGPT</a>      | 46.44| 75.65N  | 54.9N   | 35.75N  | 33.69N  |  45.3N   | 49.0N | 24.0N | 53.5N  | 13.7N  |
| <a href="#">Zero-shot-G</a>      | 40.07 |  54.8N |  50N  | 34.4N  |  26.23N |     |  |  |   |   |
| <a href="#">Zero-shot-R</a>      | 44.61N |  85.2N |   50.6N | 40.3N | 12.6N  |   25.3N  | 50.0N | 27.7N |  52.2N |  57.6N |
    FineTuningB:FineTuningBert; FineTuningR:FineTuningRoberta; PtuningB:Ptuning_RoBERTa; PtuningGPT:Ptuning_GPT; 报告的数字是每一个任务的公开测试集(test_public.json)上的实验效果；
    Zero-shot: 零样本学习。前4个任务采用GPT模型，后5个任务采用chinese_roberta_wwm_ext为基础模型。N”，代表已更新；
    五一后待榜单可以提交后，将报告在测试集（test.json）上的效果；由于CHID还在继续实验中，暂时未将CHID的分数纳入到最终的分数(Score）中。
   <a href='https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA-Gen-TensorFlow'>使用的GPT模型: NEZHA-Gen</a>

## 实验结果 vs 人类水平   Human Performance & Benchmark Results

### 1.人类水平  Human Performance

    我们采取如下方式测评人类水平。按照训练，然后测评的方式。首先，人类会在训练集上学习30个样本，
    然后我们鼓励人类标注员进行讨论和分享经验；然后，每一个人人类标注员在验证集上标注100个样本；最后，由多个人的投票得出最终预测的答案。
    
    从实验结果可以看到，人类有高达82.49分的成绩。人类在多个任务中可以取得超过80分以上的分数。在较难的指代消解任务中，人类甚至达到了高达98的分数；
    而在类别特别多的任务，如iflytek（119个类别）,csldcp(67个类别），人类仅取得了60多分的及格成绩。

### 2.测评结果  Benchmark Results

#### 2.1 模型表现分析  Analysis of Model Performance

    模型有5种不同的方式做任务，分别是使用预训练模型直接做下游任务微调、PET、RoBERTa为基础的Ptuning方式、GPT类模型为基础的Ptuning方式、
    使用RoBERTa或GPT做零样本学习。
    
    我们发现：
    1）模型潜力：最好的模型表现（54.34分）远低于人类的表现（82.49分），即比人类低了近30分。说明针对小样本场景，模型还有很大的潜力；
    2）新的学习范式：在小样本场景，新的学习方式（PET,Ptuning）的效果以较为明显的差距超过了直接调优的效果。
       如在通用的基础模型（RoBERTa）下，PET方式的学习比直接下游任务微调高了近8个点。
    3）零样本学习能力：在没有任何数据训练的情况下，零样本学习在有些任务上取得了较好的效果。如在119个类别的分类任务中，模型在没有训练的情况下
    取得了27.7的分数，与直接下游任务微调仅相差2分，而随机猜测的话仅会获得1%左右的准确率。这种想象在在67个类别的分类任务csldcp中也有表现。

#### 2.2 任务分析  Analysis of Tasks 
    我们发现，在小样本学习场景：
    不同任务对于人类和模型的难易程度相差较大。如wsc指代消解任务，对于人类非常容易（98分），但对于模型却非常困难（50分左右），只是随机猜测水平；
    而有些任务对于人类比较困难，但对于模型却不一定那么难。如csldcp有67个类别，人类只取得了及格的水平，但我们的基线模型PET在初步的实验中
    就取得了56.9的成绩。我们可以预见，模型还有不少进步能力。

## FewCLUE 有什么特点？
1、任务类型多样、具有广泛代表性。包含多个不同类型的任务，包括情感分析任务、自然语言推理、多种文本分类、文本匹配任务和成语阅读理解等。

2、研究性与应用性结合。在任务构建、数据采样阶段，即考虑到了学术研究的需要，也兼顾到实际业务场景对小样本学习的迫切需求。
如针对小样本学习中不实验结果的不稳定问题，我们采样生成了多份训练和验证集；考虑到实际业务场景类别，我们采用了多个有众多类别的任务（如50+、100+多分类），
并在部分任务中存在类别不均衡的问题。

3、时代感强。测评的主要目标是考察小样本学习，我们也同时测评了模型的零样本学习、半监督学习的能力。不仅能考察BERT类擅长语言理解的模型，
也可以同时查考了近年来发展迅速的GPT-3这类生成模型的中文版本在零样本学习、小样本学习上的能力；

此外，我们提供小样本测评完善的基础设施。
从任务设定，广泛的数据集，多个有代表性的基线模型及效果对比，一键运行脚本，小样本学习教程，到测评系统、学术论文等完整的基础设施。


## 基线模型及运行 Baselines and How to run
    目前支持4类代码：直接fine-tuning、PET、Ptuning、GPT
    
    直接fine-tuning: 
        一键运行.基线模型与代码
        1、克隆项目 
           git clone https://github.com/CLUEbenchmark/FewCLUEDatasets.git
        2、进入到相应的目录
           分类任务  
               例如：
               cd FewCLUEDatasets/baseline/models_tf/fine_tuning/bert/
        3、运行对应任务的脚本(GPU方式): 会自动下载模型并开始运行。
           bash run_classifier_multi_dataset.sh
           计算8个任务cecmmnt tnews iflytek ocnli csl cluewsc bustm csldcp，每个任务6个训练集的训练模型结果
           结果包括验证集和测试集的准确率，以及无标签测试集的生成提交文件


​      
    PET/Ptuning/GPT:
        环境准备：
          预先安装Python 3.x(或2.7), Tesorflow 1.14+, Keras 2.3.1, bert4keras。
          需要预先下载预训练模型：chinese_roberta_wwm_ext，并放入到pretrained_models目录下
        
        运行：
        1、进入到相应的目录，运行相应的代码。以ptuning为例：
           cd ./baselines/models_keras/ptuning
        2、运行代码
           python3 ptuning_iflytek.py


## NLPCC2021-小模型测评
##### 报名注册、奖励、提交样例、排行榜、赛程与测评方案

<a href='https://www.cluebenchmarks.com/NLPCC.html'>测评报名;</a>  <a href='http://tcci.ccf.org.cn/conference/2021/cfpt.php'> NLPCC2021官方链接</a>

奖励：

    比赛证书：测评前三名队伍会获得NLPCC和CCF中国信息技术技术委员会认证的证书；
    优胜队伍有机会提交测评任务的论文（Task Report），并投稿到NLPCC会议（CCF推荐会议）；
    现金奖励：第一、二、三名分别奖励1万、5千、两千五（实在智能提供）

<a href='https://www.cluebenchmarks.com/submit.html'>测评系统已开放:</a>

<img src="https://github.com/CLUEbenchmark/FewCLUE/blob/main/resources/img/fewclue_sc.png"  width="85%" height="85%" />   

<a href='https://github.com/CLUEbenchmark/FewCLUE/blob/main/resources/fewclue_submit_example'>提交样例(提交zip压缩包)</a>

### <a href='https://docs.qq.com/doc/DR2pyTGltUURmb0hk'>测评方案</a>

    比赛规则
    ● 本次测评只能使用本测评提供的数据；
    ● 禁止任何形式人工标注；
    ● 禁止仅使用开源的模型和代码，没有算法贡献的情况出现；
    ● 本次比赛只能使用单模型（单模型的定义：一个任务只能有一个预测函数，所有任务只能使用同一个预训练模型，在计算图中只能有一个预训练模型）完成任务，不能集成多个模型进行预测；
    ● 模型调优阶段( finetune)只能使用比赛提供的数据；预训练阶段可使用外部数据（方案评审环节，说明来源）
    ● 禁止为了做本次测评找大量相似文本，去做任何形式的训练。
    
    违反以上规则，一经发现，将取消测评资格。

### <a href='https://www.cluebenchmarks.com/submit.html'>赛程及提交</a>

    第一阶段（5月9日--6月4日）：只需要针对一份数据集做训练和预测：
        {train_0.json, dev_0.json, test.json}
        
        最后提交的文件为：fewclue_submit_examples.zip，包括了上面的所有文件的压缩包。
        压缩命名如：zip fewclue_submit_examples.zip *.json
        
    第二阶段（6月5日-6月18日）：需针对每一个任务的多份数据集分别做训练和预测，即
        eprstmt_predict_x.json
        csldcp_predict_x.json
        tnews_predict_x.json
        iflytek_predict_x.json
        ocnli_predict_x.json
        bustm_predict_x.json
        chid_predict_x.json
        csl_predict_x.json
        cluewsc_predict_x.json
        x包括{0,1,2,3,4,all}，即每个数据集需要在test.json上预测6次。
        
        最后将这些文件压缩，命名为 fewclue_submit_examples.zip 压缩格式文件
    
    6月25日发布测评结果---->7月15日优胜队伍测评论文(task report)提交截止


## 数据集介绍 Introduction of datasets

####   分类任务 Single Sentence Tasks
##### 1. EPRSTMT（EPR-sentiment）  电商产品评论情感分析数据集  E-commerce Product Review Dataset for Sentiment Analysis
```  电商产品评论的情感分析，根据评论来确定情感的极性，正向或负向。
     数据量：训练集（32），验证集（32），公开测试集（610），测试集（753），无标签语料（19565）
     
     例子：
     {"id": 23, "sentence": "外包装上有点磨损，试听后感觉不错", "label": "Positive"}
     每一条数据有三个属性，从前往后分别是 id,sentence,label。其中label标签，Positive 表示正向，Negative 表示负向。
```

##### 2. CSLDCP  中文科学文献学科分类数据集     
```  
     中文科学文献学科分类数据集，包括67个类别的文献类别，这些类别来自于分别归属于13个大类，范围从社会科学到自然科学，文本为文献的中文摘要。
     数据量：训练集（536），验证集（536），公开测试集（1784），测试集（2999），无标签语料（67）
     
     例子：
     {"id": 23, "sentence": "外包装上有点磨损，试听后感觉不错", "label": "Positive"}
     每一条数据有三个属性，从前往后分别是 id,sentence,label。其中label标签，Positive 表示正向，Negative 表示负向。
```


##### 3.TNEWS 今日头条中文新闻（短文本）分类数据集 Toutiao Short Text Classificaiton for News
     该数据集来自今日头条的新闻版块，共提取了15个类别的新闻，包括旅游、教育、金融、军事等。
```  数据量：训练集（240），验证集（240），公开测试集（2010），测试集（1500），无标签语料（20000）
     
     例子：
     {"label": "102", "label_des": "news_entertainment", "sentence": "江疏影甜甜圈自拍，迷之角度竟这么好看，美吸引一切事物"}
     每一条数据有三个属性，从前往后分别是 分类ID，分类名称，新闻字符串（仅含标题）。
```

##### 4.IFLYTEK 长文本分类数据集 Long Text classification
    该数据集关于app应用描述的长文本标注数据，包含和日常生活相关的各类应用主题，共119个类别："打车":0,"地图导航":1,"免费WIFI":2,"租车":3,….,"女性":115,"经营":116,"收款":117,"其他":118(分别用0-118表示)。
``` 数据量：训练集（928），验证集（690），公开测试集（1749），测试集（2279），无标签语料（7558）
    
    例子：
    {"label": "110", "label_des": "社区超市", "sentence": "朴朴快送超市创立于2016年，专注于打造移动端30分钟即时配送一站式购物平台，商品品类包含水果、蔬菜、肉禽蛋奶、海鲜水产、粮油调味、酒水饮料、休闲食品、日用品、外卖等。朴朴公司希望能以全新的商业模式，更高效快捷的仓储配送模式，致力于成为更快、更好、更多、更省的在线零售平台，带给消费者更好的消费体验，同时推动中国食品安全进程，成为一家让社会尊敬的互联网公司。,朴朴一下，又好又快,1.配送时间提示更加清晰友好2.保障用户隐私的一些优化3.其他提高使用体验的调整4.修复了一些已知bug"}
    每一条数据有三个属性，从前往后分别是 类别ID，类别名称，文本内容。
```

#### Sentence Pair Tasks 
##### 5.OCNLI 中文原版自然语言推理数据集 Original Chinese Natural Language Inference
    OCNLI，即原生中文自然语言推理数据集，是第一个非翻译的、使用原生汉语的大型中文自然语言推理数据集。
    数据量：训练集（32），验证集（32），公开测试集（2520），测试集（3000），无标签语料（20000）
    
    例子：
    {
    "level":"medium",
    "sentence1":"身上裹一件工厂发的棉大衣,手插在袖筒里",
    "sentence2":"身上至少一件衣服",
    "label":"entailment","label0":"entailment","label1":"entailment","label2":"entailment","label3":"entailment","label4":"entailment",
    "genre":"lit","prem_id":"lit_635","id":0
    }

##### 6.BUSTM 小布助手对话短文本匹配数据集 XiaoBu Dialogue Short Text Matching 
    对话短文本语义匹配数据集，源于小布助手。它是OPPO为品牌手机和IoT设备自研的语音助手，为用户提供便捷对话式服务。
    意图识别是对话系统中的一个核心任务，而对话短文本语义匹配是意图识别的主流算法方案之一。要求根据短文本query-pair，预测它们是否属于同一语义。
    
    数据量：训练集（32），验证集（32），公开测试集（1772），测试集（2000），无标签语料（4251）
    例子：
    {"id": 5, "sentence1": "女孩子到底是不是你", "sentence2": "你不是女孩子吗", "label": "1"}
    {"id": 18, "sentence1": "小影,你说话慢了", "sentence2": "那你说慢一点", "label": "0"}

#### Reading Comprehension Tasks 
##### 7.ChID 成语阅读理解填空 Chinese IDiom Dataset for Cloze Test
    以成语完形填空形式实现，文中多处成语被mask，候选项中包含了近义的成语。 https://arxiv.org/abs/1906.01265
    数据量：训练集（42），验证集（42），公开测试集（2002），测试集（2000），无标签语料（7585）
    
    例子：
    {"id": 1421, "candidates": ["巧言令色", "措手不及", "风流人物", "八仙过海", "平铺直叙", "草木皆兵", "言行一致"],
    "content": "当广州憾负北控,郭士强黯然退场那一刻,CBA季后赛悬念仿佛一下就消失了,可万万没想到,就在时隔1天后,北控外援约瑟夫-杨因个人裁决案(拖欠上一家经纪公司的费用),
    导致被禁赛,打了马布里一个#idiom#,加上郭士强带领广州神奇逆转天津,让...", "answer": 1}

##### 8.CSL 论文关键词识别 Keyword Recognition
    中文科技文献数据集(CSL)取自中文论文摘要及其关键词，论文选自部分中文社会科学和自然科学核心期刊，任务目标是根据摘要判断关键词是否全部为真实关键词（真实为1，伪造为0）。
    数据量：训练集（32），验证集（32），公开测试集（2828），测试集（3000），无标签语料（19841）
    
    例子： 
    {"id": 1, "abst": "为解决传统均匀FFT波束形成算法引起的3维声呐成像分辨率降低的问题,该文提出分区域FFT波束形成算法.远场条件下,
    以保证成像分辨率为约束条件,以划分数量最少为目标,采用遗传算法作为优化手段将成像区域划分为多个区域.在每个区域内选取一个波束方向,
    获得每一个接收阵元收到该方向回波时的解调输出,以此为原始数据在该区域内进行传统均匀FFT波束形成.对FFT计算过程进行优化,降低新算法的计算量,
    使其满足3维成像声呐实时性的要求.仿真与实验结果表明,采用分区域FFT波束形成算法的成像分辨率较传统均匀FFT波束形成算法有显著提高,且满足实时性要求.",
     "keyword": ["水声学", "FFT", "波束形成", "3维成像声呐"], "label": "1"}
     
    每一条数据有四个属性，从前往后分别是 数据ID，论文摘要，关键词，真假标签。

##### 9.CLUEWSC  WSC Winograd模式挑战中文版
    Winograd Scheme Challenge（WSC）是一类代词消歧的任务，即判断句子中的代词指代的是哪个名词。题目以真假判别的方式出现，如：  
    句子：这时候放在[床]上[枕头]旁边的[手机]响了，我感到奇怪，因为欠费已被停机两个月，现在[它]突然响了。需要判断“它”指代的是“床”、“枕头”，还是“手机”？
    从中国现当代作家文学作品中抽取，再经语言专家人工挑选、标注。  
    
    数据量：训练集（32），验证集（32），公开测试集（976），测试集（290），无标签语料（0）
    例子：  
     {"target": 
         {"span2_index": 37, 
         "span1_index": 5, 
         "span1_text": "床", 
         "span2_text": "它"}, 
     "idx": 261, 
     "label": "false", 
     "text": "这时候放在床上枕头旁边的手机响了，我感到奇怪，因为欠费已被停机两个月，现在它突然响了。"}
     "true"表示代词确实是指代span1_text中的名词的，"false"代表不是。 

## 任务构建过程与调查问卷 Construction of Tasks
任务构建过程与调查问卷

  调查问卷
    
    1.调查背景
    调查问卷主要针对任务和数据集设定进⾏，调查对象为FewCLUE⼩小样本学习交流群，获得35份有效样本。
    
    2.调查问卷主要反馈： 
    1)希望任务丰富多样、适应真实场景; 
    2)数据集数量量: 提供9个左右数据集；
    3)当个任务数量量:按类别采样16为主; 
    4)半监督学习:⼩小样本测试还应提供⼤大量量⽆无标签数据；
    5)测试零样本学习；

  任务构建过程

    任务构建过程参考了本次调查问卷，以及近年来比较有代表性的结合预训练语言模型的小样本学习的论文。

   调查问卷反馈详见：<a href='https://github.com/CLUEbenchmark/FewCLUE/blob/main/resources/questionnaire/FewCLUE%E8%B0%83%E6%9F%A5%E9%97%AE%E5%8D%B7-%E5%8F%8D%E9%A6%88%E4%BF%A1%E6%81%AF.pdf'>FewCLUE调查问卷-反馈信息</a>


## 数据集文件结构 Data set Structure
    5份训练集，对应5份验证集，1份公开测试集，1份用于提交测试集，1份无标签样本，1份合并后的训练和验证集
    
    单个数据集目录结构：
        train_0.json：训练集0
        train_1.json：训练集1
        train_2.json：训练集2
        train_3.json：训练集3
        train_4.json：训练集4
        train_few_all.json： 合并后的训练集，即训练集0-4合并去重后的结果
        
        dev_0.json：验证集0，与训练集0对应
        dev_0.json：验证集1，与训练集1对应
        dev_0.json：验证集2，与训练集2对应
        dev_0.json：验证集3，与训练集3对应
        dev_0.json：验证集4，与训练集4对应
        dev_few_all.json： 合并后的验证集，即验证集0-4合并去重后的结果
        
        test_public.json：公开测试集，用于测试，带标签
        test.json: 测试集，用于提交，不能带标签
        
        unlabeled.json: 无标签的大量样本

## 模型简介（附图）
####   1.BERT.Fine-tuning
    模型简介：
    BERT模型开创了语言模型预训练-下游任务微调的范式。结合海量数据上预训练，使得模型具有强大的泛化能力；
    通过下游任务微调，仅新引入部分参数，而不需对整个网络从头训练。
   <img src="https://github.com/CLUEbenchmark/FewCLUE/blob/main/resources/img/bert_0.jpeg"  width="83%" height="83%" />   

    MLM pre-training: 预训练，利用上下文预测[MASK]位置的信息; Fine-tuning：通过下游任务微调，获得[CLS]位置的句子语义表示，
    并预测句子的标签。见下图：
   ![alt text](https://github.com/CLUEbenchmark/FewCLUE/blob/main/resources/img/bert_2.jpeg)

    BERT的输入表示：三部分信息求和，包括字的向量、段落向量、位置向量。见下图：
   ![alt text](https://github.com/CLUEbenchmark/FewCLUE/blob/main/resources/img/bert_3.jpeg)

####    2.GPT3: Language Models are Few-Shot Learners
    模型介绍：
    GPT3: GPT-3延续单向语言模型训练方式，把模型尺寸增大到了1750亿，并且使用45TB数据进行训练。
    同时，GPT-3主要聚焦于更通用的NLP模型，解决当前BERT类模型的两个缺点：对领域内有标签数据的过分依赖；对于领域数据分布的过拟合。
    GPT-3的主要目标是用更少的领域数据、且不经过微调去解决问题。
    
    GPT3中的零样本学习、单样本学习、少样本学习。通过给模型0、1或多个示例形式实现，但没有更新模型的参数：
   ![alt text](https://github.com/CLUEbenchmark/FewCLUE/blob/main/resources/img/gpt3_1.jpeg)


####   3.PET: Pattern-Exploiting Training
    模型和示例：
    将任务转化为完形填空。包括两部分，一部分将输入文本转化为完形填空，里面包含一个需要[MASK]的部分。
       
    如输入文本“这个手机壳很不错”，转化为：这个手机壳很不错。我觉得[MASK]好。这里的[MASK]计算需要填写的部分。
    另一部分将需要预测的标签对应到一个字符上。如，有两个情感分类的标签POSITIVE和NEGATIVE。POSITIVE-->good; NEGATIVE-->bad。
     
    下面报导一则[MASK]新闻。八个月了，终于又能在赛场上看到女排姑娘们了。
    [MASK]的地方可以填“体育”，也可以填“财经”、“娱乐”，但联合概率上看“体育”最大，那么“体育”可以做为预测的标签。
    
    PET的优缺点：
    优点：将任务进行了转换后，不再需要向之前的fine-tuning阶段一样引入新的最后一层，即没有引入任何参数；将下游任务转换成与预训练的语言模型一致的形式。
    缺点：可能需要手工构造Pattern; 不同的Pattern效果差异很大

####    4.LM-BFF: Making Pre-trained Language Models Better Few-shot Learners  
    模型和示例：
    LM-BFF是一套针对小样本进行微调的技术，主要包括两部分：基于提示（prompt）进行微调，关键是如何自动化生成提示模板；
    将样本示例以上下文的形式添加到每个输入中，关键是如何对示例进行采样；
   ![alt text](https://github.com/CLUEbenchmark/FewCLUE/blob/main/resources/img/lm_bff.jpeg)
    
    LM-BFF的基本方式如上图所示，红框部分就是提示模板，篮框部分就是额外的样本示例:
   ![alt text](https://github.com/CLUEbenchmark/FewCLUE/blob/main/resources/img/lm_bff_1.jpeg)

    在“Thank you <X> me to your party <Y> week ”，T5会在<X>生成“ for inviting ”、在<Y>生成“last ”。
    然后我们就可以基于T5去填充占位符<X>和<Y>，生成提示模板T。我们选定的模板应该是使得训练集中的输出概率最大化:
   <img src="https://github.com/CLUEbenchmark/FewCLUE/blob/main/resources/img/lm_bff_2.jpeg"  width="75%" height="75%" />   
    
    LB-BFF的优缺点
    优点：结合T5的生成能力，自动化找到最优的模板，省去人工搜寻模板的过程。
    缺点：依然假设模板是自然语言的版本；非全自动化的：先找模板，然在最优模板上做任务。


​    
####    5.Ptuning: GPT Understands, Too
    模型和示例：
    人工选择模板，或者机器自动选择模板，都是比较繁琐的工作，但对效果影响很大。那有没有能自动构建模型的方式呢？
    Ptuning放弃了模板必须是自然语言的假设，通过使用BERT系列中未使用的字的表示来自动设法学习到最佳的模板；
    并且它可以只学习模板对应的参数，如10个embedding，而之前的方法一般都是需要学习所有的字的表示；
    论文中实验了GPT类的模型也可以使用Ptuning方式取得很好的文本理解效果。
       
    离散模板搜索-->连续端到端学习:
   ![alt text](https://github.com/CLUEbenchmark/FewCLUE/blob/main/resources/img/ptuning.jpeg)
    
    中文例子：
   <img src="https://github.com/CLUEbenchmark/FewCLUE/blob/main/resources/img/ptuning_2.jpeg"  width="80%" height="80%" />   
      
    这里的[u1]～[u6]，代表BERT词表里边的[unused1]～[unused6]。也就是用几个从未见过的token来构成模板，
    这里的token数目是一个超参数。

####   6.ADAPET: Improving and Simplifying Pattern Exploiting Training

    模型和示例：
    ADAPET（A Densely-supervised Approach to Pattern Exploiting Training）做为PET的改进方法，
    共提出了两种改进方式： Decoupling Label Losses和Label Conditioning。
    1、 Decoupling Label Losses：在预测label的[MASK]时，计算了词表中所有token的概率，
    选取概率最大的token做为label。
    在PET中：模型只在候选label的token上进行softmax，选取概率最大的候选label的token做为预测label。
    在ADAPET中：模型在词表所有的token上进行softmax，选取概率最大的token做为预测label。这样做的好处是
    预测label的概率的计算是考虑到了词表中其它的token的。
    
    2、 Label Conditioning：随机掩盖原句中的original token，根据label去预测这些original token。
    即，在PET中，我们是在得到input的情况下去预测正确的label；而在这里，反过来思考，我们可以在得到label后，
    去预测正确的input。
    具体地，当构造的pattern中的label正确时，我们让模型去预测original token；当label不正确时，我们不让模型去预测original token。
    以下图中句子为例（预测两个句子是否是蕴含关系）：
    sent1:Oil prices rise
    sent2:Oil prices fall back
    correct label:no
    incorrect label:yes
    如果我们随机掩盖的token是“Oil”
    对于correct label，得到pattern：[mask] price rise, no, oil price fall。我们让模型去预测[MASK]对应的token是Oil。
    对于incorrect label，得到pattern：[mask] price rise, yes, oil price fall。这时候，由于label不是正确的，所以模型不
    去预测这个[MASK]，即不计算损失。
    

  ![alt text](https://github.com/CLUEbenchmark/FewCLUE/blob/main/resources/img/ADAPET.png)


## 教程 Tutorial
   1.系列PPT分享资料，详见: <a href='https://github.com/CLUEbenchmark/FewCLUE/tree/main/resources/ppt'>PPT</a>

   2.教程（Jupyter Notebook/Google Colab），添加中。。。

## 问题 Question
    1. 问：测试系统，什么时候开发？
       答：测评系统在5月1节后才会开放。

## 贡献与参与 Contribution & Participation
    1.问：我有符合代码规范的模型代码，并经过测试，可以贡献到这个项目吗？
     答：可以的。你可以提交一个pull request，并写上说明。
    
    2.问：我正在研究小样本学习，具有较强的模型研究能力，怎么参与到此项目？
      答：发送邮件到 CLUEbenchmark@163.com，标题为：参与FewCLUE课题，并介绍一下你的研究。

   <img src="https://github.com/CLUEbenchmark/FewCLUE/blob/main/resources/img/qqq.jpeg"  width="45%" height="45%" />   

   <img src="https://github.com/CLUEbenchmark/FewCLUE/blob/main/resources/img/bq_01.jpeg"  width="45%" height="45%" />   

## 引用 Reference

1、<a href='https://arxiv.org/abs/2005.14165'>GPT3: Language Models are Few-Shot Learners</a>

2、<a href='https://arxiv.org/pdf/2009.07118.pdf'>PET: It's Not Just Size That Matters: Small Language Models Are Also Few-Shot Learners</a>

3、<a href='https://kexue.fm/archives/7764'>必须要GPT3吗？不，BERT的MLM模型也能小样本学习</a>

4、<a href="https://arxiv.org/pdf/2012.15723.pdf">LM_BFF: Making Pre-trained Language Models Better Few-shot Learners</a>

5、<a href='https://zhuanlan.zhihu.com/p/341609647'>GPT-3的最强落地方式？陈丹琦提出小样本微调框架LM-BFF，比普通微调提升11%</a>

6、<a href='https://arxiv.org/pdf/2103.10385.pdf'>论文：GPT Understands, Too</a>

7、<a href='https://kexue.fm/archives/8295'>文章：P-tuning：自动构建模版，释放语言模型潜能</a>

8、<a href='https://arxiv.org/abs/2103.11955'>ADAPET: Improving and Simplifying Pattern Exploiting Training</a>

## License

    正在添加中

## 引用
    @misc{FewCLUE,
      title={FewCLUE:Few-shot learning for Chinese Language Understanding Evaluation},
      author={CLUE benchmark},
      year={2021},
      howpublished={\url{https://github.com/CLUEbenchmark/FewCLUE}},
    }
