# FewCLUE

中文小样本学习测评基准 FewCLUE: Few-shot learning for Chinese Language Understanding Evaluation

## 简介 Intorudction 
 预训练语言模型，包括用于语言理解或文本生成的模型，通过海量文本语料上做语言模型的预训练的方式，极大提升了NLP领域上多种任务上的表现并扩展了NLP的应用。使用预训练语言模型结合成数千或上万的标注样本，在下游任务上做微调，通常可以取得在特定任务上较好的效果；但相对于机器需要的大量样本，人类可以通过极少数量的样本上的学习来学会特定的物体的识别或概念理解。
 
 小样本学习（Few-shot Learning）正是解决这类在极少数据情况下的机器学习问题。结合预训练语言模型通用和强大的泛化能力基础上，探索小样本学习最佳模型和中文上的实践，是本课题的目标。FewCLUE：中文小样本学习测评基准，基于CLUE的积累和经验，并结合少样本学习的特点和近期的发展趋势，精心设计了该测评，希望可以促进中文领域上少样本学习领域更多的研究、应用和发展。


## 任务介绍 Tasks 
包括任务统计信息的表格、什么类型的任务

## 实验结果
实验设置：训练集和验证集使用32个样本，或采样16个，测试集政策规模。使用RoBERT12层chinese_roberta_wwm_ext作为基础模型。

| 模型   | score     | cecmmnt  | bustm  | ocnli   | csldcp   | tnews | wsc | ifytek| csl | chid  |    
| :----:| :----:  | :----: |:----: |:----: |:----: |:----: |:----: |:----: |:----: |:----: |
| <a href="#">Human</a>        | - |?   | ?    |  90.3  | ？ |71.0 | 98.0 | 66.0 |  84.0|  87.1|
| <a href="https://github.com/ymcui/Chinese-BERT-wwm">TrainAll</a>        | - |-   | -     | -  |  |-  | -  | -  | - | 1- |
| <a href="https://github.com/ymcui/Chinese-BERT-wwm">FineTuning</a>        | - |70.2   | 63.1    | 35.3  |  |52.0N | 49.3N | 17.9N | 50.0N| 15.0N|
| <a href="#">PET</a>      | - | 78.6N | 64.0  | 36.0 |  |51.2N  | 50.0N| 35.1N | 55.0N | 57.5N |
| <a href="#">PtuningB</a>      | - | 88.5N | 65.4  | 35.9 |  |  48.2N  | 51.0N | 32.0N| 50.0N | 15.0N |
| <a href="#">PtuningGPT</a>      | - | 76.4？  | 60.4？   |   |   |  45.3N   | 49.0N | 24.0N | 53.5N  | 13.7N  |
| <a href="#">Zero-shot</a>      | - |   |    |   |   |   25.3N  | 50.0N | 27.7N |  52.2N |  52.2N |
| <a href="#">半监督</a>      | - |   |    |   |   |     |  |  |   |   |

    PtuningB: Ptuning_RoBERTa; PtuningGPT: Ptuning_GPT; Zero-shot: GPT类Zero-shot; TrainAll: 在5份合并后的数据上训练；半监督：小样本+无标签数据;
    N”，代表已更新。wsc: cluewsc; cecmnt：cecmmnt(sentiment)，情感2分类; bustm: bustm(opposts); 
    ?：代表新的数据集下还没有跑过实验。如跑过实验了，去掉“？”，改为N。


## FewCLUE特点
1、任务类型多样、具有广泛代表性。包含多个不同类型的任务，包括情感分析任务、自然语言推理、多种文本分类、文本匹配任务和成语阅读理解等。

2、研究性与应用性结合。在任务构建、数据采样阶段，即考虑到了学术研究的需要，也兼顾到实际业务场景对小样本学习的迫切需求。
如针对小样本学习中不实验结果的不稳定问题，我们采样生成了多份训练和验证集；考虑到实际业务场景类别，我们采用了多个有众多类别的任务（如50+、100+多分类），
并在部分任务中存在类别不均衡的问题。

3、时代感强。测评的主要目标是考察小样本学习，我们也同时测评了模型的零样本学习、半监督学习的能力。不仅能考察BERT类擅长语言理解的模型，
也可以同时查考了近年来发展迅速的GPT-3这类生成模型的中文版本在零样本学习、小样本学习上的能力；

4、完善的基础设施。我们提供从任务设定，广泛的数据集，多个有代表性的基线模型及效果对比，一键运行脚本，小样本学习教程，到测评系统、学术论文等完整的基础设施。

## 任务例子 Tasks
| Corpus   | Train     | Dev  |Test Public| Test Private | Num Labels| Unlabeled| Task | Metric | Source |
| :----:| :----:  |:----:  |:----:  |:----:  |:----:  |:----:  |:----:  |:----:  |:----:  |
|   | Single |Sentence | Tasks  |
|   EPRSTMT    | 32 | 32 | 610(太少) | 753(太少) | 2 | 19565 | Sentiment Analysis | Acc | E-Commerce Review |
|   CSLDCP    | 536 |536   | 1784| 2999 | 65? | 18111 | long text classification | Acc |Academic (CNKI) |
|   TNEWS    | 240 | 240 |2010| 1500 | 15 |20000| short text classification | Acc |news title |
|    IFLYTEK   | 928 | 690  | 1749  | 2279 | 100+  | 7558 | long text classification| Acc |App Descriptions |
|     | Sentence | Pair | Tasks |
|    OCNLI   | 32  | 32  |  2520 |  3000 | 3  | 20000 |natural language inference  |  Acc | 5 genres |
|    BUSTM   | 32 | 32  | 1772 | 2000  | 2 | semantic similarity | Acc | AI Virtual Assistant | 
|   |Machine Reading |Comprehension |Tasks |
|     CHID  | 42 |  42 | 2002 | 2000  | 7？ | 7585 |  multiple-choice, idiom | Acc  | Novel,Essay and News |
|     CSL  | 32 |  32 | 2828 | 3000 | 2? | 19841 | keyword recognition| Acc | academic (CNKI)| 
|     CLUEWSC  | 32 | 32  |  976（太少） | 290(太少）  | 2 | 0（太少）| coreference resolution  | Acc | Chinese fiction books   



## 基线模型：运行及介绍（附图） baseline
基线模型：运行及介绍 baseline

## 任务构建过程与调查问卷
任务构建过程与调查问卷

## 测评报名|提交 Submit
测评报名|提交 Submit

## 单个数据集文件结构
单个数据集文件结构

## 教程
教程（jupyter notebook/Google Colab）

## 贡献与参与
贡献与参与

##引用
