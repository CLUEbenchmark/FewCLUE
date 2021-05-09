# 生成类模型GPT的相关预测代码

## 环境要求：
1. bert4keras (0.10.5)
1. Keras (2.3.1)
1. tensorflow-gpu (1.14.0)
1. h5py 2.10
1. <a href='https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA-Gen-TensorFlow'>使用的GPT模型: NEZHA-Gen</a>

h5py：如果h5py是高版本，需要卸载h5py再安装h5py:
pip install h5py==2.10 -i https://pypi.tuna.tsinghua.edu.cn/simple/

## 已完成内容
1. p-tuning:    
   已完成任务：eprstmt,bustm,ocnli,csldcp    
   使用数据集：每个任务的0号训练集与验证集
1. zero-shot:   
   已完成任务：eprstmt,bustm,ocnli,csldcp,tnews,wsc,ifytek,csl,chid     
   使用方法：chid测试整个句子概率的方式，其他测试末尾标签出现的概率

## todo
1. 使用chid任务的零样本学习方法，测试其他几个任务
1. 使用p-tuning方法测试所有任务
1. 训练并测试所有训练集跟验证集上的数据


## 测试方式
```python
python run_gpt.py -t chid -z # 运行chid任务，并使用零样本学习的方式
python run_gpt.py -t wsc # 运行wsc任务，并使用p-tuning学习的方式
```
模型的验证集/训练集/batch size/max seq len/训练轮次/模版格式等，对结果都有影响，可根据具体自行设置。

## 已有结果
| 模型   | score     | eprstmt  | bustm  | ocnli   | csldcp   | tnews | wsc | ifytek| csl | chid  |
| :----:| :----:  | :----: |:----: |:----: |:----: |:----: |:----: |:----: |:----: |:----: |
| <a href="https://arxiv.org/pdf/2009.07118.pdf">PtuningGPT</a>      | 46.44| 75.65N  | 54.9N   | 35.75N  | 33.69N  |    |  |  |   |   |
| <a href="https://arxiv.org/abs/2005.14165">Zero-shot-G</a>      | 43.36N |  57.54N(72.2?) |  50N  | 34.4N  |  26.23N |  36.96N | 50.31N | 19.04N | 50.14N  | 65.63N  |