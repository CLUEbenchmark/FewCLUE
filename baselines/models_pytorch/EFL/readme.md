# EFL相关代码

## 环境要求：
1. python3
2. pytorch (1.7.0)
3. 预训练模型为chinese_roberta_wwm_ext_L-12_H-768_A-12


## 测试方式
```python
1. bash run_classifier_tnews.sh # tnews进行训练
2. bash run_classifier_tnews.sh predict # tnews进行预测
```
模型的验证集/训练集/batch size/max seq len/训练轮次/模版格式/learning rate等，对结果都有影响，可根据具体自行设置。

## 已有结果
| 模型   | score     | eprstmt  | bustm  | ocnli   | csldcp   | tnews | wsc | iflytek| csl | chid  |
| :----:| :----:  | :----: |:----: |:----: |:----: |:----: |:----: |:----: |:----: |:----: |
| EFL wo PT (1)   |   | 76.1 | 60.9 | 33.1 | 26.0 |47.4 |50.9 |30.5 | 50.5  |18.8|
| EFL wo PT (2)   |   |--    | 62.9 | 34.2 |45.9  |53.5 |-- |38.7 |59.8 |  --  |
| EFL (3)      |  | 85.6 |67.6  |67.5 | 46.7 |49.1? | 54.2 |44.0 |61.6  |28.8|

1. 正负样本比例1:1
2. 正负样本比例1:8,其中ocnli为三分类，构建了4倍的contradiction样本,bustm构建了4倍的负例样本
3. 采用已有的中文文本蕴含任务如cmnli或者ocnli对模型进行预先训练，再拿这个预先训练的模型作为预训练模型

#### 备注
1. 目前都是在train_0 dev_0 和test_public上做的训练和测试
2. 模版对效果影响挺大
