# EFL相关代码

## 环境要求：
1. python3
1. pytorch (1.7.0)


## 测试方式
```python
1. bash run_classifier_tnews.sh # tnews进行训练
2. bash run_classifier_tnews.sh predict # tnews进行预测
```
模型的验证集/训练集/batch size/max seq len/训练轮次/模版格式/learning rate等，对结果都有影响，可根据具体自行设置。

## 已有结果
| 模型   | score     | eprstmt  | bustm  | ocnli   | csldcp   | tnews | wsc | ifytek| csl | chid  |
| :----:| :----:  | :----: |:----: |:----: |:----: |:----: |:----: |:----: |:----: |:----: |
| EFL wo PT [^1]      | 67.9 |   |    |  | 25.9 |44.8 |  |  |   |   |
| EFL wo PT [^2]      |-- |   |    |  |41.7 |52 |  |  |   |   |
| EFL      |  |   |    |  |   | |  |  |   |   |

[^1]: 正负样本比例1:1
[^2]: 正负样本比例1:8
