ADAPET

Improving and Simplifying Pattern Exploiting Training

https://arxiv.org/pdf/2103.11955.pdf

####   ADAPET

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
   
   
#### 运行

    ADAPET: 目前支持EPRSTMT、bustm、ocnli、cluewsc、csl、chid任务
        环境准备：
            cd ./models_pytorch/ADAPET/，查看requirements.txt
        运行：
        1、进入./models_pytorch/ADAPET/目录下：
            bash bin/fewclue.sh {任务名}
            例： bash bin/fewclue.sh bustm
        2、结果保存在ADAPET/exp_out/目录下
       
       
### 如何添加新的任务？

   <a href='./adapet_修改流程.pdf'>ADAPET/adapet_修改流程.pdf</a>

        
        