这里放多种基线模型(baselines)


## 基线模型及运行 Baselines and How to run
    目前支持4类代码：直接fine-tuning、PET、Ptuning、GPT、ADAPET
    
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



~~~
ADAPET: 目前支持EPRSTMT、bustm、ocnli、cluewsc、csl、chid任务
	环境准备：
		cd ./models_pytorch/ADAPET/，查看requirements.txt
	运行：
	1、进入./models_pytorch/ADAPET/目录下：
		bash bin/fewclue.sh {任务名}
		例： bash bin/fewclue.sh bustm
	2、结果保存在ADAPET/exp_out/目录下
~~~

