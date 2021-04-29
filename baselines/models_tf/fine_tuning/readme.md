这里放直接fine-tuning的代码



一键运行.基线模型与代码 Baseline with codes
---------------------------------------------------------------------
    使用方式：
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
      
       
一键运行.基线模型与代码 Baseline with codes
---------------------------------------------------------------------
    使用方式：
    1、克隆项目 
       git clone https://github.com/CLUEbenchmark/FewCLUEDatasets.git
    2、进入到相应的目录
       分类任务  
           例如：
           cd FewCLUEDatasets/baseline/models_tf/fine_tuning/bert/
    3、运行对应任务的脚本(GPU方式): 会自动下载模型并开始运行。
       bash run_classifier_xxx.sh
       如运行 bash run_classifier_ecomments.sh 会开始cecmmnt任务的训练和评估
   
 
