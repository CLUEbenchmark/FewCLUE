# LM-BFF for FewCLUE

This repository implements [FewCLUE](https://github.com/CLUEbenchmark/FewCLUE) tasks with [LM-BFF]((https://arxiv.org/pdf/2012.15723.pdf)) (Making Pre-trained Language Models Better Few-shot Learners). Please read the following documents to know how to use this repository and add your own dataset.

本代码实现了 [LM-BFF]((https://arxiv.org/pdf/2012.15723.pdf)) 在 [FewCLUE](https://github.com/CLUEbenchmark/FewCLUE) 任务上训练、验证和预测。以下文档讲介绍如何运行本代码并添加自定义数据集。

## 实验结果 :bar_chart:

| 模型   | score     | eprstmt  | bustm  | ocnli   | csldcp   | tnews | wsc | ifytek| csl | chid  |
| :----:| :----:  | :----: |:----: |:----: |:----: |:----: |:----: |:----: |:----: |:----: |
| <a href="https://arxiv.org/abs/2004.05986">Human</a>        | 82.49 |90.0N  | 88.0N    |  90.3N  | 68.0N |71.0N | 98.0N | 66.0N |  84.0N|  87.1N|
| <a href="https://github.com/ymcui/Chinese-BERT-wwm">FineTuningB</a>        | 39.35 |61.9N   | 54.1N   | 33.6N  | 25.6N |40.5N | 50.3N |22.6N | 50.5N| 15.0N|
| <a href="https://github.com/google-research/bert">FineTuningR</a>        | | 63.2N |55.5N   | 33.5N    | 35.7N  | 49.3N |49.6N | 32.8N |50.0N | |
| <a href="https://arxiv.org/pdf/2009.07118.pdf">PET</a>      | 57.36 | 87.2N | 64.0  | 43.9N | 56.9N |53.7N  | 59.2N| 35.1N | 55.0N | 61.3N |
| <a href="https://arxiv.org/pdf/2009.07118.pdf">PtuningB</a>      | 51.81| 88.5N | 65.4  | 35.0N | 44.4N |  48.2N  | 51.0N | 32.0N| 50.0N | 57.6N |
| <a href="https://arxiv.org/pdf/2009.07118.pdf">PtuningGPT</a>      | 46.44| 75.65N  | 54.9N   | 35.75N  | 33.69N  |  45.3N   | 49.0N | 24.0N | 53.5N  | 13.7N  |
| <a href="https://arxiv.org/abs/2005.14165">Zero-shot-G</a>      | 43.36N |  57.54N |  50N  | 34.4N  |  26.23N |  36.96N | 50.31N | 19.04N | 50.14N  | 65.63N  |
| <a href="https://arxiv.org/abs/2005.14165">Zero-shot-R</a>      | 44.61N |  85.2N |   50.6N | 40.3N | 12.6N  |   25.3N  | 50.0N | 27.7N |  52.2N |  57.6N |
| <a href="https://arxiv.org/pdf/2012.15723.pdf">LM-BFF</a> | 55.79 | 84.59 | 54.06 | 43.10 |  53.64 |  56.27 | 51.84 | 46.14 | 51.16 | 61.3 |

注：

- 模板生成模型 `uer/t5-base-chinese-cluecorpussmall`, 分类模型 `hfl/chinese-roberta-wwm-ext`。
- 使用 `Auto-T`，即只自动生成模板，`beam=30`。
- `few_shot_type=prompt`，即 Prompt-based fine-tuning。预实验发现原文效果最好的 Prompt-based fine-tuning with demonstrations 效果并不佳。
- 当前训练、验证、测试集使用了 train_0，dev_0，test_public
- 由于 `chid` 任务不需要模板，因此 LM-BFF 退化成了 PET，在此直接引用了 PET 的结果。

## 运行步骤 :page_with_curl:

1. 安装实验环境  
   运行本代码需要两个虚拟环境 `lm-bff` 以及 `lm-bff-gen`。若使用 `conda`，可按照以下命令安装环境，

   ```
   conda create -n lm-bff python=3.7
   pip install -r ./requirements.txt

   conda create -n lm-bff-gen --clone lm-bff
   pip install -U transformers
   ```

2. 准备数据
   
   ```shell
   # 切换到 FewCLUE 根目录
   cd ../../..
   python baselines/models_pytorch/LM-BFF/convert_format.py
   python baselines/models_pytorch/LM-BFF/tools/copy_fewclue_datasets.py
   ```

   该脚本将所有文件转换为没有 heading 的 csv 格式，保存在 `$ROOT_DIR/datasets/lm-bff` 下，接着将所有文件复制到 `./data/k-shot` 下准备使用。

3. 生成模板
   在 `generate_template.sh` 中将 `TASK` 变量设为你想使用的数据集。接着运行命令，

   ```
   conda activate lm-bff-gen
   bash generate_template.sh
   ```
   
   生成的模板将会保存在 `my_auto_template` 下。运行以下脚本以生成符合格式的 `-clean.txt` 模板。

   ```
   python tools/clean_t5_template.py --task_name tnews csl tnews ...
   ```

4. 模板评估  
   在 `generate_template.sh` 设置以下变量和参数，

   ```
   # 要实验的任务
   TASK=cluewsc
   # 文本长度限制，推荐数值可以参考 run_my_experiment.sh
   TASK_EXTRA="--max_seq_len 300 --first_sent_limit 256"
   # ...
   # 任务对应的 label word mapping
   --mapping "{False:'否',True:'是'}" 
   ```

   设置后运行脚本 `bash run_template_experiment.sh`. 该脚本将会对每个 candidate 模板进行训练，求验证集指标后对模板进行排序，保存在 `my_auto_template/$TASK`。

5. 训练模型  
   在 `run_my_experiment.sh` 中设置一下变量，

   ```
   TASK=cluewsc
   TASK_EXTRA="--max_seq_len 270 --first_sent_limit 256"
   MAPPING="{False:'否',True:'是'}"
   ```

   运行脚本 `bash run_my_experiment.sh` 即可开始训练。训练结束后可在 terminal 看到输出的实验结果。

6. 获取实验结果  
   所有的运行结果可以在 `./log` 中查找。

## 添加自定义数据集 :bulb:

   若使用 vscode，推荐安装 [Todo Tree](https://marketplace.visualstudio.com/items?itemName=Gruntfuggly.todo-tree) 插件，方便寻找需要修改的代码位置。

1. 将数据集文件保存到 FewCLUE 的文件夹中，格式和 FewCLUE 相同即可。

2. 在 `src/processors.py` 添加数据集的 `DataProcessor`. 如果是 single sentence task, 可以直接在 `TextClassificationProcessor` 下添加，
   
   ```python
   def get_labels(self):
        """See base class."""
        # TODO: 文本分类数据集在此添加 labels，下同 (这里指原数据集中出现的 label)
        # ...
        elif self.task_name == "eprstmt":
            return ['Negative', 'Positive']
        elif self.task_name == "iflytek":
            return list(range(119))
        elif self.task_name == 'tnews':
            return [100, 101, 102, 103, 104, 106, 107, 108, 109, 110, 112, 113, 114, 115, 116]
   
   # ...
   def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        # TODO: example 添加语句
        examples = []
        for (i, line) in enumerate(lines):
            # ...
            elif self.task_name in ['eprstmt']:
                examples.append(InputExample(guid=guid, text_a=line[1], label=line[2]))
            elif self.task_name in ['iflytek']:
                examples.append(InputExample(guid=guid, text_a=line[2], label=line[0]))
            elif self.task_name in ['tnews']:
   ```

   如果是 sentence pair task，需要添加一个 `DataProcessor` 的子类，以 `csl` 为例，

   ```python
   class CslProcessor(DataProcessor):
    """Processor for the CSL data set."""
   # ...
      def _create_examples(self, lines, set_type):
          """Creates examples for the training, dev and test sets."""
          examples = []
          for (i, line) in enumerate(lines):
              guid = "%s-%s" % (set_type, i)
              text_a = line[1]
              text_b = ','.join(eval(line[2]))
              label = line[3]
              examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
          return examples
   ```
   
   接着在三个 mapping 变量中加入新数据集，
   
   ```python
   processors_mapping = {
      # ...
      "eprstmt": TextClassificationProcessor("eprstmt"),
      "iflytek": TextClassificationProcessor("iflytek"),
      "tnews": TextClassificationProcessor("tnews"),
      "ocnli": OcnliProcessor(),
      "bustm": BustmProcessor(),
      "csldcp": TextClassificationProcessor("csldcp"),
      "csl": CslProcessor(),
      "cluewsc": TextClassificationProcessor("cluewsc"),
   }

   num_labels_mapping = {
       # ...
       "ocnli": 3,
       "bustm": 2,
       "csldcp": 67,
       "csl": 2,
       "cluewsc": 2,
   }

   output_modes_mapping = {
       # ...
       "ocnli": "classification",
       "bustm": "classification",
       "csldcp": "classification",
       "csl": "classification",
       "cluewsc": "classification",
   }

   # Return a function that takes (task_name, preds, labels) as inputs
   compute_metrics_mapping = {
       # ...
       "csldcp": text_classification_metrics,
       "csl": text_classification_metrics,
       "cluewsc": text_classification_metrics,
   }
   ```

3. 修改 `tools/generate_template.py`。在 `load_datasets` 方法中添加数据集，
   
   ```python
   # TODO: 添加自定义任务的格式
    elif task in ["eprstmt"]:
        lines = pd.read_csv(os.path.join(data_dir, 'train.csv'), header=None).values.tolist()
        dataset = []
        for line in lines:
            dataset.append({'label': line[-1], 'text': [line[1]]})
   # ...
   ```

   在 `search_template` 中加入你的 label word mapping，
   
   ```python
   def search_template(model, tokenizer, task_name, k, seed, beam, output_dir, data_dir):
    # ...
    # Manual label word mappings
    # TODO: 在此添加 label word mappings
    map_of_mapping = {
        "eprstmt": {"Negative": "差", "Positive": "好"},
        "ocnli": {'contradiction':'不','neutral':'或','entailment':'是'},
        # ...
   ```

   如果你的任务是 singel sentence，加到以下位置，

   ```python
   # TODO: 添加相应的 single sentence tasks
   # if task_name in ['SST-2', 'sst-5', 'mr', 'cr', 'subj', 'trec', 'CoLA', 'mpqa']:
   if task_name in ['SST-2', 'sst-5', 'mr', 'cr', 'subj', 'trec', 'CoLA', 'mpqa', 'eprstmt', 'iflytek', "tnews",  "csldcp",  'cluewsc']: 
   ```

   Sentence pair 任务加到以下位置，

   ```python
   # TODO: 在此添加 sentence pair 任务，转成 Chinese T5 的格式
   elif task_name in ['MRPC', 'QQP', 'STS-B', 'MNLI', 'SNLI', 'QNLI', 'RTE', 'ocnli', 'bustm', 'csl']:
   ```

4. 修改 `tools/sort_template.py`。在 `main` 函数下添加任务，
   
   ```python
   # TODO: 添加 CLUE 任务
   elif condition['task_name'] == 'eprstmt':
       args.key = 'eprstmt_dev_eval_acc'
       args.test_key = 'eprstmt_test_eval_acc'
       print_name = condition['task_name']
   elif condition['task_name'] == 'iflytek':
       args.key = 'iflytek_dev_eval_acc'
       args.test_key = 'iflytek_test_eval_acc'
       print_name = condition['task_name']
   elif condition['task_name'] == 'tnews':
       args.key = 'tnews_dev_eval_acc'
       args.test_key = 'tnews_test_eval_acc'
       print_name = condition['task_name']
   ```

   之后就可以按照上节方式训练自定义数据集了。:rocket:

## Citation

Original implementation: [princeton-nlp/LM-BFF](https://github.com/princeton-nlp/LM-BFF)

```bibtex
@inproceedings{gao2021making,
   title={Making Pre-trained Language Models Better Few-shot Learners},
   author={Gao, Tianyu and Fisch, Adam and Chen, Danqi},
   booktitle={Association for Computational Linguistics (ACL)},
   year={2021}
}
```
