import argparse
import os
import torch
import numpy as np
from transformers import *

from src.data.Batcher import Batcher
from src.utils.Config import Config
from src.utils.util import device
from src.adapet import adapet
from src.eval.eval_model import new_test_eval

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', "--exp_dir", required=True)
    args = parser.parse_args()

    config_file = os.path.join(args.exp_dir, "config.json")
    config = Config(config_file, mkdir=False)

    tokenizer = BertTokenizer.from_pretrained(config.pretrained_weight)
    # tokenizer = AutoTokenizer.from_pretrained(config.pretrained_weight)
    batcher = Batcher(config, tokenizer, config.dataset)
    dataset_reader = batcher.get_dataset_reader()

    model = adapet(config, tokenizer, dataset_reader).to(device)
    if os.path.exists(os.path.join(args.exp_dir, "best_model.pt")):
        model.load_state_dict(torch.load(os.path.join(args.exp_dir, "best_model.pt")))
        acc,logits = new_test_eval(config, model, batcher)
        print('best_model_test_acc:{}'.format(acc))
        with open(os.path.join(args.exp_dir, "result.txt"), 'a') as f:
            f.write('best_model_test_acc={}\n'.format(acc))

    model = adapet(config, tokenizer, dataset_reader).to(device)
    if os.path.exists(os.path.join(args.exp_dir, "final_model.pt")):
        model.load_state_dict(torch.load(os.path.join(args.exp_dir, "final_model.pt")))
        acc,logits = new_test_eval(config, model, batcher)
        print('final_test_acc:{}'.format(acc))
        with open(os.path.join(args.exp_dir, "result.txt"), 'a') as f:
            f.write('final_model_test_acc={}\n'.format(acc))
    # os.remove(os.path.join(args.exp_dir, "final_model.pt"))
    # os.remove(os.path.join(args.exp_dir, "best_model.pt"))

