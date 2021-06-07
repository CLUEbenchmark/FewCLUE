import pandas as pd
import numpy as np
import argparse
import pathlib
import os
from typing import Dict
import json
from collections import defaultdict
from tqdm import tqdm


def load_data(filename):
    samples = defaultdict(list)
    f = open(filename, "r", encoding="utf-8")
    for line in f:
        s = json.loads(line.strip())
        for k, v in s.items():
            samples[k].append(v)
    f.close()
    samples = pd.DataFrame(samples)
    return samples


def load_datasets(data_dir, tasks) -> Dict:
    datasets = {}
    for task in tasks:
        datasets[task] = {}
        files_dir = data_dir / task
        for fname in files_dir.iterdir():
            if (
                fname.is_file()
                and fname.suffix == ".json"
                and fname.stem.split("_")[0] in ["train", "dev", "test"]
            ):
                samples = load_data(fname)
                datasets[task][fname.stem] = samples
    return datasets


def convert_datasets(datasets: Dict, output_dir):
    for task in tqdm(datasets.keys(), desc="Converting datasets"):
        task_output_dir = output_dir / task
        task_output_dir.mkdir(parents=True, exist_ok=True)
        for fname in datasets[task].keys():
            datasets[task][fname].to_csv(
                output_dir / task / (fname + ".csv"), header=False, index=False
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=[
            "iflytek",
            "eprstmt",
            "tnews",
            "ocnli",
            "bustm",
            "csldcp",
            "cluewsc",
            "csl"
        ],
        help="Task names",
    )

    parser.add_argument(
        "--data_dir", type=str, default="datasets", help="Path to original data"
    )
    parser.add_argument(
        "--output_dir", type=str, default="datasets/lm-bff", help="Output path"
    )

    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = pathlib.Path(args.data_dir)

    datasets = load_datasets(data_dir, args.tasks)
    convert_datasets(datasets, output_dir)


if __name__ == "__main__":
    main()
