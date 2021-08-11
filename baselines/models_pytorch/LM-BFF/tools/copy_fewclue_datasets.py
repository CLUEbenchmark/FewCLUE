import pathlib
import shutil
from tqdm import tqdm
import argparse


def main(splits):
    project_path = (pathlib.Path(__file__).parent / "../../../..").resolve()
    dataset_path = project_path / "datasets" / "lm-bff"
    dst_dataset_path = (pathlib.Path(__file__).parent / "../data/k-shot").resolve()
    
    for subpath in tqdm(dataset_path.iterdir(), desc="Copying files into LM-BFF directory"):
        for split in args.splits:
            task_name = subpath.name
            dst_task_path = dst_dataset_path / str(split) / task_name / "16-13"
            dst_task_path.mkdir(parents=True, exist_ok=True)
            for fpath in subpath.iterdir():
                if (
                    fpath.is_file()
                    and fpath.name in [f"train_{split}.csv", f"dev_{split}.csv", "test_public.csv"]
                ):
                    dst_fname = fpath.name.split("_")[0] + ".csv"
                    shutil.copy(fpath, dst_task_path / dst_fname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--splits', default=[0, 1, 2, 3, 4, "few_all"], nargs="+", help='Dataset splits to be copied to LM-BFF folder.')

    args = parser.parse_args()

    main(args.splits)