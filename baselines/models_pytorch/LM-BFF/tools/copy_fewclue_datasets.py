import pathlib
import shutil
from tqdm import tqdm


project_path = (pathlib.Path(__file__).parent / "../../../..").resolve()
dataset_path = project_path / "datasets" / "lm-bff"
dst_dataset_path = (pathlib.Path(__file__).parent / "../data/k-shot").resolve()

for subpath in tqdm(dataset_path.iterdir(), desc="Copying files into LM-BFF directory"):
    task_name = subpath.name
    dst_task_path = dst_dataset_path / task_name / "16-13"
    dst_task_path.mkdir(parents=True, exist_ok=True)
    for fpath in subpath.iterdir():
        if (
            fpath.is_file()
            and fpath.name in ["train_0.csv", "dev_0.csv", "test_public.csv"]
        ):
            dst_fname = fpath.name.split("_")[0] + ".csv"
            shutil.copy(fpath, dst_task_path / dst_fname)