import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, nargs='+', default=[13], help="Data split seeds")
    parser.add_argument('--task_name', type=str, nargs='+', default=['eprstmt', 'bustm', 'ocnli', 'csldcp', 'tnews', 'cluewsc', 'iflytek', 'csl'], help="Task names")
    parser.add_argument('--template_dir', type=str, default="my_auto_template", help='Template directory')
    parser.add_argument('--k', type=int, default=16, help="Number of training instances per label")
    parser.add_argument('--splits', default=[0, 1, 2, 3, 4, "few_all"], nargs="+", help='Dataset splits to be copied to LM-BFF folder.')

    args = parser.parse_args()

    for seed in args.seed:
        for task_name in args.task_name:
            for split in args.splits:
                fpath = Path(f"{args.template_dir}/{split}/{task_name}/{args.k}-{seed}.txt")
                if not fpath.exists():
                    continue
                tmps = []
                with open(fpath, 'r') as f:
                    for line in f:
                        tmp_str = line.strip()
                        # 去除奇怪的符号
                        tmp_str = tmp_str.replace("[UNK]", "")
                        tmp_str = tmp_str.replace("[SEP]", "")
                        # 模板在第一个 *sep+* 后即可截止
                        end = tmp_str.find("*sep+*")
                        if end != -1:
                            tmp_str = tmp_str[:end + len('*sep+*')]

                            # n 个 token 的 label word 需要 n 个 *mask*
                            if task_name in ["iflytek", 'csldcp', "tnews"]:
                                mask_start = tmp_str.find("*mask*")
                                tmp_str = tmp_str[: mask_start] + "*mask" + tmp_str[mask_start:]
                            
                            tmps.append(tmp_str)
                        else:
                            # 没有 *sep+*，不保存该模板
                            pass
                    
                with open(fpath.parent / (fpath.stem + "-clean.txt"), 'w') as f:
                    for t in tmps:
                        f.write(t + '\n')


if __name__ == '__main__':
    main()