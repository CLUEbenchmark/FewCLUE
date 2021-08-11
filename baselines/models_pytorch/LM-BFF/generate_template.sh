# TASK=tnews

for TASK in eprstmt bustm ocnli csldcp tnews cluewsc iflytek csl
do
    for split in 1 2 3 4 "few_all"
    do
        echo "Generate templates for task $TASK, split $split"
        python tools/generate_template.py \
            --data_dir data/k-shot/$split \
            --output_dir my_auto_template/$split \
            --task_name $TASK \
            --seed 13 \
            --beam 30 \
            --t5_model uer/t5-base-chinese-cluecorpussmall \
        # --task_name csldcp \
        # 中文 T5 的 beam 可以设大一点，因为会生成很多重复的模板
        # --task_name iflytek \
        # --task_name bustm \
        # --task_name iflytek \
        # --t5_model google/mt5-base \
        # --beam 30
        # --task_name tnews \
        # --task_name eprstmt \
    done
    python tools/clean_t5_template.py --task_name $TASK
done
