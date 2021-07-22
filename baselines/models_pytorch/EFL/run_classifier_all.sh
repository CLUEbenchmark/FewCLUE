#! /bin/bash
task_name_list=(bustm chid iflytek csldcp csl eprstmt ocnli tnews wsc)
for task_name in ${task_name_list[@]};do
    shell_file_name=run_classifier_$task_name.sh
    for index in 0 1 2 3 4 few_all;do
        train_file_name=train_$index.json
        dev_file_name=dev_$index.json
        for test_file_name in test_public.json test.json;do
            bash -x $shell_file_name $train_file_name $dev_file_name $test_file_name
            bash -x $shell_file_name $train_file_name $dev_file_name $test_file_name predict
        done
    done
done
