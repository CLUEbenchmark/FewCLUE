task=$1
echo $task
for i in 0 1 2 3 4 few_all
do
    python ./baselines/models_keras/pet/pet_$task.py --train_set_index $i
done