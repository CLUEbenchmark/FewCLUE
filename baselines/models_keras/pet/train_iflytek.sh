for i in 0 1 2 3 4 few_all
do
    python ./baselines/models_keras/pet/pet_iflytek.py --train_set_index $i
done