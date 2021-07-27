for i in 1 2 3 4 5 few_all
do
    python ./baselines/models_keras/pet/pet_tnews.py --train_set_index $i
done