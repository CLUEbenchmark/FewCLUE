MODEL=$1
K=16
# TASK=eprstmt
# TODO: 添加 tasks

# python tools/get_sbert_embedding.py --sbert_model $MODEL --task SST-2 sst-5 mr cr mpqa subj trec CoLA MRPC QQP STS-B MNLI SNLI QNLI RTE
# python tools/get_sbert_embedding.py --sbert_model $MODEL --seed 42 --do_test --task SST-2 sst-5 mr cr mpqa subj trec CoLA MRPC QQP STS-B MNLI SNLI QNLI RTE

# for seed in 13 21 87 100
# do
#     for task in SST-2 sst-5 mr cr mpqa subj trec CoLA MRPC QQP STS-B MNLI SNLI QNLI RTE
#     do
#         cp data/k-shot/$task/$K-42/test_sbert-$MODEL.npy  data/k-shot/$task/$K-$seed/
#     done

#     cp data/k-shot/MNLI/$K-42/test_matched_sbert-$MODEL.npy  data/k-shot/MNLI/$K-$seed/
#     cp data/k-shot/MNLI/$K-42/test_mismatched_sbert-$MODEL.npy  data/k-shot/MNLI/$K-$seed/
# done

python tools/get_sbert_embedding.py --sbert_model $MODEL --seed 13 --task eprstmt iflytek tnews ocnli
python tools/get_sbert_embedding.py --sbert_model $MODEL --seed 13 --do_test --task eprstmt iflytek tnews ocnli

for seed in 13
do
    for task in eprstmt iflytek tnews ocnli
    do
        cp data/k-shot/$task/$K-13/test_sbert-$MODEL.npy  data/k-shot/$task/$K-$seed/
    done
done
