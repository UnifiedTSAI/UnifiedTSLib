python main.py \
    --micro_batch_size 1 \
    --global_batch_size 1 \
    --inner_batch_ratio 1 \
    --model_name autoformer \
    --from_scratch \
    -o logs/autoformer_pretrain \
    -d Data/pretrain_datasets/train/data_etth1_train.jsonl \
    --channel_mixing True \
