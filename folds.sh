#!/bin/bash
timestamp=$(date +"%Y/%m/%d, %H:%M")

lr=${LR:-0.0001}
k_folds=${K_FOLDS:-5}
batch_size=${BATCH_SIZE:-8}
epochs=${EPOCHS:-30}
vocab_size=${VOCAB_SIZE:-256}
max_seq_length=${MAX_SEQ_LENGTH:-100}
dataset=${DATASET:-"fashion-dataset"}

dataset_path="my-datasets/${dataset}"
split_path="splits/${dataset}/k${k_folds}"
tokenizer_path="tokenizers/tokenizer_${vocab_size}.pickle"
model_path="weights/clip_kern_b${batch_size}_e${epochs}_v${vocab_size}_s${max_seq_length}_${dataset}"

mkdir -p $model_path

for (( k=0; k<$k_folds; k++ ))
do
    python3 main.py train_test \
        --batch_size=$batch_size \
        --epochs=$epochs \
        --lr=$lr \
        --model_path=$model_path \
        --dataset_path=$dataset_path \
        --tokenizer_path=$tokenizer_path \
        --split_path=$split_path \
        --k=$k \
        --vocab_size=$vocab_size \
        --max_seq_length=$max_seq_length \
        --wandb_name="clip_kern_b${batch_size}_e${epochs}_lr{$lr}_v${vocab_size}_s${max_seq_length}_k${k}_${dataset}" \
        --wandb_group="${timestamp}_clip_kern_b${batch_size}_e${epochs}_lr{$lr}_v${vocab_size}_s${max_seq_length}_kf${k_folds}_${dataset}"
done
