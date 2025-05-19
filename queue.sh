#!/bin/bash
rm slurm_log/*.out slurm_log/*.err

lrs=(0.001)
batch_sizes=(32)
datasets=("fashion-dataset") # Datasets: "fashion-dataset" "HOMUS-parsed" "dataset" "MTD-custom-noq"
epochs=20
vocab_sizes=(256)
max_seq_length=100
k_folds=5

for lr in "${lrs[@]}"; do
    for batch_size in "${batch_sizes[@]}"; do
        for dataset in "${datasets[@]}"; do
            for vocab_size in "${vocab_sizes[@]}"; do
                job_file="job_${dataset}_lr${lr}_bs${batch_size}_vs${vocab_size}.slurm"
                sed "s/{LR}/${lr}/g; s/{BATCH_SIZE}/${batch_size}/g; s/{DATASET}/${dataset}/g; s/{EPOCHS}/${epochs}/g; s/{VOCAB_SIZE}/${vocab_size}/g; s/{MAX_SEQ_LENGTH}/${max_seq_length}/g; s/{K_FOLDS}/${k_folds}/g" \
                    job_template.slurm > $job_file
                sbatch $job_file
                rm $job_file
            done
	done
    done
done
