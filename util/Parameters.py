import numpy as np
import torch
import datetime
import wandb

emb_dim = 16
img_width = 9
img_size = (100, 100)
patch_size = (25, 25)
n_channels = 1
img_layers = 6
img_heads = 3
vocab_size = 1024
text_width = 32
max_seq_length = 100
text_heads = 8
text_layers = 4
lr = 5e-4
epochs = 32
batch_size = 16
k = 0
k_folds = 5
seed = 42
workspace = "/workspace"
dataset = "MTD-custom-noq"
dataset_path = f"{workspace}/my-datasets/{dataset}"
split_path = f"{workspace}/splits/{dataset}/k{k_folds}"
model_path = f"{workspace}/clip_kern_b{batch_size}_e{epochs}_v{vocab_size}_s{max_seq_length}.pt"
tokenizer_path = f"{workspace}/tokenizers/tokenizer_{vocab_size}.pickle"
gpu_log_path = f"{workspace}/gpu/{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wandb_name = f"clip_kern_b{batch_size}_e{epochs}_lr{lr}_v{vocab_size}_s{max_seq_length}_k{k}_{dataset}", 
wandb_group = f"clip_kern_b{batch_size}_e{epochs}_lr{lr}_v{vocab_size}_s{max_seq_length}_kf{k_folds}_{dataset}"
