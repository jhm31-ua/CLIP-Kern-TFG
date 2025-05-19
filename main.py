import fire
import numpy as np
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torch
import wandb

from util import Parameters
from model.CLIPModel import CLIPModel
from data.KernDataModule import KernDataModule

class Main:
    def __init__(self,
                 batch_size = Parameters.batch_size,
                 epochs = Parameters.epochs,
                 lr = Parameters.lr,
                 model_path = Parameters.model_path,
                 tokenizer_path = Parameters.tokenizer_path,
                 dataset_path = Parameters.dataset_path,
                 split_path = Parameters.split_path,
                 vocab_size = Parameters.vocab_size,
                 max_seq_length = Parameters.max_seq_length,
                 k = Parameters.k,
                 wandb_name = Parameters.wandb_name,
                 wandb_group = Parameters.wandb_group):
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.dataset_path = dataset_path
        self.split_path = split_path
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.k = k
        self.wandb_name = wandb_name
        self.wandb_group = wandb_group

        self.data_module = KernDataModule(self.k, self.split_path, self.dataset_path, self.tokenizer_path, self.max_seq_length, self.batch_size)
        self.model = CLIPModel(Parameters.emb_dim, Parameters.img_width, Parameters.img_size, Parameters.patch_size,
                               Parameters.n_channels, Parameters.img_layers, Parameters.img_heads, self.vocab_size,
                               Parameters.text_width, self.max_seq_length, Parameters.text_heads,
                               Parameters.text_layers, self.lr)

        wandb_logger = WandbLogger(
            project = "clip-kern-clean",
            group = self.wandb_group, 
            name = self.wandb_name,
            log_model = True,
            config = {  
                "seed": Parameters.seed,
                "learning_rate": self.lr,
                "epochs": self.epochs,
                "batch": self.batch_size,
                "dataset": self.dataset_path,
                "tokenizer": self.tokenizer_path,
                "split": self.split_path,
                "k": self.k,
                "vocab_size": self.vocab_size,
                "max_seq_length": self.max_seq_length,
            }
        )

        best_model_callback = ModelCheckpoint(monitor = "validation_loss", mode = "min", save_top_k = 1, dirpath = self.model_path, filename=f"best_model-k={self.k}-{{epoch}}-{{validation_loss:.2f}}")
        self.trainer = pl.Trainer(callbacks = [best_model_callback], logger = wandb_logger, max_epochs = self.epochs, accelerator = "auto", devices = 1)

        self.best_ckpt_path = None
        best_model_callback.to_save_callback = lambda path: setattr(self, 'best_ckpt_path', path)

    def train(self):
        self.trainer.fit(self.model, self.data_module)

    def test(self):
        klist = [1, 5, 10]
        for k in klist:
            self.trainer.topk = k
            self.trainer.test(self.model, self.data_module, ckpt_path = self.best_ckpt_path)

    def train_test(self):
        self.train()
        self.test()

if __name__ == "__main__":
    fire.Fire(Main) # Ejemplo de uso: python main.py --batch_size=64 --epochs=20 --lr=0.0001 train

