import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.optim as optim
import torch.nn as nn
import pytorch_lightning as pl
import wandb
from sklearn.manifold import TSNE
from torchvision.transforms import ToPILImage

from model.ImageEncoder import ImageEncoder
from model.TextEncoder import TextEncoder
from model.BPEKernTokenizer import BPEKernTokenizer
from util import Parameters, Functions

class CLIPModel(pl.LightningModule):
    def __init__(self, emb_dim, img_width, img_size, patch_size, n_channels, img_layers, img_heads, vocab_size, text_width, max_seq_length, text_heads, text_layers, lr):
        super().__init__()

        self.image_encoder = ImageEncoder(img_width, img_size, patch_size, n_channels, img_layers, img_heads, emb_dim)
        self.text_encoder = TextEncoder(vocab_size, text_width, max_seq_length, text_heads, text_layers, emb_dim)

        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.emb_dim = emb_dim
        self.max_seq_length = max_seq_length
        self.lr = lr

        self.save_hyperparameters()

    def forward(self, image, text, mask = None):
        image_features = self.image_encoder(image)
        text_features = self.text_encoder(text, mask)

        logits = (image_features @ text_features.transpose(-2, -1)) * torch.exp(self.temperature)

        labels = torch.arange(logits.shape[0], device = Parameters.device)
        loss_image = nn.functional.cross_entropy(logits.transpose(-2, -1), labels)
        loss_text = nn.functional.cross_entropy(logits, labels)
        loss = (loss_image + loss_text) / 2 # TODO: Mirar alternativas para la pérdida

        return loss

    def training_step(self, batch, batch_idx):
        img, cap, mask = (batch["image"], batch["caption"], batch["mask"])

        if batch_idx == 0 and self.current_epoch == 0:
            table = wandb.Table(columns = ["Image", "Caption", "Mask"])
            for i in range(img.shape[0]):
                table.add_data(
                    wandb.Image(ToPILImage()(img[i].squeeze(0))),
                    ", ".join(map(str, cap[i].tolist())),
                    ", ".join(map(str, mask[i][0].tolist()))
                )
            self.logger.experiment.log({"batch_example": table})

        loss = self(img, cap, mask) 
        self.log("train_loss", loss, on_step = False, on_epoch = True, prog_bar = True)

        return loss

    def validation_step(self, batch):
        img, cap, mask = (batch["image"], batch["caption"], batch["mask"])

        loss = self(img, cap, mask) 
        self.log("validation_loss", loss, on_step = False, on_epoch = True, prog_bar = True)

        return loss

    def on_test_start(self):
        num_test_samples = len(self.trainer.datamodule.test_dataloader().dataset)

        self.image_features = torch.zeros((num_test_samples, self.emb_dim), device = self.device)
        self.text_features = torch.zeros((num_test_samples, self.emb_dim), device = self.device)
        self.labels = torch.zeros((num_test_samples, self.max_seq_length), dtype = torch.long, device = self.device)
        self.masks = torch.zeros((num_test_samples, self.max_seq_length), dtype = torch.bool, device = self.device)
        
        self.test_idx = 0
    
    def test_step(self, batch, batch_idx): 
        img, cap, mask = (batch["image"], batch["caption"], batch["mask"])

        image_features = self.image_encoder(img)
        text_features = self.text_encoder(cap, mask = mask)
        image_features /= image_features.norm(dim = -1, keepdim = True)
        text_features /= text_features.norm(dim = -1, keepdim = True)

        batch_size = image_features.shape[0]

        self.image_features[self.test_idx:self.test_idx + batch_size] = image_features
        self.text_features[self.test_idx:self.test_idx + batch_size] = text_features
        self.labels[self.test_idx:self.test_idx + batch_size] = cap
        self.masks[self.test_idx:self.test_idx + batch_size] = mask[:, 0]

        self.test_idx += batch_size

        return image_features, text_features, cap, mask

    def on_test_epoch_end(self):
        similarity = (100.0 * self.image_features @ self.text_features.T).softmax(dim = -1)
        indices = Functions.unique_topk(self.trainer.topk, similarity, self.labels)

        tokenizer = BPEKernTokenizer(filepath = self.trainer.datamodule.tokenizer_path, load = True)
        pred = [
            [tokenizer.encode(
                    open(os.path.join(self.trainer.datamodule.dataset_path, f"{self.trainer.datamodule.test_set.data_files[int(i)]}.krn"), 'r').read().strip(),
                    max_seq_length=self.max_seq_length
                ) for i in sample
            ] for sample in indices
        ]

        pred_labels = torch.stack([torch.stack([e[0] for e in batch]) for batch in pred]).to(Parameters.device)
        pred_masks = torch.stack([torch.stack([e[1] for e in batch]) for batch in pred]).to(Parameters.device)

        table = wandb.Table(columns = ["Predictions", "True Label", "Correct?"])
        correct, total = 0, self.labels.size(0)
        for i in range(total):
            guess = torch.any(torch.all(pred_labels[i] == self.labels[i].unsqueeze(0), dim = 1)).item()
            table.add_data(
                [tokenizer.decode(pred_labels[i][j], pred_masks[i][j]) for j in range(self.trainer.topk)], 
                tokenizer.decode(self.labels[i], self.masks[i]), 
                guess
            )
            if guess:
                correct += 1

        accuracy = 100 * correct / total
        self.log(f"top{self.trainer.topk}_accuracy", accuracy, prog_bar = False)
        self.logger.experiment.log({f"comparison_top{self.trainer.topk}": table})
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr = self.lr)

        
