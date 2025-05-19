import torch
import torch.nn as nn
import torch.nn.functional as F

from model.PositionalEmbedding import PositionalEmbedding
from model.TransformerEncoder import TransformerEncoder

class ImageEncoder(nn.Module):
    def __init__(self, width, img_size, patch_size, n_channels, n_layers, n_heads, emb_dim):
        super().__init__()

        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, "img_size dimensions must be divisible by patch_size dimensions"
        assert width % n_heads == 0, "width must be divisible by n_heads"

        self.n_patches = (img_size[0] * img_size[1]) // (patch_size[0] * patch_size[1])
        self.max_seq_length = self.n_patches + 1
        self.linear_project = nn.Conv2d(n_channels, width, kernel_size = patch_size, stride = patch_size)
        self.norm_layer = nn.LayerNorm(width)
        self.cls_token = nn.Parameter(torch.randn(1, 1, width))
        self.positional_embedding = PositionalEmbedding(width, self.max_seq_length)
        self.encoder = nn.ModuleList([TransformerEncoder(width, n_heads) for _ in range(n_layers)])
        self.projection = nn.Parameter(torch.randn(width, emb_dim))

    def forward(self, x):
        x = self.linear_project(x)
        x = x.flatten(2).transpose(1, 2)

        x = torch.cat((self.cls_token.expand(x.size()[0], -1, -1), x), dim = 1)
        x = self.positional_embedding(x)

        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            x = self.norm_layer(x)

        x = x[:, 0, :]

        if self.projection is not None:
            x = F.dropout(x, p = 0.05, training = self.training)
            x = x @ self.projection

        x = x / torch.norm(x, dim = -1, keepdim = True)

        return x