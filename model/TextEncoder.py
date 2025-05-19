import torch
import torch.nn as nn

from model.PositionalEmbedding import PositionalEmbedding 
from model.TransformerEncoder import TransformerEncoder

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, width, max_seq_length, n_heads, n_layers, emb_dim):
        super().__init__()

        self.max_seq_length = max_seq_length
        self.encoder_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = PositionalEmbedding(width, max_seq_length)
        self.encoder = nn.ModuleList([TransformerEncoder(width, n_heads) for _ in range(n_layers)])
        self.projection = nn.Parameter(torch.randn(width, emb_dim))

    def forward(self, text, mask = None):
        x = self.encoder_embedding(text)
        x = self.positional_embedding(x)

        for encoder_layer in self.encoder:
            x = encoder_layer(x, mask = mask)
        
        x = x[torch.arange(text.shape[0]), torch.sub(torch.sum(mask[:, 0], dim = 1), 1)]

        if self.projection is not None:
            x = x @ self.projection

        x = x / torch.norm(x, dim = -1, keepdim = True)

        return x