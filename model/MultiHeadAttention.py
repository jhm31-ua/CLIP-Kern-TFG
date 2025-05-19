import torch
import torch.nn as nn

from model.AttentionHead import AttentionHead

class MultiHeadAttention(nn.Module):
    def __init__(self, width, n_heads):
        super().__init__()

        self.head_size = width // n_heads
        self.W_o = nn.Linear(width, width)
        self.heads = nn.ModuleList([AttentionHead(width, self.head_size) for _ in range(n_heads)])

    def forward(self, x, mask = None):
        out = torch.cat([head(x, mask = mask) for head in self.heads], dim = -1)
        out = self.W_o(out)

        return out
