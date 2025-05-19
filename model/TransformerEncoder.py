import torch.nn as nn

from model.MultiHeadAttention import MultiHeadAttention

class TransformerEncoder(nn.Module):
    def __init__(self, width, n_heads, r_mlp = 4):
        super().__init__()
        self.width = width
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(width)
        self.mha = MultiHeadAttention(width, n_heads)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(self.width, self.width * r_mlp),
            nn.GELU(),
            nn.Linear(self.width * r_mlp, self.width)
        )

    def forward(self, x, mask = None):
        x = x + self.mha(self.norm1(x), mask = mask)
        x = x + self.mlp(self.norm2(x))

        return x