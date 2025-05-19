import numpy as np
import torch
import torch.nn as nn

class PositionalEmbedding(nn.Module):
    def __init__(self, width, max_seq_length):
        super().__init__()

        pe = torch.zeros(max_seq_length, width)

        for pos in range(max_seq_length):
            for i in range(width):
                if i % 2 == 0:
                    pe[pos][i] = np.sin(pos / (10000 ** (i / width)))
                else:
                    pe[pos][i] = np.cos(pos / (10000 ** ((i - 1) / width)))

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe
        return x