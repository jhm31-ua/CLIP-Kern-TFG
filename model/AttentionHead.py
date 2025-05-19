import torch
import torch.nn as nn

class AttentionHead(nn.Module):
    def __init__(self, width, head_size):
        super().__init__()
        self.head_size = head_size

        self.query = nn.Linear(width, head_size)
        self.key = nn.Linear(width, head_size)
        self.value = nn.Linear(width, head_size)

    def forward(self, x, mask = None):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention = Q @ K.transpose(-2,-1)
        attention = attention / (self.head_size ** 0.5)

        if mask is not None:
            attention = attention.masked_fill(mask == 0, float("-inf"))

        attention = torch.softmax(attention, dim = -1)
        attention = attention @ V
        return attention