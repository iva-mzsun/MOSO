import math
import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len, use_norm):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

        if use_norm is False:
            self.norm = lambda x: x
        else:
            self.norm = nn.LayerNorm(d_model)


    def forward(self, x):
        '''
            x: [B, L]
            return: [1, L, d_model]
        '''
        return self.norm(self.pe[:, :x.size(1)])

class SpatialEmbedding(nn.Module):
    def __init__(self, d_model, h, w, use_norm):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(h * w, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, w).repeat(h).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

        if use_norm is False:
            self.norm = lambda x: x
        else:
            self.norm = nn.LayerNorm(d_model)


    def forward(self, x):
        '''
            x: [B, L] / [B, L, V]
            return: [1, L, d_model]
        '''
        return self.norm(self.pe[:, :x.size(1)])
