import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from src.utils import *
from src.modules.Embedding import PositionalEmbedding, SpatialEmbedding

from ipdb import set_trace as st
from torch.distributions import Categorical

class StackTransformer(nn.Module):
    def __init__(self, model_opt, T, vocab_size):
        super(StackTransformer, self).__init__()
        self.t = T

        # Embedding for 1) sequential positions; 2) input tokens;
        self.tok_embed = nn.Embedding(vocab_size, model_opt.embed_dim)

        self.pos_embed = PositionalEmbedding(model_opt.embed_dim,
                                             model_opt.max_input_length, use_norm=True)
        self.spatial_embed_mo = SpatialEmbedding(model_opt.embed_dim,
                                                 int(math.sqrt(model_opt.mo_token_length)),
                                                 int(math.sqrt(model_opt.mo_token_length)), use_norm=True)
        self.temporal_embed_mo = PositionalEmbedding(model_opt.embed_dim, T, use_norm=True)

        # model layers
        self.pre_mapping = nn.Sequential(*[
            nn.Linear(model_opt.embed_dim, model_opt.hidden_dim),
            nn.GELU(),
            nn.Dropout(model_opt.dropout)
        ])

        self.encoder_m = nn.Sequential(
            *[Encoder_Mo_Layer(model_opt.hidden_dim, model_opt.immediate_hidden_dim,
                               model_opt.encoder_opt, t=T, dropout=model_opt.dropout)
              for _ in range(model_opt.mo_num_layer)])
        self.predicting = nn.Linear(in_features=model_opt.hidden_dim, out_features=vocab_size)

        # weight initialize
        self.apply(weights_init)

    def forward_mo(self, xmo):
        # Sequence/Spatial position embedding for MO
        xmo = rearrange(xmo, 'b (t l) -> (b t) l', t=self.t)
        positional_embed_mo = self.pos_embed(xmo)
        spatial_embed_mo = self.spatial_embed_mo(xmo)
        input_mo = self.tok_embed(xmo) + spatial_embed_mo + positional_embed_mo  # [BT, L]
        input_mo = self.pre_mapping(input_mo)
        input_mo = rearrange(input_mo, '(b t) l d -> b t l d', t=self.t)
        stacktwo_input = rearrange(input_mo, 'b t l d -> b (t l) d')
        stacktwo_output = self.encoder_m(stacktwo_input)
        # extract xmo logits
        stacktwo_output = rearrange(stacktwo_output, 'b (t l) d -> (b t) l d', t=self.t)
        logits_mo = stacktwo_output  # [B * T, mol, hidden_dim]
        logits_mo = rearrange(logits_mo, '(b t) l d -> b (t l) d', t=self.t)
        ## Predict token logits
        logits_mo = self.predicting(logits_mo)  # [B, T * mol, Vocab_size]
        return logits_mo

    def sample(self, logits):
        """
        logits: B, L, V
        return sample_ids - [B, L]
        """

        b, l, v = logits.shape
        logits_flatten = rearrange(logits, 'b l v -> (b l) v')
        sampled_ids = Categorical(logits=logits_flatten).sample().to(torch.int32)
        sampled_ids = rearrange(sampled_ids, '(b l) -> b l', b=b)
        return sampled_ids

    def forward(self, xmo):
        '''
            xmo: [B, T * mo_token_length], tokens of mo in range [mo_vocab_s, cmd_vocab_s)
        '''
        ## Predict logits_mo
        logits_mo = self.forward_mo(xmo)

        return logits_mo


def weights_init(m):
    classname = m.__class__.__name__
    if "Linear" in classname or "Embedding" == classname:
        print(f"Initializing Module {classname}.")
        nn.init.trunc_normal_(m.weight.data, 0.0, 0.02)
    # elif "Parameter" in classname:
    #     return nn.init.trunc_normal_(m, 0.0, 0.02)

class Encoder_Layer(nn.Module):
    """
    A layer for Transformer encoder using MultiHeadAttention and MLP along with skip connections and LayerNorm
    """
    def __init__(self, dim, hidden_dim, encoder_opt, dropout=0.1):
        super().__init__()
        self.encoder_opt = encoder_opt
        self.MultiHeadAttention = nn.MultiheadAttention(dim, num_heads=encoder_opt.num_head, dropout=dropout)
        self.LayerNorm1 = nn.LayerNorm(dim, eps=1e-12)
        self.LayerNorm2 = nn.LayerNorm(dim, eps=1e-12)
        self.MLP = nn.Sequential(*[
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim)
        ])
        self.dropout = nn.Dropout(dropout)

        if encoder_opt.get('switch_ln', False) is True:
            self.LayerNorm3 = nn.LayerNorm(dim, eps=1e-12)
            self.LayerNorm4 = nn.LayerNorm(dim, eps=1e-12)
        else:
            self.LayerNorm3 = nn.Identity()
            self.LayerNorm4 = nn.Identity()

    def forward(self, x, attn_mask=None):
        '''
            input x: [batch, seq_length, dim]
            attn_mask: [target_seq_length, source_seq_length]
        '''

        # 1、MultiHead Attention
        z = self.LayerNorm1(x)
        h = rearrange(z, 'b l c -> l b c')
        attn, _ = self.MultiHeadAttention(h, h, h, attn_mask=attn_mask)
        attn = rearrange(attn, 'l b c -> b l c')
        attn = self.LayerNorm3(attn)
        x = x + self.dropout(attn)

        # 2、MLP
        z = self.LayerNorm2(x)
        mlp = self.MLP(z)
        mlp = self.LayerNorm4(mlp)
        x = x + self.dropout(mlp)
        return x


class Encoder_Mo_Layer(nn.Module):
    """
    A layer for Transformer encoder using MultiHeadAttention and MLP along with skip connections and LayerNorm
    """

    def __init__(self, dim, hidden_dim, encoder_opt, t, dropout=0.1):
        super().__init__()
        self.t = t
        self.encoder_opt = encoder_opt
        self.MultiHeadAttention_temporal = nn.MultiheadAttention(dim, num_heads=encoder_opt.num_head, dropout=dropout)
        self.MultiHeadAttention_spatial = nn.MultiheadAttention(dim, num_heads=encoder_opt.num_head, dropout=dropout)
        self.LayerNorm1 = nn.LayerNorm(dim, eps=1e-12)
        self.LayerNorm2 = nn.LayerNorm(dim, eps=1e-12)
        self.LayerNorm3 = nn.LayerNorm(dim, eps=1e-12)
        self.MLP = nn.Sequential(*[
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim)
        ])
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, attn_mask=None):
        '''
            input x: [batch, t * seq_length, dim]
            attn_mask: [target_seq_length, source_seq_length]
        '''
        x = rearrange(x, 'b (t l) c -> b t l c', t=self.t)
        b, t, l, c = x.shape

        # 1、Spatial MultiHead Attention
        sx = rearrange(x, 'b t l c -> (b t) l c')
        z = self.LayerNorm1(sx)
        h = rearrange(z, 'bt l c -> l bt c')
        attn, _ = self.MultiHeadAttention_spatial(h, h, h, attn_mask=attn_mask)
        attn = rearrange(attn, 'l (b t) c -> b t l c', b=b, t=t)
        x = x + self.dropout(attn)

        # 2 Temporal MultiHead Attention
        tx = rearrange(x, 'b t l c -> (b l) t c')
        z = self.LayerNorm2(tx)
        h = rearrange(z, 'bl t c -> t bl c')
        attn, _ = self.MultiHeadAttention_temporal(h, h, h, attn_mask=attn_mask)
        attn = rearrange(attn, 't (b l) c -> b t l c', b=b, l=l)
        x = x + self.dropout(attn)

        # 2、MLP
        x = rearrange(x, 'b t l c -> b (t l) c')
        z = self.LayerNorm3(x)
        mlp = self.MLP(z)
        x = x + self.dropout(mlp)
        return x

