import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from src.utils import *
from src.modules.Embedding import PositionalEmbedding, SpatialEmbedding

from ipdb import set_trace as st

class StackTransformer(nn.Module):
    def __init__(self, model_opt, T, vocab_size):
        super(StackTransformer, self).__init__()
        self.t = T

        # Embedding for 1) sequential positions; 2) input tokens;
        self.tok_embed = nn.Embedding(vocab_size, model_opt.embed_dim)

        self.pos_embed = PositionalEmbedding(model_opt.embed_dim,
                                             model_opt.max_input_length, use_norm=True)
        self.spatial_embed_bg = SpatialEmbedding(model_opt.embed_dim,
                                                 int(math.sqrt(model_opt.bg_token_length)),
                                                 int(math.sqrt(model_opt.bg_token_length)), use_norm=True)
        self.spatial_embed_id = SpatialEmbedding(model_opt.embed_dim,
                                                 int(math.sqrt(model_opt.id_token_length)),
                                                 int(math.sqrt(model_opt.id_token_length)), use_norm=True)
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

        self.encoder_c = nn.Sequential(
            *[Encoder_Layer(model_opt.hidden_dim, model_opt.immediate_hidden_dim, model_opt.encoder_opt)
              for _ in range(model_opt.num_layer)])
        self.encoder_m = nn.Sequential(
            *[Encoder_Mo_Layer(model_opt.hidden_dim, model_opt.immediate_hidden_dim, model_opt.encoder_opt, t=T)
              for _ in range(model_opt.mo_num_layer)])
        self.predicting = nn.Linear(in_features=model_opt.hidden_dim, out_features=vocab_size)

        # weight initialize
        self.apply(weights_init)

    def forward(self, xbg, xid, hmo, xmo=None):
        '''
            xbg: [B, bg_token_length], tokens of bg in range [bg_vocab_s, id_vocab_s)
            xid: [B, id_token_length], tokens of id in range [id_vocab_s, mo_vocab_s)
            xmo: [B, T * mo_token_length], tokens of mo in range [mo_vocab_s, cmd_vocab_s)
            hmo: [B, T], command tokens in range [mo_vocab_s, vocab_size)
        '''
        ## Stage One:
        # Spatial postion embedding for BG/ID
        spatial_embed_bg = self.spatial_embed_bg(xbg) # [1, bg_token_length, d_model]
        spatial_embed_id = self.spatial_embed_bg(xid) # [1, id_token_length, d_model]
        bg_embed = self.tok_embed(xbg) + spatial_embed_bg # [B, bg_token_length, d_model]
        id_embed = self.tok_embed(xid) + spatial_embed_id # [B, id_token_length, d_model]
        # Temporal position embedding for HMO
        temporal_embed_mo = self.temporal_embed_mo(hmo)  # [1, T, d_model]
        hmo_embed = self.tok_embed(hmo) + temporal_embed_mo # [B, T, d_model]
        # Sequence position embedding for [xbg, xid, hmo]
        position_embed = self.pos_embed(torch.cat([xbg, xid, hmo], dim=1)) # [1, bg+id+T, d_model]
        input_bgidhmo = torch.cat([bg_embed, id_embed, hmo_embed], dim=1) + position_embed
        # pre_mapping input embeddings and feeding encoder_c
        stackone_input = self.pre_mapping(input_bgidhmo) # [B, bg+id+T, hidden_dim]
        stackone_output = self.encoder_c(stackone_input) # [B, bg+id+T, hidden_dim]
        # extract bg/id/hmo logits
        logits_bg = stackone_output[:, :xbg.shape[1], :] # [B, bgl, d]
        logits_id = stackone_output[:, xbg.shape[1]:xbg.shape[1]+xid.shape[1], :] # [B, idl, d]
        ## Predict token logits
        logits_bg = self.predicting(logits_bg)  # [B, bgl, Vocab_size]
        logits_id = self.predicting(logits_id)  # [B, idl, Vocab_size]

        if xmo is not None:
            ## Stage Two: Predict logits_mo
            zmo = stackone_output[:, xbg.shape[1] + xid.shape[1]:, :]  # [B, T, hidden_dim]
            zmo = rearrange(zmo, 'b (t l) d -> b t l d', t=self.t, l=1)
            # Sequence/Spatial position embedding for MO
            xmo = rearrange(xmo, 'b (t l) -> (b t) l', t=self.t)
            positional_embed_mo = self.pos_embed(xmo)
            spatial_embed_mo = self.spatial_embed_mo(xmo)
            input_mo = self.tok_embed(xmo) + spatial_embed_mo + positional_embed_mo # [BT, L]
            input_mo = self.pre_mapping(input_mo)
            input_mo = rearrange(input_mo, '(b t) l d -> b t l d', t=self.t)
            # Concat with ZMO
            stacktwo_input = torch.cat([zmo, input_mo], dim=2)
            stacktwo_input = rearrange(stacktwo_input, 'b t l d -> b (t l) d')
            stacktwo_output = self.encoder_m(stacktwo_input)
            # extract xmo logits
            stacktwo_output = rearrange(stacktwo_output, 'b (t l) d -> (b t) l d', t=self.t)
            logits_mo = stacktwo_output[:, 1:, :] # [B * T, mol, hidden_dim]
            logits_mo = rearrange(logits_mo, '(b t) l d -> b (t l) d', t=self.t)
            ## Predict token logits
            logits_mo = self.predicting(logits_mo)  # [B, T * mol, Vocab_size]
        else:
            logits_mo = None

        return logits_bg, logits_id, logits_mo


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

