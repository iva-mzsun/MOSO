import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat, reduce


class PatchwiseAttention(nn.Module):
    def __init__(self, ngroup_c, ngroup_m, n_head, d_model, d_k=None, d_v=None, dropout=0):
        super(PatchwiseAttention, self).__init__()
        """
        feat_c: 8 x 32 x 32 -> 8 x (4 x 8) x (4 x 8)
        feat_m: 8 x 8 x 8 -> 8 x (4 x 2) x (4 x 2) (x 3) 
        """
        self.ngroup_c = ngroup_c
        self.ngroup_m = ngroup_m
        # self.ngroup_t = ngroup_t
        self.attention_module = MultiHeadAttention(n_head=n_head, d_model=d_model,
                                                   d_k=d_k, d_v=d_v, dropout=dropout)

    def forward(self, feat_c, feat_m):
        """
         feat_c: [ B, 1, C, HC, WC ]
         feat_m: [ B, T, C, HM, WM ]
        """
        B, T, C, HM, WM = feat_m.shape
        _, _, _, HC, WC = feat_c.shape
        feat_c = repeat(feat_c.squeeze(1), 'B C HC WC->B T C HC WC', T=T)

        q = rearrange(feat_c, 'B T C (GH H) (GW W) -> (B T GH GW) (H W) C',
                      GH=self.ngroup_c, GW=self.ngroup_c)

        feat_m_mid = feat_m.unsqueeze(2) # [B T 1 C H W]
        feat_m_pre = torch.zeros_like(feat_m_mid)
        feat_m_pre[:, 1:] = feat_m_mid[:, :-1]
        feat_m_nxt = torch.zeros_like(feat_m_mid)
        feat_m_nxt[:, :-1] = feat_m_mid[:, 1:]
        feat_m = torch.cat([feat_m_pre, feat_m_mid, feat_m_nxt], dim=2) # [B T 3 C H W]

        kv = rearrange(feat_m, 'B T N C (GH H) (GW W) -> (B T GH GW) (H W N) C',
                       GH=self.ngroup_m, GW=self.ngroup_m)

        feat_c, attn = self.attention_module(q, kv, kv)
        feat_c = rearrange(feat_c, '(B T GH GW) (H W) C -> B T C (GH H) (GW W)',
                           B=B, T=T, GH=self.ngroup_c, H=HC // self.ngroup_c)

        return feat_c, attn

class PatchwiseSelfAttention3D(nn.Module):
    def __init__(self, window_hw, window_t, n_head, d_model, d_k=None, d_v=None, dropout=0):
        super().__init__()
        """
        feat_m: 8 x 8 x 8 -> 8 x (4 x 2) x (4 x 2) (x 3) 
        """
        self.window_h = window_hw
        self.window_w = window_hw
        self.window_t = window_t
        self.attention_module = MultiHeadAttention(n_head=n_head, d_model=d_model,
                                                   d_k=d_k, d_v=d_v, dropout=dropout)

    def forward(self, feat_m):
        """
         feat_m: [ B, T, C, H, W ]
        """
        B, T, C, H, W = feat_m.shape

        qkv = rearrange(feat_m, 'B (GT T) C (GH H) (GW W) -> (B GT GH GW) (T H W) C',
                       GH=H // self.window_h, GW=W // self.window_w, GT=T // self.window_t)

        feat_m, attn = self.attention_module(qkv, qkv, qkv)
        feat_m = rearrange(feat_m, '(B GT GH GW) (T H W) C -> B (GT T) C (GH H) (GW W)',
                           GH=H // self.window_h, GW=W // self.window_w, GT=T // self.window_t,
                           B=B, T=self.window_t, H=self.window_h, W=self.window_w)

        return feat_m


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):
        """

        :param q: [b, len_q, d_model]
        :param k: [b, len_k, d_model]
        :param v: [b, len_v, d_model]
        :return:
        """

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn

