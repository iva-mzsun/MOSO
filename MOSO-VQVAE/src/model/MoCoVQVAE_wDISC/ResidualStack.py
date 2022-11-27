import torch
import torch.nn as nn
import torch.nn.functional as F
from .Attentions import PatchwiseSelfAttention3D
from .Swin_Block import Mlp
from einops import rearrange

class ResidualStack(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, add_BN=False):
        super(ResidualStack, self).__init__()

        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        self._layers = nn.ModuleList()
        if add_BN is True:
            for i in range(num_residual_layers):
                curlayer = nn.Sequential(
                    nn.Conv2d(num_hiddens, num_residual_hiddens,
                              kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(num_residual_hiddens),
                    nn.GELU(),
                    nn.Conv2d(num_residual_hiddens, num_hiddens,
                              kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(num_hiddens),
                    nn.GELU(),
                )
                self._layers.append(curlayer)
        else:
            for i in range(num_residual_layers):
                curlayer = nn.Sequential(
                    nn.Conv2d(num_hiddens, num_residual_hiddens,
                              kernel_size=3, stride=1, padding=1),
                    nn.GELU(),
                    nn.Conv2d(num_residual_hiddens, num_hiddens,
                              kernel_size=1, stride=1, padding=0),
                    nn.GELU()
                )
                self._layers.append(curlayer)

    def forward(self, inputs):
        h = inputs
        for layer in self._layers:
            z = layer(h)
            h = h + z
        return F.gelu(h)

class AttnResidualBlock(nn.Module):
    def __init__(self, num_hiddens, window_hw, window_t, n_head, d_model, d_kv):
        super().__init__()

        self.norm1 = nn.LayerNorm(num_hiddens)
        self.attn = PatchwiseSelfAttention3D(window_hw, window_t, n_head, d_model, d_k=d_kv, d_v=d_kv)

        self.norm2 = nn.LayerNorm(num_hiddens)
        self.mlp = Mlp(in_features=num_hiddens, hidden_features=num_hiddens*2, act_layer=nn.GELU)

    def forward(self, x):
        """
        param x:    [B T C H W]
        return:     [B T C H W]
        """
        B, T, C, H, W = x.shape
        shortcut = x
        tx = rearrange(x, 'b t c h w -> b (t h w) c')
        tx = self.norm1(tx)
        tx = rearrange(tx, 'b (t h w) c -> b t c h w', t=T, h=H, w=W)
        tx = self.attn(tx)
        x = shortcut + tx

        tx = rearrange(x, 'b t c h w -> b (t h w) c')
        tx = self.mlp(self.norm2(tx))
        tx = rearrange(tx, 'b (t h w) c -> b t c h w', t=T, h=H, w=W)
        x = x + tx
        return x




class ResidualStack_UP(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, add_BN=False):
        super(ResidualStack_UP, self).__init__()
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        self._layers = nn.ModuleList()
        if add_BN is True:
            for i in range(num_residual_layers):
                curlayer = nn.Sequential(
                    nn.ConvTranspose2d(num_hiddens, num_residual_hiddens,
                                       kernel_size=6, stride=2, padding=2),
                    nn.BatchNorm2d(num_residual_hiddens),
                    nn.GELU(),
                    nn.Conv2d(num_residual_hiddens, num_residual_hiddens,
                              kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(num_residual_hiddens),
                    nn.GELU(),
                    nn.Conv2d(num_residual_hiddens, num_hiddens,
                              kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(num_hiddens),
                    nn.GELU(),
                )
                self._layers.append(curlayer)
        else:
            for i in range(num_residual_layers):
                curlayer = nn.Sequential(
                    nn.ConvTranspose2d(num_hiddens, num_residual_hiddens,
                                       kernel_size=6, stride=2, padding=2),
                    nn.GELU(),
                    nn.Conv2d(num_hiddens, num_residual_hiddens,
                              kernel_size=3, stride=1, padding=1),
                    nn.GELU(),
                    nn.Conv2d(num_residual_hiddens, num_hiddens,
                              kernel_size=1, stride=1, padding=0),
                    nn.GELU()
                )
                self._layers.append(curlayer)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, inputs):
        h = inputs
        for layer in self._layers:
            z = layer(h)
            hx2 = self.upsample(h)
            h = hx2 + z
        return h