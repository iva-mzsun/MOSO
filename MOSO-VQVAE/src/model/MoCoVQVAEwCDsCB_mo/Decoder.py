import torch, ipdb
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import repeat, rearrange
from .ResidualStack import ResidualStack_UP, ResidualStack
from .MergeModule import MergeModule, MergeModule_simple
from .Attentions import PatchwiseAttention


class Decoder(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
                 ds_content, ds_motion, ds_identity, ds_background, dropout=0.1):
        """
        ds_diff_cm = ds_content - ds_motion
        HC = HM * 2**ds_diff_cm
        ds_decoder = ds_content
        """
        super(Decoder, self).__init__()
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        # Step 3: Get content feature
        self._pre_layer = nn.Conv2d(self._num_hiddens, self._num_hiddens,
                                    kernel_size=3, padding=1, stride=1)
        self._residual = ResidualStack(self._num_hiddens,
                                       self._num_residual_layers,
                                       self._num_residual_hiddens, True)

        # Step 4: upsample and Output reconstructed video frames
        self.downsample_factor = ds_motion
        self._layers = nn.ModuleList()
        for i in range(self.downsample_factor):
            if i == self.downsample_factor - 1:
                curlayer = nn.ConvTranspose2d(self._num_hiddens, self._num_hiddens // 2,
                                              kernel_size=5, stride=2, padding=2)
            else:
                curlayer = nn.ConvTranspose2d(self._num_hiddens, self._num_hiddens,
                                              kernel_size=5, stride=2, padding=2)

            self._layers.append(curlayer)

        self._last_layer = nn.Conv2d(self._num_hiddens // 2, 3,
                                     kernel_size=3, stride=1, padding=1)

    def forward(self, feat_mo):
        """
        feat_bg: [B, 1, D, H_BG, W_BG]
        feat_id: [B, 1, D, H_ID, W_ID]
        feat_mo: [B, T, D, H_MO, W_MO]
        """

        # Step 2: Merge feat_bg, feat_id, feat_mo
        feat_out = feat_mo

        # Step 3: Get upsampled content feature
        B, T, C, H, W = feat_out.shape
        feat_out = rearrange(feat_out, 'B T C H W->(B T) C H W')
        feat_out = self._residual(self._pre_layer(feat_out))

        # Step 4: Output reconstrued video frames
        z = feat_out
        shape = np.array(feat_out.shape)
        for i in range(self.downsample_factor):
            h = self._layers[i](z, output_size=shape * (2 ** (i + 1)))
            z = F.gelu(h)

        rec = self._last_layer(z)
        x_rec = rearrange(rec, '(B T) C H W -> B T C H W', T=T)
        return x_rec, None, None