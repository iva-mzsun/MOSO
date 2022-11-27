import torch
import torch.nn as nn
import torch.nn.functional as F
from .ResidualStack import ResidualStack
from einops import rearrange


class Encoder_Identity(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
                 ds_content, T, suf_method="avg_pool"):
        """
        T: The number of frames in each input video.
        """
        super(Encoder_Identity, self).__init__()
        self._ds_m = ds_content
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        # Step 1: downsample for each video frame
        input_channels = 3
        self._layers = nn.ModuleList()
        for i in range(ds_content):
            curlayer = nn.Sequential(
                nn.Conv2d(input_channels, self._num_hiddens,
                          kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(self._num_hiddens),
                nn.GELU(),
                nn.Conv2d(self._num_hiddens, self._num_hiddens,
                          kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(self._num_hiddens),
                nn.GELU()
            )
            input_channels = self._num_hiddens
            self._layers.append(curlayer)

        # Step 2: Lower the dimension of concatenated frame features.
        self._suf_method = suf_method.lower()
        if self._suf_method == "avg_pool":
            self._suf_layer = nn.AvgPool1d(T)
        elif self._suf_method == "conv":
            self._suf_layer = nn.Sequential(
                nn.Conv2d(T * self._num_hiddens, self._num_hiddens,
                          kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(self._num_hiddens),
                nn.GELU()
            )
        else:
            assert ValueError(f"No Implementation for Encoder_Content's suf_layer: {self._suf_method}!")

        # Step 3: Get output
        self._residual = ResidualStack(self._num_hiddens, self._num_residual_layers,
                                       self._num_residual_hiddens, True)

    def forward(self, x_id):
        """
        输入T帧视频帧，保留帧间差大于p的像素，帧间差小于p的像素为0；
        输出1张Token图；
        :param x: [B, T, C, H, W]
        :return: [B, 1, D, H', W']
        """
        B, T, C, H, W = x_id.shape
        # Step 1: 降低每帧的图像分辨率。  xs=[B, T, D, H // 2**ds, W // 2**ds]
        xs = rearrange(x_id, 'b t c h w -> (b t) c h w')
        for i in range(self._ds_m):
            h = self._layers[i](xs)
            xs = F.relu(h)
        _, D, HS, WS = xs.shape

        # Step 2: Dimension维度Concatenation。 z=[B, D', HS, WS]
        if self._suf_method == "avg_pool":
            xs = rearrange(xs, '(b t) d hs ws -> (b hs ws) d t', b=B)
            zs = self._suf_layer(xs)
            zs = torch.squeeze(zs)
            z = rearrange(zs, '(b hs ws) d -> b d hs ws',
                          b=B, d=D, hs=HS, ws=WS)
        elif self._suf_method == "conv":
            xs = rearrange(xs, '(b t) d hs ws -> b (t d) hs ws', b=B)
            z = self._suf_layer(xs)
        else:
            z = None
            assert ValueError(f"No Implementation for Encoder_Content's suf_layer: {self._suf_method}!")

        # Step 3: 获取输出的Content Feature.
        z_ = self._residual(z)
        z_ = z_.unsqueeze(1)
        return z_