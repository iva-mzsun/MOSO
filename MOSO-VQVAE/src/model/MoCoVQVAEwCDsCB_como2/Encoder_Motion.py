import torch
import torch.nn as nn
import torch.nn.functional as F
from .ResidualStack import ResidualStack, AttnResidualBlock
from einops import rearrange
from .Attentions import MultiHeadAttention, PatchwiseSelfAttention3D
from .Swin_Block import SwinBlock

class Encoder_Motion(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, ds_motion,
                 n_head, d_model, d_kv, time_head):
        super(Encoder_Motion, self).__init__()
        self._ds_m = ds_motion
        self._time_head = time_head
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        # Step 1: downsample for each video frame
        input_channels = 3
        self._layers = nn.ModuleList()
        for i in range(ds_motion):
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

        # Step 2: Self-attention
        self._attention = MultiHeadAttention(n_head=n_head, d_model=d_model, d_k=d_kv, d_v=d_kv)

        # Step 3: Get output
        self._residual = ResidualStack(self._num_hiddens, self._num_residual_layers,
                                       self._num_residual_hiddens, True)

    def forward(self, x_mo):
        """
        输入T帧视频，得到T张Token图。
        :param x: [B, T, C, H, W]
        :return: [B, T, D, H', W']
        """
        B, T, C, H, W = x_mo.shape
        # Step 1: 降低每帧的图像分辨率。  xs=[B, T, D, H // 2**ds, W // 2**ds]
        xs = rearrange(x_mo, 'b t c h w -> (b t) c h w')
        for i in range(self._ds_m):
            h = self._layers[i](xs)
            xs = F.relu(h)
        _, D, HS, WS = xs.shape

        # Step 2: Self-attention
        _, _, H, W = xs.shape
        z = rearrange(xs, '(B N T) C H W -> (B N H W) T C', B=B, T=self._time_head)
        z_, attn_ = self._attention(z, z, z)
        z = rearrange(z_, '(B N H W) T C -> (B N T) C H W', H=H, W=W, N=T // self._time_head)

        # Step 3: 获取输出的Motion Feature.
        z_ = self._residual(z)
        z_ = rearrange(z_, '(B T) C H W -> B T C H W', B=B)
        return z_

class Encoder_Motion_TA(nn.Module):
    '''
    Time-agnostic Motion Encoder
    '''
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, ds_motion,
                 n_head, d_model, d_kv):
        super(Encoder_Motion_TA, self).__init__()
        self._ds_m = ds_motion
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        # Step 1: downsample for each video frame
        input_channels = 3
        self._layers = nn.ModuleList()
        for i in range(ds_motion):
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

        # Step 2: Get output
        self._residual = ResidualStack(self._num_hiddens, self._num_residual_layers,
                                       self._num_residual_hiddens, True)

    def forward(self, x_mo):
        """
        输入T帧视频，得到T张Token图。
        :param x: [B, T, C, H, W]
        :return: [B, T, D, H', W']
        """
        B, T, C, H, W = x_mo.shape
        # Step 1: 降低每帧的图像分辨率。  xs=[B, T, D, H // 2**ds, W // 2**ds]
        xs = rearrange(x_mo, 'b t c h w -> (b t) c h w')
        for i in range(self._ds_m):
            h = self._layers[i](xs)
            xs = F.relu(h)
        _, D, HS, WS = xs.shape

        # Step 2: 获取输出的Motion Feature.
        z_ = self._residual(xs)
        z_ = rearrange(z_, '(B T) C H W -> B T C H W', B=B)
        return z_


class Encoder_Motion_Attn(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, ds_motion, n_head, d_model, d_kv):
        super().__init__()
        self._ds_m = ds_motion
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers

        # Step 1: downsample for each video frame
        input_channels = 3
        self._layers = nn.ModuleList()
        for i in range(ds_motion):
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

        # Step 2: Attention Residual
        self._attns = nn.ModuleList()
        for i in range(num_residual_layers):
            curblock0 = AttnResidualBlock(num_hiddens=num_hiddens,
                                          window_hw=4, window_t=1,
                                          n_head=n_head, d_model=d_model, d_kv=d_kv)
            self._attns.append(curblock0)
            curblock1 = AttnResidualBlock(num_hiddens=num_hiddens,
                                          window_hw=2, window_t=2,
                                          n_head=n_head, d_model=d_model, d_kv=d_kv)
            self._attns.append(curblock1)
            curblock2 = AttnResidualBlock(num_hiddens=num_hiddens,
                                          window_hw=1, window_t=8,
                                          n_head=n_head, d_model=d_model, d_kv=d_kv)
            self._attns.append(curblock2)

    def forward(self, x_mo):
        """
        输入T帧视频，得到T张Token图。
        :param x: [B, T, C, H, W]
        :return: [B, T, D, H', W']
        """
        B, T, C, H, W = x_mo.shape
        # Step 1: 降低每帧的图像分辨率。  xs=[B, T, D, H // 2**ds, W // 2**ds]
        xs = rearrange(x_mo, 'b t c h w -> (b t) c h w')
        for i in range(self._ds_m):
            h = self._layers[i](xs)
            xs = F.relu(h)
        _, D, HS, WS = xs.shape

        # Step 2: Self-attention
        _, _, H, W = xs.shape
        z = rearrange(xs, '(B T) C H W -> B T C H W', B=B)
        for i in range(3 * self._num_residual_layers):
            h = self._attns[i](z)
            z = h
        return z

class Encoder_Motion_Attn_Res(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
                 ds_motion, n_head, d_model, d_kv):
        super().__init__()
        self._ds_m = ds_motion
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        # Step 1: downsample for each video frame
        input_channels = 3
        self._layers = nn.ModuleList()
        for i in range(ds_motion):
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

        # Step 2: Attention Residual
        self._attns = nn.ModuleList()
        for i in range(num_residual_layers):
            curblock0 = AttnResidualBlock(num_hiddens=num_hiddens,
                                          window_hw=4, window_t=1,
                                          n_head=n_head, d_model=d_model, d_kv=d_kv)
            self._attns.append(curblock0)
            curblock1 = AttnResidualBlock(num_hiddens=num_hiddens,
                                          window_hw=2, window_t=2,
                                          n_head=n_head, d_model=d_model, d_kv=d_kv)
            self._attns.append(curblock1)
            curblock2 = AttnResidualBlock(num_hiddens=num_hiddens,
                                          window_hw=1, window_t=8,
                                          n_head=n_head, d_model=d_model, d_kv=d_kv)
            self._attns.append(curblock2)

        # Step 3: Get output
        self._residual = ResidualStack(self._num_hiddens, self._num_residual_layers,
                                       self._num_residual_hiddens, True)

    def forward(self, x_mo):
        """
        输入T帧视频，得到T张Token图。
        :param x: [B, T, C, H, W]
        :return: [B, T, D, H', W']
        """
        B, T, C, H, W = x_mo.shape
        # Step 1: 降低每帧的图像分辨率。  xs=[B, T, D, H // 2**ds, W // 2**ds]
        xs = rearrange(x_mo, 'b t c h w -> (b t) c h w')
        for i in range(self._ds_m):
            h = self._layers[i](xs)
            xs = F.relu(h)
        _, D, HS, WS = xs.shape

        # Step 2: Self-attention
        _, _, H, W = xs.shape
        z = rearrange(xs, '(B T) C H W -> B T C H W', B=B)
        for i in range(3 * self._num_residual_layers):
            h = self._attns[i](z)
            z = h

        # Step 3: 获取输出的Motion Feature.
        z = rearrange(z, 'B T C H W -> (B T) C H W', B=B)
        z_ = self._residual(z)
        z_ = rearrange(z_, '(B T) C H W -> B T C H W', B=B)
        return z_