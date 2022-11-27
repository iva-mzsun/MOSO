import torch, ipdb
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import repeat, rearrange

class MergeModule_simple(nn.Module):
    def __init__(self, d_model, ds_diff_cm):
        super(MergeModule_simple, self).__init__()

        self.ds_diff_cm = ds_diff_cm
        # layers for getting the weight
        self._fc_l1 = nn.Sequential(
            nn.Conv2d(3 * d_model, d_model, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model, d_model, kernel_size=1)
        )

        self._fc = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, feat_bg, feat_id, feat_m):
        """
         feat_bg: [ B, T, C, HC, WC ]
         feat_id: [ B, T, C, HC, WC]
         feat_m: [ B, T, C, HM, WM ], HC = 4 * HM
        """
        _, T, _, _, _ = feat_m.shape

        # get multi-scale factors, l3: 8, l2: 16, l1: 32
        feat_m_l3 = rearrange(feat_m, 'B T C H W->(B T) C H W')
        feat_m_l2 = self.upsample(feat_m_l3)
        feat_m_l1 = self.upsample(feat_m_l2)

        feat_bg_l1 = rearrange(feat_bg, 'B T C H W->(B T) C H W')
        feat_id_l1 = rearrange(feat_id, 'B T C H W->(B T) C H W')

        # get multi-scale weight
        weight_l1 = self._fc_l1(torch.cat([feat_m_l1, feat_bg_l1, feat_id_l1], dim=1))
        weight = F.sigmoid(self._fc(weight_l1))
        feat_out = weight * feat_bg_l1 + (1 - weight) * feat_id_l1
        feat_out = rearrange(feat_out, '(B T) C H W->B T C H W', T=T)
        return feat_out

class MergeModule(nn.Module):
    def __init__(self, d_model, ds_diff_cm):
        super(MergeModule, self).__init__()
        self.ds_diff_cm = ds_diff_cm
        if ds_diff_cm == 2:
            pass
        elif ds_diff_cm == 1:
            self._mo_ds_1 = nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        elif ds_diff_cm == 0:
            self._mo_ds_1 = nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1)
            self._mo_ds_2 = nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        else:
            raise ValueError(f"No implemention for ds_diff_cm: {ds_diff_cm}")

        # layers for downsample
        self._mo_ds_1 = nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        self._mo_ds_2 = nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1)

        # layers for getting the weight
        self._fc_l1 = nn.Sequential(
            nn.Conv2d(2 * d_model, d_model, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model, d_model, kernel_size=1)
        )
        self._fc_l2 = nn.Sequential(
            nn.Conv2d(2 * d_model, d_model, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model, d_model, kernel_size=1)
        )
        self._fc_l3 = nn.Sequential(
            nn.Conv2d(2 * d_model, d_model, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model, d_model, kernel_size=1)
        )

        self._fc = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, feat_co, feat_m):
        """
         feat_bg: [ B, T, C, HC, WC ]
         feat_id: [ B, T, C, HC, WC]
         feat_m: [ B, T, C, HM, WM ], HC = 4 * HM
        """
        _, T, _, _, _ = feat_m.shape

        # get multi-scale factors, l3: 8, l2: 16, l1: 32
        if self.ds_diff_cm == 2:
            feat_m_l3 = rearrange(feat_m, 'B T C H W->(B T) C H W')
            feat_m_l2 = self.upsample(feat_m_l3)
            feat_m_l1 = self.upsample(feat_m_l2)
        elif self.ds_diff_cm == 1:
            feat_m_l2 = rearrange(feat_m, 'B T C H W->(B T) C H W')
            feat_m_l1 = self.upsample(feat_m_l2)
            feat_m_l3 = self._mo_ds_1(feat_m_l2)
        elif self.ds_diff_cm == 0:
            feat_m_l1 = rearrange(feat_m, 'B T C H W->(B T) C H W')
            feat_m_l2 = self._mo_ds_1(feat_m_l1)
            feat_m_l3 = self._mo_ds_2(feat_m_l2)

        feat_co_l1 = rearrange(feat_co, 'B T C H W->(B T) C H W')
        feat_co_l2 = self._mo_ds_1(feat_co_l1)
        feat_co_l3 = self._mo_ds_2(feat_co_l2)

        # get multi-scale weight
        feat_l3 = self._fc_l3(torch.cat([feat_m_l3, feat_co_l3], dim=1))
        feat_l2 = self._fc_l2(torch.cat([feat_m_l2, feat_co_l2], dim=1))
        feat_l2 = feat_l2 + self.upsample(feat_l3)
        feat_l1 = self._fc_l1(torch.cat([feat_m_l1, feat_co_l1], dim=1))
        feat_l1 = feat_l1 + self.upsample(feat_l2)

        feat_out = self._fc(feat_l1)
        feat_out = rearrange(feat_out, '(B T) C H W->B T C H W', T=T)
        return feat_out
