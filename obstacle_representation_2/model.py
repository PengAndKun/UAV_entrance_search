from __future__ import annotations

import os
from typing import Dict

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class APlus2AffordanceNet(nn.Module):
    """Small RGB-D U-Net style model with direction and flyover heads."""

    def __init__(self, *, geometry_dim: int, num_directions: int = 6, mask_channels: int = 2) -> None:
        super().__init__()
        self.enc1 = ConvBlock(4, 24)
        self.enc2 = ConvBlock(24, 48)
        self.enc3 = ConvBlock(48, 96)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(96, 48, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(96, 48)
        self.up1 = nn.ConvTranspose2d(48, 24, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(48, 24)
        self.mask_head = nn.Conv2d(24, mask_channels, kernel_size=1)

        self.geometry_encoder = nn.Sequential(
            nn.Linear(geometry_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
        )
        self.direction_head = nn.Sequential(
            nn.Linear(96 + 32, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.15),
            nn.Linear(96, 64),
            nn.ReLU(inplace=True),
        )
        self.direction_logits = nn.Linear(64, num_directions)
        self.score_logits = nn.Linear(64, num_directions)
        self.flyover_head = nn.Linear(64, 1)

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor, geometry: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = torch.cat([rgb, depth], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.up2(e3)
        if d2.shape[-2:] != e2.shape[-2:]:
            d2 = F.interpolate(d2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = F.interpolate(d1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        pooled = F.adaptive_avg_pool2d(e3, (1, 1)).flatten(1)
        geom = self.geometry_encoder(geometry)
        fused = self.direction_head(torch.cat([pooled, geom], dim=1))
        return {
            "mask_logits": self.mask_head(d1),
            "direction_logits": self.direction_logits(fused),
            "score_logits": self.score_logits(fused),
            "flyover_delta": self.flyover_head(fused).squeeze(1),
        }
