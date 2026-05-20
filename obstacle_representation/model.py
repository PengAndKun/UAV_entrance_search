from __future__ import annotations

from typing import Dict

import torch
from torch import nn


def _small_image_encoder(in_channels: int, out_dim: int = 64) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(64, out_dim),
        nn.ReLU(inplace=True),
    )


class SchemeAObstacleNet(nn.Module):
    def __init__(self, *, num_labels: int, geometry_dim: int) -> None:
        super().__init__()
        self.rgb_encoder = _small_image_encoder(3, out_dim=64)
        self.geometry_encoder = nn.Sequential(
            nn.Linear(geometry_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
        )
        self.label_head = nn.Linear(64, num_labels)
        self.flyover_head = nn.Linear(64, 1)

    def forward(self, rgb: torch.Tensor, geometry: torch.Tensor) -> Dict[str, torch.Tensor]:
        rgb_feat = self.rgb_encoder(rgb)
        geom_feat = self.geometry_encoder(geometry)
        fused = self.fusion(torch.cat([rgb_feat, geom_feat], dim=1))
        return {
            "label_logits": self.label_head(fused),
            "flyover_logits": self.flyover_head(fused).squeeze(1),
        }


class SchemeAPlusObstacleNet(nn.Module):
    def __init__(self, *, num_labels: int, geometry_dim: int) -> None:
        super().__init__()
        self.rgb_encoder = _small_image_encoder(3, out_dim=64)
        self.depth_encoder = _small_image_encoder(1, out_dim=48)
        self.geometry_encoder = nn.Sequential(
            nn.Linear(geometry_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Linear(144, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.15),
            nn.Linear(96, 64),
            nn.ReLU(inplace=True),
        )
        self.label_head = nn.Linear(64, num_labels)
        self.flyover_head = nn.Linear(64, 1)

    def forward(self, rgb: torch.Tensor, geometry: torch.Tensor, depth: torch.Tensor) -> Dict[str, torch.Tensor]:
        rgb_feat = self.rgb_encoder(rgb)
        depth_feat = self.depth_encoder(depth)
        geom_feat = self.geometry_encoder(geometry)
        fused = self.fusion(torch.cat([rgb_feat, depth_feat, geom_feat], dim=1))
        return {
            "label_logits": self.label_head(fused),
            "flyover_logits": self.flyover_head(fused).squeeze(1),
        }
