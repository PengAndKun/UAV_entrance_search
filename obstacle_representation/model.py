from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class SchemeAObstacleNet(nn.Module):
    def __init__(self, *, num_labels: int, geometry_dim: int) -> None:
        super().__init__()
        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
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
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
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
