"""
model.py — SpectrogramCNN Architecture
========================================
Architecture:
  Input: (B, 1, n_mels, T)   ← single-channel log-mel spectrogram

  Block 1: Conv2d(1→32)   + BN + ReLU + MaxPool2d(2×2) + Dropout2d(0.20)
  Block 2: Conv2d(32→64)  + BN + ReLU + MaxPool2d(2×2) + Dropout2d(0.20)
  Block 3: Conv2d(64→128) + BN + ReLU + MaxPool2d(2×2) + Dropout2d(0.25)
  Block 4: Conv2d(128→256)+ BN + ReLU + MaxPool2d(2×2) + Dropout2d(0.30)

  Global Average Pooling  → (B, 256)
  FC: Linear(256→256) + BN1d + ReLU + Dropout(0.5)
      Linear(256→num_classes)

  Output: logits (B, 2)   [0=bonafide/human, 1=spoof/synthetic]

Design notes:
  - bias=False in all Conv2d when followed by BatchNorm (standard practice)
  - Dropout2d drops entire feature maps — better for spatial features
  - Progressive dropout rate: 0.20 → 0.20 → 0.25 → 0.30
  - GAP removes fixed spatial-size requirement (works with any input length)
  - FC block uses BN1d for stable deep training
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Convolutional building block:
      Conv2d → BatchNorm2d → ReLU → MaxPool2d → Dropout2d

    Args:
        in_channels  : Number of input channels.
        out_channels : Number of output (filter) channels.
        kernel_size  : Convolution kernel size (default 3×3).
        padding      : Padding to preserve spatial resolution before pooling.
        pool_size    : MaxPool kernel and stride (halves spatial dims).
        dropout      : Spatial dropout probability (Dropout2d).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        pool_size: int = 2,
        dropout: float = 0.20,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,          # BN handles bias
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_size, stride=pool_size),
            nn.Dropout2d(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SpectrogramCNN(nn.Module):
    """
    4-block CNN classifier for spectrogram-based speech deepfake detection.

    Input shape : (B, 1, n_mels, T)  — 2-D log-mel spectrogram
    Output shape: (B, num_classes)    — raw logits

    Binary label convention:
        0 → bonafide (human / genuine speech)
        1 → spoof    (AI-generated / synthetic speech)
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_dropout: float = 0.20,
    ):
        super().__init__()

        # ── 4 Convolutional Blocks ─────────────────────────────────
        self.block1 = ConvBlock(in_channels, 32,  dropout=base_dropout)
        self.block2 = ConvBlock(32,  64,           dropout=base_dropout)
        self.block3 = ConvBlock(64,  128,          dropout=base_dropout + 0.05)
        self.block4 = ConvBlock(128, 256,          dropout=base_dropout + 0.10)

        # ── Global Average Pooling ─────────────────────────────────
        # Collapses (B, 256, H', W') → (B, 256) regardless of input size
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # ── Fully Connected Classifier ─────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(256, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.50),
            nn.Linear(256, num_classes),
        )

        # ── Weight Initialisation ──────────────────────────────────
        self._init_weights()

    def _init_weights(self):
        """Kaiming He init for Conv layers; Xavier for Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Spectrogram tensor.
               Accepted shapes:
                 (B, 1, n_mels, T)  — standard (channel already present)
                 (B, n_mels, T)     — channel dim added automatically

        Returns:
            logits: (B, num_classes)
        """
        # Auto-add channel dim if missing
        if x.dim() == 3:
            x = x.unsqueeze(1)   # (B, n_mels, T) → (B, 1, n_mels, T)

        x = self.block1(x)       # (B,  32, H/2,   W/2)
        x = self.block2(x)       # (B,  64, H/4,   W/4)
        x = self.block3(x)       # (B, 128, H/8,   W/8)
        x = self.block4(x)       # (B, 256, H/16,  W/16)

        x = self.gap(x)          # (B, 256, 1, 1)
        x = x.flatten(1)         # (B, 256)

        logits = self.classifier(x)   # (B, num_classes)
        return logits

    def count_parameters(self) -> int:
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
