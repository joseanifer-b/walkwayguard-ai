"""Spatio-Temporal Autoencoder (STAE): Encoder -> ConvLSTM -> Decoder (Section 5.3, Table 5.2).

The classes below follow the source code in the report's appendix.
`STAE(use_convlstm=False)` gives the spatial-only autoencoder used in the Table 6.7 ablation.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """CONV2D - BATCHNORM - RELU block used in the spatial encoder."""

    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SpatialEncoder(nn.Module):
    """Applies Conv-BN-ReLU blocks independently to each frame."""

    def __init__(self, in_ch=3, channels=(32, 64, 128)):
        super().__init__()
        layers, c_in = [], in_ch
        for c_out in channels:
            layers.append(ConvBlock(c_in, c_out))
            c_in = c_out
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, C, H, W)
        return self.encoder(x)  # -> (B, d, h, w)


class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell modelling motion while preserving spatial layout (Eq. 2)."""

    def __init__(self, in_ch, hidden_ch, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(in_ch + hidden_ch, 4 * hidden_ch, kernel_size, padding=pad)
        self.hidden_ch = hidden_ch

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.split(gates, self.hidden_ch, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c


class Decoder(nn.Module):
    """Mirrors the encoder with transposed convolutions to reconstruct frames."""

    def __init__(self, channels=(128, 64, 32), out_ch=3):
        super().__init__()
        layers, c_in = [], channels[0]
        for c_out in list(channels[1:]) + [out_ch]:
            layers.append(nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1))
            layers.append(nn.ReLU(inplace=True))
            c_in = c_out
        self.decoder = nn.Sequential(*layers[:-1], nn.Sigmoid())

    def forward(self, h):
        return self.decoder(h)


class STAE(nn.Module):
    """Spatio-Temporal Autoencoder: Encoder -> ConvLSTM -> Decoder."""

    def __init__(self, use_convlstm: bool = True):
        super().__init__()
        self.use_convlstm = use_convlstm
        self.encoder = SpatialEncoder()
        self.convlstm = ConvLSTMCell(in_ch=128, hidden_ch=128) if use_convlstm else None
        self.decoder = Decoder()

    def forward(self, x_seq):
        # x_seq: (B, T, C, H, W)
        B, T, C, H, W = x_seq.shape
        h = torch.zeros(B, 128, H // 8, W // 8, device=x_seq.device)
        c = torch.zeros_like(h)
        recon = []
        for t in range(T):
            feat = self.encoder(x_seq[:, t])
            if self.use_convlstm:
                h, c = self.convlstm(feat, h, c)
            else:
                h = feat
            recon.append(self.decoder(h))
        return torch.stack(recon, dim=1)  # (B, T, C, H, W)


def reconstruction_error(x, x_hat):
    """Per-frame MSE reconstruction error, Eq. (4)/(5) of the design. -> (B, T)"""
    return ((x - x_hat) ** 2).flatten(2).mean(dim=2)


def regularity_score(errors):
    """Min-max normalise errors into a bounded [0, 1] regularity score, Eq. (5)."""
    e_min, e_max = errors.min(), errors.max()
    return 1 - (errors - e_min) / (e_max - e_min + 1e-8)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


ARCHITECTURE_TABLE = [
    {"component": "Input", "parameter": "Sequence length T = 10, 128 x 128 x 3", "role": "Sliding-window video clip"},
    {"component": "Encoder", "parameter": "3 Conv-BN-ReLU blocks, 3 x 3, stride 2", "role": "Spatial feature extraction"},
    {"component": "Encoder", "parameter": "Filters 32 -> 64 -> 128", "role": "Increasing feature depth and downsampling"},
    {"component": "Temporal", "parameter": "1 ConvLSTM layer, 128 hidden channels, 3 x 3", "role": "Motion and direction modelling"},
    {"component": "Decoder", "parameter": "3 transposed conv layers, 4 x 4, stride 2; 128 -> 64 -> 32 -> 3", "role": "Frame reconstruction"},
    {"component": "Loss", "parameter": "MSE reconstruction loss", "role": "End-to-end training objective"},
    {"component": "Scoring", "parameter": "Min-max normalised reconstruction error", "role": "Regularity / anomaly score"},
    {"component": "Threshold", "parameter": "tau (validation-tuned)", "role": "Final anomaly decision"},
]
