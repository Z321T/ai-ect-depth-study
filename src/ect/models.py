"""Fixed 1D classifiers for float32 signals in [batch, I/Q, 250] layout."""

import torch
from torch import nn


def _validate_input(signals: torch.Tensor) -> None:
    if not isinstance(signals, torch.Tensor):
        raise TypeError('signals must be a float32 torch.Tensor with shape [B, 2, 250]')
    if signals.ndim != 3 or tuple(signals.shape[1:]) != (2, 250):
        raise ValueError('signals must have shape [B, 2, 250] (channels first)')
    if signals.dtype != torch.float32:
        raise TypeError('signals must have dtype float32')


class CNN1D(nn.Module):
    """Three Conv-BN-ReLU-pool blocks followed by global average pooling."""

    def __init__(self, num_classes: int):
        super().__init__()
        layers = []
        in_channels = 2
        for out_channels in (32, 64, 128):
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
            ])
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        _validate_input(signals)
        features = self.pool(self.features(signals)).squeeze(-1)
        return self.classifier(features)


class _BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(signals)
        features = self.relu(self.bn1(self.conv1(signals)))
        features = self.bn2(self.conv2(features))
        return self.relu(features + residual)


class ResNet1D(nn.Module):
    """Two basic blocks per stage, with stage widths 32, 64 and 128."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, padding=3, stride=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        stages = []
        in_channels = 32
        for out_channels, stride in ((32, 1), (64, 2), (128, 2)):
            stages.append(nn.Sequential(
                _BasicBlock(in_channels, out_channels, stride),
                _BasicBlock(out_channels, out_channels, 1),
            ))
            in_channels = out_channels
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        _validate_input(signals)
        features = self.pool(self.stages(self.stem(signals))).squeeze(-1)
        return self.classifier(features)


def build_model(name: str = 'cnn', num_classes: int = 20) -> nn.Module:
    """Build a fresh CNN or ResNet returning unnormalized class logits."""
    if name not in ('cnn', 'resnet'):
        raise ValueError("model name must be 'cnn' or 'resnet'")
    if isinstance(num_classes, bool) or not isinstance(num_classes, int):
        raise TypeError('num_classes must be an integer >= 2')
    if num_classes < 2:
        raise ValueError('num_classes must be an integer >= 2')
    return CNN1D(num_classes) if name == 'cnn' else ResNet1D(num_classes)
