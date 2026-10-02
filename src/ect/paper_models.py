"""ResNeXt1D reconstruction registered in paper_components_v1.

This implements the documented blueprint and engineering assumptions, not
an assertion of numerical equivalence to the unavailable author code.
"""

import torch
from torch import nn
from torch.nn import functional as F


def _same_padding(length: int, kernel: int, stride: int,
                  dilation: int = 1) -> tuple[int, int]:
    """TensorFlow SAME: ceil output length, with extra padding on the right."""
    output_length = (length + stride - 1) // stride
    total = max((output_length - 1) * stride + dilation * (kernel - 1) + 1 - length, 0)
    left = total // 2
    return left, total - left


class _SameConv1d(nn.Conv1d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, groups: int = 1):
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=0, groups=groups, bias=True)

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        padding = _same_padding(signals.shape[-1], self.kernel_size[0],
                                self.stride[0], self.dilation[0])
        return super().forward(F.pad(signals, padding))


class _SameMaxPool1d(nn.MaxPool1d):
    def __init__(self):
        super().__init__(kernel_size=3, stride=2, padding=0)

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        padding = _same_padding(signals.shape[-1], self.kernel_size, self.stride)
        return super().forward(F.pad(signals, padding, value=float('-inf')))


def _batch_norm(channels: int) -> nn.BatchNorm1d:
    return nn.BatchNorm1d(channels, eps=1e-3, momentum=0.01)


class _PaperBottleneck(nn.Module):
    def __init__(self, in_channels: int, width: int, stride: int,
                 projection: bool, omit_input_preactivation: bool = False):
        super().__init__()
        layers = []
        if not omit_input_preactivation:
            layers.extend([_batch_norm(in_channels), nn.ReLU()])
        layers.extend([
            _SameConv1d(in_channels, width, kernel_size=1),
            _batch_norm(width), nn.ReLU(),
            _SameConv1d(width, width, kernel_size=3, stride=stride, groups=5),
            _batch_norm(width), nn.ReLU(),
            _SameConv1d(width, 2 * width, kernel_size=1),
        ])
        self.main = nn.Sequential(*layers)
        self.shortcut = (_SameConv1d(in_channels, 2 * width, kernel_size=1, stride=stride)
                         if projection else nn.Identity())

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        # The projection consumes raw block input; addition has no activation.
        residual = self.shortcut(signals)
        return self.main(signals) + residual


def _validate_input(signals: torch.Tensor) -> None:
    if not isinstance(signals, torch.Tensor):
        raise TypeError('signals must be a float32 torch.Tensor with shape [B, 2, 224]')
    if signals.ndim != 3 or tuple(signals.shape[1:]) != (2, 224) or signals.shape[0] < 1:
        raise ValueError('signals must have shape [B, 2, 224] with B >= 1 (channels first)')
    if signals.dtype != torch.float32:
        raise TypeError('signals must have dtype float32')
    if not torch.isfinite(signals).all():
        raise ValueError('signals must contain only finite values')


class PaperResNeXt1D(nn.Module):
    """Four preactivation stages mapping float32 [B, 2, 224] to logits.

    Three blocks per stage implement the 38-layer blueprint (134654 trainable
    parameters for 20 classes). Two implement the 26-layer cross-check (93754).
    The reported paper table's 38-layer 1.35e6 count is not forced by widening.
    SAME alignment, biases and BN settings follow the registered assumptions.
    """

    def __init__(self, num_classes: int = 20, blocks_per_stage: int = 3):
        super().__init__()
        if isinstance(num_classes, bool) or not isinstance(num_classes, int):
            raise TypeError('num_classes must be an integer >= 2')
        if num_classes < 2:
            raise ValueError('num_classes must be an integer >= 2')
        if isinstance(blocks_per_stage, bool) or not isinstance(blocks_per_stage, int):
            raise TypeError('blocks_per_stage must be an integer, 2 or 3')
        if blocks_per_stage not in (2, 3):
            raise ValueError('blocks_per_stage must be 2 (26 layers) or 3 (38 layers)')
        self.num_classes = num_classes
        self.blocks_per_stage = blocks_per_stage
        self.stem = nn.Sequential(
            _SameConv1d(2, 6, kernel_size=3), _batch_norm(6), nn.ReLU(),
        )
        self.stem_pool = _SameMaxPool1d()
        stages = []
        in_channels = 6
        for stage_index, width in enumerate((10, 20, 40, 80)):
            blocks = []
            for block_index in range(blocks_per_stage):
                first_block = block_index == 0
                stride = 2 if stage_index > 0 and first_block else 1
                blocks.append(_PaperBottleneck(
                    in_channels, width, stride, projection=first_block,
                    omit_input_preactivation=stage_index == 0 and first_block,
                ))
                in_channels = 2 * width
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        self.final = nn.Sequential(_batch_norm(160), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(160, num_classes, bias=True)
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        _validate_input(signals)
        features = self.stem_pool(self.stem(signals))
        features = self.final(self.stages(features))
        return self.classifier(self.pool(features).squeeze(-1))
