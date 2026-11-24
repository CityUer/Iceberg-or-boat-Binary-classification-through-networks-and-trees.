"""CNN architectures with configurable incidence-angle fusion.

Angle fusion strategies implemented here (this is the E01 axis):

  none     - angle ignored; pure image model (E00 reproduction)
  scalar   - angle broadcast as a constant extra input channel
  channel  - angle (plus derived spatial terms) as extra channels
  mlp      - late fusion: CNN embedding concatenated with an angle MLP
  film     - FiLM: angle MLP predicts per-channel scale/shift inside the CNN

Every model returns logits shaped (N,).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

ANGLE_MODES = ("none", "scalar", "channel", "mlp", "film")


# --------------------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------------------


def make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "leaky_relu":
        return nn.LeakyReLU(0.1, inplace=True)
    if name == "elu":
        return nn.ELU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "silu" or name == "swish":
        return nn.SiLU(inplace=True)
    if name == "mish":
        return nn.Mish(inplace=True)
    raise ValueError(f"unknown activation {name!r}")


def make_pool(name: str) -> nn.Module:
    name = name.lower()
    if name == "avg":
        return nn.AvgPool2d(2)
    if name == "max":
        return nn.MaxPool2d(2)
    raise ValueError(f"unknown pooling {name!r}")


class ConvBlock(nn.Module):
    """Two 3x3 convolutions with normalization, activation, pooling and dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        dropout: float = 0.0,
        activation: str = "relu",
        pool: str = "avg",
        norm: str = "batch",
        double: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        layers += [
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            _norm(norm, out_channels),
            make_activation(activation),
        ]
        if double:
            layers += [
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                _norm(norm, out_channels),
                make_activation(activation),
            ]
        if pool != "none":
            layers.append(make_pool(pool))
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    """Pre-activation residual block with a 1x1 projection shortcut."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.act = make_activation(activation)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.dropout(out)
        return self.act(out + residual)


def _norm(kind: str, channels: int) -> nn.Module:
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    if kind == "group":
        groups = 8 if channels % 8 == 0 else 4 if channels % 4 == 0 else 1
        return nn.GroupNorm(groups, channels)
    if kind == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"unknown norm {kind!r}")


# --------------------------------------------------------------------------------------
# Angle encoders
# --------------------------------------------------------------------------------------


class AngleMLP(nn.Module):
    """Small MLP that turns a scalar angle into an embedding."""

    def __init__(self, out_dim: int, *, hidden: int = 32, activation: str = "relu") -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            make_activation(activation),
            nn.Linear(hidden, hidden),
            make_activation(activation),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, angle: torch.Tensor) -> torch.Tensor:
        return self.net(angle.reshape(-1, 1))


class FiLM(nn.Module):
    """Feature-wise linear modulation: angle embedding -> per-channel scale and shift."""

    def __init__(self, channels: int, *, hidden: int = 32, activation: str = "relu") -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(1, hidden),
            make_activation(activation),
            nn.Linear(hidden, hidden),
            make_activation(activation),
        )
        self.to_scale = nn.Linear(hidden, channels)
        self.to_shift = nn.Linear(hidden, channels)

    def forward(self, angle: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(angle.reshape(-1, 1))
        scale = 1.0 + self.to_scale(features)
        shift = self.to_shift(features)
        return scale, shift

    @staticmethod
    def apply(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        return x * scale.reshape(-1, x.shape[1], 1, 1) + shift.reshape(-1, x.shape[1], 1, 1)


# --------------------------------------------------------------------------------------
# Angle channel construction
# --------------------------------------------------------------------------------------


def build_angle_channels(
    angle_norm: torch.Tensor,
    images: torch.Tensor,
    *,
    mode: str,
    radius: int = 12,
    sharpness: float = 22.0,
) -> torch.Tensor:
    """Expand the scalar angle into spatial channels.

    `scalar`  - one channel filled with the constant normalised angle.
    `channel` - constant angle plus centre-distance and a radial proximity map,
                letting the network reason about angle jointly with position.
    """

    batch, _, height, width = images.shape
    constant = angle_norm.reshape(batch, 1, 1, 1).expand(batch, 1, height, width)
    if mode == "scalar":
        return constant

    device = images.device
    ys = torch.arange(height, device=device, dtype=torch.float32) - (height - 1) / 2.0
    xs = torch.arange(width, device=device, dtype=torch.float32) - (width - 1) / 2.0
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    distance = torch.sqrt(grid_x**2 + grid_y**2)
    distance = distance / distance.max().clamp(min=1e-6)
    proximity = torch.exp(-((distance / (radius / max(height, width))) ** 2) * sharpness)
    distance = distance.reshape(1, 1, height, width).expand(batch, 1, height, width)
    proximity = proximity.reshape(1, 1, height, width).expand(batch, 1, height, width)
    return torch.cat([constant, distance, proximity], dim=1)


# --------------------------------------------------------------------------------------
# Main model
# --------------------------------------------------------------------------------------


class IcebergNet(nn.Module):
    """Configurable SAR classifier with optional incidence-angle fusion.

    The angle can enter at three different depths:
      - as extra input channels (scalar / channel)
      - as FiLM modulation after each conv stage (film)
      - concatenated with the pooled embedding (mlp)
    """

    def __init__(
        self,
        *,
        in_channels: int = 2,
        width: int = 32,
        depth: int = 3,
        dropout: float = 0.15,
        head_dropout: float = 0.4,
        activation: str = "relu",
        pool: str = "avg",
        norm: str = "batch",
        double_conv: bool = True,
        residual: bool = False,
        angle_mode: str = "none",
        angle_embed: int = 32,
        angle_channels: str = "channel",
        angle_mlp_hidden: int = 32,
        use_angle_in_head: bool = True,
    ) -> None:
        super().__init__()
        if angle_mode not in ANGLE_MODES:
            raise ValueError(f"unknown angle_mode {angle_mode!r}")
        self.angle_mode = angle_mode
        self.angle_channels = angle_channels
        self.use_angle_in_head = use_angle_in_head and angle_mode in ("scalar", "channel")

        extra = 0
        if angle_mode == "scalar":
            extra = 1
        elif angle_mode == "channel":
            extra = 3
        conv_in = in_channels + extra

        block_cls = ResidualBlock if residual else ConvBlock
        self.blocks = nn.ModuleList()
        channels = [conv_in] + [width * (2**i) for i in range(depth)]
        for index in range(depth):
            if residual:
                self.blocks.append(
                    block_cls(
                        channels[index],
                        channels[index + 1],
                        stride=2,
                        dropout=dropout,
                        activation=activation,
                    )
                )
            else:
                self.blocks.append(
                    block_cls(
                        channels[index],
                        channels[index + 1],
                        dropout=dropout,
                        activation=activation,
                        pool=pool,
                        norm=norm,
                        double=double_conv,
                    )
                )

        final_channels = channels[-1]
        feature_dim = final_channels * 2  # average + max pooling

        if angle_mode == "film":
            self.films = nn.ModuleList(
                [FiLM(ch, hidden=angle_mlp_hidden, activation=activation) for ch in channels[1:]]
            )
        if angle_mode in ("mlp", "film"):
            self.angle_encoder = AngleMLP(
                angle_embed, hidden=angle_mlp_hidden, activation=activation
            )
        if angle_mode == "mlp":
            feature_dim += angle_embed
        if self.use_angle_in_head:
            feature_dim += 1

        self.head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            make_activation(activation),
            nn.Dropout(head_dropout),
            nn.Linear(64, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, images: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        if self.angle_mode == "scalar":
            x = torch.cat(
                [images, build_angle_channels(angles, images, mode="scalar")], dim=1
            )
        elif self.angle_mode == "channel":
            x = torch.cat(
                [images, build_angle_channels(angles, images, mode=self.angle_channels)], dim=1
            )
        else:
            x = images

        for index, block in enumerate(self.blocks):
            x = block(x)
            if self.angle_mode == "film":
                scale, shift = self.films[index](angles)
                x = FiLM.apply(x, scale, shift)

        pooled = torch.cat(
            [F.adaptive_avg_pool2d(x, 1).flatten(1), F.adaptive_max_pool2d(x, 1).flatten(1)],
            dim=1,
        )

        if self.angle_mode == "mlp":
            pooled = torch.cat([pooled, self.angle_encoder(angles)], dim=1)
        if self.use_angle_in_head:
            pooled = torch.cat([pooled, angles.reshape(-1, 1)], dim=1)

        return self.head(pooled).squeeze(1)


# --------------------------------------------------------------------------------------
# Parameter counting
# --------------------------------------------------------------------------------------


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------------------
# MixUp / label smoothing helpers (used by later experiments)
# --------------------------------------------------------------------------------------


def mixup_batch(
    images: torch.Tensor,
    angles: torch.Tensor,
    labels: torch.Tensor,
    *,
    alpha: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MixUp for binary logits; returns mixed inputs and soft targets."""

    if alpha <= 0:
        return images, angles, labels
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    lam = max(lam, 1.0 - lam)
    permutation = torch.randperm(images.size(0), device=images.device)
    images = lam * images + (1.0 - lam) * images[permutation]
    angles = lam * angles + (1.0 - lam) * angles[permutation]
    labels = lam * labels + (1.0 - lam) * labels[permutation]
    return images, angles, labels


def bce_with_logits_soft(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """BCE that accepts soft (mixed) targets."""

    return F.binary_cross_entropy_with_logits(logits, targets)
