"""Core utilities for the 2026-09-17 round.

Design rules fixed BEFORE any experiment runs (see NEXT_ROUND_EXPERIMENT_PLAN_917.md):

1. One frozen stratified 5-fold split, shared by every candidate.
2. Every preprocessing statistic (image mean/std, angle mean/std/fill) is fitted
   ONLY on the training part of the fold it is used for.
3. OOF probabilities are written per (config, fold, seed) and aligned by sample id.
4. Training budget is IDENTICAL across candidates: same epochs, same schedule.
   No validation-loss-based early stopping and no validation-loss LR scheduling,
   so the reported OOF number contains no label-dependent selection.
5. The historical recipe (plateau scheduler + early stopping) is available as a
   separate, explicitly-labelled variant for reproducing published numbers.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------

L1_DIR = Path(__file__).resolve().parent
BESTEXP_DIR = L1_DIR.parent
DATA_DIR = BESTEXP_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
NR_DIR = L1_DIR
FOLDS_PATH = L1_DIR / "folds.json"

# `models.py` (copied from experiments/common/models.py) sits next to this file.
if str(L1_DIR) not in sys.path:
    sys.path.insert(0, str(L1_DIR))

IMAGE_SIDE = 75
KAGGLE_EPS = 1e-15

# Frozen protocol constants -- do not change mid-round.
FOLD_SEED = 20260917
N_FOLDS = 5

# Fixed training budget, identical for every candidate.
#
# Chosen from the 30-epoch pilot (runs_pilot_ep30/, 8 runs of a0_base):
# mean validation log loss is minimised near epoch 13 and degrades steadily
# afterwards (epoch 13 = 0.2259 vs epoch 30 = 0.2848). Epochs 12-17 form a broad,
# noisy plateau, so 15 is taken as the budget rather than the sharp argmin.
# The plan (section 2.2) explicitly allows determining the training length from
# the folds. The SAME constant is used for every candidate, so paired comparisons
# stay fair. Disclosed in PHASE0_AUDIT.md.
DEFAULT_EPOCHS = 15
DEFAULT_BATCH = 32


# --------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Split:
    images: np.ndarray
    ids: np.ndarray
    angles: np.ndarray
    labels: np.ndarray | None


def load_split(split: str) -> Split:
    images = np.load(CACHE_DIR / f"{split}_images.npy", mmap_mode="r")
    ids = np.load(CACHE_DIR / f"{split}_ids.npy")
    angles = np.load(CACHE_DIR / f"{split}_angles.npy")
    labels = None
    path = CACHE_DIR / f"{split}_labels.npy"
    if path.is_file():
        labels = np.load(path)
    return Split(images=images, ids=np.asarray(ids), angles=np.asarray(angles),
                 labels=None if labels is None else np.asarray(labels))


def file_hash(path: Path, *, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------------------
# Frozen folds
# --------------------------------------------------------------------------------------


def make_folds(labels: np.ndarray) -> list[dict[str, Any]]:
    from sklearn.model_selection import StratifiedKFold

    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)
    folds: list[dict[str, Any]] = []
    for index, (train_idx, valid_idx) in enumerate(splitter.split(np.zeros(len(labels)), labels)):
        folds.append({
            "fold": index,
            "train": [int(x) for x in train_idx],
            "valid": [int(x) for x in valid_idx],
        })
    return folds


def build_and_save_folds(split: Split) -> dict[str, Any]:
    assert split.labels is not None
    folds = make_folds(split.labels)
    payload = {
        "fold_seed": FOLD_SEED,
        "n_folds": N_FOLDS,
        "n_samples": int(len(split.labels)),
        "positive_rate": float(np.mean(split.labels)),
        "data_hashes": {
            name: file_hash(CACHE_DIR / name)
            for name in ("train_images.npy", "train_ids.npy", "train_angles.npy",
                         "train_labels.npy")
        },
        "folds": folds,
    }
    FOLDS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_folds() -> dict[str, Any]:
    if not FOLDS_PATH.is_file():
        raise FileNotFoundError(f"folds not built yet: {FOLDS_PATH}")
    return json.loads(FOLDS_PATH.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------
# Fold-fitted preprocessing
# --------------------------------------------------------------------------------------


@dataclass
class FoldStats:
    """Every statistic here is fitted on the fold's TRAIN indices only."""

    image_mean: np.ndarray
    image_std: np.ndarray
    angle_mean: float
    angle_std: float
    angle_fill: float
    diff_mean: float = 0.0
    diff_std: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "image_mean": np.asarray(self.image_mean).tolist(),
            "image_std": np.asarray(self.image_std).tolist(),
            "angle_mean": float(self.angle_mean),
            "angle_std": float(self.angle_std),
            "angle_fill": float(self.angle_fill),
            "diff_mean": float(self.diff_mean),
            "diff_std": float(self.diff_std),
        }


def fit_fold_stats(images: np.ndarray, angles: np.ndarray,
                   train_idx: Sequence[int]) -> FoldStats:
    idx = np.asarray(train_idx, dtype=np.int64)
    selected = np.asarray(images[idx], dtype=np.float64)
    mean = selected.mean(axis=(0, 2, 3))
    std = np.sqrt(np.maximum((selected ** 2).mean(axis=(0, 2, 3)) - mean ** 2, 0.0))
    std = np.maximum(std, 1e-6)

    diff = selected[:, 0] - selected[:, 1]
    diff_mean = float(diff.mean())
    diff_std = float(max(diff.std(), 1e-6))

    raw = np.asarray(angles, dtype=np.float64)[idx]
    observed = raw[~np.isnan(raw)]
    if len(observed) == 0:
        raise ValueError("fold has no observed angles")
    angle_mean = float(observed.mean())
    angle_std = float(max(observed.std(), 1e-6))
    angle_fill = float(np.median(observed))
    return FoldStats(mean.astype(np.float32), std.astype(np.float32),
                     angle_mean, angle_std, angle_fill, diff_mean, diff_std)


# --------------------------------------------------------------------------------------
# Channel construction
# --------------------------------------------------------------------------------------

CHANNEL_MODES = ("hh_hv", "hh_hv_diff")


def build_channels(image: torch.Tensor, mode: str) -> torch.Tensor:
    """`image` is (C=2, H, W) raw. Returns (C', H, W) before standardisation."""

    if mode == "hh_hv":
        return image
    if mode == "hh_hv_diff":
        hh = image[0:1]
        hv = image[1:2]
        return torch.cat([hh, hv, hh - hv], dim=0)
    raise ValueError(f"unknown channel mode {mode!r}")


def channel_count(mode: str) -> int:
    return 2 if mode == "hh_hv" else 3


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------


class NRDataset(Dataset):
    """Images + angle with fold-fitted standardisation.

    Optional `preprocess` switches mirror the original pipeline's `image_log` and
    `subtract_background`, applied in the same order (log first, then standardise,
    then background subtraction).
    """

    def __init__(self, images: np.ndarray, angles: np.ndarray, labels: np.ndarray | None,
                 indices: Sequence[int], stats: FoldStats, *, channel_mode: str,
                 preprocess: dict[str, Any] | None = None) -> None:
        self.images = images
        self.angles = angles
        self.labels = labels
        self.indices = (
            np.arange(len(images), dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )
        self.stats = stats
        self.channel_mode = channel_mode
        self.preprocess = dict(preprocess or {})
        self.image_log = bool(self.preprocess.get("image_log", False))
        self.subtract_background = bool(self.preprocess.get("subtract_background", False))
        self.mean = torch.as_tensor(stats.image_mean, dtype=torch.float32).reshape(-1, 1, 1)
        self.std = torch.as_tensor(stats.image_std, dtype=torch.float32).reshape(-1, 1, 1)
        self.diff_mean = float(stats.diff_mean)
        self.diff_std = float(stats.diff_std)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        source = int(self.indices[item])
        raw = torch.from_numpy(np.array(self.images[source], dtype=np.float32, copy=True))

        if self.image_log:
            # Shift to positive then compress: tames the bright point-scatterer tail.
            raw = torch.log1p(torch.clamp(raw - raw.min() + 1.0, min=0.0))

        image = build_channels(raw, self.channel_mode)
        if self.channel_mode == "hh_hv_diff":
            base = image[:2]
            base = (base - self.mean) / self.std
            diff = (image[2:3] - self.diff_mean) / self.diff_std
            image = torch.cat([base, diff], dim=0)
        else:
            image = (image - self.mean) / self.std

        if self.subtract_background:
            image = subtract_background(image)

        angle = float(self.angles[source])
        if np.isnan(angle):
            angle = self.stats.angle_fill
        angle = (angle - self.stats.angle_mean) / self.stats.angle_std
        label = torch.tensor(float(self.labels[source]), dtype=torch.float32) \
            if self.labels is not None else torch.tensor(0.0)
        return image, torch.tensor(angle, dtype=torch.float32), label


def subtract_background(image: torch.Tensor) -> torch.Tensor:
    """Cheap high-pass: subtract a 3x3 box blur (ocean background) from the image."""

    pooled = F.avg_pool2d(image.unsqueeze(0), kernel_size=3, stride=1, padding=1)
    return image - pooled.squeeze(0)


def fit_diff_stats(images: np.ndarray, train_idx: Sequence[int]) -> tuple[float, float]:
    """Mean/std of the HH-HV difference channel on fold-train data."""

    idx = np.asarray(train_idx, dtype=np.int64)
    selected = np.asarray(images[idx], dtype=np.float64)
    diff = selected[:, 0] - selected[:, 1]
    return float(diff.mean()), float(max(diff.std(), 1e-6))


def collate(batch):
    images = torch.stack([b[0] for b in batch])
    angles = torch.stack([b[1] for b in batch])
    labels = torch.stack([b[2] for b in batch])
    return images, angles, labels


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------


def kaggle_log_loss(labels, probabilities) -> float:
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    clipped = np.clip(probabilities, KAGGLE_EPS, 1.0 - KAGGLE_EPS)
    return float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)))


def roc_auc(labels, probabilities) -> float:
    labels = np.asarray(labels).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if len(np.unique(labels)) < 2:
        return float("nan")
    order = np.argsort(probabilities)
    ranks = np.empty(len(probabilities), dtype=np.float64)
    ranks[order] = np.arange(1, len(probabilities) + 1)
    pos = labels == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if not n_pos or not n_neg:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def brier(labels, probabilities) -> float:
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    return float(np.mean((probabilities - labels) ** 2))


# --------------------------------------------------------------------------------------
# Model zoo
# --------------------------------------------------------------------------------------


def build_model(name: str, *, in_channels: int, angle_mode: str,
                activation: str = "mish", width: int = 32, depth: int = 3,
                dropout: float = 0.15, head_dropout: float = 0.4,
                **extra) -> nn.Module:
    """Model factory. CNN-specific args carry defaults so non-CNN entries can omit them."""

    if name == "cnn":
        from models import IcebergNet
        return IcebergNet(in_channels=in_channels, width=width, depth=depth,
                          dropout=dropout, head_dropout=head_dropout,
                          activation=activation, angle_mode=angle_mode)
    if name == "resnet18":
        return ResNet18Angle(in_channels=in_channels, angle_mode=angle_mode,
                             pretrained=bool(extra.get("pretrained", False)),
                             input_size=int(extra.get("input_size", 112)),
                             head_dropout=head_dropout, angle_embed=extra.get("angle_embed", 32))
    raise ValueError(f"unknown model {name!r}")


class ResNet18Angle(nn.Module):
    """ResNet-18 backbone with an incidence-angle-aware head.

    The backbone can be ImageNet-pretrained or randomly initialised; the two are
    compared directly so that any gain is attributed correctly.
    """

    def __init__(self, *, in_channels: int = 3, angle_mode: str = "film",
                 pretrained: bool = False, input_size: int = 112,
                 head_dropout: float = 0.4, angle_embed: int = 32) -> None:
        super().__init__()
        from torchvision.models import resnet18

        weights = None
        if pretrained:
            from torchvision.models import ResNet18_Weights
            weights = ResNet18_Weights.IMAGENET1K_V1
        backbone = resnet18(weights=weights)

        if in_channels != 3:
            old = backbone.conv1
            new = nn.Conv2d(in_channels, old.out_channels, old.kernel_size,
                            old.stride, old.padding, bias=False)
            with torch.no_grad():
                if weights is not None:
                    new.weight.copy_(old.weight[:, :in_channels] * (3.0 / in_channels))
            backbone.conv1 = new

        self.input_size = input_size
        self.angle_mode = angle_mode
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.feature_dim = backbone.fc.in_features

        self.angle_encoder = nn.Sequential(
            nn.Linear(1, angle_embed), nn.ReLU(inplace=True),
            nn.Linear(angle_embed, angle_embed), nn.ReLU(inplace=True),
        )
        self.use_angle = angle_mode in ("mlp", "film", "scalar", "channel")
        self.head = nn.Sequential(
            nn.Linear(self.feature_dim + (angle_embed if self.use_angle else 0), 64),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(64, 1),
        )

    def forward(self, images: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        if images.shape[-1] != self.input_size:
            images = F.interpolate(images, size=(self.input_size, self.input_size),
                                   mode="bilinear", align_corners=False)
        x = self.stem(images)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        if self.use_angle:
            x = torch.cat([x, self.angle_encoder(angles.reshape(-1, 1))], dim=1)
        return self.head(x).squeeze(1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class TrainResult:
    oof_probabilities: np.ndarray
    valid_indices: np.ndarray
    curve: list[dict[str, float]]
    best_epoch: int
    best_valid_logloss: float
    final_valid_logloss: float
    elapsed_seconds: float
    parameter_count: int
    trainable_at_start: int
    n_train: int


@torch.inference_mode()
def predict_tta(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    """8-view rotation/flip test-time augmentation.

    Inference-only: it cannot affect training. Motivated by the fact that target
    orientation in SAR is arbitrary, so the decision should be approximately
    invariant to the dihedral group.
    """

    model.eval()
    chunks: list[np.ndarray] = []
    for batch in loader:
        images, angles = batch[0], batch[1]
        images = images.to(device)
        angles = angles.to(device)
        accumulated = torch.zeros(images.shape[0], device=device)
        views = 0
        for rotations in range(4):
            rotated = torch.rot90(images, rotations, dims=(-2, -1))
            for hflip in (False, True):
                view = torch.flip(rotated, dims=(-1,)) if hflip else rotated
                accumulated += torch.sigmoid(model(view, angles).reshape(-1))
                views += 1
        chunks.append((accumulated / views).cpu().numpy())
    return np.concatenate(chunks).astype(np.float64)


def train_one(
    *,
    images: np.ndarray,
    angles: np.ndarray,
    labels: np.ndarray,
    train_idx: Sequence[int],
    valid_idx: Sequence[int],
    stats: FoldStats,
    channel_mode: str,
    model_name: str,
    model_kwargs: dict[str, Any],
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 0,
    schedule: str = "cosine",
    device: str = "cpu",
    torch_threads: int = 2,
    progress: bool = False,
    preprocess: dict[str, Any] | None = None,
    tta: bool = False,
) -> TrainResult:
    """Train once. With schedule='cosine' the final epoch is the OOF prediction."""

    import time

    torch.set_num_threads(torch_threads)
    seed_everything(seed)
    started = time.perf_counter()

    train_ds = NRDataset(images, angles, labels, train_idx, stats,
                         channel_mode=channel_mode, preprocess=preprocess)
    valid_ds = NRDataset(images, angles, labels, valid_idx, stats,
                         channel_mode=channel_mode, preprocess=preprocess)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed),
                              num_workers=0, collate_fn=collate)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                              num_workers=0, collate_fn=collate)

    dev = torch.device(device)
    model = build_model(model_name, in_channels=channel_count(channel_mode),
                        **model_kwargs).to(dev)
    trainable_at_start = count_parameters(model)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=weight_decay)

    if schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif schedule == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2)
    else:
        scheduler = None

    curve: list[dict[str, float]] = []
    best_valid, best_epoch, best_probs = float("inf"), 0, None
    final_probs = None

    for epoch in range(1, epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for batch_images, batch_angles, batch_labels in train_loader:
            batch_images = batch_images.to(dev)
            batch_angles = batch_angles.to(dev)
            batch_labels = batch_labels.to(dev)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_images, batch_angles).reshape(-1)
            loss = criterion(logits, batch_labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.detach()) * logits.shape[0]
            seen += logits.shape[0]

        model.eval()
        probs, trues = [], []
        with torch.inference_mode():
            for batch_images, batch_angles, batch_labels in valid_loader:
                logits = model(batch_images.to(dev), batch_angles.to(dev)).reshape(-1)
                probs.append(torch.sigmoid(logits).cpu().numpy())
                trues.append(batch_labels.numpy())
        probs = np.concatenate(probs)
        trues = np.concatenate(trues)
        valid_loss = kaggle_log_loss(trues, probs)

        curve.append({
            "epoch": epoch,
            "train_loss": running / max(seen, 1),
            "valid_logloss": valid_loss,
            "valid_auc": roc_auc(trues, probs),
            "valid_brier": brier(trues, probs),
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        if valid_loss < best_valid - 1e-9:
            best_valid, best_epoch, best_probs = valid_loss, epoch, probs.copy()
        final_probs = probs

        if progress and (epoch == 1 or epoch % 5 == 0 or epoch == epochs):
            print(f"    epoch {epoch:3d}/{epochs}  train={curve[-1]['train_loss']:.4f} "
                  f"valid={valid_loss:.4f} auc={curve[-1]['valid_auc']:.4f} "
                  f"best={best_valid:.4f}@{best_epoch}", flush=True)

        if schedule == "plateau" and scheduler is not None:
            scheduler.step(valid_loss)
        elif schedule == "cosine" and scheduler is not None:
            scheduler.step()

    elapsed = time.perf_counter() - started
    chosen = final_probs if schedule == "cosine" else best_probs
    if tta:
        # Re-predict the validation fold with 8-view TTA at the final weights.
        chosen = predict_tta(model, valid_loader, dev)
        elapsed = time.perf_counter() - started
    return TrainResult(
        oof_probabilities=chosen,
        valid_indices=np.asarray(valid_idx, dtype=np.int64),
        curve=curve,
        best_epoch=best_epoch,
        best_valid_logloss=best_valid,
        final_valid_logloss=curve[-1]["valid_logloss"],
        elapsed_seconds=elapsed,
        parameter_count=count_parameters(model),
        trainable_at_start=trainable_at_start,
        n_train=len(train_idx),
    )
