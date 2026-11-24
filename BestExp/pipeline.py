"""BestExp — 共享的流水线代码（数据、特征、后处理）。

所有常量与 `experiments/phase4/` 里的原实现**逐一对应**，改动任何一项都会导致复现失败。

    KS             = (1, 3, 5)          ← 最近 K 个角度组
    DECIMALS       = 4                  ← 组表精度（4 位小数）
    L1 OOF 种子    = 0..6（7 个）        ← a0_base 的 7 种子平均
    L1 集成        = a0_noangle(s0..s2) 与 b_member_gelu_film(s0..s2) 等权平均
    L2 特征        = arm C 去掉 angle_missing → 24 列
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CACHE = DATA / "cache"
L1_OOF = DATA / "l1_oof"
L1_TEST = DATA / "l1_test"

N_FOLDS = 5
KS = (1, 3, 5)
DECIMALS = 4
L1_OOF_SEEDS = list(range(7))          # a0_base 的 7 个种子
L1_ENSEMBLE_MEMBERS = ("a0_noangle", "b_member_gelu_film")
L1_ENSEMBLE_SEEDS = (0, 1, 2)


# ---------------------------------------------------------------- 数据读取

def load_split(split: str) -> dict:
    images = np.load(CACHE / f"{split}_images.npy", mmap_mode="r")
    ids = np.load(CACHE / f"{split}_ids.npy")
    angles = np.load(CACHE / f"{split}_angles.npy")
    labels = None
    p = CACHE / f"{split}_labels.npy"
    if p.is_file():
        labels = np.load(p)
    return {"images": images, "ids": np.asarray(ids),
            "angles": np.asarray(angles),
            "labels": None if labels is None else np.asarray(labels)}


def l1_train_oof() -> np.ndarray:
    """7 种子平均的 OOF（折种子 20260917）。"""

    n = 1604
    rows = []
    for s in L1_OOF_SEEDS:
        v = np.full(n, np.nan)
        for f in range(N_FOLDS):
            d = L1_OOF / f"a0_base_f{f}_s{s}_cosine"
            v[np.load(d / "valid_idx.npy")] = np.load(d / "oof.npy")
        if np.isnan(v).any():
            raise ValueError(f"OOF 不完整: seed {s}")
        rows.append(v)
    return np.vstack(rows).mean(axis=0)


def l1_test_ensemble() -> np.ndarray:
    """L1 集成：a0_noangle 与 b_member_gelu_film 各自 3 种子平均后再等权平均。

    公开包未分发 data/l1_test/，此时回退到发布的冻结集成预测。
    """
    frozen = Path(__file__).resolve().parent.parent / "weights" / "l1" / \
        "l1_ensemble_test_predictions.csv"
    if not L1_TEST.is_dir() and frozen.is_file():
        return pd.read_csv(frozen, dtype={"id": str}).set_index("id")["is_iceberg"].values

    per = []
    for cfg in L1_ENSEMBLE_MEMBERS:
        vs = []
        for s in L1_ENSEMBLE_SEEDS:
            f = L1_TEST / f"{cfg}_s{s}" / "submission.csv"
            vs.append(pd.read_csv(f, dtype={"id": str}).set_index("id")["is_iceberg"].values)
        per.append(np.vstack(vs).mean(axis=0))
    return np.vstack(per).mean(axis=0)


# ---------------------------------------------------------------- 特征构造

def band_feats(images, batch: int = 512) -> np.ndarray:
    """HH/HV 原始统计：每个波段 min/mean/median/max/std，共 10 维。"""

    out = []
    for i in range(0, len(images), batch):
        a = np.asarray(images[i:i + batch], dtype=np.float32)
        hh, hv = a[:, 0], a[:, 1]
        out.append(np.stack([
            hh.min(axis=(1, 2)), hh.mean(axis=(1, 2)), np.median(hh, axis=(1, 2)),
            hh.max(axis=(1, 2)), hh.std(axis=(1, 2)),
            hv.min(axis=(1, 2)), hv.mean(axis=(1, 2)), np.median(hv, axis=(1, 2)),
            hv.max(axis=(1, 2)), hv.std(axis=(1, 2)),
        ], axis=1))
    return np.concatenate(out)


def group_tables(tr_ang, ytr) -> dict:
    """每个精度下的组索引 -> [n, n_ship, n_ice]。"""

    tables = {}
    for d in (4, 3, 2):
        k = np.round(tr_ang, d)
        ok = ~np.isnan(k)
        tab: dict[float, list[int]] = {}
        for key, lab in zip(k[ok], ytr[ok]):
            rec = tab.setdefault(float(key), [0, 0, 0])
            rec[0] += 1
            rec[1] += int(lab == 0.0)
            rec[2] += int(lab == 1.0)
        tables[d] = tab
    return tables


def label_features(angles, tables, own_labels=None, decimals=DECIMALS) -> np.ndarray:
    """角度组标签特征（6 维）。

    训练行传 own_labels → 留一法，行看不到自己的标签；
    测试行传 None → 用全组统计。
    """

    k = np.round(angles, decimals)
    n = len(k)
    cnt = np.zeros(n); tr = np.zeros(n); ship = np.zeros(n); ice = np.zeros(n)
    has = np.zeros(n)
    for i, key in enumerate(k):
        rec = tables[decimals].get(float(key))
        if rec is None or np.isnan(key):
            continue
        c, s, ic = rec
        if own_labels is not None and not np.isnan(own_labels[i]):
            c -= 1
            s -= int(own_labels[i] == 0.0)
            ic -= int(own_labels[i] == 1.0)
        cnt[i] = c; ship[i] = s; ice[i] = ic
        tr[i] = c
        has[i] = 1.0
    rate = np.where(tr > 0, ice / np.maximum(tr, 1e-9), 0.5)
    return np.stack([cnt, tr, ship, ice, rate, has], axis=1)


def geom_features(angles, tables, decimals=DECIMALS) -> np.ndarray:
    """到最近 K 个不同训练组的距离 + 该组大小（6 维，不用标签）。"""

    known = np.unique(np.round([x for x in angles if not np.isnan(x)], decimals))
    n = len(angles)
    out = np.zeros((n, len(KS) * 2))
    for i, a in enumerate(angles):
        if np.isnan(a):
            continue
        dist = np.abs(known - a)
        order = np.argsort(dist)[: max(KS)]
        for j, K in enumerate(KS):
            if K <= len(order):
                key = float(known[order[K - 1]])
                rec = tables[decimals].get(key, [0, 0, 0])
                out[i, 2 * j] = dist[order[K - 1]]
                out[i, 2 * j + 1] = rec[0]
    return out


def build_features(split: str, tables: dict, p_l1: np.ndarray):
    """arm C 的 24 列（已去掉 angle_missing）。"""

    s = load_split(split)
    ang = s["angles"]
    own = s["labels"].astype(float) if split == "train" else None
    X = np.hstack([
        p_l1[:, None],
        ang[:, None],
        band_feats(s["images"]),
        geom_features(ang, tables),
        label_features(ang, tables, own_labels=own),
    ])
    names = (["p_l1", "angle"]
             + [f"band{i}" for i in range(10)]
             + [f"k{k}_{t}" for k in KS for t in ("dist", "size")]
             + ["g_count", "g_train", "g_ship", "g_ice", "g_rate", "g_has"])
    assert X.shape[1] == len(names) == 24, (X.shape, len(names))
    return X, names, s


# ---------------------------------------------------------------- 后处理

ADJ_DECIMALS = 1        # 后处理用的组精度
ADJ_BASE_T = 0.30
ADJ_POWER = 2.0
ADJ_CLIP = (0.2, 4.0)


def bounded_adjust(p: np.ndarray, te_ang: np.ndarray,
                   tr_ang: np.ndarray, ytr: np.ndarray) -> np.ndarray:
    """有界调整：仅纯角度组；T = base_T * (n / median_n)^power，clip 到 [0.2x, 4x]。"""

    k = np.round(tr_ang, ADJ_DECIMALS)
    g: dict[float, list[float]] = {}
    for key, lab in zip(k[~np.isnan(k)], ytr[~np.isnan(k)]):
        g.setdefault(float(key), []).append(float(lab))
    ice = {key: len(v) for key, v in g.items() if all(x == 1.0 for x in v)}
    ship = {key: len(v) for key, v in g.items() if all(x == 0.0 for x in v)}

    kt = np.round(te_ang, ADJ_DECIMALS)
    is_ice = np.array([(not np.isnan(x)) and float(x) in ice for x in kt])
    is_ship = np.array([(not np.isnan(x)) and float(x) in ship for x in kt])
    hit = is_ice | is_ship
    n = np.array([(ice.get(float(x), 0) if float(x) in ice
                   else ship.get(float(x), 0)) if not np.isnan(x) else 0
                  for x in kt]).astype(float)
    target = np.where(is_ice, 1.0, 0.0)

    med = np.median(n[hit])
    T = np.clip(ADJ_BASE_T * ((n[hit] / med) ** ADJ_POWER),
                ADJ_BASE_T * ADJ_CLIP[0], ADJ_BASE_T * ADJ_CLIP[1])
    out = p.copy()
    cur, tgt = p[hit], target[hit]
    close = np.abs(tgt - cur) <= T
    idx = np.flatnonzero(hit)[close]
    out[idx] = np.clip(target[idx], 0.005, 0.995)
    return out, int(hit.sum()), int(close.sum())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
