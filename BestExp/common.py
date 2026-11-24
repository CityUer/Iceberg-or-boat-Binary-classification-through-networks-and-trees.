"""BestExp — 后处理层共享工具（自包含，无任何绝对路径）。

对应最终版本 **private 0.08488 / public 0.09227**（榜单第 2）。

路径约定（全部相对仓库根）
    BestExp/data/cache/   ← scripts/download_data.py 生成（需你自己的 Kaggle token）
    weights/l1/           ← L1 六个 CNN + 冻结集成预测
    weights/l2/           ← L2 XGBoost
    weights/expert/       ← 跨域专家

设计原则
--------
* **baseline 原封不动**：直接用发布仓库的冻结产物
  （`weights/l1/l1_ensemble_test_predictions.csv` + `weights/l2/l2.json`），
  经 `pipeline.py` 的 blend(w=0.40) + `bounded_adjust` 复现最终版本；
  用 `w_blend=W_BLEND_BASE`(0.10) 可复现 0.10127 那个里程碑。
* **分区只用推理时可得的信息**：角度、图像统计、模型输出，绝不用测试标签。
* 训练侧的一切评估用 **OOF**（L2 五折 + 折外组表后处理）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent          # BestExp/
REPO = ROOT.parent                              # 仓库根
CACHE = ROOT / "data" / "cache"                 # 数据缓存（用户自行下载后构建）
WEIGHTS = REPO / "weights"                      # 发布的权重
MODEL = ROOT / "model"                          # L2 模型（weights/l2 的副本）

sys.path.insert(0, str(ROOT))
import pipeline as P  # noqa: E402

W_BLEND = 0.40          # L2/L1 融合权重：(1-w)·L2 + w·L1（R6 扫描：0.10 欠拟合）
W_BLEND_BASE = 0.10     # 发布仓库原始口径，仅用于复现 0.10127 里程碑
N_FOLDS = 5
FOLD_SEED = 20260923


# ------------------------------------------------------------------ 数据

def load_all():
    tr = P.load_split("train")
    te = P.load_split("test")
    return tr, te


def images_of(split: str) -> np.ndarray:
    return np.asarray(np.load(CACHE / f"{split}_images.npy", mmap_mode="r"))


def real_mask() -> np.ndarray:
    """评分集掩码：`inc_angle` 为 4 位小数（原始字符串长度 ≤ 7）的行。

    由 `scripts/download_data.py` 生成 `test_real_mask.npy`（3424 行）。
    这是本题最关键的先验：排行榜只评这 3424 行。
    """
    f = CACHE / "test_real_mask.npy"
    if not f.is_file():
        raise SystemExit(
            f"缺少 {f}\n"
            "  请先运行：python scripts/download_data.py\n"
            "  （或 --cache-only，如果 train.json / test.json 已经放在 data/ 下）")
    return np.load(f)


# ------------------------------------------------------------------ baseline

def l1_test_frozen() -> np.ndarray:
    """冻结的 L1 集成测试预测（0.10127 流水线用的那一份）。"""
    f = WEIGHTS / "l1" / "l1_ensemble_test_predictions.csv"
    if f.is_file():
        return pd.read_csv(f, dtype={"id": str}).set_index("id")["is_iceberg"].values
    return P.l1_test_ensemble()


def l2_test(p_l1: np.ndarray):
    """L2 XGBoost 测试预测（用发布的冻结模型）。"""
    import xgboost as xgb

    tr_ang = np.load(CACHE / "train_angles.npy")
    ytr = np.load(CACHE / "train_labels.npy").astype(float)
    tables = P.group_tables(tr_ang, ytr)
    Xte, names, test = P.build_features("test", tables, p_l1)

    model = xgb.XGBClassifier()
    model.load_model(WEIGHTS / "l2" / "l2.json")
    p = np.clip(model.predict_proba(Xte)[:, 1], 0.005, 0.995)
    return p, Xte, names, test, tables


def baseline_test(w_blend: float | None = None):
    """复现 baseline：返回 (p_final, 各中间量 dict)。

    p_final = bounded_adjust(clip((1-w)·p_L2 + w·p_L1))
      w = W_BLEND (0.40) → 最终版本；w = W_BLEND_BASE (0.10) → private 0.10127
    """
    w = W_BLEND if w_blend is None else float(w_blend)
    tr_ang = np.load(CACHE / "train_angles.npy")
    ytr = np.load(CACHE / "train_labels.npy").astype(float)

    p_l1 = l1_test_frozen()
    p_l2, Xte, names, test, tables = l2_test(p_l1)
    p_blend = np.clip((1.0 - w) * p_l2 + w * p_l1, 0.005, 0.995)
    p_final, n_hit, n_adj = P.bounded_adjust(p_blend, test["angles"], tr_ang, ytr)
    return p_final, dict(p_l1=p_l1, p_l2=p_l2, p_blend=p_blend,
                         Xte=Xte, names=names, test=test, tables=tables,
                         n_hit=n_hit, n_adj=n_adj)


# ------------------------------------------------------------------ OOF 流水线

def l2_oof(p_l1_oof: np.ndarray, tables: dict, n_folds: int = N_FOLDS,
           seed: int = FOLD_SEED, params: dict | None = None):
    """五折 OOF 的 L2 预测（特征用全表标签特征，与测试构造一致）。"""
    import xgboost as xgb
    from sklearn.model_selection import StratifiedKFold

    Xtr, names, s = P.build_features("train", tables, p_l1_oof)
    ytr = s["labels"].astype(float)

    prm = dict(n_estimators=400, max_depth=2, learning_rate=0.05, subsample=0.60,
               colsample_bytree=0.55, min_child_weight=1, reg_lambda=0.1,
               random_state=42, n_jobs=8, eval_metric="logloss", tree_method="hist")
    if params:
        prm.update(params)

    oof = np.full(len(ytr), np.nan)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for f, (itr, iva) in enumerate(skf.split(Xtr, ytr)):
        m = xgb.XGBClassifier(**prm)
        m.fit(Xtr[itr], ytr[itr])
        oof[iva] = m.predict_proba(Xtr[iva])[:, 1]
    return np.clip(oof, 0.005, 0.995), Xtr, names, ytr, skf


def bounded_adjust_foldout(p: np.ndarray, ang: np.ndarray, y: np.ndarray,
                           fold_id: np.ndarray):
    """折外组表版有界调整：第 k 折用 folds != k 建组表（模拟测试情形）。"""
    out = p.copy()
    for k in np.unique(fold_id):
        m = fold_id == k
        sub, _, _ = P.bounded_adjust(p[m], ang[m], ang[~m], y[~m])
        out[m] = sub
    return out


def baseline_oof():
    """OOF 版 baseline（L1 OOF + L2 OOF + blend + 折外后处理）。"""
    from sklearn.model_selection import StratifiedKFold

    tr_ang = np.load(CACHE / "train_angles.npy")
    ytr = np.load(CACHE / "train_labels.npy").astype(float)
    tables = P.group_tables(tr_ang, ytr)
    p_l1 = P.l1_train_oof()

    p_l2, Xtr, names, y, skf = l2_oof(p_l1, tables)
    fold_id = np.zeros(len(y), dtype=int)
    for f, (_, iva) in enumerate(skf.split(Xtr, y)):
        fold_id[iva] = f

    p_blend = np.clip((1.0 - W_BLEND) * p_l2 + W_BLEND * p_l1, 0.005, 0.995)
    p_final = bounded_adjust_foldout(p_blend, tr_ang, ytr, fold_id)
    return p_final, dict(p_l1=p_l1, p_l2=p_l2, p_blend=p_blend,
                         Xtr=Xtr, names=names, y=ytr, fold_id=fold_id,
                         tables=tables, angles=tr_ang)


# ------------------------------------------------------------------ 图像统计

def image_stats(split: str, batch: int = 256) -> pd.DataFrame:
    """推理时可得、且不需要标签的图像统计量。"""
    imgs = images_of(split)
    rows = []
    for i in range(0, len(imgs), batch):
        a = np.asarray(imgs[i:i + batch], dtype=np.float32)
        hh, hv = a[:, 0], a[:, 1]
        d = hh - hv
        s = hh + hv
        rec = {}
        for nm, b in (("hh", hh), ("hv", hv), ("dif", d), ("sum", s)):
            rec[f"{nm}_min"] = b.min(axis=(1, 2))
            rec[f"{nm}_max"] = b.max(axis=(1, 2))
            rec[f"{nm}_mean"] = b.mean(axis=(1, 2))
            rec[f"{nm}_med"] = np.median(b, axis=(1, 2))
            rec[f"{nm}_std"] = b.std(axis=(1, 2))
        rec["hh_range"] = rec["hh_max"] - rec["hh_min"]
        rec["hv_range"] = rec["hv_max"] - rec["hv_min"]
        rec["dif_p25"] = np.percentile(d, 25, axis=(1, 2))
        rec["dif_p75"] = np.percentile(d, 75, axis=(1, 2))
        rec["dif_frac_pos"] = (d > 0).mean(axis=(1, 2))
        q90 = np.percentile(hh, 90, axis=(1, 2))
        rec["hh_hi_conc"] = q90 - rec["hh_mean"]
        gx = np.abs(np.diff(hh, axis=2)).mean(axis=(1, 2))
        gy = np.abs(np.diff(hh, axis=1)).mean(axis=(1, 2))
        rec["hh_edge"] = 0.5 * (gx + gy)
        gx2 = np.abs(np.diff(hv, axis=2)).mean(axis=(1, 2))
        gy2 = np.abs(np.diff(hv, axis=1)).mean(axis=(1, 2))
        rec["hv_edge"] = 0.5 * (gx2 + gy2)
        rows.append(pd.DataFrame(rec))
    df = pd.concat(rows, ignore_index=True)
    df["angle_missing"] = np.isnan(np.load(CACHE / f"{split}_angles.npy")).astype(int)
    return df


# ------------------------------------------------------------------ 评估

def logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-15, 1 - 1e-15)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def group_keys(angles: np.ndarray, decimals: int = 1) -> np.ndarray:
    return np.round(np.asarray(angles, dtype=float), decimals)


def pure_group_map(tr_ang: np.ndarray, ytr: np.ndarray, decimals: int = 1):
    """返回 (ice_keys, ship_keys)：纯冰山 / 纯船 的角度组。"""
    from collections import defaultdict

    k = group_keys(tr_ang, decimals)
    ice, ship = {}, {}
    g = defaultdict(list)
    for key, lab in zip(k, ytr):
        if not np.isnan(key):
            g[float(key)].append(float(lab))
    for key, v in g.items():
        if all(x == 1.0 for x in v):
            ice[key] = len(v)
        elif all(x == 0.0 for x in v):
            ship[key] = len(v)
    return ice, ship


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8")
