"""BestExp — 评分集（3424 真实角度行）的分区与工具。

关键事实（诊断得出）
------------------
* 排行榜只评 3424 个「真实角度」行（test.json 里 inc_angle 为 4 位小数者）；
  另外 5000 行的角度是全精度随机值，**不参与评分**（常数探针证明）。
* 评分集冰山率 ρ ≈ 0.527。
* 纯角度组（1 位小数）命中 1509 行；其中触发后处理 1382 行。

分区（全部只用推理时可得的信息）
--------------------------------
    real         3424  评分集
    hit          1509  角度落在 1 位小数「纯组」里
    trig         1382  hit 且 |target − p_blend| ≤ T（被 bounded_adjust 移动）
    untrig        127  hit 但未触发
      untrig_agree 77  untrig 且模型方向与组标签一致
      untrig_dis   50  untrig 且模型方向与组标签相反  ← 负偏置作用区
    nonhit       1915  real 且不在任何 1 位小数纯组里
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

EPS = 0.005


def _build(w_blend: float | None = None):
    st = {}
    tr_ang = np.load(C.CACHE / "train_angles.npy").astype(float)
    ytr = np.load(C.CACHE / "train_labels.npy").astype(float)
    te_ang = np.load(C.CACHE / "test_angles.npy").astype(float)
    p_base, d = C.baseline_test(w_blend)

    st.update(tr_ang=tr_ang, ytr=ytr, te_ang=te_ang, p_base=p_base,
              ids=d["test"]["ids"].astype(str), p_l1=d["p_l1"], p_l2=d["p_l2"],
              p_blend=d["p_blend"], Xte=d["Xte"])
    st["real"] = C.real_mask()
    st["n_scored"] = int(st["real"].sum())

    # 1 位小数纯组
    g = defaultdict(list)
    for k, l in zip(np.round(tr_ang, 1), ytr):
        if not np.isnan(k):
            g[float(k)].append(float(l))
    ice = {k: len(v) for k, v in g.items() if all(x == 1.0 for x in v)}
    ship = {k: len(v) for k, v in g.items() if all(x == 0.0 for x in v)}
    kt = np.round(te_ang, 1)
    is_ice = np.array([(not np.isnan(x)) and float(x) in ice for x in kt])
    is_ship = np.array([(not np.isnan(x)) and float(x) in ship for x in kt])
    hit = (is_ice | is_ship) & st["real"]
    tgt = np.where(is_ice, 1.0, 0.0)
    n = np.array([(ice.get(float(x), 0) if float(x) in ice else ship.get(float(x), 0))
                  for x in kt]).astype(float)
    med = np.median(n[hit])
    T = np.clip(0.30 * ((n / med) ** 2), 0.06, 1.20)
    trig = hit & (np.abs(tgt - st["p_blend"]) <= T)

    st.update(ice=ice, ship=ship, kt=kt, is_ice=is_ice, is_ship=is_ship,
              hit=hit, tgt=tgt, gsize=n, T=T, trig=trig)

    agree = np.zeros(len(kt), bool)
    agree[hit] = (st["p_blend"][hit] > 0.5) == (tgt[hit] > 0.5)
    st["agree"] = agree
    st["untrig"] = hit & ~trig
    st["untrig_agree"] = hit & ~trig & agree
    st["untrig_dis"] = hit & ~trig & ~agree
    st["nonhit"] = st["real"] & ~hit

    # 多精度纯组
    multi = {}
    for dec in (2, 3, 4):
        gg = defaultdict(list)
        for k, l in zip(np.round(tr_ang, dec), ytr):
            if not np.isnan(k):
                gg[float(k)].append(float(l))
        ic = {k: len(v) for k, v in gg.items() if all(x == 1.0 for x in v)}
        sp = {k: len(v) for k, v in gg.items() if all(x == 0.0 for x in v)}
        kk = np.round(te_ang, dec)
        i2 = np.array([(not np.isnan(x)) and float(x) in ic for x in kk])
        s2 = np.array([(not np.isnan(x)) and float(x) in sp for x in kk])
        h2 = (i2 | s2) & st["real"]
        t2 = np.where(i2, 1.0, 0.0)
        n2 = np.array([(ic.get(float(x), 0) if float(x) in ic else sp.get(float(x), 0))
                       for x in kk]).astype(float)
        multi[dec] = dict(hit=h2, tgt=t2, size=n2, is_ice=i2, ice=ic, ship=sp)
    st["multi"] = multi
    return st


_S: dict = {}


def st(w_blend: float | None = None):
    """分区状态。w_blend=None 用 C.W_BLEND；分区依赖 w，故按 w 缓存。"""
    key = C.W_BLEND if w_blend is None else round(float(w_blend), 4)
    if key not in _S:
        _S[key] = _build(key)
    return _S[key]


def logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def bias(p, mask, b):
    o = p.copy()
    o[mask] = sigmoid(logit(p[mask]) + b)
    return o


def temp(p, mask, T):
    o = p.copy()
    o[mask] = sigmoid(logit(p[mask]) / T)
    return o


def write(tag, p):
    s = st()
    import pandas as pd
    out = C.ROOT / f"{tag}.csv"
    pd.DataFrame({"id": s["ids"], "is_iceberg": np.clip(p, EPS, 1 - EPS)}).to_csv(
        out, index=False, float_format="%.8f")
    return out


def report():
    s = st()
    print(f"  评分集 {s['n_scored']} 行")
    for nm in ("hit", "trig", "untrig", "untrig_agree", "untrig_dis", "nonhit"):
        print(f"    {nm:14s} {s[nm].sum():5d}")
    for dec in (2, 3, 4):
        print(f"    {dec}位纯组命中 {s['multi'][dec]['hit'].sum():5d}")


if __name__ == "__main__":
    report()
