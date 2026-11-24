"""LINE 1 — 专家/证据模块（可复用）。

三类「区间专家证据」：
  1. `soft_group(dec, alpha)`   : 精确角度组的 Beta 后验 → logit 平移
  2. `knn_shift(K, alpha, w)`   : 角度最近 K 个 train 行的冰山率 → logit 平移
  3. `bias(mask, b)`            : 区间校准偏置
  4. `bounded`                  : 原有后处理（纯组硬确认）
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import regions as R

PI = 0.4695
_CACHE: dict = {}


def ctx():
    if _CACHE:
        return _CACHE
    # ⚠️ 专家是**冻结产物**：其训练分区（trig 伪标签行）固定在发布口径 w=0.10。
    # 最终 pipeline 用 w=0.40 分区，但只取本 ctx 的角度/标签（与 w 无关），故二者解耦。
    s = R.st(R.C.W_BLEND_BASE)
    _CACHE.update(s)
    # 排序后的 train 角度
    a = s["tr_ang"]; y = s["ytr"]
    ok = ~np.isnan(a)
    o = np.argsort(a[ok])
    _CACHE["A_sorted"] = a[ok][o]
    _CACHE["Y_sorted"] = y[ok][o]
    _CACHE["A_orig"] = np.flatnonzero(ok)[o]
    # 各精度的组计数
    for d in (1, 2, 3, 4):
        cnt = defaultdict(lambda: [0, 0])
        for key, lab in zip(np.round(a, d), y):
            if not np.isnan(key):
                rec = cnt[float(key)]
                rec[0] += 1
                rec[1] += int(lab == 1.0)
        _CACHE[f"cnt{d}"] = cnt
    return _CACHE


def _sh_from_rate(r, alpha):
    return np.clip(alpha * (R.logit(np.clip(r, 0.02, 0.98)) - R.logit(np.array([PI]))[0]),
                   -2.5, 2.5)


def group_rate(dec):
    c = ctx()
    cnt = c[f"cnt{dec}"]
    kt = np.round(c["te_ang"], dec)
    k = np.zeros(len(kt)); n = np.zeros(len(kt))
    for i, x in enumerate(kt):
        if np.isnan(x):
            continue
        rec = cnt.get(float(x))
        if rec:
            n[i], k[i] = rec
    return (k + 1.0) / (n + 2.0)


def group_shift(dec, alpha):
    return _sh_from_rate(group_rate(dec), alpha)


def knn_rate(K, wfun=None, dec=None):
    c = ctx()
    A, Y = c["A_sorted"], c["Y_sorted"]
    te = c["te_ang"]
    rate = np.full(len(te), PI)
    idx = np.searchsorted(A, np.nan_to_num(te, nan=0.0))
    for i, a in enumerate(te):
        if np.isnan(a):
            continue
        j = idx[i]
        lo, hi = max(0, j - K - 2), min(len(A), j + K + 2)
        cand = np.arange(lo, hi)
        d = np.abs(A[cand] - a)
        sel = cand[np.argsort(d)[:K]]
        if wfun is None:
            rate[i] = Y[sel].mean()
        else:
            w = wfun(np.abs(A[sel] - a))
            rate[i] = (Y[sel] * w).sum() / w.sum()
    return rate


def knn_shift(K, alpha, wfun=None):
    return _sh_from_rate(knn_rate(K, wfun), alpha)


def bias(p, mask, b):
    return R.bias(p, mask, b)


def move(p, mask, q, tgt=None):
    c = ctx()
    t = c["tgt"] if tgt is None else tgt
    o = p.copy()
    tt = t[mask]
    o[mask] = np.where(tt > 0.5, q, 1 - q)
    return o


def build(shifts=None, nh_b=0.0, e2=None, apply_mask=None, ud_b=0.0, extra_bias=None):
    """在 baseline 上叠加若干变换。shifts = [(shift_array, mask), ...]"""
    c = ctx()
    p = c["p_base"]
    o = p.copy()
    for sh, mask in (shifts or []):
        mm = c["nonhit"] if mask is None else mask
        o[mm] = R.sigmoid(R.logit(p[mm]) + sh[mm])
    if ud_b:
        o = bias(o, c["untrig_dis"], ud_b)
    if nh_b:
        o = bias(o, c["nonhit"], nh_b)
    if e2 is not None:
        o = move(o, c["untrig_agree"], e2)
    return o


def wdist(c0):
    return lambda d: 1.0 / (d + c0)
