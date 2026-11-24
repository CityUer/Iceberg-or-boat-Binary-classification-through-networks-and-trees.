"""BestExp — 推理并生成提交文件。

流程：L2 预测 → 与 L1 集成混合 → 有界调整 → submission.csv

    混合:  p = 0.9 * p_L2 + 0.1 * p_L1_ensemble        （w = 0.10）
    后处理: 仅纯角度组（1 位小数），keep_if_far，目标 0.995/0.005，
            T = 0.30 * (n / median_n)^2，clip 到 [0.06, 1.20]

用法:  python predict.py [--out submission.csv]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import pipeline as P

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "model"
W_BLEND = 0.10          # 诚实集成的权重（在 LB 上选）


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--l1", default=None,
                    help="L1 集成目录（含 submission.csv）。默认用发布的冻结集成预测；"
                         "重训后用 l1_out/ENS_l1/")
    args = ap.parse_args()

    import xgboost as xgb

    t0 = time.perf_counter()
    print("=" * 74)
    print("BestExp / predict.py — 推理并生成提交文件")
    print("=" * 74)

    # ---- 特征 -------------------------------------------------------------
    tr_ang = np.load(P.CACHE / "train_angles.npy")
    ytr = np.load(P.CACHE / "train_labels.npy").astype(float)
    tables = P.group_tables(tr_ang, ytr)

    if args.l1:
        src = Path(args.l1) / "submission.csv"
        print(f"  L1 集成来源: {src}")
        p_te_l1 = pd.read_csv(src, dtype={"id": str}).set_index("id")["is_iceberg"].values
    else:
        p_te_l1 = P.l1_test_ensemble()
    Xte, names, test = P.build_features("test", tables, p_te_l1)
    print(f"  测试特征矩阵: {Xte.shape}")
    print(f"  p_l1(集成) 均值 {p_te_l1.mean():.5f}")

    saved = json.loads((MODEL / "feature_names.json").read_text(encoding="utf-8"))
    if saved != names:
        raise SystemExit(f"特征名与训练时不一致，复现失败\n  训练: {saved}\n  推理: {names}")
    print(f"  特征名与训练时一致 ✓（{len(names)} 列）")

    # ---- L2 预测 ----------------------------------------------------------
    model = xgb.XGBClassifier()
    model.load_model(MODEL / "l2.json")
    p_l2 = np.clip(model.predict_proba(Xte)[:, 1], 0.005, 0.995)
    print(f"  p_L2 均值 {p_l2.mean():.5f}")

    # ---- 混合 -------------------------------------------------------------
    p = np.clip((1 - W_BLEND) * p_l2 + W_BLEND * p_te_l1, 0.005, 0.995)
    print(f"  混合后均值 {p.mean():.5f}   (w = {W_BLEND})")

    # ---- 有界调整 ---------------------------------------------------------
    p_final, n_hit, n_adj = P.bounded_adjust(p, test["angles"], tr_ang, ytr)
    print(f"  有界调整: 命中 {n_hit}，实际移动 {n_adj}，均值 {p_final.mean():.5f}")

    # ---- 写出 -------------------------------------------------------------
    out = ROOT / args.out
    pd.DataFrame({"id": test["ids"], "is_iceberg": p_final}).to_csv(
        out, index=False, float_format="%.8f")
    print(f"  → {out}")
    print(f"  sha256 {P.sha256(out)}")
    print(f"  用时 {time.perf_counter() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
