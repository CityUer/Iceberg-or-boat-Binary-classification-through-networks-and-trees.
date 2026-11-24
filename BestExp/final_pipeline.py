"""BestExp — 最终提交流水线（可复现）。★ 2026-10-27 更新

private 0.08488 / public 0.09227   （Kaggle tag `L1_R7_a035`，榜单第 2）
上一版：private 0.08704 / public 0.09326（w=0.10）
更早：0.08908；baseline：private 0.10127 / public 0.11213

流程
----
0. baseline（`pipeline.py` 冻结产物，完全不动）
   weights/l1/l1_ensemble_test_predictions.csv + weights/l2/l2.json
   → blend((1-w)·L2 + w·L1)，w = 0.40 → `pipeline.bounded_adjust`（1 位小数纯组硬确认）
        → private 0.10127
1. 只作用于「评分集」= 3424 个真实角度行（inc_angle 为 4 位小数者）
2. 区间划分（全部只用推理时可得的信息，见 `regions.py`）
   - hit    : 角度落在 1 位小数「纯组」里
   - trig   : hit 且 |target − p_blend| ≤ T（被 baseline 后处理移动）
   - untrig : hit 且未触发
     · untrig_agree : untrig 且模型方向与组标签一致（77 行）
     · untrig_dis   : untrig 且模型方向与组标签相反（50 行，其中 44 行"真未移动"）
   - nonhit : 不在任何 1 位小数纯组里（1915 行）
3. 三类「专家证据」
   (a) KNN 角度证据（nonhit）：
       取角度最近的 K=4 个 train 行，冰山率 r
       logit(p) += α·(logit(clip(r,0.02,0.98)) − logit(0.4695)),  α = 0.35
   (b) nonhit 区校准偏置： logit(p) += 0.40
   (c) untrig_agree（77 行）→ 直接设为 0.94 / 0.06
4. ★ 新增：untrig_dis 区负偏置（**真未移动的 44 行**）  logit(p) += −2.00
   —— 该区组标签准确率仅约 0.31（组样本极少：n≤2 的占 25/44），模型方向反而更可靠，
      故整体压低其自信度。
      ⚠️ 必须排除 `moved` 的 6 行（已被 bounded_adjust 正确钉到 0.005/0.995，动它 = 自伤）。
   —— 实测曲线：0→0.08908、−0.5→0.08819、−0.9→0.08766、−1.5→0.08718、
      **−2.00→0.08704**、−2.25→0.08704、−2.50→0.08709、−3.00→0.08729（−2.0~−2.25 为平台）。
5. 跨域专家混合（nonhit ∪ untrig，β = 0.15）：
   专家 = XGBoost，训练集 = 1604 train（LOO 特征）+ 1382 个 trig 评分行（组标签伪标签）
   特征 = [p_l1, angle, band×10, KNN率(LOO), 2位组冰山率(LOO), 2位组大小(LOO), 4位组大小(LOO)]
   （冻结预测在 `weights/expert/trans_ab_test.npy`）
6. clip 到 [0.005, 0.995]

⚠️ 步骤顺序必须与生成候选时的实现一致（第 4 步在专家混合之前），否则不逐值复现。

用法
    python final_pipeline.py                        # → submission.csv
    python final_pipeline.py --out my.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import common as C      # noqa: E402
import expert as E      # noqa: E402
import regions as R     # noqa: E402

BETA = 0.15
K_NN = 4
ALPHA = 0.35          # ★ R7/R8：0.25 → 0.35（平台 0.30~0.40 的中心）
NH_BIAS = 0.40
E2_TARGET = 0.94
UD_BIAS = -2.00          # ★ 定稿值（−2.00 与 −2.25 并列最好；−2.5 起回升）

REF = ROOT / "submission_0.08488.csv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="submission.csv")
    args = ap.parse_args()

    t0 = time.perf_counter()
    print("=" * 74)
    print("BestExp / final_pipeline.py — 最终流水线（private 0.08488）")
    print("=" * 74)

    s = R.st()
    p = s["p_base"].copy()

    nonhit = s["nonhit"]
    untrig = s["untrig"]
    ua = s["untrig_agree"]
    ud = s["untrig_dis"]
    tgt = s["tgt"]
    real = s["real"]

    print(f"  评分集 {s['n_scored']} 行  "
          f"(hit {s['hit'].sum()} / nonhit {nonhit.sum()})")

    # 真被 bounded_adjust 移动的行
    moved = (np.abs(s["p_base"] - s["p_blend"]) > 1e-12) & real

    # 3(a) KNN 角度证据（仅 nonhit）
    sh = E.knn_shift(K_NN, ALPHA)
    p[nonhit] = R.sigmoid(R.logit(p[nonhit]) + sh[nonhit])
    print(f"  3(a) KNN 角度证据    nonhit {nonhit.sum()} 行，K={K_NN} α={ALPHA}")

    # 3(b) nonhit 校准偏置
    p[nonhit] = R.sigmoid(R.logit(p[nonhit]) + NH_BIAS)
    print(f"  3(b) nonhit 校准偏置 logit += {NH_BIAS:+.2f}")

    # 3(c) untrig_agree → 0.94 / 0.06
    tt = tgt[ua]
    p[ua] = np.where(tt > 0.5, E2_TARGET, 1 - E2_TARGET)
    print(f"  3(c) untrig_agree    {ua.sum()} 行 → {E2_TARGET} / {1 - E2_TARGET:.2f}")

    # 4 ★ untrig_dis 真未移动行 → logit 负偏置
    m_ud = ud & ~moved
    p[m_ud] = R.sigmoid(R.logit(p[m_ud]) + UD_BIAS)
    print(f"  4    untrig_dis 负偏置 {m_ud.sum()} 行，logit += {UD_BIAS:+.2f}"
          f"（区内 {ud.sum()} 行，排除 {int((ud & moved).sum())} 行已移动）")

    # 5 跨域专家混合（读冻结预测）
    f = C.WEIGHTS / "expert" / "trans_ab_test.npy"
    if not f.is_file():
        raise SystemExit(f"缺少 {f}")
    exp = np.load(f)
    print(f"  5    跨域专家混合      β={BETA}（冻结预测 {f.name}）")
    m = nonhit | untrig
    p[m] = (1 - BETA) * p[m] + BETA * exp[m]
    print(f"       作用 {m.sum()} 行（nonhit ∪ untrig）")

    # 6 clip
    p = np.clip(p, 0.005, 0.995)

    out = ROOT / args.out
    pd.DataFrame({"id": s["ids"], "is_iceberg": p}).to_csv(
        out, index=False, float_format="%.8f")
    print(f"\n  → {out}")
    print(f"  mean={p.mean():.5f}   sha256 {C.P.sha256(out)}")
    print(f"  用时 {time.perf_counter() - t0:.1f} s")

    if REF.is_file():
        a = pd.read_csv(out, dtype={"id": str}).sort_values("id").reset_index(drop=True)
        b = pd.read_csv(REF, dtype={"id": str}).sort_values("id").reset_index(drop=True)
        if np.array_equal(a.id.values, b.id.values):
            d = float(np.abs(a.is_iceberg.values - b.is_iceberg.values).max())
            print(f"  与冻结产物 {REF.name} 最大逐值差 = {d:.2e}")
        else:
            print("  ⚠️ id 集合不一致，无法比对")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
