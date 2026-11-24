"""BestExp — 从头重训 L1（6 个 CNN），生成 L1 集成。

外部用户没有我们冻结的 L1 预测文件时，用这个脚本重训。
重训约 **1 小时**（CPU，6 个模型 × 约 10 分钟）。

    CNN-A  a0_noangle         seeds 0, 1, 2
    CNN-B  b_member_gelu_film seeds 0, 1, 2
    → 各自 3 种子平均，再等权平均 = L1 集成

⚠️ 无法保证逐位一致
    CNN 训练含非确定性（cuDNN/cuBLAS、数据加载顺序、浮点归约顺序）。
    同机同环境通常可复现到 1e-4 量级，跨机可能更大。
    实测该量级的差异对最终 logloss 影响 < 0.001。

用法
    python train_l1.py                  # 训全部 6 个
    python train_l1.py --configs a0_noangle --seeds 0
    python train_l1.py --list
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "l1"))

import nrcore as nr            # noqa: E402
import configs as cfg          # noqa: E402

OUT = ROOT / "l1_out"

# ★ 折种子：实测为 20260917（不是 p4.py 里写的 42 —— 那是过期值）
FOLD_SEED = 20260917
N_FOLDS = 5
EPOCHS = 15                    # 冻结的 epoch 预算
BATCH = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
SCHEDULE = "cosine"

ENSEMBLE_MEMBERS = ("a0_noangle", "b_member_gelu_film")
ENSEMBLE_SEEDS = (0, 1, 2)


def train_full(config: str, seed: int, threads: int) -> Path:
    """全量数据重训 → 测试集预测。"""

    import torch

    spec = cfg.get(config)
    # ★ 必须在建模型/建 DataLoader 之前播种 —— 否则权重初始化用的是未播种的 RNG。
    #   原版 full4.py 也是开头就调 nr.seed_everything(seed)。
    torch.set_num_threads(threads)
    nr.seed_everything(seed)

    tr = nr.load_split("train")
    te = nr.load_split("test")
    labels = np.asarray(tr.labels, dtype=np.float64)
    all_idx = np.arange(len(labels), dtype=np.int64)

    stats = nr.fit_fold_stats(tr.images, tr.angles, all_idx)
    # ★ 顺序必须与原版 full4.py 完全一致：dataset → loader → model。
    #   构造 DataLoader 会消耗全局 torch RNG，若先建模型再建 loader，
    #   模型初始化用的随机状态就与原版不同。
    ds = nr.NRDataset(tr.images, tr.angles, tr.labels, all_idx, stats,
                      channel_mode=spec["channel_mode"],
                      preprocess=spec.get("preprocess"))
    loader = torch.utils.data.DataLoader(
        ds, batch_size=BATCH, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0, collate_fn=nr.collate)
    model = nr.build_model(spec["model_name"],
                           in_channels=nr.channel_count(spec["channel_mode"]),
                           **spec["model_kwargs"])
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = torch.nn.BCEWithLogitsLoss()

    for _ in range(EPOCHS):
        model.train()
        for x, a, y in loader:
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x, a).reshape(-1), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()

    model.eval()
    tds = nr.NRDataset(te.images, te.angles, None, np.arange(len(te.ids)), stats,
                       channel_mode=spec["channel_mode"],
                       preprocess=spec.get("preprocess"))
    tl = torch.utils.data.DataLoader(tds, batch_size=BATCH, shuffle=False,
                                     num_workers=0, collate_fn=nr.collate)
    out = []
    with torch.inference_mode():
        for batch in tl:
            out.append(torch.sigmoid(model(batch[0], batch[1]).reshape(-1)).numpy())
    p = np.concatenate(out)

    d = OUT / f"{config}_s{seed}"
    d.mkdir(parents=True, exist_ok=True)
    # ★ 保存权重：L1 无法逐位复现，权重必须随仓库一起发布
    torch.save(model.state_dict(), d / "weights.pt")
    pd.DataFrame({"id": te.ids, "is_iceberg": p}).to_csv(
        d / "submission.csv", index=False, float_format="%.8f")
    (d / "meta.json").write_text(json.dumps({
        "config": config, "seed": seed, "epochs": EPOCHS, "batch_size": BATCH,
        "learning_rate": LR, "weight_decay": WEIGHT_DECAY, "schedule": SCHEDULE,
        "channel_mode": spec["channel_mode"], "model_kwargs": spec["model_kwargs"],
        "fold_seed": FOLD_SEED, "p_mean": float(p.mean()),
        "parameter_count": int(sum(x.numel() for x in model.parameters()
                                   if x.requires_grad)),
        "weights_file": "weights.pt",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return d


def build_ensemble() -> Path:
    """把 6 个模型按 3 种子平均后等权平均，写 L1 集成文件。"""

    tr = nr.load_split("test")
    per = []
    for config in ENSEMBLE_MEMBERS:
        vs = []
        for s in ENSEMBLE_SEEDS:
            f = OUT / f"{config}_s{s}" / "submission.csv"
            vs.append(pd.read_csv(f, dtype={"id": str}).set_index("id")
                      ["is_iceberg"].reindex(tr.ids).values)
        per.append(np.vstack(vs).mean(axis=0))
    ens = np.vstack(per).mean(axis=0)

    d = OUT / "ENS_l1"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": tr.ids, "is_iceberg": ens}).to_csv(
        d / "submission.csv", index=False, float_format="%.8f")
    print(f"  L1 集成 p_mean = {ens.mean():.5f}  → {d/'submission.csv'}")
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(ENSEMBLE_MEMBERS))
    ap.add_argument("--seeds", default=",".join(str(s) for s in ENSEMBLE_SEEDS))
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--ensemble-only", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    jobs = [(c, s) for c in configs for s in seeds]

    if args.list:
        print(f"折种子 {FOLD_SEED}｜{EPOCHS} epoch｜batch {BATCH}｜lr {LR}｜wd {WEIGHT_DECAY}")
        for c, s in jobs:
            spec = cfg.get(c)
            print(f"  {c}_s{s}  {spec['model_name']}  {spec['model_kwargs']}")
        return 0

    if not args.ensemble_only:
        print("=" * 78)
        print(f"BestExp / train_l1.py — 重训 L1（{len(jobs)} 个模型）")
        print("=" * 78)
        print(f"  折种子 {FOLD_SEED}｜{EPOCHS} epoch｜batch {BATCH}｜lr {LR}｜"
              f"wd {WEIGHT_DECAY}｜schedule {SCHEDULE}")
        t0 = time.perf_counter()
        for i, (c, s) in enumerate(jobs, 1):
            d = OUT / f"{c}_s{s}"
            if (d / "submission.csv").is_file():
                print(f"  [{i}/{len(jobs)}] {c}_s{s} 已存在，跳过")
                continue
            print(f"  [{i}/{len(jobs)}] {c}_s{s} 训练中…", flush=True)
            t1 = time.perf_counter()
            train_full(c, s, args.threads)
            print(f"      完成，用时 {(time.perf_counter()-t1)/60:.1f} min", flush=True)
        print(f"  全部完成，总用时 {(time.perf_counter()-t0)/60:.1f} min")

    print()
    print("构造 L1 集成：")
    build_ensemble()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
