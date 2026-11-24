"""BestExp — inference using the published weights in `weights/`.

This is the path a teammate takes when they clone the repo and want to
reproduce the score **without retraining anything**:

    1. load the 6 L1 CNN weight files from weights/l1/
    2. run them on the test images, average -> L1 ensemble
    3. feed the L1 ensemble into the L2 pipeline (train-free) -> submission.csv

Verified score: **private logloss 0.10127** (Kaggle ref 56451534).

Usage
    python predict_from_weights.py                  # uses ../weights by default
    python predict_from_weights.py --weights ../weights --out submission.csv
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

import nrcore as nr        # noqa: E402
import configs as cfg      # noqa: E402
import pipeline as P       # noqa: E402

DEFAULT_WEIGHTS = ROOT.parent / "weights"
BATCH = 32


def l1_ensemble_from_weights(weights_dir: Path, threads: int) -> np.ndarray:
    """Run the 6 published CNNs on the test set and average them."""

    import torch

    torch.set_num_threads(threads)

    tr = nr.load_split("train")
    te = nr.load_split("test")

    # The published CNNs were trained on ALL labelled data, so the
    # standardisation statistics come from the full training set.
    all_idx = np.arange(len(tr.labels), dtype=np.int64)
    stats = nr.fit_fold_stats(tr.images, tr.angles, all_idx)

    members: list[np.ndarray] = []
    for wdir in sorted((weights_dir / "l1").iterdir()):
        wf = wdir / "weights.pt"
        if not wf.is_file():
            continue
        meta = json.loads((wdir / "meta.json").read_text(encoding="utf-8"))
        spec = cfg.get(meta["config"])

        model = nr.build_model(spec["model_name"],
                               in_channels=nr.channel_count(spec["channel_mode"]),
                               **spec["model_kwargs"])
        model.load_state_dict(torch.load(wf, map_location="cpu"))
        model.eval()

        ds = nr.NRDataset(te.images, te.angles, None,
                          np.arange(len(te.ids)), stats,
                          channel_mode=spec["channel_mode"],
                          preprocess=spec.get("preprocess"))
        dl = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=False,
                                         num_workers=0, collate_fn=nr.collate)
        out = []
        with torch.inference_mode():
            for x, a, _ in dl:
                out.append(torch.sigmoid(model(x, a).reshape(-1)).numpy())
        p = np.concatenate(out)
        members.append(p)
        print(f"    {wdir.name:<26} p_mean={p.mean():.6f}")

    # per-config 3-seed average, then equal-weight average across configs
    per_cfg: dict[str, list[np.ndarray]] = {}
    for wdir, p in zip(sorted((weights_dir / "l1").iterdir()), members):
        c = json.loads((wdir / "meta.json").read_text(encoding="utf-8"))["config"]
        per_cfg.setdefault(c, []).append(p)
    cfg_means = [np.vstack(v).mean(axis=0) for v in per_cfg.values()]
    return np.vstack(cfg_means).mean(axis=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    import xgboost as xgb

    t0 = time.perf_counter()
    weights_dir = Path(args.weights)
    print("=" * 74)
    print("BestExp / predict_from_weights.py — inference from published weights")
    print("=" * 74)
    print(f"  weights dir: {weights_dir}")

    # ---- L1: run the published CNNs ---------------------------------------
    print("\n  [L1] running the 6 published CNNs:")
    p_l1 = l1_ensemble_from_weights(weights_dir, args.threads)
    print(f"    L1 ensemble p_mean = {p_l1.mean():.6f}")

    # ---- features ---------------------------------------------------------
    tr_ang = np.load(P.CACHE / "train_angles.npy")
    ytr = np.load(P.CACHE / "train_labels.npy").astype(float)
    tables = P.group_tables(tr_ang, ytr)
    Xte, names, test = P.build_features("test", tables, p_l1)

    saved = json.loads((weights_dir / "l2" / "feature_names.json")
                       .read_text(encoding="utf-8"))
    if saved != names:
        raise SystemExit(f"feature mismatch\n  trained: {saved}\n  inferred: {names}")
    print(f"\n  [L2] feature matrix {Xte.shape}, names match training ✓")

    # ---- L2 ---------------------------------------------------------------
    model = xgb.XGBClassifier()
    model.load_model(weights_dir / "l2" / "l2.json")
    p_l2 = np.clip(model.predict_proba(Xte)[:, 1], 0.005, 0.995)
    print(f"    p_L2 mean {p_l2.mean():.6f}")

    # ---- blend ------------------------------------------------------------
    p = np.clip(0.9 * p_l2 + 0.1 * p_l1, 0.005, 0.995)
    print(f"    blended mean {p.mean():.6f}   (w = 0.10)")

    # ---- bounded adjustment ----------------------------------------------
    p_final, n_hit, n_adj = P.bounded_adjust(p, test["angles"], tr_ang, ytr)
    print(f"    bounded adjust: {n_hit} in pure groups, {n_adj} moved, "
          f"mean {p_final.mean():.6f}")

    # ---- write ------------------------------------------------------------
    out = ROOT / args.out
    pd.DataFrame({"id": test["ids"], "is_iceberg": p_final}).to_csv(
        out, index=False, float_format="%.8f")
    print(f"\n  -> {out}")
    print(f"  sha256 {P.sha256(out)}")
    print(f"  elapsed {time.perf_counter() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
