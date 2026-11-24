"""下载竞赛数据并构建 .npy 缓存。

用法
    python scripts/download_data.py              # 下载 + 构建缓存
    python scripts/download_data.py --cache-only # 只构建缓存（json 已在）
    python scripts/download_data.py --force      # 强制重建

前置条件
    Kaggle API token 放在 `~/.kaggle/access_token`（或 `~/.kaggle/kaggle.json`）。
    见 https://github.com/Kaggle/kaggle-api#api-credentials

产物（写入 BestExp/data/cache/）
    train_images.npy   (1604, 2, 75, 75)  float32   # 通道 0 = band_1(HH)，通道 1 = band_2(HV)
    test_images.npy    (8424, 2, 75, 75)  float32
    train_angles.npy   (1604,)            float32   # inc_angle，缺失为 NaN
    test_angles.npy    (8424,)            float32
    train_labels.npy   (1604,)            float32   # is_iceberg
    train_ids.npy      (1604,)            <U64
    test_ids.npy       (8424,)            <U64     # 与 sample_submission.csv 顺序一致
    test_angle_len.npy (8424,)            int64    # inc_angle 原始字符串长度
    test_real_mask.npy (8424,)            bool     # ★ 评分集掩码（3424 行）
    train_metadata.json / test_metadata.json

★ 关于 test_real_mask.npy
    本题最关键的先验：排行榜只评 inc_angle 为 4 位小数（原始字符串长度 ≤ 7）的
    **3424 行**；另外 5000 行的角度是全精度随机值，不参与评分。
    掩码由 test.json 的原始角度字符串长度判定（长度 < 14 即为真实角度），
    与历史诊断脚本逐位一致（已核对）。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
# 数据放 BestExp/data/cache/ —— 这是 pipeline.py 读取的位置
DATA = ROOT / "BestExp" / "data"
CACHE = DATA / "cache"
PROJ = ROOT / "data"          # 原始 json 的下载目录
COMPETITION = "statoil-iceberg-classifier-challenge"
IMAGE_SIDE = 75


def download() -> None:
    """用 Kaggle CLI 下载竞赛数据到 Projects/。"""

    print("从 Kaggle 下载竞赛数据…")
    r = subprocess.run(
        [sys.executable, "-m", "kaggle", "competitions", "download",
         "-c", COMPETITION, "-p", str(PROJ), "--force"],
        capture_output=True, text=True)
    if r.returncode != 0:
        print("下载失败：")
        print((r.stdout or "") + (r.stderr or ""))
        raise SystemExit(1)

    import zipfile
    z = PROJ / f"{COMPETITION}.zip"
    if z.is_file():
        print(f"解压 {z.name}…")
        with zipfile.ZipFile(z) as f:
            f.extractall(PROJ)


def build(split: str, require_labels: bool, force: bool) -> None:
    """把 json 转成 .npy 缓存。"""

    src = PROJ / f"{split}.json"
    if not src.is_file():
        raise SystemExit(f"找不到 {src}，先运行不带 --cache-only 的命令")

    dst = CACHE / f"{split}_images.npy"
    if dst.is_file() and not force:
        print(f"  {split}: 缓存已存在，跳过（--force 可强制重建）")
        return

    print(f"  构建 {split} 缓存…")
    records = json.loads(src.read_text())
    n = len(records)
    images = np.empty((n, 2, IMAGE_SIDE, IMAGE_SIDE), dtype=np.float32)
    angles = np.full(n, np.nan, dtype=np.float32)
    ids = np.empty(n, dtype="<U64")
    labels = np.empty(n, dtype=np.float32) if require_labels else None

    for i, rec in enumerate(records):
        # band_1 = HH, band_2 = HV；原始是长度 5625 的扁平列表
        images[i, 0] = np.asarray(rec["band_1"], dtype=np.float32).reshape(IMAGE_SIDE, IMAGE_SIDE)
        images[i, 1] = np.asarray(rec["band_2"], dtype=np.float32).reshape(IMAGE_SIDE, IMAGE_SIDE)
        a = rec.get("inc_angle")
        if a is not None and a != "na":
            angles[i] = float(a)
        ids[i] = rec["id"]
        if require_labels:
            labels[i] = float(rec["is_iceberg"])

    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(CACHE / f"{split}_images.npy", images)
    np.save(CACHE / f"{split}_angles.npy", angles)
    np.save(CACHE / f"{split}_ids.npy", ids)
    if require_labels:
        np.save(CACHE / f"{split}_labels.npy", labels)

    (CACHE / f"{split}_metadata.json").write_text(json.dumps({
        "cache_version": 1,
        "source": str(src),
        "source_size": src.stat().st_size,
        "require_labels": require_labels,
        "records": n,
        "image_shape": list(images.shape),
        "channel_order": ["band_1_HH", "band_2_HV"],
        "angle_missing": int(np.isnan(angles).sum()),
    }, indent=2), encoding="utf-8")

    print(f"    images {images.shape} {images.dtype}")
    print(f"    angles {angles.shape}（缺失 {int(np.isnan(angles).sum())}）")
    print(f"    ids    {ids.shape}")


def angle_len_pass() -> None:
    """流式扫描 test.json 的原始字节，记录每个 inc_angle 字符串的长度。

    长度 < 14 → 真实角度（4 位小数，如 `34.9664`，长度 7）→ 参与评分；
    长度 ≥ 14 → 合成角度（全精度随机，如 `32.615072051740334`）→ 不参与评分。
    """

    src = PROJ / "test.json"
    if not src.is_file():
        raise SystemExit(f"找不到 {src}，先运行不带 --cache-only 的命令")

    dst = CACHE / "test_real_mask.npy"
    if dst.is_file():
        print("  real_mask: 已存在，跳过（--force 可强制重建）")
        return

    print("  扫描 test.json 角度字符串长度…")
    pat = re.compile(rb'"inc_angle":"?([^",}]+)')
    lens: list[int] = []
    with open(src, "rb") as f:
        tail = b""
        while True:
            buf = f.read(1 << 24)
            if not buf:
                break
            data = tail + buf
            lens.extend(len(m.group(1)) for m in pat.finditer(data))
            tail = data[-64:]

    lens = np.asarray(lens, dtype=np.int64)
    real = lens < 14
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(CACHE / "test_angle_len.npy", lens)
    np.save(CACHE / "test_real_mask.npy", real)
    print(f"    记录 {len(lens)} 行；评分集（真实角度）{int(real.sum())} 行，"
          f"合成角度 {int((~real).sum())} 行")


def verify() -> None:
    """健全性检查。"""

    print()
    print("健全性检查：")
    import pandas as pd

    tr = np.load(CACHE / "train_images.npy", mmap_mode="r")
    te = np.load(CACHE / "test_images.npy", mmap_mode="r")
    y = np.load(CACHE / "train_labels.npy")
    tr_ids = np.load(CACHE / "train_ids.npy")
    te_ids = np.load(CACHE / "test_ids.npy")

    print(f"  train {tr.shape}   test {te.shape}")
    print(f"  训练集冰山比例 {y.mean():.4f}（基准 0.4695）")

    sub = PROJ / "sample_submission.csv"
    if sub.is_file():
        s = pd.read_csv(sub, dtype={"id": str})
        ok = np.array_equal(te_ids, s["id"].values)
        print(f"  test_ids 与 sample_submission 顺序一致: {ok}")
        if not ok:
            raise SystemExit("  ❌ id 顺序不一致，复现会失败")

    rm = CACHE / "test_real_mask.npy"
    if rm.is_file():
        n = int(np.load(rm).sum())
        print(f"  评分集（真实角度）行数 {n}（应为 3424）")
        if n != 3424:
            raise SystemExit("  ❌ real_mask 行数不对，复现会失败")

    print("  ✅ 缓存就绪")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-only", action="store_true",
                    help="跳过下载，直接用已放在 data/ 下的 train.json / test.json")
    ap.add_argument("--json-dir", default=None,
                    help="已下载的 train.json / test.json 所在目录（默认 <repo>/data）")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.json_dir:
        global PROJ
        PROJ = Path(args.json_dir).resolve()

    print("=" * 70)
    print("CS4486 — 数据下载与缓存构建")
    print("=" * 70)
    print(f"  原始 json 目录: {PROJ}")
    print(f"  缓存目标目录  : {CACHE}")
    print()

    if not args.cache_only:
        download()

    build("train", require_labels=True, force=args.force)
    build("test", require_labels=False, force=args.force)
    angle_len_pass()
    verify()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
