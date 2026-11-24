# Iceberg or boat? Binary classification with network and trees. (By Team 24)

**Final result.** On the Kaggle *Statoil/C-CORE Iceberg Classifier Challenge*
(binary log loss, lower is better), this repository reproduces a **private score of
0.08488** and a **public score of 0.09227** --- rank **2 of 3,330 teams** on the
official 2018 private leaderboard (submission tag `L1_R7_a035`). The score is the
minimum of a submission-based hyperparameter search and is therefore an optimistic
point estimate; differences below ~0.0026 (the model-variation standard deviation)
should be treated as ties.

## 1. Environment

Tested on Windows 11, Python 3.11, CPU only (PyTorch CPU build; no GPU/CUDA needed).
Full inference takes about 20 seconds.

```bash
conda env create -f environment.yml
conda activate iceberg_classify
```

Or with plain pip:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -c "import torch, xgboost; print(torch.__version__, xgboost.__version__)"
# expect: 2.14.0+cpu 3.2.0
```

## 2. Data

The competition data is **not** included (it exceeds GitHub's size limit). Place the
three raw files in a `data/` directory at the repository root:

```
<repo root>/
└── data/
    ├── train.json              (187 MB)
    ├── test.json               (1.45 GB)
    └── sample_submission.csv   (118 KB)
```

Then build the `.npy` cache (takes a few minutes):

```bash
# If you already have the JSON files locally:
python scripts/download_data.py --cache-only --json-dir /path/to/your/data

# Or let the script download them from Kaggle
# (requires your token at ~/.kaggle/access_token, see
#  https://github.com/Kaggle/kaggle-api#api-credentials):
python scripts/download_data.py
```

The script writes the cache to `BestExp/data/cache/` and also rebuilds
`test_real_mask.npy`.

## 3. Reproduce the final submission

```bash
python BestExp/final_pipeline.py
```

This writes `BestExp/submission.csv`. All model predictions are frozen under
`weights/`, so nothing is trained and the six CNNs never need to be retrained.

## 4. Verify

The script prints the SHA-256 of the generated file and compares it against the
frozen expected output `BestExp/submission_0.08488.csv`; the maximum per-row
difference should be `0.00e+00` and the hash should be:

```
29f24c089a364730a5a908a930d28b92f06db97781dec03a74ec925027826642
```

The pipeline is deterministic and reproduces the submitted file exactly. The six
CNNs are not bitwise reproducible because of floating-point reduction order, which
is why their trained weights and frozen predictions are shipped.

## 5. (Optional) Submit to Kaggle

```bash
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
kaggle competitions submit -c statoil-iceberg-classifier-challenge \
  -f BestExp/submission.csv -m "L1_R7_a035"
```

Expected private score: **0.08488**.
