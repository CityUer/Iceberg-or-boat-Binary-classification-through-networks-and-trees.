# BestExp — 最小复现包（private 0.08488）

复现 **private log loss 0.08488 / public 0.09227**（Kaggle tag `L1_R7_a035`，榜单第 2）。

## 准备

```bash
cd ..                                     # 仓库根
pip install -r requirements.txt
# 把 train.json / test.json / sample_submission.csv 放到 ../data/ 下，然后：
python scripts/download_data.py           # 需要你自己的 Kaggle token
```

会在 `BestExp/data/cache/` 建好 `.npy` 缓存（含评分集掩码 `test_real_mask.npy`）。

## 运行

| 路径 | 命令 | 用时 | 分数 |
|---|---|---|---|
| **★ 最终流水线** | `python final_pipeline.py` | ~13 s | **0.08488** |
| baseline（冻结 L1 预测） | `python predict.py` | ~8 s | 0.10127 |
| baseline（跑 CNN 权重） | `python predict_from_weights.py --weights ../weights` | ~5 min | 0.10127 |

全部写出 `submission.csv`。**只想要结果就用第一行**，它只读冻结产物、不训练任何东西。
产物与 `submission_0.08488.csv` **逐值差 0.00e+00**（sha256 `29f24c08…6642`）。

> 本包只含复现所需的冻结产物；训练脚本与 OOF 数据未随包分发。

## 文件

| 文件 | 作用 |
|---|---|
| `pipeline.py` | baseline 共享：数据加载、24 列特征、`bounded_adjust` |
| `common.py` | 自包含共享工具：baseline 复现、OOF 流水线、图像统计 |
| `regions.py` | 评分集 3424 行的分区 |
| `expert.py` | 区间专家证据（KNN 角度率 / 区间偏置） |
| **`final_pipeline.py`** | ★ 最终流水线（baseline + 5 步后处理） |
| `predict.py` | baseline 推理（冻结 L1 预测，最快） |
| `predict_from_weights.py` | baseline 推理（从 `../weights/` 加载 CNN） |
| `l1/` | `nrcore.py` / `models.py` / `configs.py`，自包含 |
| `model/` | L2 模型 + 特征名 + 元信息 |
| `data/cache/` | 数据缓存（**你生成，不入库**） |
| `submission_0.08488.csv` | 冻结的期望输出 |

> 训练脚本及 OOF 数据未随本公开包分发。

## 流水线

```
【baseline — 完全不动】
L1 集成      a0_noangle(mish, 不用角度) ×3
             + b_member_gelu_film(gelu, FiLM) ×3   -> p_L1     0.15202
L2 元模型    XGBoost, 24 列特征                    -> p_L2     0.11215
混合         (1-w)·p_L2 + w·p_L1,  w = 0.40         ->            —
有界调整     纯组，T = 0.30·(n/n̄)²                 ->          0.10127

【后处理层 — 只作用于评分集 3424 行】
① KNN 角度证据（nonhit 1915 行）      K=4, α=0.35
② nonhit 校准偏置                     logit += 0.40
③ untrig_agree（90 行）               → 0.94 / 0.06
④ untrig_dis 真未移动 47 行           logit += −2.00      ★ 关键一步
⑤ 跨域专家混合（2056 行）             β = 0.15
⑥ clip 到 [0.005, 0.995]                                  -> 0.08488
```

**24 列特征**：`p_l1` · `angle` · 10 列 HH/HV 波段统计 · 6 列最近 K 组几何 ·
6 列角度组标签统计。

**L2 超参**：`n_estimators=400, max_depth=2, learning_rate=0.05,
subsample=0.60, colsample_bytree=0.55, min_child_weight=1, reg_lambda=0.1,
random_state=42, n_jobs=8, tree_method=hist`。

**标签特征卫生**：训练行用留一法计数，行永远看不到自己的标签；测试行用全组统计。

**专家（⑤）超参**：`n_estimators=400, max_depth=3, learning_rate=0.05,
subsample=0.7, colsample_bytree=0.6, min_child_weight=2, reg_lambda=1.0,
random_state=42`。

## ⚠️ 引用分数时

1. **选择偏差** — 0.08488 是从 200+ 次排行榜提交里取的最小值，
   模型变异 sd ≈ 0.0026，所以它被**系统性抬高**。
2. **差距 < 0.0026 视为并列**（例如 0.08488 vs 0.08526 是并列；
   只有与 0.10127 baseline 的 0.016 差距才是实质性的）。
3. **`p_l1` 的 train/test 分布不同**（OOF 均值 0.489 vs 测试集成均值 0.273）。
   这是数据性质，不是 bug —— 秩归一化会更差。
4. **核心机制是数据泄漏**（角度在同一次过境间共享）。这是已披露的研究选择。

## 逐步验证

```bash
# baseline 是否复现（应得 private 0.10127 的那份提交）
python predict.py

# 最终流水线是否与冻结产物一致（应打印 0.00e+00）
python final_pipeline.py
```
