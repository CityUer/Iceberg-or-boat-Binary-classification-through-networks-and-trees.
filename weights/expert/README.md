# 跨域专家（后处理步骤 ⑤）

一个 XGBoost 模型，用来在 `nonhit ∪ untrig` 区（2042 行）替换掉 baseline 的预测。
在最终流水线里以 **β = 0.15** 的权重混合：

```
p = 0.85 · p_prev + 0.15 · p_expert
```

## 它为什么存在

baseline 的 L1（CNN）和 L2（XGBoost）都只在 1604 个训练样本上拟合，
而测试集有 8424 行、且分布不同（`p_l1` 的 OOF 均值 0.489 vs 测试集成均值 0.273）。
这个专家换了个训练域：**把 1382 个 `trig` 评分行当额外训练数据**。

`trig` = 角度落在 1 位小数纯组、且 `bounded_adjust` 确实触发过的行。
这些行的组标签准确率约 **99.8%**，所以可以当**伪标签**用。

## 特征（16 列，顺序敏感）

| # | 名称 | 含义 |
|---|---|---|
| 0 | `p_l1` | L1 集成预测 |
| 1 | `angle` | `inc_angle` |
| 2–11 | `band0..9` | HH/HV 的 min/mean/median/max/std（10 列）|
| 12 | `knn_rate` | 角度最近 K=4 个 train 行的冰山率（train 行用 LOO）|
| 13 | `g2_rate` | 2 位小数组的冰山率（Beta 平滑，train 行 LOO）|
| 14 | `g2_size` | 2 位小数组的样本数（train 行 LOO）|
| 15 | `g4_size` | 4 位小数组的样本数（train 行 LOO）|

**刻意不含** 4 位小数精确组的标签特征 —— 那正是泄漏源，会主导训练。

## 训练配置（本包只发布产物，不含训练脚本）

| 项 | 值 |
|---|---|
| 训练集 A | 1604 train 中角度非缺失的 1471 行（LOO 特征）|
| 训练集 B | 1382 个 `trig` 评分行（伪标签）|
| 超参 | `n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.7, colsample_bytree=0.6, min_child_weight=2, reg_lambda=1.0, random_state=42, n_jobs=8, tree_method=hist` |

## 文件

| 文件 | 说明 |
|---|---|
| `expert_ab.json` | XGBoost 模型（约 407 KB）|
| `feature_names.json` | 16 列特征名，顺序敏感 |
| `trans_ab_test.npy` | 8424 行测试预测，`float32`，**冻结** |

## 复现性

✅ `trans_ab_test.npy` 是冻结产物，`final_pipeline.py` 直接读取它，
最终提交与 `submission_0.08488.csv` 逐值差 **0.00e+00**。
