# 模型权重（private log loss 0.08488）

## 为什么随仓库发布

六个 L1 CNN **无法逐位复现** —— CNN 训练带有不可约的浮点非确定性。
所以它们的训练权重和冻结预测直接发布，复现时无需训练。
L2、后处理层、跨域专家都是确定性的，其产物一并冻结。

## 内容

```
weights/
├── l1/                              L1：六个 CNN（3 种子 × 2 配置）
│   ├── a0_noangle_s{0,1,2}/             weights.pt + meta.json
│   ├── b_member_gelu_film_s{0,1,2}/     weights.pt + meta.json
│   └── l1_ensemble_test_predictions.csv  冻结的 L1 集成测试预测
├── l2/                              L2：XGBoost
│   ├── l2.json
│   ├── feature_names.json           24 列特征名（顺序敏感）
│   └── train_meta.json
└── expert/                          跨域专家（后处理步骤 ⑤）
    ├── expert_ab.json               XGBoost 模型
    ├── feature_names.json           16 列特征名（顺序敏感）
    └── trans_ab_test.npy            8424 行测试预测（float32，冻结）
```

## 怎么用

复现最终结果只需一条命令（读取全部冻结产物）：

```bash
python BestExp/final_pipeline.py
# -> BestExp/submission.csv，private 0.08488
```

也可以单独加载某一层：

```python
# L2（XGBoost）
import json, xgboost as xgb
model = xgb.XGBClassifier()
model.load_model("weights/l2/l2.json")
names = json.load(open("weights/l2/feature_names.json"))   # 列顺序必须一致
p_l2 = model.predict_proba(X)[:, 1]

# 跨域专家（XGBoost，16 列）
expert = xgb.XGBClassifier()
expert.load_model("weights/expert/expert_ab.json")

# L1 集成预测（最省事，直接用冻结概率）
import pandas as pd
p_l1 = pd.read_csv("weights/l1/l1_ensemble_test_predictions.csv",
                   dtype={"id": str}).set_index("id")["is_iceberg"].values
```

CNN 权重可用 `BestExp/l1/` 里的模型定义加载；输入必须用全训练集的统计量标准化。

## 注意事项

1. `l2.json` 是用 `random_state=42, n_jobs=8` 训的；加载不受影响，
   重训必须用同样的值（`n_jobs` 会影响直方图构建顺序）。
2. 特征列顺序很重要，每次都要和 `feature_names.json` 核对。
3. `trans_ab_test.npy` 是冻结的专家输出，`final_pipeline.py` 默认读它，保证逐位一致。
4. 每个 `meta.json` 记录 config、seed、超参、参数量（303,681 / 324,001）、
   `fold_seed`（20260917）和该模型的预测均值。

## 验证

| 版本 | private | Kaggle tag |
|---|---|---|
| **本仓库（最终）** | **0.08488** | `L1_R7_a035` |
| 交付 baseline | 0.10127 | ref 56451534 |

后处理层与 `submission_0.08488.csv` 逐值差 **0.00e+00**
（sha256 `29f24c08…6642`）。
