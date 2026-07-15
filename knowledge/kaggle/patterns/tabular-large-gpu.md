# Pattern: tabular-large-gpu-accelerated

适用于大规模表格数据（GPU 加速）。

## 匹配条件

| 维度 | 值 |
|------|-----|
| n_samples | > 500K |
| n_features | 任何 |
| feature_types | 表格数据 |
| metric | 任何 |
| gpu_available | true (必须) |

## 推荐算法栈

### 主模型
1. **LightGBM (GPU)** — `device='gpu'`, `gpu_platform_id=0`, `gpu_device_id=0`
2. **XGBoost (GPU)** — `tree_method='gpu_hist'`, `gpu_id=0`
3. **CatBoost (GPU)** — `task_type='GPU'`
4. **cuML RandomForest** — RAPIDS GPU RandomForest, `n_estimators=1000`

### 数据处理
- **cuDF** (RAPIDS) — GPU 加速 DataFrame（pandas API 兼容）
- **dask-cudf** — 多 GPU 分布式 DataFrame

### 自动基线
- **AutoGluon** — `presets='best_quality'` (时间充裕) or `presets='medium_quality_faster_train'`
- **FLAML** — `estimator_list=['lgbm', 'xgboost', 'catboost']`

## 大规模特有问题

| 问题 | 解决方案 |
|------|---------|
| 内存不足 | LGBM + 降采样 + bagging; cuDF 分块处理 |
| 训练太慢 | GPU 加速; 先用 10% 数据跑 baseline; FLAML 快速搜索 |
| 特征太多 | 先用 LGBM feature_importance 筛到 top-500; 再上全模型 |
| CV 太慢 | 降 k-fold 到 3; GroupKFold > StratifiedKFold 速度; 时间序列不要打乱 |
| 提交太慢 | 提前准备好 test 特征，用全量数据只训一个最终模型 |

## 参考来源

- NVIDIA Kaggle Grandmasters Playbook: cuDF + cuML + GPU GBDT
- RAPIDS 官方 Kaggle 示例
