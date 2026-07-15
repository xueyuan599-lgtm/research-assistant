# Pattern: image-classification-transfer-learning

## 匹配条件

| 维度 | 值 |
|------|-----|
| problem_type | image_classification (binary / multiclass / multilabel) |
| n_samples | 1K – 1M images |
| image_size | 任何（中等分辨率为主） |
| metric | accuracy / auc / f1 / logloss |

## 推荐算法栈

### Tier 1: 预训练模型（必选）
1. **EfficientNet** (b0-b7) — 效率之王，Kaggle 最常用
2. **ConvNeXt** — 2020s CNN 新范式，超越 ViT
3. **ResNet / ResNeXt** — 经典基线
4. **ViT / Swin Transformer** — 大样本 + 高分辨率时

### Tier 2: 技巧
5. **timm** (PyTorch Image Models) — 900+ 预训练模型库，一键切换 backbones
6. **Albumentations** — 数据增强库（比 torchvision transforms 更快更丰富）
7. **MixUp / CutMix** — 正则化增强
8. **TTA (Test Time Augmentation)** — 测试时增强，5-10x 预测取平均

### Tier 3: 进阶
9. **ArcFace / AdaCos** — 度量学习 head（细粒度分类）
10. **Class-Balanced Loss / Focal Loss** — 类别不平衡
11. **Multi-Sample Dropout** — 集成 dropout 预测

## 典型 Pipeline

```
1. 加载 timm 预训练模型
2. 替换分类 head
3. 数据增强: Albumentations (RandomResizedCrop, HorizontalFlip, ColorJitter, MixUp)
4. 训练: cosine annealing + warmup + AdamW/AdamP
5. 集成: 不同 backbone 的预测加权平均
6. TTA: 5x (hflip + 多尺度)
```

## CV 策略

图像竞赛中 CV 必须分层：
- StratifiedKFold (按 target 分层)
- GroupKFold (按 patient/source 等分，防止泄漏)
- 注意 train/test 可能来自不同分布 → adversarial validation

## 已知成功案例

| 竞赛 | 最佳排名 | 核心方法 |
|------|---------|---------|
| Cassava Leaf Disease | top 1% | EfficientNet + ArcFace + MixUp |
| SIIM-ISIC Melanoma | top 3% | EfficientNet ensemble + TTA |
| Google Landmark | top 5% | ArcFace + global descriptor |

## 参考来源

- timm (rwightman) GitHub
- Albumentations 官方文档
- Kaggle 图像竞赛顶级方案合集
