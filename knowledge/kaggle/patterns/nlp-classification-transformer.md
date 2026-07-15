# Pattern: nlp-classification-transformer

## 匹配条件

| 维度 | 值 |
|------|-----|
| problem_type | text_classification / ner / qa / similarity |
| n_samples | 1K – 1M texts |
| text_length | 短文本 (<512 tokens) / 长文本 (>512 tokens) |
| language | 英语/多语言 |
| metric | accuracy / auc / f1 / logloss |

## 推荐算法栈

### Tier 1: 预训练 Transformer（必选）
1. **DeBERTa-v3** — SOTA 文本理解，Kaggle NLP 竞赛首选
2. **RoBERTa** — 经典强大
3. **ModernBERT** — 2024 新基线，2x 速度 + 更长上下文 (8K)
4. **DeBERTa-v3-large** — 大样本时

### Tier 2: 效率优化
5. **sentence-transformers** — 句子级 embedding（语义相似度任务）
6. **LoRA / QLoRA** — 参数高效微调（大模型时）
7. **ONNX / TensorRT** — 推理加速

### Tier 3: 传统基线
8. **TF-IDF + Logistic Regression** — 必做基线（有时很能打）
9. **TF-IDF + LightGBM** — 同上
10. **fastText** — 速度快、baseline 强

## 典型 Pipeline

```
1. Tokenizer: AutoTokenizer (HuggingFace)
2. 模型: DeBERTa-v3-base (AutoModelForSequenceClassification)
3. 训练: AdamW + linear warmup + cosine decay
4. 集成: 不同预训练模型的预测平均
5. TTA: 文本增强（回译/NLPAug）+ 预测平均
```

## 评估与推理加速

| 场景 | 方法 |
|------|------|
| 推理速度瓶颈 | ONNX Runtime, TensorRT, quantization |
| 长文本 (>512 tokens) | Longformer, BigBird, 滑窗 pooling |
| 多语言 | XLM-RoBERTa, mDeBERTa |
| 小样本 | 数据增强 (EDA/back-translation), few-shot prompting |

## 已知成功案例

| 竞赛 | 最佳排名 | 核心方法 |
|------|---------|---------|
| Jigsaw Toxic Comment | top 1% | RoBERTa ensemble |
| NBME Clinical NLP | top 3% | DeBERTa-v3 + extra features |
| CommonLit Readability | top 5% | DeBERTa + RoBERTa ensemble |
| USPPPM Patent | top 10% | TF-IDF + LightGBM (模型简单反而赢) |

## 参考来源

- HuggingFace Transformers 官方文档
- DeBERTa-v3 论文 (He et al., 2023)
- ModernBERT (Answer.ai, 2024)
