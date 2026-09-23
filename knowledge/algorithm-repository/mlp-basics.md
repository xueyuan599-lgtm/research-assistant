---
title: MLP — 多层感知机 / 全连接神经网络基础
type:
  - deep-learning
domain:
  - generic-ml
---
# MLP — 多层感知机 / 全连接神经网络基础

- **来源**: 周志华.《机器学习》（西瓜书）第 5 章神经网络. 清华大学出版社.
- **方法类别**: 深度学习

## 数学设定

### 网络结构与前向传播

MLP 是**全连接前馈网络**：输入层 → 若干隐藏层 → 输出层，层间全连接、层内无连接。

设输入 $x \in \mathbb{R}^d$，第 $\ell$ 隐藏层神经元数为 $d_\ell$，权重矩阵 $W^{(\ell)} \in \mathbb{R}^{d_\ell \times d_{\ell-1}}$，偏置 $b^{(\ell)}$，激活函数 $\sigma$。前向传播：

$$
z^{(\ell)} = W^{(\ell)} a^{(\ell-1)} + b^{(\ell)}, \qquad a^{(\ell)} = \sigma(z^{(\ell)})
$$

输出层对回归为线性（$a^{(L)} = z^{(L)}$），对二分类为 sigmoid，对多分类为 softmax。

### 激活函数

| 激活 | 公式 $\sigma(z)$ | 特性 | 常用场景 |
|------|-----------------|------|---------|
| **ReLU** | $\max(0, z)$ | 计算快、缓解梯度消失、稀疏激活 | 隐藏层首选 |
| Sigmoid | $1/(1+e^{-z})$ | 输出 $(0,1)$，易梯度消失 | 二分类输出层 |
| tanh | $\frac{e^z-e^{-z}}{e^z+e^{-z}}$ | 输出 $(-1,1)$，零中心 | 隐藏层备选 |
| softmax | $\frac{e^{z_k}}{\sum_j e^{z_j}}$ | 归一化概率 | 多分类输出层 |

现代实践：隐藏层几乎用 ReLU（或其变体 Leaky-ReLU），饱和型 sigmoid/tanh 因梯度消失仅用于输出层。

### 损失函数

- 回归（MSE）：$L = \frac{1}{n}\sum_i \|y_i - \hat{y}_i\|^2$
- 二分类（交叉熵）：$L = -\frac{1}{n}\sum_i [y_i \log p_i + (1-y_i)\log(1-p_i)]$
- 多分类：$L = -\frac{1}{n}\sum_i \sum_k y_{ik} \log p_{ik}$

### 反向传播（Backpropagation）

训练用**梯度下降**：$\theta \leftarrow \theta - \eta \nabla_\theta L$。梯度由**反向传播**高效计算——先从输出层按链式法则回传误差 $\delta^{(L)} = \nabla_{a^{(L)}} L \odot \sigma'(z^{(L)})$，再逐层回传：

$$
\delta^{(\ell)} = (W^{(\ell+1)\top} \delta^{(\ell+1)}) \odot \sigma'(z^{(\ell)})
$$

权重梯度 $\nabla_{W^{(\ell)}} L = \delta^{(\ell)} a^{(\ell-1)\top}$。批量梯度下降用全部样本，随机梯度下降（SGD）用单样本/小批量（mini-batch），小批量在效率与稳定性间折中。

### 正则化与防过拟合

| 方法 | 机制 | 实现 |
|------|------|------|
| **Dropout** | 训练时随机丢弃神经元（概率 $p$），测试时全保留并缩放 | `torch.nn.Dropout(p)` |
| **早停 (Early Stopping)** | 验证集指标不再提升即停，防过拟合关键 | callback / patience |
| **批归一化 (BatchNorm)** | 每批归一化激活，加速收敛、适当正则 | `torch.nn.BatchNorm1d` |
| **权重衰减 (L2)** | 对权重加 $\lambda\|W\|^2$ 惩罚 | optim 的 weight_decay |
| **L1/L2 正则** | 同传统机器学习 | — |
| 数据增强/更多数据 | 增大有效样本 | 合成/扰动 |

### 超参数网格搜索

- 隐藏层结构：`[(64,), (128,64), (256,128,64)]`
- 激活：ReLU / tanh；优化器：Adam（学习率 $\eta \in [1e-4, 1e-2]$）
- Dropout：$p \in [0,0.5]$；批大小：`[16,32,64]`；epochs：由早停决定
- 用 `GridSearchCV`/`Optuna` 在小网格上搜索，配 5 折 CV + 早停

## 关键假设

- **函数光滑可微**：损失与激活需可微以使反向传播成立
- **特征尺度相似**：数值特征需标准化/归一化（树模型不需要，DNN 强烈需要）
- **样本量充足**：容量大、参数多，小样本易过拟合，需强正则
- **目标相对平滑**：见 `tabular-vs-deep.md`，DNN 对不平滑/阶梯目标不适配

## 适用场景

- **何时值得用 NN（vs 树模型）**：
  - 数据量大（≥ 数十万，尤其上百万）且特征同质稠密
  - 时序数据（配合 LSTM/Transformer）
  - 高维非线性、可端到端学特征表示（图像/文本/语音）
  - 需要可微架构做后续扩展（多任务、强化学习策略头、输入梯度解释）
- **作树模型的对照基线**：国赛全模型对比中，MLP 是不可或缺的"深度基线"
- **小数据防过拟合策略**：深层 -> 加深正则（Dropout/早停/BatchNorm）+ 增大数据 + 保守容量

### 不适用 / 注意

- **小样本（< 1000）**：容量大易过拟合，树/线性模型更稳
- **异质轴对齐表格 + 中样本**：见 `tabular-vs-deep.md`，GBDT 占优
- **黑盒风险**：缺乏内置可解释性，需配 SHAP（见 `shap.md`）补解释
- **长训练时间**：调参与迭代开销大，国赛时间预算下慎做主模型

## 实现要点

### sklearn（快速基线）

```python
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

model = make_pipeline(
    StandardScaler(),  # DNN 必须标准化
    MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        alpha=1e-3,                 # L2 权重衰减
        max_iter=500,
        early_stopping=True,        # 内置早停
        learning_rate_init=1e-3,
        random_state=42,
    )
)
model.fit(X_train, y_train)
```

### PyTorch 模板（torch 模板）

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=(128, 64), out_dim=1, p=0.3):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(p)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))   # 回归线性输出
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

# 训练主干（回归 MSE）
model = MLP(in_dim=X.shape[1])
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.MSELoss()
for ep in range(200):
    model.train()
    opt.zero_grad()
    loss = loss_fn(model(torch.tensor(Xs, dtype=torch.float32)),
                   torch.tensor(y, dtype=torch.float32))
    loss.backward()
    opt.step()
    # 每 ep 用验证集早停（略）
```

要点：
- **标准化先行**：DNN 对尺度敏感，`StandardScaler` 是硬前提。
- **早停 + dropout + weight_decay**：小数据防过拟合三件套，缺一不可。
- **固定 seed**：确保可复现，国赛纪律要求。

## 国赛应用

### "何时用 NN、怎么和树对比、怎么避免黑盒扣分"写作模板

> 考虑到本问样本量为 $n$（若 < 10 万，倾向树）且特征含类别/数值异质（见 `tabular-vs-deep.md`），我们将 NN 定位为**对照基线**而非主模型。MLP 在相同 5 折 CV 下 CV-RMSE = 0.xx，略逊于 LightGBM 的 0.xx（Δ = 0.0x），佐证表格数据下树模型的优势；在数据量更大/特征更同质的场景 3 中，MLP 提升至与树模型相当（CV-AUC 0.xx vs 0.xx）。

消除黑盒顾虑：对 MLP 与 GBDT 统一施加 SHAP 归因，两模型对关键影响因子的方向与排序一致，说明结论不依赖特定黑盒架构。网络结构、激活、正则均列于附录与超参表，seed 固定，可完全复现。

要点：
1. **NN 参与对比但不喧宾夺主**：跑它证明"树模型胜出是实证结论"，同时回应"为何不用更复杂模型"。
2. **统一 SHAP 解释**：给黑盒模型配归因，避免"黑盒不可解释"扣分。
3. **完整报告超参与 seed**：深度学习可复现性必须交代清楚，这是评委核验点。

## 参考文献

- 周志华. 《机器学习》[M]. 清华大学出版社, 2016.（第 5 章神经网络、第 6 章支持向量机）
