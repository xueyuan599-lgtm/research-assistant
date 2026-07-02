# Transformer — 自注意力机制

- **来源**: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention Is All You Need. *Advances in Neural Information Processing Systems*, 30, 5998–6008.
- **DOI**: 10.48550/arXiv.1706.03762
- **方法类别**: 深度学习 / 序列建模

## 数学设定

### 核心思想
Transformer 完全摒弃循环和卷积，仅依赖 **自注意力（Self-Attention）** 机制建模序列中任意两个位置的依赖关系。每个位置可以**直接**关注所有其他位置，消除了 RNN 的路径长度瓶颈。

### Scaled Dot-Product Attention
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

其中 $Q \in \mathbb{R}^{n \times d_k}$（查询）, $K \in \mathbb{R}^{m \times d_k}$（键）, $V \in \mathbb{R}^{m \times d_v}$（值）。

- $QK^\top$ 计算 pairwise 相似度，$\sqrt{d_k}$ 缩放防止 softmax 梯度消失
- $\text{softmax}$ 按行归一化为注意力权重
- 输出是值向量的加权和，权重由 query-key 相似度决定

**自注意力特例**：当 $Q=K=V$ 均为同一序列的投影时，称为 self-attention，$n=m$。

### Multi-Head Attention
将注意力拆分为 $h$ 个"头"，每个头在不同子空间学习不同的关注模式：

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W_O
$$

其中每个头：
$$
\text{head}_i = \text{Attention}(QW_Q^i, KW_K^i, VW_V^i)
$$

投影矩阵维度：
- $W_Q^i \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W_K^i \in \mathbb{R}^{d_{\text{model}} \times d_k}$
- $W_V^i \in \mathbb{R}^{d_{\text{model}} \times d_v}$, $W_O \in \mathbb{R}^{h \cdot d_v \times d_{\text{model}}}$

通常 $d_k = d_v = d_{\text{model}} / h$（默认 $d_{\text{model}} = 512$, $h = 8$, $d_k = d_v = 64$）。

多头机制使得模型可以同时关注不同位置的不同表示子空间（如语法、语义、位置等不同层面）。

### Position-wise Feed-Forward Network
每个位置独立使用相同的两层全连接网络（含 ReLU 激活）：

$$
\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
$$

或者写作：
$$
\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2
$$

其中 $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$, $W_2 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$。默认 $d_{\text{ff}} = 2048$。

FFN 等价于两个核大小为 1 的卷积，等价于为每个位置学习一个非线性变换。$d_{\text{ff}}$ 远大于 $d_{\text{model}}$，先升维再降维。

### Positional Encoding
由于自注意力是**排列等变**（permutation equivariant）的，必须注入位置信息：

$$
\begin{aligned}
\text{PE}_{(pos, 2i)} &= \sin\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right) \\
\text{PE}_{(pos, 2i+1)} &= \cos\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)
\end{aligned}
$$

其中 $pos$ 是位置索引，$i$ 是维度索引。

**设计原理**：
- 每个维度对应一个正弦/余弦对，频率从 $2\pi$ 到 $10000 \cdot 2\pi$ 递减
- 对任意固定偏移 $k$，$\text{PE}_{pos+k}$ 可表示为 $\text{PE}_{pos}$ 的线性函数（利用三角恒等式），使模型能学习相对位置
- 无需学习，可外推到任意长度

### Layer Normalization
$$
\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sigma} + \beta
$$

其中 $\mu = \frac{1}{d}\sum_{j=1}^{d} x_j$, $\sigma = \sqrt{\frac{1}{d}\sum_{j=1}^{d} (x_j - \mu)^2 + \epsilon}$。

与 BatchNorm 不同，LayerNorm 在**特征维度**归一化（同一样本跨特征），适合序列变长场景。

### Residual Connection
每个子层后接残差连接 + 层归一化。

原始（Post-LN，Vaswani 原文）：
$$
\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))
$$

改进（Pre-LN，更稳定）：
$$
\text{output} = x + \text{Sublayer}(\text{LayerNorm}(x))
$$

Pre-LN 允许更深的堆叠，梯度更稳定，是目前的主流实践。

### Encoder 架构
Encoder 由 $N$ 个相同层堆叠（原文 $N=6$），每层包含：

1. **Multi-Head Self-Attention**（残差 + LayerNorm）
2. **Position-wise FFN**（残差 + LayerNorm）

所有子层输出维度均为 $d_{\text{model}}$。

$$
\text{Enc}_i(x) = \text{FFN}(\text{LN}(x' + \text{Attn}(\text{LN}(x'), \text{LN}(x'), \text{LN}(x'))))
$$

（Pre-LN 版本的精确顺序）

### Decoder 架构
Decoder 由 $N$ 个相同层堆叠（原文 $N=6$），每层包含：

1. **Masked Multi-Head Self-Attention**: 使用 look-ahead mask 防止当前位置看到未来信息（自回归生成）
2. **Cross-Attention**: 以 Decoder 状态为 $Q$，Encoder 输出为 $K, V$，建立源-目标对齐
3. **Position-wise FFN**

每层均含残差连接和 LayerNorm。

**关键设计**：Decoder 的 cross-attention 是 Transformer 的核心创新，它使解码时每个位置都能灵活地关注编码器的任意位置（而非像传统 seq2seq 那样只关注最后一个隐状态）。

### 架构图示

```
Encoder (×N)                    Decoder (×N)
┌─────────────────┐            ┌──────────────────────┐
│  Output          │            │  Output              │
│      ↑           │            │      ↑               │
│  [LayerNorm]     │            │  [LayerNorm]         │
│      ↑           │            │      ↑               │
│  FFN             │            │  FFN                 │
│      ↑           │            │      ↑               │
│  [LayerNorm]     │            │  [LayerNorm]         │
│      ↑           │            │      ↑               │
│  MHA Self-Attn   │            │  Cross-Attention     │
│      ↑           │            │  (Q=dec, K=enc, V=enc)│
│  [LayerNorm]      │            │      ↑               │
│      ↑           │            │  [LayerNorm]         │
│  + (residual)    │            │      ↑               │
│      ↑           │            │  Masked Self-Attn    │
│  Input Embedding │            │      ↑               │
│      ↑           │            │  [LayerNorm]         │
│  Positional Enc  │            │      ↑               │
└─────────────────┘            │  + (residual)        │
                                │      ↑               │
                                │  Output Embedding    │
                                │      ↑               │
                                │  Positional Enc      │
                                └──────────────────────┘
```

### 复杂度分析

| 模块 | 计算复杂度 | 参数量 | 说明 |
|------|-----------|--------|------|
| Self-Attention | $\mathcal{O}(n^2 \cdot d_k)$ | $\mathcal{O}(d_{\text{model}}^2)$ | $n$ 为序列长度，$d_k = d_{\text{model}}/h$ |
| FFN | $\mathcal{O}(n \cdot d_{\text{model}} \cdot d_{\text{ff}})$ | $\mathcal{O}(d_{\text{model}} \cdot d_{\text{ff}})$ | 逐位置计算，可高度并行 |
| 总复杂度（单层） | $\mathcal{O}(n^2 \cdot d + n \cdot d^2)$ | $\mathcal{O}(d^2)$ | 相比 RNN $\mathcal{O}(n \cdot d^2)$ 的平方长度依赖 |

**关键对比**：
- RNN: $\mathcal{O}(n \cdot d^2)$ — 线性步数，但无法并行
- CNN: $\mathcal{O}(k \cdot n \cdot d^2)$ — 线性步数，但感受野受限
- Transformer: $\mathcal{O}(n^2 \cdot d)$ — 平方长度，但完全并行，且最大路径长度 $\mathcal{O}(1)$（任意位置一步可达）

## 关键假设

- **全局依赖假设**：序列中任意两个位置之间的依赖都需要被直接建模，而非通过隐状态逐步传递
- **排列不变性需要破除**：自注意力本身是排列等变的，必须通过位置编码显式注入顺序信息
- **二次复杂度可接受**：假设序列长度 $n$ 满足 $n^2 \cdot d < n \cdot d^2$，即 $n < d$ 时 Transformer 效率不低于 RNN（实际中通常成立）
- **大规模数据驱动**：Transformer 缺乏 RNN/CNN 的内置归纳偏置（如局部性、平移不变性），需要更多数据或大规模预训练来习得这些模式
- **多头互补**：不同注意力头学习不同的关注模式，且 $h$ 个头的总计算量与单头（$d_k = d_{\text{model}}$）近似相同
- **残差+归一化稳定训练**：深层堆叠依赖残差连接和 LayerNorm 的组合来维持梯度流动

## 适用场景

- **自然语言处理**：机器翻译（原文任务）、文本生成（GPT 系列）、文本理解（BERT）、命名实体识别、情感分析等
- **多模态学习**：视觉-语言模型（CLIP, BLIP）、图文生成、视频理解
- **计算机视觉**：Vision Transformer (ViT) — 将图像分 patch 作为 token 序列
- **语音处理**：语音识别（Speech-Transformer）、语音合成（Tacotron2 + Transformer）
- **代码生成**：CodeBERT, CodeGPT, GitHub Copilot
- **时间序列预测**：Informer, Autoformer, PatchTST — 修改注意力机制适应时序
- **科学数据**：AlphaFold2 (蛋白质结构预测)、分子生成、气象预报

### 不适用

- **极长序列**（$n > 10^4$）：原始 $\mathcal{O}(n^2)$ 注意力不可行
  - 替代方案：Longformer（滑动窗 + 全局注意力）、BigBird（稀疏注意力）、Performer（核近似线性注意力）或 Mamba（状态空间模型）
- **小数据集**（无预训练）：Transformer 缺少归纳偏置，容易过拟合
  - 替代方案：LSTM/GRU + 正则化，或使用预训练模型做 fine-tune
- **实时边缘端部署**：Transformer 参数量大、计算密集
  - 替代方案：蒸馏版（DistilBERT, TinyBERT）或 MobileNet-style 轻量网络
- **简单任务**：当 RNN/CNN 已能很好解决时，Transformer 是过度工程
- **需要严格自回归约束的低延迟场景**：transformer 逐 token 生成的开销高于 LSTM

## 实现要点

### 关键超参数

| 参数 | 典型值 | 说明 |
|------|--------|------|
| $d_{\text{model}}$ | 512 (base), 1024 (large) | 所有子层的输入/输出维度 |
| $n_{\text{heads}}$ | 8 (base), 16 (large) | 注意力头数，$d_k = d_{\text{model}} / n_{\text{heads}}$ |
| $n_{\text{layers}}$ | 6 (base), 12 (large) | Encoder/Decoder 堆叠层数 |
| $d_{\text{ff}}$ | 2048 (base), 4096 (large) | FFN 中间层维度 |
| dropout | 0.1 | 各子层输出和注意力权重上的 dropout |
| warmup\_steps | 4000 | 学习率预热的步数 |
| label\_smoothing | 0.1 | 目标分布的 label smoothing |

### 设计决策

1. **$d_k = d_v = d_{\text{model}} / n_{\text{heads}}$**：头维度通常固定为 64，增大 $d_{\text{model}}$ 时应同等增加 $n_{\text{heads}}$
2. **学习率调度**：
   $$
   lr = d_{\text{model}}^{-0.5} \cdot \min(step^{-0.5}, \; step \cdot warmup\_steps^{-1.5})
   $$
   先线性预热，再按步数平方根衰减
3. **Label Smoothing**: 将 one-hot 目标分布替换为 $y'_{c} = (1-\epsilon)\mathbb{I}[c=y] + \epsilon/C$，$\epsilon=0.1$，防止过拟合
4. **Masking**:
   - **Padding mask**: 将填充位置（pad token）的注意力权重置为 $-\infty$，防止模型关注无效位置
   - **Look-ahead mask**: Decoder 自注意力中，位置 $i$ 只能看到位置 $\leq i$ 的信息（下三角矩阵）
5. **Pre-LN vs Post-LN**:
   - Post-LN（原文）：LayerNorm 在残差之后，梯度容易爆炸，需要 careful 的 warmup
   - Pre-LN（当前主流）：LayerNorm 在每个子层之前，训练更稳定，warmup 需求小，适合更深模型
6. **Flash Attention**: 通过 tiling 和 recomputation 将 $\mathcal{O}(n^2)$ 显存优化为 $\mathcal{O}(n)$，训练速度提升 2-4 倍。推荐使用 PyTorch 2.0+ 原生 `F.scaled_dot_product_attention`（自动采用 Flash Attention）

### 代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


# ==============================================================================
# 1. Scaled Dot-Product Attention
# ==============================================================================
class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    """

    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (batch, n_heads, seq_len_q, d_k)
            key:   (batch, n_heads, seq_len_k, d_k)
            value: (batch, n_heads, seq_len_k, d_v)
            mask:  (batch, 1, seq_len_q, seq_len_k) or broadcastable
        Returns:
            output: (batch, n_heads, seq_len_q, d_v)
            attn:   (batch, n_heads, seq_len_q, seq_len_k)
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, value)
        return output, attn


# ==============================================================================
# 2. Multi-Head Attention
# ==============================================================================
class MultiHeadAttention(nn.Module):
    """Multi-Head Attention

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W_O
    head_i = Attention(Q W_Q^i, K W_K^i, V W_V^i)
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # d_k = d_v = d_model / n_heads

        # 投影矩阵（将三个投影合并为一个 linear 提高效率）
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        self.attention = ScaledDotProductAttention(dropout)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (batch, seq_len_q, d_model)
            key:   (batch, seq_len_k, d_model)
            value: (batch, seq_len_k, d_model)
            mask:  (batch, seq_len_q, seq_len_k) or broadcastable
        Returns:
            output: (batch, seq_len_q, d_model)
            attn:   (batch, n_heads, seq_len_q, seq_len_k)
        """
        batch_size = query.size(0)

        # 1. 线性投影 + 切分为多头
        # (batch, seq_len, d_model) -> (batch, seq_len, n_heads, d_k)
        Q = self.W_Q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # mask 扩展为多头形式
        if mask is not None:
            # (batch, seq_len_q, seq_len_k) -> (batch, 1, seq_len_q, seq_len_k)
            mask = mask.unsqueeze(1)

        # 2. 计算注意力
        attn_output, attn_weights = self.attention(Q, K, V, mask)

        # 3. 拼接多头
        # (batch, n_heads, seq_len, d_k) -> (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous() \
            .view(batch_size, -1, self.d_model)

        # 4. 输出投影
        output = self.W_O(attn_output)
        return output, attn_weights


# ==============================================================================
# 3. Position-wise Feed-Forward Network
# ==============================================================================
class PositionwiseFFN(nn.Module):
    """Position-wise Feed-Forward Network

    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.W_1 = nn.Linear(d_model, d_ff)
        self.W_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        # 使用 xavier 初始化（Vaswani 原文设置）
        nn.init.xavier_uniform_(self.W_1.weight)
        nn.init.xavier_uniform_(self.W_2.weight)
        nn.init.constant_(self.W_1.bias, 0.0)
        nn.init.constant_(self.W_2.bias, 0.0)

    def forward(self, x):
        return self.W_2(self.dropout(F.relu(self.W_1(x))))


# ==============================================================================
# 4. Positional Encoding
# ==============================================================================
class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding

    PE(pos, 2i)   = sin(pos / 10000^{2i / d_model})
    PE(pos, 2i+1) = cos(pos / 10000^{2i / d_model})
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # 预计算 PE 矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x + positional encoding
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ==============================================================================
# 5. Encoder Layer & Encoder
# ==============================================================================
class TransformerEncoderLayer(nn.Module):
    """单层 Encoder（Pre-LN 结构，训练更稳定）"""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-LN: LayerNorm before each sublayer
        # Self-Attention with residual
        attn_out, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        x = x + self.dropout1(attn_out)

        # FFN with residual
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout2(ffn_out)
        return x


class TransformerEncoder(nn.Module):
    """Transformer Encoder — N 层堆叠"""

    def __init__(self, d_model, n_heads, d_ff, n_layers=6, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# ==============================================================================
# 6. Decoder Layer & Decoder
# ==============================================================================
class TransformerDecoderLayer(nn.Module):
    """单层 Decoder（Pre-LN 结构）"""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Masked Self-Attention (防止看到未来 token)
        attn_out, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), tgt_mask)
        x = x + self.dropout1(attn_out)

        # Cross-Attention: Q=decoder, K=V=encoder
        cross_out, _ = self.cross_attn(self.norm2(x), enc_output, enc_output, src_mask)
        x = x + self.dropout2(cross_out)

        # FFN
        ffn_out = self.ffn(self.norm3(x))
        x = x + self.dropout3(ffn_out)
        return x


class TransformerDecoder(nn.Module):
    """Transformer Decoder — N 层堆叠"""

    def __init__(self, d_model, n_heads, d_ff, n_layers=6, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return self.norm(x)


# ==============================================================================
# 7. 完整 Transformer（Encoder-Decoder）
# ==============================================================================
class Transformer(nn.Module):
    """完整 Transformer 模型

    含 embedding、positional encoding、encoder、decoder、输出投影。
    """

    def __init__(self,
                 src_vocab_size,
                 tgt_vocab_size,
                 d_model=512,
                 n_heads=8,
                 d_ff=2048,
                 n_layers=6,
                 dropout=0.1,
                 max_len=5000):
        super().__init__()

        self.d_model = d_model

        # Embedding（共享权重）
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)

        # Positional Encoding
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)

        # Encoder & Decoder
        self.encoder = TransformerEncoder(d_model, n_heads, d_ff, n_layers, dropout)
        self.decoder = TransformerDecoder(d_model, n_heads, d_ff, n_layers, dropout)

        # 输出投影（词汇表大小）
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

        # 参数初始化
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Args:
            src: (batch, src_seq_len) — 源语言 token IDs
            tgt: (batch, tgt_seq_len) — 目标语言 token IDs（含 <sos>）
            src_mask: (batch, 1, 1, src_seq_len) padding mask
            tgt_mask: (batch, 1, tgt_seq_len, tgt_seq_len) look-ahead + padding mask
        Returns:
            logits: (batch, tgt_seq_len, tgt_vocab_size)
        """
        # Embedding + 缩放（缩小 embedding 方差）
        src = self.src_embed(src) * math.sqrt(self.d_model)
        tgt = self.tgt_embed(tgt) * math.sqrt(self.d_model)

        # Positional Encoding
        src = self.pos_enc(src)
        tgt = self.pos_enc(tgt)

        # Encoder-Decoder
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)

        # 输出投影
        logits = self.output_proj(dec_output)
        return logits

    def encode(self, src, src_mask=None):
        """编码器单独使用（可用于提取表示）"""
        src = self.src_embed(src) * math.sqrt(self.d_model)
        src = self.pos_enc(src)
        return self.encoder(src, src_mask)

    def decode(self, tgt, enc_output, src_mask=None, tgt_mask=None):
        """解码器单独使用"""
        tgt = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_enc(tgt)
        return self.decoder(tgt, enc_output, src_mask, tgt_mask)


# ==============================================================================
# 辅助函数：生成 attention mask
# ==============================================================================
def generate_padding_mask(seq, pad_idx=0):
    """生成 padding mask（True=有效位置）"""
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)


def generate_lookahead_mask(size):
    """生成 look-ahead mask（下三角矩阵，防止看到未来）"""
    mask = torch.triu(torch.ones(size, size), diagonal=1) == 0
    return mask.to(dtype=torch.bool)  # (seq_len, seq_len)


def generate_tgt_mask(tgt, pad_idx=0):
    """合并 padding mask + look-ahead mask"""
    tgt_pad_mask = generate_padding_mask(tgt, pad_idx)  # (batch, 1, 1, tgt_len)
    seq_len = tgt.size(1)
    look_ahead = generate_lookahead_mask(seq_len).to(tgt.device)  # (tgt_len, tgt_len)
    # look_ahead 广播到 batch 维度
    return tgt_pad_mask & look_ahead  # (batch, 1, tgt_len, tgt_len)


# ==============================================================================
# 使用示例 1: 从零实现的 Transformer — 简单 Copy Task
# ==============================================================================
def run_copy_task_example():
    """Copy Task: 模型学习将输入序列复制到输出

    目标: <sos> x1 x2 ... xn <eos>
    输入: x1 x2 ... xn
    这是验证 seq2seq 模型能否学习基本对齐的经典测试。
    """
    print("=" * 60)
    print("示例 1: 从零实现 Transformer — Copy Task")
    print("=" * 60)

    # 超参数（小模型，便于快速运行）
    vocab_size = 20       # 词汇表大小（含 <pad>, <sos>, <eos>）
    d_model = 64
    n_heads = 4
    d_ff = 256
    n_layers = 3
    dropout = 0.1
    max_len = 30
    batch_size = 32
    seq_len = 8           # 序列长度
    n_epochs = 50
    pad_idx = 0
    sos_idx = 1
    eos_idx = 2

    # 数据生成器
    def generate_copy_batch(batch_size, seq_len, vocab_size):
        """生成 copy task 数据"""
        # 随机序列（不含特殊 token）
        data = torch.randint(3, vocab_size, (batch_size, seq_len))
        # Encoder 输入: 原始序列
        src = data.clone()
        # Decoder 输入: <sos> + 原始序列
        tgt = torch.cat([torch.full((batch_size, 1), sos_idx, dtype=torch.long), data], dim=1)
        # Decoder 目标: 原始序列 + <eos>
        tgt_y = torch.cat([data, torch.full((batch_size, 1), eos_idx, dtype=torch.long)], dim=1)
        return src, tgt, tgt_y

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers,
        dropout=dropout,
        max_len=max_len
    ).to(device)

    # 学习率调度（Vaswani 原文调度器）
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
    total_steps = n_epochs * 100  # 假设每 epoch 100 步
    warmup_steps = 400

    def lr_lambda(step):
        step = max(step, 1)
        return (d_model ** (-0.5)) * min(step ** (-0.5), step * (warmup_steps ** (-1.5)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    model.train()
    for epoch in range(n_epochs):
        total_loss = 0.0
        n_batches = 100
        for _ in range(n_batches):
            src, tgt, tgt_y = generate_copy_batch(batch_size, seq_len, vocab_size)
            src, tgt, tgt_y = src.to(device), tgt.to(device), tgt_y.to(device)

            src_mask = generate_padding_mask(src, pad_idx)
            tgt_mask = generate_tgt_mask(tgt, pad_idx)

            logits = model(src, tgt, src_mask, tgt_mask)

            # 忽略 <sos> 位置的损失
            loss = criterion(logits[:, 1:].reshape(-1, vocab_size),
                             tgt_y[:, 1:].reshape(-1))
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪（防止爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_loss = total_loss / n_batches
            print(f"  Epoch {epoch + 1:2d}/{n_epochs}, Loss: {avg_loss:.4f}")

    # 推理测试
    model.eval()
    with torch.no_grad():
        src, tgt, tgt_y = generate_copy_batch(4, seq_len, vocab_size)
        src, tgt = src.to(device), tgt.to(device)
        src_mask = generate_padding_mask(src, pad_idx)
        tgt_mask = generate_tgt_mask(tgt, pad_idx)

        logits = model(src, tgt, src_mask, tgt_mask)
        preds = logits.argmax(dim=-1)

        print("\n  推理示例:")
        for i in range(min(2, src.size(0))):
            src_tokens = src[i].cpu().tolist()
            tgt_tokens = tgt_y[i].cpu().tolist()
            pred_tokens = preds[i].cpu().tolist()
            print(f"  Input:    {src_tokens}")
            print(f"  Target:   {tgt_tokens}")
            print(f"  Predicted:{pred_tokens}")
            print()


# ==============================================================================
# 使用示例 2: 基于 nn.Transformer 的标准用法
# ==============================================================================
def run_pytorch_transformer_example():
    """演示 PyTorch 内置 nn.Transformer 的标准使用方式"""
    print("=" * 60)
    print("示例 2: PyTorch nn.Transformer 标准用法")
    print("=" * 60)

    vocab_size = 30
    d_model = 128
    nhead = 4
    num_encoder_layers = 3
    num_decoder_layers = 3
    dim_feedforward = 512
    batch_size = 16
    seq_len = 10

    # 内置 Transformer
    transformer = nn.Transformer(
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=0.1,
        activation='relu',
        batch_first=True  # (batch, seq, d_model) 格式
    )

    # Embedding
    src_embed = nn.Embedding(vocab_size, d_model)
    tgt_embed = nn.Embedding(vocab_size, d_model)
    pos_enc = PositionalEncoding(d_model, max_len=50, dropout=0.1)
    output_proj = nn.Linear(d_model, vocab_size)

    # 假数据
    src = torch.randint(3, vocab_size, (batch_size, seq_len))
    tgt = torch.randint(3, vocab_size, (batch_size, seq_len))

    # Embedding + Positional Encoding
    src_emb = pos_enc(src_embed(src) * math.sqrt(d_model))
    tgt_emb = pos_enc(tgt_embed(tgt) * math.sqrt(d_model))

    # nn.Transformer 内置 mask 生成
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len)
    src_key_padding_mask = (src == 0)
    tgt_key_padding_mask = (tgt == 0)

    # 前向
    output = transformer(
        src=src_emb,
        tgt=tgt_emb,
        tgt_mask=tgt_mask,
        src_key_padding_mask=src_key_padding_mask,
        tgt_key_padding_mask=tgt_key_padding_mask,
        memory_key_padding_mask=src_key_padding_mask
    )

    logits = output_proj(output)
    print(f"  输入形状: src={src.shape}, tgt={tgt.shape}")
    print(f"  输出形状: {logits.shape}  (batch, seq_len, vocab_size)")
    print(f"  参数分解:")
    print(f"    d_model={d_model}, nhead={nhead}, n_layers=enc={num_encoder_layers}+dec={num_decoder_layers}")
    print(f"    总参数量: {sum(p.numel() for p in transformer.parameters()):,}")
    print()


# ==============================================================================
# 使用示例 3: Encoder-only (BERT-style) 序列分类
# ==============================================================================
class BertStyleClassifier(nn.Module):
    """Encoder-only Transformer 用于序列分类"""

    def __init__(self, vocab_size, d_model=128, n_heads=4, d_ff=512,
                 n_layers=4, n_classes=2, dropout=0.1, max_len=5000):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        self.encoder = TransformerEncoder(d_model, n_heads, d_ff, n_layers, dropout)
        self.pooler = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh()
        )
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x, mask=None):
        # Embedding
        x = self.embed(x) * math.sqrt(self.embed.embedding_dim)
        x = self.pos_enc(x)

        # Encoding
        x = self.encoder(x, mask)

        # 取 [CLS] token (位置 0) 做分类
        cls_token = x[:, 0, :]
        pooled = self.pooler(cls_token)
        logits = self.classifier(pooled)
        return logits


def run_classification_example():
    """Encoder-only Transformer 序列分类演示"""
    print("=" * 60)
    print("示例 3: Encoder-only (BERT-style) 序列分类")
    print("=" * 60)

    vocab_size = 50
    d_model = 64
    n_heads = 4
    d_ff = 256
    n_layers = 3
    n_classes = 2
    batch_size = 16
    seq_len = 8
    n_epochs = 30

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BertStyleClassifier(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers,
        n_classes=n_classes,
        dropout=0.1,
        max_len=50
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # 合成数据：奇数 token 计数 -> class 1
    model.train()
    for epoch in range(n_epochs):
        src = torch.randint(3, vocab_size, (batch_size, seq_len))
        # 标签: 序列中奇数 token 数量为偶数为 0，奇数为 1
        labels = (src % 2).sum(dim=1) % 2

        src, labels = src.to(device), labels.to(device)
        mask = generate_padding_mask(src, pad_idx=0)

        logits = model(src, mask)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            preds = logits.argmax(dim=-1)
            acc = (preds == labels).float().mean()
            print(f"  Epoch {epoch + 1:2d}/{n_epochs}, Loss: {loss.item():.4f}, Acc: {acc.item():.3f}")

    print()


# ==============================================================================
# 主程序
# ==============================================================================
if __name__ == "__main__":
    print("PyTorch version:", torch.__version__)
    print()

    run_copy_task_example()
    run_pytorch_transformer_example()
    run_classification_example()

    print("所有示例运行完成。")
```

## 参考文献

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention Is All You Need. *Advances in Neural Information Processing Systems*, 30, 5998–6008.

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL-HLT*, 4171–4186.

Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., ... & Amodei, D. (2020). Language Models are Few-Shot Learners. *Advances in Neural Information Processing Systems*, 33, 1877–1901.

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR*.

Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., ... & Lample, G. (2023). LLaMA: Open and Efficient Foundation Language Models. *arXiv preprint arXiv:2302.13971*.
