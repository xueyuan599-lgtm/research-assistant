# LSTM/GRU — 长短期记忆网络 / 门控循环单元

- **来源**: Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780.
- **DOI**: 10.1162/neco.1997.9.8.1735
- **方法类别**: 深度学习 / 序列建模
- **相关文献**: Cho, K. et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. *EMNLP*.

## 数学设定

### Vanilla RNN 与梯度问题

标准循环神经网络（RNN）在时间步 $t$ 的隐状态更新为：

$$
h_t = \tanh(W_h h_{t-1} + W_x x_t + b)
$$

其中 $x_t \in \mathbb{R}^{d}$ 为输入，$h_t \in \mathbb{R}^{H}$ 为隐状态。在反向传播时，梯度通过时间反向传播（BPTT）涉及 Jacobian 的连乘：

$$
\frac{\partial L}{\partial h_1} = \frac{\partial L}{\partial h_T} \prod_{t=2}^{T} \frac{\partial h_t}{\partial h_{t-1}} = \frac{\partial L}{\partial h_T} \prod_{t=2}^{T} W_h^\top \cdot \text{diag}(\tanh'(h_{t-1}))
$$

由于 $\tanh' \in (0, 1]$，当 $W_h$ 的特征值 < 1 时梯度趋零（**梯度消失**），> 1 时梯度爆炸（**梯度爆炸**）。LSTM 通过门控机制和细胞状态直通路径解决了这一问题。

### LSTM 单元

LSTM 引入**细胞状态** $C_t$ 作为信息高速公路，通过三个门控单元控制信息流。

**遗忘门**（Forget Gate）：决定丢弃多少旧细胞状态

$$
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
$$

**输入门**（Input Gate）：决定写入多少新信息

$$
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
$$

**候选细胞状态**（Candidate Cell State）：

$$
\tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)
$$

**细胞状态更新**（核心创新）：遗忘旧信息 + 写入新信息

$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
$$

其中 $\odot$ 为逐元素乘法。$C_t$ 的梯度在时间上有一条**无衰减直通路径**（$f_t$ 可接近 1，使梯度畅通），这是 LSTM 缓解梯度消失的关键。

**输出门**（Output Gate）：决定输出多少细胞状态到隐状态

$$
o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
$$

**隐状态**：

$$
h_t = o_t \odot \tanh(C_t)
$$

示意图（单细胞信息流）：

```
x_t ──┐
       ├── [concat] ──▶ σ ──▶ f_t ──▶ × ──┐
h_{t-1}──┘                       C_{t-1}──┘
                                          │
       ┌── [concat] ──▶ σ ──▶ i_t ──▶ × ──┼──▶ C_t ──▶ tanh ──▶ × ──▶ h_t
       │                                  │                    ▲
       └── [concat] ──▶ tanh ──▶ C̃_t ──▶ +        o_t ──▶───┘
                                                     ▲
                                            [concat] ──▶ σ
                                                     ▲
                                               x_t, h_{t-1}
```

### GRU 单元（简化变体）

GRU 将遗忘门和输入门合并为**更新门**，并合并细胞状态和隐状态，参数量更少、计算更快。

**重置门**（Reset Gate）：控制忽略过去隐状态的程度

$$
r_t = \sigma(W_r \cdot [h_{t-1}, x_t])
$$

**更新门**（Update Gate）：控制从前一状态继承多少信息

$$
z_t = \sigma(W_z \cdot [h_{t-1}, x_t])
$$

**候选隐状态**：

$$
\tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t])
$$

**最终隐状态**：更新门平衡新旧信息

$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$

GRU 参数约为 LSTM 的 $3/4$，在中小规模数据上往往与 LSTM 性能相当，且训练更快。

### 双向 LSTM（BiLSTM）

在每个时间步，同时运行一个**前向** LSTM（从左到右）和一个**后向** LSTM（从右到左），将两个方向的隐状态拼接：

$$
\overrightarrow{h}_t = \text{LSTM}_{\text{fwd}}(x_t, \overrightarrow{h}_{t-1})
$$
$$
\overleftarrow{h}_t = \text{LSTM}_{\text{bwd}}(x_t, \overleftarrow{h}_{t+1})
$$
$$
h_t = [\overrightarrow{h}_t; \overleftarrow{h}_t]
$$

适用于需要完整上下文的任务（如文本分类、命名实体识别），但不适合在线预测。

### 堆叠 LSTM（Stacked LSTM / Deep LSTM）

将多层 LSTM 垂直堆叠，第 $l$ 层的输出作为第 $l+1$ 层的输入：

$$
h_t^{(1)} = \text{LSTM}^{(1)}(x_t, h_{t-1}^{(1)})
$$
$$
h_t^{(l)} = \text{LSTM}^{(l)}(h_t^{(l-1)}, h_{t-1}^{(l)}), \quad l = 2, \dots, L
$$

深层结构可学习更抽象的时序特征，但需更多数据和正则化。

### BPTT（Backpropagation Through Time）

将 RNN 沿时间轴展开为深度前馈网络，使用标准反向传播。关键操作：

1. 沿时间正向传播，存储每步的隐状态和输出
2. 从最后时间步反向传播误差到初始时间步
3. 梯度是所有时间步的梯度之和：
   $$
   \frac{\partial L}{\partial W} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial W}
   $$

### 梯度裁剪（Gradient Clipping）

当梯度范数超过阈值 $\theta$ 时缩放到该阈值：

$$
g \leftarrow \frac{\theta}{\|g\|} \cdot g \quad \text{if } \|g\| > \theta
$$

这是防止梯度爆炸的标准做法，常用 $\theta \in [0.25, 1.0]$。

## 关键假设

- **时序依赖性**：数据点之间存在顺序依赖关系，且依赖结构可用固定步长的隐状态近似
- **固定时间粒度**：采样间隔均匀或可标准化为均匀间隔
- **足够数据量**：LSTM 参数量大，需要充足的训练数据防止过拟合（通常至少数千序列）
- **序列长度适中**：单层 LSTM 可在约 100–500 步内有效，更长序列需注意力机制或 Transformer
- **梯度裁剪必要**：即使有 LSTM 的门控机制，深层/长序列 BPTT 仍需梯度裁剪保证训练稳定

## 适用场景

- **时间序列预测**：股票、天气、电力负荷、交通流量等（通常 BiLSTM 或 Seq2Seq LSTM）
- **自然语言处理**：文本分类、命名实体识别、机器翻译、情感分析
- **语音识别与合成**：声学建模、语音信号处理
- **异常检测**：时序异常模式识别（如传感器故障、金融欺诈检测）
- **视频分析**：动作识别、帧序列建模
- **生成任务**：文本生成、音乐生成、代码生成

### 不适用

- **非序列数据**：无时间/顺序结构的数据，使用 MLP 或 CNN 更高效
- **极长序列（>1000 步）**：梯度仍可能衰减，且计算代价高，使用 Transformer + 自注意力更合适
- **小数据集**：LSTM 参数量大（单层四组权重矩阵），容易过拟合
- **可解释性要求极高**：LSTM 内部机制复杂，难以解释预测原因（可用注意力机制辅助）
- **实时性要求极高的在线系统**：BPTT 计算量大，推理延时高于简单方法
- **低频数据 / 采样不规则**：需先做插值对齐，或使用神经 ODE 等连续时间模型

## 实现要点

### 关键超参数

| 参数 | 范围 | 默认值 | 说明 |
|------|------|--------|------|
| $hidden\_size$ | [32, 512] | 128 | 隐状态维度，越大容量越大 |
| $num\_layers$ | [1, 4] | 2 | 堆叠层数，2–3 层最常见 |
| $dropout$ | [0, 0.5] | 0.2 | 层间 dropout（非循环连接） |
| $seq\_len$ | [10, 200] | 64 | 输入序列长度（回溯窗口） |
| $batch\_size$ | [16, 256] | 64 | 小批量大小 |
| $learning\_rate$ | [1e-4, 1e-2] | 1e-3 | Adam 优化器常用 1e-3 |
| $clip\_norm$ | [0.25, 5.0] | 1.0 | 梯度裁剪阈值 |
| $bidirectional$ | True / False | False | 是否使用双向 LSTM |

### 调优经验

1. **先定序列长度**：用自相关分析（ACF/PACF）确定合理的回溯窗口，过短丢失信息，过长引入噪声
2. **层数选择**：2 层是性价比最高的起点；3 层以上需大量数据和正则化，且收益递减
3. **隐状态维度**：先 128，参考同类数据集的论文调整；增大 $hidden\_size$ 需同步增大 batch size
4. **梯度裁剪**：**始终开启**（通常 1.0），这是 LSTM 训练的标配而非可选项
5. **学习率调度**：使用 ReduceLROnPlateau 或 CosineAnnealingLR，避免 loss 平台期
6. **Early Stopping**：patience=10–20 epoch，防止过拟合
7. **双向的选择**：如果任务可访问完整序列（如文本分类），用 BiLSTM；在线预测只能用单向
8. **Stateful 模式**：长序列截断训练时，跨 batch 传递隐状态可保持长期依赖，但需要手动 reset

### 代码

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


# =========================================================================
# 1. 从零实现 LSTM 单元（纯 NumPy，单步前向传播）
# =========================================================================

class LSTMCell:
    """LSTM 单元 — 单时间步前向传播（NumPy 实现，仅供教学参考）"""
    
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 初始化权重（Xavier 初始化）
        def _init(shape):
            std = np.sqrt(2.0 / (shape[0] + shape[1]))
            return np.random.randn(*shape) * std
        
        # 四个门共享输入拼接 [h_{t-1}, x_t]
        concat_dim = hidden_size + input_size
        
        self.W_f = _init((hidden_size, concat_dim))
        self.b_f = np.zeros(hidden_size)
        
        self.W_i = _init((hidden_size, concat_dim))
        self.b_i = np.zeros(hidden_size)
        
        self.W_c = _init((hidden_size, concat_dim))
        self.b_c = np.zeros(hidden_size)
        
        self.W_o = _init((hidden_size, concat_dim))
        self.b_o = np.zeros(hidden_size)
    
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -100, 100)))
    
    def forward(self, x_t, h_prev, c_prev):
        """
        x_t:    (input_size,)
        h_prev: (hidden_size,)
        c_prev: (hidden_size,)
        """
        concat = np.concatenate([h_prev, x_t])  # (hidden_size + input_size,)
        
        f_t = self.sigmoid(self.W_f @ concat + self.b_f)
        i_t = self.sigmoid(self.W_i @ concat + self.b_i)
        c_tilde = np.tanh(self.W_c @ concat + self.b_c)
        c_t = f_t * c_prev + i_t * c_tilde
        o_t = self.sigmoid(self.W_o @ concat + self.b_o)
        h_t = o_t * np.tanh(c_t)
        
        return h_t, c_t


# =========================================================================
# 2. PyTorch LSTM 模型 — 序列预测
# =========================================================================

class LSTMModel(nn.Module):
    """PyTorch LSTM 模型 — 序列预测
    
    params:
        input_size:    输入特征维度
        hidden_size:   隐状态维度
        num_layers:    LSTM 堆叠层数
        output_size:   输出维度（预测目标维度）
        dropout:       层间 dropout 概率（仅在 num_layers > 1 时生效）
        bidirectional: 是否使用双向 LSTM
    """
    
    def __init__(self, input_size=1, hidden_size=128, num_layers=2,
                 output_size=1, dropout=0.2, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)
    
    def forward(self, x, hidden=None):
        """
        x: (batch_size, seq_len, input_size)
        returns: (batch_size, output_size)
        """
        lstm_out, (h_n, c_n) = self.lstm(x, hidden)
        
        # 取最后一层在所有时间步的输出 → 只取最后一个时间步
        # lstm_out: (batch, seq_len, num_directions * hidden_size)
        last_out = lstm_out[:, -1, :]  # (batch, num_directions * hidden_size)
        
        out = self.fc(last_out)  # (batch, output_size)
        return out


# =========================================================================
# 3. 序列数据准备（滑动窗口）
# =========================================================================

def create_sequences(data, seq_len=64, pred_len=1):
    """将时序数据转换为监督学习的 (X, y) 样本
    
    Args:
        data:     (total_samples, n_features) — 单变量也可传入 (N, 1)
        seq_len:  输入序列长度（回溯窗口）
        pred_len: 预测步长（默认 1 步预测）
    
    Returns:
        X: (num_samples, seq_len, n_features)
        y: (num_samples, pred_len, n_features)
    """
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len:i + seq_len + pred_len])
    return np.array(X), np.array(y)


# =========================================================================
# 4. 训练函数（含梯度裁剪）
# =========================================================================

def train_model(model, train_loader, val_loader, epochs=100, lr=1e-3,
                clip_norm=1.0, patience=15, device='cpu'):
    """训练 LSTM 模型，支持 Early Stopping 和梯度裁剪
    
    Args:
        model:        PyTorch LSTM 模型
        train_loader: 训练 DataLoader
        val_loader:   验证 DataLoader
        epochs:       最大训练轮数
        lr:           学习率
        clip_norm:    梯度裁剪阈值
        patience:     Early Stopping 容忍轮数
        device:       计算设备
    
    Returns:
        train_losses, val_losses: 训练和验证损失历史
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_state = None
    counter = 0
    
    for epoch in range(epochs):
        # --- 训练阶段 ---
        model.train()
        epoch_train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch.squeeze(-1))
            loss.backward()
            
            # 梯度裁剪（LSTM 训练标配）
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            
            optimizer.step()
            epoch_train_loss += loss.item() * X_batch.size(0)
        
        avg_train_loss = epoch_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        
        # --- 验证阶段 ---
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch.squeeze(-1))
                epoch_val_loss += loss.item() * X_batch.size(0)
        
        avg_val_loss = epoch_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:3d} | Train Loss: {avg_train_loss:.6f} "
                  f"| Val Loss: {avg_val_loss:.6f}")
    
    # 恢复最优参数
    model.load_state_dict(best_state)
    return train_losses, val_losses


# =========================================================================
# 5. 预测函数（Teacher Forcing 模式推理）
# =========================================================================

def forecast(model, history, forecast_steps=50, device='cpu'):
    """多步预测（Teacher Forcing：用真实观测更新窗口）
    
    自回归预测（无 Teacher Forcing）也类似，只需将预测值拼接回 history。
    
    Args:
        model:         已训练 LSTM 模型
        history:       (seq_len, n_features) 作为初始输入窗口
        forecast_steps: 预测步数
        device:         计算设备
    
    Returns:
        predictions: (forecast_steps, n_features)
    """
    model.eval()
    model = model.to(device)
    
    predictions = []
    window = history.copy()  # (seq_len, n_features)
    
    with torch.no_grad():
        for _ in range(forecast_steps):
            # 当前窗口 → 预测下一时间步
            X = torch.FloatTensor(window).unsqueeze(0).to(device)  # (1, seq_len, n_features)
            y_pred = model(X)  # (1, output_size)
            
            # 存储预测值
            pred_np = y_pred.cpu().numpy().flatten()
            predictions.append(pred_np)
            
            # 滑动窗口：去掉最早的时间步，加入最新预测
            window = np.vstack([window[1:], pred_np])
    
    return np.array(predictions)


# =========================================================================
# 6. 完整使用示例 — 正弦波预测
# =========================================================================

if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    
    # --- 生成数据 ---
    t = np.linspace(0, 100, 2000)
    data = np.sin(t) + 0.1 * np.random.randn(len(t))
    data = data.reshape(-1, 1)  # (2000, 1)
    
    # --- 划分训练/验证/测试 ---
    train_ratio = 0.7
    val_ratio = 0.15
    n_total = len(data)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_data = data[:n_train]
    val_data = data[n_train:n_train + n_val]
    test_data = data[n_train + n_val:]
    
    # --- 构造序列样本 ---
    seq_len = 32
    X_train, y_train = create_sequences(train_data, seq_len, pred_len=1)
    X_val, y_val = create_sequences(val_data, seq_len, pred_len=1)
    
    print(f"Train samples: {X_train.shape[0]}, Val samples: {X_val.shape[0]}")
    
    # --- DataLoader ---
    batch_size = 64
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
        batch_size=batch_size, shuffle=False
    )
    
    # --- 初始化模型 ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = LSTMModel(
        input_size=1,
        hidden_size=64,
        num_layers=2,
        output_size=1,
        dropout=0.2,
        bidirectional=False
    )
    print(model)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
    
    # --- 训练 ---
    train_losses, val_losses = train_model(
        model, train_loader, val_loader,
        epochs=100, lr=1e-3, clip_norm=1.0, patience=15,
        device=device
    )
    
    # --- 损失曲线 ---
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Training / Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('outputs/lstm_loss_curve.png', dpi=150)
    plt.show()
    
    # --- 多步预测（自回归） ---
    history = data[n_train + n_val - seq_len: n_train + n_val]  # (seq_len, 1)
    true_future = data[n_train + n_val: n_train + n_val + 200]  # 真实未来值
    
    predictions = forecast(model, history, forecast_steps=200, device=device)
    
    # --- 可视化预测结果 ---
    plt.figure(figsize=(12, 5))
    plt.plot(true_future, label='True', color='black', alpha=0.8)
    plt.plot(predictions, label='LSTM Forecast', color='red', linestyle='--')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('LSTM Multi-Step Forecast on Sine Wave')
    plt.legend()
    plt.grid(True)
    plt.savefig('outputs/lstm_forecast.png', dpi=150)
    plt.show()
    
    mse = np.mean((predictions.flatten() - true_future.flatten()) ** 2)
    print(f"\nForecast MSE: {mse:.6f}")
```


### 基于 PyTorch Lightning 的生产用法

```python
import pytorch_lightning as pl
from torchmetrics import MeanSquaredError


class LitLSTM(pl.LightningModule):
    """PyTorch Lightning 封装的 LSTM 模型"""
    
    def __init__(self, input_size=1, hidden_size=128, num_layers=2,
                 output_size=1, dropout=0.2, lr=1e-3, clip_norm=1.0):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.clip_norm = clip_norm
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.criterion = nn.MSELoss()
        self.metric = MeanSquaredError()
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y.squeeze(-1))
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y.squeeze(-1))
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'}
        }
    
    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None):
        # Lightning 内置梯度裁剪
        pass


# 使用：
# model = LitLSTM()
# trainer = pl.Trainer(max_epochs=100, gradient_clip_val=1.0)
# trainer.fit(model, train_loader, val_loader)
```

## 参考文献

Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780.

Cho, K., van Merrienboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. In *Proceedings of EMNLP*, 1724–1734.

Graves, A. (2012). *Supervised Sequence Labelling with Recurrent Neural Networks*. Studies in Computational Intelligence, Springer.

Gers, F. A., Schmidhuber, J., & Cummins, F. (2000). Learning to Forget: Continual Prediction with LSTM. *Neural Computation*, 12(10), 2451–2471.

Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural Machine Translation by Jointly Learning to Align and Translate. In *ICLR*.
