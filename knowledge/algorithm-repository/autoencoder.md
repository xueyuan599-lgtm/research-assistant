# Autoencoder / VAE — 自编码器与变分自编码器

- **来源**: Rumelhart, D.E., Hinton, G.E., & Williams, R.J. (1986). Learning representations by back-propagating errors. *Nature*, 323(6088), 533–536. / Kingma, D.P. & Welling, M. (2014). Auto-Encoding Variational Bayes. *ICLR*.
- **DOI**: 10.1038/323533a0 / 10.48550/arXiv.1312.6114
- **方法类别**: 深度学习 / 无监督学习 / 生成模型

## 数学设定

### 标准自编码器 (Autoencoder)

自编码器由编码器 $\phi$ 和解码器 $\psi$ 组成，通过重建误差学习低维表示：

$$
\phi: \mathcal{X} \to \mathcal{Z}, \quad \psi: \mathcal{Z} \to \hat{\mathcal{X}}
$$

- **编码器**：$z = \phi(x) = f(W_e x + b_e)$，其中 $f$ 为非线性激活（如 ReLU）
- **解码器**：$\hat{x} = \psi(z) = g(W_d z + b_d)$，其中 $g$ 为激活（线性 / Sigmoid）

**重建损失**（欠完备自编码器，$\dim(\mathcal{Z}) \ll \dim(\mathcal{X})$）：

$$
\mathcal{L}_{\text{AE}} = \|x - \psi(\phi(x))\|^2 = \sum_{i=1}^{D} (x_i - \hat{x}_i)^2
$$

对于二值化 / 归一化数据，使用二元交叉熵损失：

$$
\mathcal{L}_{\text{BCE}} = -\sum_{i=1}^{D} \big[x_i \log \hat{x}_i + (1 - x_i) \log(1 - \hat{x}_i)\big]
$$

### 去噪自编码器 (Denoising AE, Vincent et al. 2008)

对输入注入噪声，迫使模型学习鲁棒表示：

$$
\tilde{x} = x + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2 I) \quad \text{或} \quad \tilde{x} \sim \text{Mask}(x, p)
$$

$$
\mathcal{L}_{\text{DAE}} = \mathbb{E}_{x \sim p(x), \tilde{x} \sim q(\tilde{x}|x)} \big[\|x - \psi(\phi(\tilde{x}))\|^2\big]
$$

其中 $q(\tilde{x}|x)$ 是噪声分布。模型从 corrupted 输入重建原始未污染数据。

### 稀疏自编码器 (Sparse AE)

在损失中加入稀疏性正则化，迫使隐层神经元大部分处于抑制状态：

$$
\mathcal{L}_{\text{SAE}} = \|x - \hat{x}\|^2 + \lambda \cdot \text{KL}(\rho \,\|\, \hat{\rho})
$$

其中 $\rho$ 为目标稀疏度（接近 0 的小值），$\hat{\rho}_j = \frac{1}{N}\sum_{i=1}^{N} a_j(x_i)$ 为第 $j$ 个隐层单元的平均激活值，KL 散度为：

$$
\text{KL}(\rho \,\|\, \hat{\rho}_j) = \rho \log\frac{\rho}{\hat{\rho}_j} + (1 - \rho) \log\frac{1 - \rho}{1 - \hat{\rho}_j}
$$

### 收缩自编码器 (Contractive AE, Rifai et al. 2011)

在损失中加入编码器 Jacobian 的 Frobenius 范数，迫使隐层表示对输入微小变化不敏感：

$$
\mathcal{L}_{\text{CAE}} = \|x - \hat{x}\|^2 + \lambda \left\|\frac{\partial h}{\partial x}\right\|_F^2
$$

其中 $\frac{\partial h}{\partial x}$ 是编码器输出对输入的 Jacobian 矩阵，$\|\cdot\|_F$ 为 Frobenius 范数。

### 变分自编码器 (Variational Autoencoder, Kingma & Welling 2014)

VAE 是生成式模型，引入概率框架——学习数据的潜在分布 $p(x)$。

#### 生成模型（解码器）
- 先验分布：$p(z) = \mathcal{N}(0, I)$
- 条件分布：$p_\theta(x|z)$ 由解码器神经网络参数化
  - 连续数据：$p_\theta(x|z) = \mathcal{N}(\mu_\theta(z), \sigma_\theta^2(z) I)$
  - 二值数据：$p_\theta(x|z) = \text{Bernoulli}(\pi_\theta(z))$

#### 推理模型（编码器）
使用变分后验近似真实后验（后者难以计算）：

$$
q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x) I)
$$

#### 证据下界 (ELBO)

目标函数为最大化对数边际似然的证据下界：

$$
\log p_\theta(x) \geq \mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \text{KL}(q_\phi(z|x) \,\|\, p(z))
$$

展开为两项：

1. **重建项**：$\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]$ —— 解码器重建质量
2. **KL 正则项**：$\text{KL}(q_\phi(z|x) \,\|\, p(z))$ —— 后验与先验的接近程度

#### 重参数化技巧 (Reparameterization Trick)

为使采样操作可微，将 $z \sim q_\phi(z|x)$ 重写为确定性变换：

$$
z = \mu_\phi(x) + \sigma_\phi(x) \odot \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)
$$

其中 $\odot$ 为逐元素乘法。这使得梯度可经 $\mu_\phi$ 和 $\sigma_\phi$ 反向传播。

#### 高斯 KL 闭式解

当 $q_\phi(z|x) = \mathcal{N}(\mu, \sigma^2 I)$、$p(z) = \mathcal{N}(0, I)$ 时，KL 散度有解析形式：

$$
\text{KL}(\mathcal{N}(\mu, \sigma^2) \,\|\, \mathcal{N}(0, 1)) = \frac{1}{2} \sum_{j=1}^{d} \big(\mu_j^2 + \sigma_j^2 - 1 - \log \sigma_j^2\big)
$$

其中 $d$ 为潜在空间维度。

### $\beta$-VAE (Higgins et al. 2017)

引入 $\beta > 1$ 增强解耦表示学习：

$$
\mathcal{L}_{\beta\text{-VAE}} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \beta \cdot \text{KL}(q_\phi(z|x) \,\|\, p(z))
$$

$\beta$ 增大强制学习更分解的潜在表示（对 disentanglement 有利，但重建质量会下降）。

### VAE 的梯度

$$
\nabla_\theta \mathcal{L} \approx \nabla_\theta \frac{1}{L} \sum_{\ell=1}^{L} \log p_\theta(x \mid z^{(\ell)}), \quad z^{(\ell)} = \mu_\phi(x) + \sigma_\phi(x) \odot \varepsilon^{(\ell)}
$$

$$
\nabla_\phi \mathcal{L} \approx \frac{1}{L} \sum_{\ell=1}^{L} \nabla_\phi \log p_\theta(x \mid z^{(\ell)}) - \nabla_\phi \text{KL}(q_\phi(z|x) \,\|\, p(z))
$$

实践中通常 $L = 1$。

## 关键假设

- 高维观测数据存在一个低维潜在流形（manifold hypothesis），编码器可有效捕捉
- 编码-解码的重建误差是有意义的无监督学习目标
- VAE 额外假设：潜在变量服从指定的先验分布（通常是各向同性高斯）
- 去噪自编码器假设：对输入注入噪声不会改变数据的语义结构
- 潜在空间各维度独立（VAE 的对角协方差假设）——真实数据未必满足
- $\beta$-VAE 假设：增大 KL 权重能促使维度解耦，但缺乏理论保证

## 适用场景

- **非线性降维**：相比 PCA，自编码器可学习非线性流形结构
- **异常检测**：重建误差高 = 异常样本（训练集无异常时）
- **去噪**：DAE 训练后可直接用于数据去噪
- **特征提取 / 预训练**：编码器输出作为下游任务的输入特征
- **数据生成**：VAE 可从先验采样生成新数据（图像、文本嵌入等）
- **表示学习**：$\beta$-VAE 用于学习解耦表示（disentangled representation）
- **缺失值填补**：利用重建能力补全部分遮挡 / 缺失的数据

### 不适用

- **直接用于判别任务**：分类等任务应使用编码器提取特征 + 监督学习头
- **高保真图像生成**：VAE 生成图像偏模糊，此类任务首选 GAN / Diffusion 模型
- **对潜在空间可解释性要求极高**：VAE 的潜在维度虽可遍历采样，但语义解释需 post-hoc 分析
- **小样本场景**：自编码器通常需要大量无标签数据才能学到有意义的表示
- **实时推理 / 资源受限设备**：编解码双网络计算量较大

## 实现要点

### 关键超参数

| 参数 | 范围 | 默认值 | 说明 |
|------|------|--------|------|
| $latent\_dim$ | [2, 256] | 32 | 瓶颈维度，决定压缩程度 |
| $hidden\_dims$ | — | [512, 256, 128] | 编码器各隐层维度（对称解码器） |
| $\beta$ (VAE) | [0, 10] | 1.0 | KL 权重，$\beta > 1$ 增强解耦 |
| $learning\_rate$ | [1e-5, 1e-3] | 1e-3 | 优化器学习率 |
| $batch\_size$ | [32, 512] | 128 | 训练批大小 |
| $n\_epochs$ | [50, 500] | 100 | 训练轮数 |

### 调优经验

1. **瓶颈维度 ($latent\_dim$)** 是最重要的超参数——太小丢失信息，太大则退化为恒等映射（过完备时需加正则）
2. **重建损失选择**：连续数据用 MSE，像素值归一化到 [0,1] 时用 BCE
3. **KL Annealing**（VAE 关键技巧）：训练初期 $\beta$ 从 0 逐渐增长到目标值，避免 posterior collapse（KL 项迅速归零导致模型忽略潜在变量）
4. **Posterior Collapse 监测**：若 KL 散度持续接近 0，表示编码器退化为先验，需调整 $\beta$ 或使用 free bits 技巧
5. **架构对称性**：解码器通常镜像编码器结构，但也可更浅（因生成比编码容易）
6. **Batch Normalization** 通常加在编码器隐层，解码器最后一层不应加 BN（输出需保持原始数据尺度）
7. **过完备问题**：当 $\dim(\mathcal{Z}) \ge \dim(\mathcal{X})$ 且无正则时，模型可能学到恒等映射而非有用特征，此时需加稀疏 / 收缩正则

### 代码

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class Autoencoder(nn.Module):
    """自编码器 — 欠完备非线性降维"""

    def __init__(self, input_dim, latent_dim=32, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        # 编码器：input_dim → hidden_dims → latent_dim
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(nn.BatchNorm1d(h_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(0.2))
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # 解码器：latent_dim → hidden_dims[::-1] → input_dim
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in hidden_dims[::-1]:
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.BatchNorm1d(h_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        # 最后一层不使用 BN 和 ReLU（输出需保持原始数据尺度）
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def reconstruct(self, x):
        """重建并返回重建误差"""
        x_hat = self.forward(x)
        loss = torch.mean((x - x_hat) ** 2, dim=1)
        return x_hat, loss

    def fit(self, X, val_X=None, epochs=100, batch_size=128,
            lr=1e-3, weight_decay=1e-5, verbose=True):
        """完整的训练循环"""
        device = next(self.parameters()).device
        dataset = TensorDataset(torch.FloatTensor(X))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
        )

        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                x_hat = self.forward(batch)
                loss = nn.functional.mse_loss(x_hat, batch)
                loss.backward()
                # 梯度裁剪防止训练不稳定
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item() * batch.size(0)

            train_loss = epoch_loss / len(X)
            history['train_loss'].append(train_loss)

            # 验证集评估
            if val_X is not None:
                self.eval()
                with torch.no_grad():
                    val_tensor = torch.FloatTensor(val_X).to(device)
                    val_x_hat = self.forward(val_tensor)
                    val_loss = nn.functional.mse_loss(val_x_hat, val_tensor).item()
                history['val_loss'].append(val_loss)
                scheduler.step(val_loss)
            else:
                scheduler.step(train_loss)

            if verbose and (epoch + 1) % 10 == 0:
                msg = f"Epoch [{epoch+1}/{epochs}]  Train Loss: {train_loss:.6f}"
                if val_X is not None:
                    msg += f"  Val Loss: {val_loss:.6f}"
                print(msg)

        return history


class VAE(Autoencoder):
    """变分自编码器 — 生成式变分推理"""

    def __init__(self, input_dim, latent_dim=32, hidden_dims=None, beta=1.0):
        super().__init__(input_dim, latent_dim, hidden_dims)
        self.beta = beta

        # VAE 编码器输出均值和 log 方差（覆盖 Autoencoder 的编码器）
        prev_dim = input_dim
        for h_dim in hidden_dims:
            prev_dim = h_dim  # 走到最后一层隐层维度
        self.mu_layer = nn.Linear(prev_dim, latent_dim)
        self.logvar_layer = nn.Linear(prev_dim, latent_dim)

    def encode(self, x):
        """编码器输出变为 mu 和 logvar"""
        h = x
        for layer in self.encoder[:-1]:  # 去掉最后一层线性层
            h = layer(h)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """重参数化技巧：z = mu + sigma * eps"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def kl_divergence(self, mu, logvar):
        """高斯 KL 闭式解：KL(N(mu,sigma^2) || N(0,1))"""
        return -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

    def reconstruct(self, x):
        """重建并返回重建误差（用于异常检测）"""
        x_hat, mu, logvar = self.forward(x)
        recon_loss = torch.mean((x - x_hat) ** 2, dim=1)
        kl_loss = self.kl_divergence(mu, logvar)
        return x_hat, recon_loss, kl_loss

    def sample(self, n_samples, device='cpu'):
        """从先验采样生成新数据"""
        self.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.mu_layer.out_features).to(device)
            samples = self.decode(z)
        return samples

    def fit(self, X, val_X=None, epochs=100, batch_size=128,
            lr=1e-3, weight_decay=1e-5, beta_anneal=True, verbose=True):
        """
        VAE 训练循环（含 KL annealing 支持）

        beta_anneal: 是否从 0 线性增长 beta 到设定值
        """
        device = next(self.parameters()).device
        dataset = TensorDataset(torch.FloatTensor(X))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
        )

        history = {'train_loss': [], 'val_loss': [],
                   'train_recon': [], 'train_kl': []}

        for epoch in range(epochs):
            # KL annealing：beta 逐步从 0 增长到目标值
            if beta_anneal:
                beta_current = self.beta * min(1.0, epoch / (epochs * 0.3))
            else:
                beta_current = self.beta

            self.train()
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_kl = 0.0
            n_total = 0

            for (batch,) in loader:
                batch = batch.to(device)
                batch_size_ = batch.size(0)
                n_total += batch_size_

                optimizer.zero_grad()
                x_hat, mu, logvar = self.forward(batch)
                recon_loss = nn.functional.mse_loss(x_hat, batch, reduction='sum')
                kl_loss = self.kl_divergence(mu, logvar).sum()
                loss = recon_loss + beta_current * kl_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                epoch_recon += recon_loss.item()
                epoch_kl += kl_loss.item()

            # 平均到样本
            history['train_loss'].append(epoch_loss / n_total)
            history['train_recon'].append(epoch_recon / n_total)
            history['train_kl'].append(epoch_kl / n_total)

            # 验证集
            if val_X is not None:
                self.eval()
                with torch.no_grad():
                    val_tensor = torch.FloatTensor(val_X).to(device)
                    val_x_hat, val_mu, val_logvar = self.forward(val_tensor)
                    val_recon = nn.functional.mse_loss(val_x_hat, val_tensor).item()
                    val_kl = self.kl_divergence(val_mu, val_logvar).mean().item()
                    val_loss = val_recon + beta_current * val_kl
                history['val_loss'].append(val_loss)
                scheduler.step(val_loss)
            else:
                scheduler.step(history['train_loss'][-1])

            if verbose and (epoch + 1) % 10 == 0:
                msg = (f"Epoch [{epoch+1}/{epochs}]  "
                       f"Loss: {history['train_loss'][-1]:.4f}  "
                       f"Recon: {history['train_recon'][-1]:.4f}  "
                       f"KL: {history['train_kl'][-1]:.4f}  "
                       f"beta: {beta_current:.3f}")
                if val_X is not None:
                    msg += f"  Val Loss: {val_loss:.4f}"
                print(msg)

        return history


class DenoisingAE(Autoencoder):
    """去噪自编码器：在输入注入噪声后重建原始数据"""

    def __init__(self, input_dim, latent_dim=32, hidden_dims=None, noise_factor=0.1):
        super().__init__(input_dim, latent_dim, hidden_dims)
        self.noise_factor = noise_factor

    def forward(self, x):
        """训练时注入高斯噪声"""
        if self.training:
            noise = torch.randn_like(x) * self.noise_factor
            x_corrupted = x + noise
        else:
            x_corrupted = x
        z = self.encoder(x_corrupted)
        return self.decoder(z)


# =====================
# 使用示例
# =====================
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # 1. 生成合成高维数据
    X, y = make_classification(
        n_samples=2000, n_features=50, n_informative=20,
        n_redundant=10, random_state=42
    )
    X = StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 2. 标准自编码器
    print("=" * 50)
    print("Standard Autoencoder")
    print("=" * 50)
    ae = Autoencoder(input_dim=50, latent_dim=8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae.to(device)
    history = ae.fit(X_train, val_X=X_test, epochs=50, verbose=True)

    # 编码与重建
    X_tensor = torch.FloatTensor(X_test).to(device)
    z_test = ae.encode(X_tensor)
    X_recon, recon_errors = ae.reconstruct(X_tensor)
    print(f"\nLatent shape: {z_test.shape}")
    print(f"Mean reconstruction error: {recon_errors.mean().item():.6f}")

    # 3. 变分自编码器
    print("\n" + "=" * 50)
    print("Variational Autoencoder")
    print("=" * 50)
    vae = VAE(input_dim=50, latent_dim=8, beta=1.0)
    vae.to(device)
    vae_history = vae.fit(X_train, val_X=X_test, epochs=50, verbose=True,
                          beta_anneal=True)

    # 生成新数据
    generated = vae.sample(10, device=device)
    print(f"\nGenerated samples shape: {generated.shape}")

    # 4. 异常检测示例：在测试集混入异常
    print("\n" + "=" * 50)
    print("Anomaly Detection with AE")
    print("=" * 50)
    X_normal = X_train.copy()
    X_anomaly = X_test.copy()
    # 在一些样本中添加大偏差作为异常
    np.random.seed(42)
    anomaly_idx = np.random.choice(len(X_anomaly), size=20, replace=False)
    X_anomaly[anomaly_idx] += np.random.randn(20, 50) * 5.0

    ae_anomaly = Autoencoder(input_dim=50, latent_dim=8)
    ae_anomaly.to(device)
    ae_anomaly.fit(X_normal, epochs=30, verbose=False)

    # 计算重建误差
    _, errors = ae_anomaly.reconstruct(torch.FloatTensor(X_anomaly).to(device))
    errors = errors.detach().cpu().numpy()

    # 取阈值（训练集误差的 95 分位数）
    _, train_errors = ae_anomaly.reconstruct(torch.FloatTensor(X_normal).to(device))
    train_errors = train_errors.detach().cpu().numpy()
    threshold = np.percentile(train_errors, 95)

    y_pred = (errors > threshold).astype(int)
    y_true = np.zeros(len(X_anomaly))
    y_true[anomaly_idx] = 1
    precision = np.mean(y_pred[y_true == 1])  # 异常中被检出比例
    print(f"Detection rate on anomalies: {precision:.2f} "
          f"(threshold = P95 of normal: {threshold:.4f})")
```

### 基于 PyTorch Lightning 的生产用法

```python
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class LightningVAE(pl.LightningModule):
    """生产级 VAE — PyTorch Lightning 封装"""

    def __init__(self, input_dim, latent_dim=32,
                 hidden_dims=None, beta=1.0, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.beta = beta
        self.lr = lr

        # 编码器
        encoder_layers = []
        prev = input_dim
        for h in (hidden_dims or [512, 256, 128]):
            encoder_layers.extend([
                nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h)
            ])
            prev = h
        self.encoder_shared = nn.Sequential(*encoder_layers)
        self.mu_layer = nn.Linear(prev, latent_dim)
        self.logvar_layer = nn.Linear(prev, latent_dim)

        # 解码器
        decoder_layers = []
        prev = latent_dim
        for h in (hidden_dims or [128, 256, 512]):
            decoder_layers.extend([
                nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h)
            ])
            prev = h
        decoder_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder_shared(x)
        return self.mu_layer(h), self.logvar_layer(h)

    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def training_step(self, batch, batch_idx):
        x, = batch
        x_hat, mu, logvar = self(x)
        recon_loss = F.mse_loss(x_hat, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())
        loss = (recon_loss + self.beta * kl_loss) / x.size(0)
        self.log_dict({
            'train_loss': loss,
            'train_recon': recon_loss / x.size(0),
            'train_kl': kl_loss / x.size(0),
        }, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, = batch
        x_hat, mu, logvar = self(x)
        recon_loss = F.mse_loss(x_hat, x)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu ** 2 - logvar.exp())
        self.log_dict({
            'val_loss': recon_loss + self.beta * kl_loss,
            'val_recon': recon_loss,
            'val_kl': kl_loss,
        }, prog_bar=True)
        return recon_loss + self.beta * kl_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            }
        }


# Lightning 使用
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.preprocessing import StandardScaler

    X, _ = make_classification(n_samples=5000, n_features=100, random_state=42)
    X = StandardScaler().fit_transform(X).astype(np.float32)

    train_dl = DataLoader(TensorDataset(torch.from_numpy(X[:4000])),
                          batch_size=128, shuffle=True)
    val_dl = DataLoader(TensorDataset(torch.from_numpy(X[4000:])),
                        batch_size=128)

    model = LightningVAE(input_dim=100, latent_dim=16, beta=1.0)
    trainer = pl.Trainer(max_epochs=50, accelerator='auto',
                         enable_checkpointing=True)
    trainer.fit(model, train_dl, val_dl)
```

## 参考文献

- Rumelhart, D.E., Hinton, G.E., & Williams, R.J. (1986). Learning representations by back-propagating errors. *Nature*, 323(6088), 533–536.
- Kingma, D.P. & Welling, M. (2014). Auto-Encoding Variational Bayes. *ICLR*.
- Rezende, D.J., Mohamed, S., & Wierstra, D. (2014). Stochastic Backpropagation and Approximate Inference in Deep Generative Models. *ICML*.
- Vincent, P., Larochelle, H., Bengio, Y., & Manzagol, P.A. (2008). Extracting and composing robust features with denoising autoencoders. *ICML*.
- Rifai, S., Vincent, P., Muller, X., Glorot, X., & Bengio, Y. (2011). Contractive auto-encoders: Explicit invariance during feature extraction. *ICML*.
- Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., Botvinick, M., Mohamed, S., & Lerchner, A. (2017). beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework. *ICLR*.
