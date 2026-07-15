# Direct Preference Optimization (DPO)

**Source**: Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2024). Direct preference optimization: Your language model is secretly a reward model. *Advances in Neural Information Processing Systems (NeurIPS), 36*. https://arxiv.org/abs/2305.18290

**Category**: Machine Learning / Reinforcement Learning from Human Feedback / LLM Alignment

## Mathematical Setup

### Background: KL-Constrained RLHF

Standard reinforcement learning from human feedback (RLHF) optimizes a policy $\pi_\theta$ under a KL-divergence constraint:

$$
\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(y|x)} \left[ r_\phi(x, y) \right] - \beta \, \mathbb{D}_{\text{KL}}[\pi_\theta(y|x) \,\|\, \pi_{\text{ref}}(y|x)]
$$

where $r_\phi(x, y)$ is a learned reward model, $\pi_{\text{ref}}$ is the reference (SFT) policy, and $\beta > 0$ controls how far the policy can deviate. This requires: (1) training a reward model on preference data, (2) running PPO or similar RL algorithm.

### Closed-Form Optimal Policy

The key insight in Rafailov et al. (2023) is that the optimal policy for the KL-constrained reward maximization problem has a **closed-form solution**:

$$
\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} r(x, y)\right)
$$

where $Z(x) = \sum_y \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} r(x, y)\right)$ is the partition function.

Rewriting to solve for $r(x, y)$:

$$
r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)
$$

### Bradley-Terry Preference Model

Under the Bradley-Terry model, the probability that $y_w$ is preferred over $y_l$ given prompt $x$ is:

$$
p^*(y_w \succ y_l \,|\, x) = \sigma(r(x, y_w) - r(x, y_l))
$$

where $\sigma$ is the logistic sigmoid.

### DPO Objective

Substituting the reward expression into the Bradley-Terry model eliminates the need for a separate reward model:

$$
p_\theta(y_w \succ y_l \,|\, x) = \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)
$$

This gives the DPO loss:

$$
\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]
$$

Define the implicit reward:

$$
\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}
$$

Then the DPO loss simplifies to a binary cross-entropy:

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma\left( \hat{r}_\theta(x, y_w) - \hat{r}_\theta(x, y_l) \right) \right]
$$

### Gradient Analysis

The gradient of the DPO loss with respect to $\theta$ is:

$$
\nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta \, \mathbb{E} \left[ \underbrace{\sigma(\hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w))}_{\text{weight } w_\theta(x, y_w, y_l)} \left( \underbrace{\nabla_\theta \log \pi_\theta(y_w|x)}_{\text{increase } p(y_w)} - \underbrace{\nabla_\theta \log \pi_\theta(y_l|x)}_{\text{decrease } p(y_l)} \right) \right]
$$

The weight $w_\theta$ is high when the model incorrectly prefers $y_l$ over $y_w$, giving more update to misranked pairs -- a form of **adaptive margin** similar to a learned margin in hinge loss.

## Key Assumptions

| Assumption | Formalization | Implication |
|-----------|--------------|-------------|
| **Bradley-Terry preference model** | $p(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))$ | Preferences depend only on reward differences; no context effects |
| **KL-constrained optimization** | $\max_\pi \mathbb{E}[r] - \beta \, \mathbb{D}_{\text{KL}}[\pi \|\pi_{\text{ref}}]$ | Assumes the unconstrained optimal policy is a valid distribution |
| **Access to reference policy** | $\pi_{\text{ref}}(y|x)$ is available for all $(x, y)$ | Requires storing a frozen copy of the SFT model |
| **Transitivity of preferences** | Bradley-Terry implies transitive preferences | May not hold for multi-dimensional preferences |

## Applicable Scenarios

**When to use:**
- Aligning language models with human preferences (harmlessness, helpfulness, style)
- Any task currently using RLHF where training stability is a concern
- Preference optimization without access to online RL infrastructure
- Fine-tuning on pairwise preference datasets

**When NOT to use:**
- When online exploration is important (DPO is offline, uses fixed preference data)
- When reward signals are available directly (e.g., game scores) rather than preferences
- When the reference policy is unavailable or too expensive to store
- When preferences are non-transitive (e.g., multi-dimensional with trade-offs)

**Comparison:** DPO matches PPO-based RLHF quality with significantly simpler training (no reward model, no value function, no PPO clipping). However, it is an offline algorithm and cannot explore novel responses.

## Algorithm / Method Details

### Training Procedure

1. **Preference Dataset**: Collect triples $(x, y_w, y_l)$ where $y_w$ is preferred to $y_l$ given prompt $x$.

2. **Reference Log-Probabilities**: For each batch, compute $\log \pi_{\text{ref}}(y_w|x)$ and $\log \pi_{\text{ref}}(y_l|x)$ using a frozen reference model. These can be precomputed and cached.

3. **Policy Log-Probabilities**: Compute $\log \pi_\theta(y_w|x)$ and $\log \pi_\theta(y_l|x)$ using the current policy.

4. **Loss Computation**:
   $$
   \mathcal{L}_{\text{DPO}} = -\frac{1}{B} \sum_{i=1}^B \log \sigma\left(\beta \left( \log \frac{\pi_\theta(y_{w,i}|x_i)}{\pi_{\text{ref}}(y_{w,i}|x_i)} - \log \frac{\pi_\theta(y_{l,i}|x_i)}{\pi_{\text{ref}}(y_{l,i}|x_i)} \right) \right)
   $$

5. **Backward Pass**: Standard gradient computation with respect to $\theta$.

### Inference

Inference is identical to standard language model generation. No reward model or value function is needed.

### Complexity Analysis

- **Memory**: Requires storing two models during training (policy + reference), vs three for RLHF (policy, reference, reward model, and optionally value function).
- **Compute**: Each batch requires 2 forward passes (policy on chosen and rejected) plus 2 forward passes (reference on chosen and rejected). Throughput is approximately 2x faster than PPO-based RLHF.

## Implementation Details

### Key Hyperparameters

| Parameter | Typical Value | Tuning Guide |
|-----------|--------------|--------------|
| $\beta$ | 0.1-0.5 | Lower = more aggressive alignment, higher = stays closer to reference |
| Learning rate | 5e-7 to 1e-6 (for LLMs) | Very small LR is critical; DPO is sensitive to overfitting |
| Batch size | 32-128 | Larger batches help stabilize the implicit reward margin |
| SFT warmup | 1 epoch of SFT on chosen responses | Important to prevent the policy from generating degenerate tokens |
| Label smoothing | 0-0.1 | Helps when preferences are noisy |

### Numerical Considerations

- Use the **log-sigmoid** formulation to avoid numerical overflow: `-F.logsigmoid(beta * (chosen_logratios - rejected_logratios))`.
- Precompute reference log-probabilities for the entire dataset before training to reduce memory.
- Monitor the **implicit reward margin**: $\hat{r}_\theta(x, y_w) - \hat{r}_\theta(x, y_l)$ should increase over training. If it saturates, increase $\beta$.
- Watch for **reward hacking**: if the implicit reward grows unbounded while generation quality degrades, the KL penalty is insufficient (increase $\beta$ or reduce LR).

### Recommended Libraries

- **Hugging Face TRL**: `trl.DPOTrainer` provides a full implementation.
- **Axolotl**: Configuration-based DPO fine-tuning.
- **Unsloth**: Memory-optimized DPO for consumer GPUs.

## Python Implementation

```python
"""
Minimal implementation of Direct Preference Optimization (DPO).
Based on: Rafailov et al. (2023) https://arxiv.org/abs/2305.18290

Trains a small transformer language model with DPO on synthetic preference data.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from dataclasses import dataclass


# ============================================================
# Minimal Transformer for demonstration
# ============================================================

@dataclass
class GPTConfig:
    vocab_size: int = 1000
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    block_size: int = 64
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    """Single causal self-attention layer."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # Causal mask
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    """Feed-forward network with GELU."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.gelu(self.c_fc(x)))


class TransformerBlock(nn.Module):
    """Single transformer block."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """Minimal GPT-style language model."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Tie weights
        self.lm_head.weight = self.transformer.wte.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        return_logits: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len)
            return_logits: If True, return logits directly
        Returns:
            If return_logits: logits (batch, seq_len, vocab_size)
            Else: log probabilities (batch, seq_len, vocab_size)
        """
        device = input_ids.device
        b, t = input_ids.shape
        assert t <= self.config.block_size, f"Sequence length {t} exceeds block size {self.config.block_size}"

        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        tok_emb = self.transformer.wte(input_ids)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        if return_logits:
            return logits

        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs

    def get_log_probs(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Get log probabilities for each token in the sequence.
        Args:
            input_ids: (batch, seq_len)
        Returns:
            log_probs: (batch, seq_len)
        """
        log_probs = self.forward(input_ids)  # (batch, seq_len, vocab_size)
        # Gather log probs of actual tokens
        return log_probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)  # (batch, seq_len)


# ============================================================
# DPO Training
# ============================================================

class DPOTrainer:
    """
    Direct Preference Optimization trainer.
    
    Trains a policy model pi_theta to align with preferences,
    while staying close to a reference model pi_ref.
    """
    def __init__(
        self,
        policy: nn.Module,
        ref_policy: Optional[nn.Module] = None,
        beta: float = 0.1,
        lr: float = 5e-6,
        device: str = "cpu",
    ):
        self.policy = policy.to(device)
        self.ref_policy = ref_policy if ref_policy is not None else self._copy_model(policy)
        self.ref_policy.to(device)
        # Freeze reference model
        for p in self.ref_policy.parameters():
            p.requires_grad = False
        
        self.beta = beta
        self.device = device
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=lr)

    def _copy_model(self, model: nn.Module) -> nn.Module:
        """Deep copy a model."""
        import copy
        return copy.deepcopy(model)

    @staticmethod
    def dpo_loss(
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
        beta: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute DPO loss.

        Args:
            policy_chosen_logps: (batch,) log prob of chosen responses under policy
            policy_rejected_logps: (batch,) log prob of rejected responses under policy
            ref_chosen_logps: (batch,) log prob of chosen responses under reference
            ref_rejected_logps: (batch,) log prob of rejected responses under reference
            beta: KL penalty coefficient

        Returns:
            loss: scalar
            accuracy: fraction of pairs where chosen > rejected in implicit reward
        """
        # Compute implicit reward differences
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        
        logits = pi_logratios - ref_logratios  # (batch,)
        
        # DPO loss: -log(sigmoid(beta * logits))
        loss = -F.logsigmoid(beta * logits).mean()
        
        # Accuracy: how often does the model prefer the correct response?
        with torch.no_grad():
            accuracy = (logits > 0).float().mean().item()
        
        return loss, accuracy

    def train_step(
        self,
        prompt_ids: torch.Tensor,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
    ) -> dict:
        """
        Single DPO training step.

        Args:
            prompt_ids: (batch, prompt_len)
            chosen_ids: (batch, response_len) -- preferred continuation
            rejected_ids: (batch, response_len) -- dispreferred continuation

        Returns:
            stats dict with loss and accuracy
        """
        batch_size = prompt_ids.shape[0]
        
        # Concatenate prompt + response
        chosen_input_ids = torch.cat([prompt_ids, chosen_ids], dim=1)
        rejected_input_ids = torch.cat([prompt_ids, rejected_ids], dim=1)
        prompt_len = prompt_ids.shape[1]

        # Forward pass: get log probabilities
        policy_chosen_logps = self.policy.get_log_probs(chosen_input_ids)
        policy_rejected_logps = self.policy.get_log_probs(rejected_input_ids)

        with torch.no_grad():
            ref_chosen_logps = self.ref_policy.get_log_probs(chosen_input_ids)
            ref_rejected_logps = self.ref_policy.get_log_probs(rejected_input_ids)

        # Mask out prompt tokens -- only compute loss on response tokens
        policy_chosen_logps = policy_chosen_logps[:, prompt_len:].sum(dim=1)
        policy_rejected_logps = policy_rejected_logps[:, prompt_len:].sum(dim=1)
        ref_chosen_logps = ref_chosen_logps[:, prompt_len:].sum(dim=1)
        ref_rejected_logps = ref_rejected_logps[:, prompt_len:].sum(dim=1)

        # Compute DPO loss
        loss, accuracy = self.dpo_loss(
            policy_chosen_logps, policy_rejected_logps,
            ref_chosen_logps, ref_rejected_logps,
            self.beta,
        )

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "accuracy": accuracy,
            "reward_margin": (policy_chosen_logps - policy_rejected_logps).mean().item(),
        }


# ============================================================
# Training demo with synthetic data
# ============================================================

def create_synthetic_preference_data(
    vocab_size: int = 100,
    n_samples: int = 1000,
    seq_len: int = 16,
    prompt_len: int = 6,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create synthetic preference data where "chosen" responses are
    defined as sequences with higher token values.

    We assign each token a "quality" score. Chosen responses have
    higher average quality.
    """
    torch.manual_seed(seed)
    
    prompts = torch.randint(0, vocab_size, (n_samples, prompt_len))
    
    # Token "quality" scores: some tokens are inherently better
    token_quality = torch.randn(vocab_size)
    
    chosen_responses = []
    rejected_responses = []
    
    for i in range(n_samples):
        # Generate two candidate responses
        cand1 = torch.randint(0, vocab_size, (seq_len,))
        cand2 = torch.randint(0, vocab_size, (seq_len,))
        
        # Compute average quality
        q1 = token_quality[cand1].mean()
        q2 = token_quality[cand2].mean()
        
        if q1 > q2:
            chosen_responses.append(cand1)
            rejected_responses.append(cand2)
        else:
            chosen_responses.append(cand2)
            rejected_responses.append(cand1)
    
    return (
        prompts,
        torch.stack(chosen_responses),
        torch.stack(rejected_responses),
    )


def test_dpo():
    """Run a complete DPO training demo."""
    print("=" * 60)
    print("DPO Training Demo")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Training hyperparameters
    vocab_size = 100
    n_epochs = 5
    batch_size = 32
    seq_len = 12
    prompt_len = 4
    beta = 0.2
    
    # Create synthetic preference data
    prompts, chosen, rejected = create_synthetic_preference_data(
        vocab_size=vocab_size,
        n_samples=1000,
        seq_len=seq_len,
        prompt_len=prompt_len,
    )
    print(f"Data: {len(prompts)} preference pairs")
    print(f"Prompt shape: {prompts.shape}, Chosen shape: {chosen.shape}, Rejected shape: {rejected.shape}")
    
    # Create models
    config = GPTConfig(
        vocab_size=vocab_size,
        n_embd=64,
        n_head=2,
        n_layer=2,
        block_size=32,
    )
    
    # First train a "SFT" model on chosen responses (warmup)
    sft_model = GPT(config).to(device)
    sft_optimizer = torch.optim.AdamW(sft_model.parameters(), lr=1e-3)
    
    print("\n[SFT Warmup] Training on chosen responses...")
    sft_losses = []
    for epoch in range(5):
        perm = torch.randperm(len(prompts))
        epoch_loss = 0.0
        for i in range(0, len(prompts), batch_size):
            idx = perm[i:i+batch_size]
            input_ids = torch.cat([prompts[idx], chosen[idx]], dim=1).to(device)
            
            log_probs = sft_model(input_ids)
            # Mask out prompt
            log_probs_resp = log_probs[:, prompt_len:]
            chosen_tokens = chosen[idx].to(device)
            loss = F.nll_loss(
                log_probs_resp.reshape(-1, vocab_size),
                chosen_tokens.reshape(-1),
            )
            
            sft_optimizer.zero_grad()
            loss.backward()
            sft_optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / (len(prompts) / batch_size)
        sft_losses.append(avg_loss)
        print(f"  Epoch {epoch+1}/5, SFT Loss: {avg_loss:.4f}")
    
    # Now apply DPO
    print("\n[DPO Training] Aligning with preferences...")
    trainer = DPOTrainer(
        policy=GPT(config).to(device),
        ref_policy=GPT(config).to(device),
        beta=beta,
        lr=5e-6,
        device=device,
    )
    
    # Initialize both models with SFT weights
    trainer.policy.load_state_dict(sft_model.state_dict())
    trainer.ref_policy.load_state_dict(sft_model.state_dict())
    
    # Train
    dpo_stats = {"loss": [], "accuracy": [], "reward_margin": []}
    for epoch in range(n_epochs):
        perm = torch.randperm(len(prompts))
        epoch_stats = {"loss": 0.0, "accuracy": 0.0, "reward_margin": 0.0}
        n_batches = 0
        
        for i in range(0, len(prompts), batch_size):
            idx = perm[i:i+batch_size]
            stats = trainer.train_step(
                prompts[idx].to(device),
                chosen[idx].to(device),
                rejected[idx].to(device),
            )
            for k in epoch_stats:
                epoch_stats[k] += stats.get(k, 0.0)
            n_batches += 1
        
        for k in epoch_stats:
            epoch_stats[k] /= max(n_batches, 1)
            dpo_stats[k].append(epoch_stats[k])
        
        print(f"  Epoch {epoch+1}/{n_epochs}: "
              f"Loss={epoch_stats['loss']:.4f}, "
              f"Acc={epoch_stats['accuracy']:.4f}, "
              f"Margin={epoch_stats['reward_margin']:.4f}")
    
    print("\n[Results]")
    print(f"  Final DPO loss: {dpo_stats['loss'][-1]:.4f}")
    print(f"  Final accuracy: {dpo_stats['accuracy'][-1]:.4f}")
    print(f"  Final reward margin: {dpo_stats['reward_margin'][-1]:.4f}")
    print("Training complete!")


if __name__ == "__main__":
    test_dpo()
```

## References

Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2024). Direct preference optimization: Your language model is secretly a reward model. *Advances in Neural Information Processing Systems (NeurIPS), 36*. https://arxiv.org/abs/2305.18290

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C. L., Mishkin, P., ... & Lowe, R. (2022). Training language models to follow instructions with human feedback. *Advances in Neural Information Processing Systems (NeurIPS), 35*. https://arxiv.org/abs/2203.02155

Ziegler, D. M., Stiennon, N., Wu, J., Brown, T. B., Radford, A., Amodei, D., ... & Irving, G. (2019). Fine-tuning language models from human preferences. *arXiv preprint arXiv:1909.08593*. https://arxiv.org/abs/1909.08593

Ethayarajh, K., Xu, W., Muennighoff, N., Jurafsky, D., & Kiela, D. (2024). KTO: Model alignment as prospect theoretic optimization. *arXiv preprint arXiv:2402.01306*. https://arxiv.org/abs/2402.01306

Azar, M. G., Guo, Z. D., Piot, B., Munos, R., Rowland, M., Valko, M., & Calandriello, D. (2024). A general theoretical paradigm to understand learning from human preferences. *International Conference on Artificial Intelligence and Statistics (AISTATS)*. https://arxiv.org/abs/2310.12036
