"""上下文饱和度估算工具 — 基于 tiktoken cl100k_base

用法:
    from context_monitor import estimate_tokens, read_context_log, check_saturation

    # 估算文本 token 数
    tokens = estimate_tokens("某段文本")

    # 记录本次贡献到上下文日志
    write_context_log("literature.search", tokens, "outputs/checkpoints/")

    # 检查当前饱和度
    saturated, pct = check_saturation("outputs/checkpoints/")
    if saturated:
        print(f"上下文饱和度 {pct:.0f}%，建议切割")
"""

import json
import os
from pathlib import Path

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
except Exception:
    _enc = None

# 默认上下文窗口（Claude Code 典型值，可通过环境变量覆盖）
MAX_CONTEXT_TOKENS = int(os.environ.get("MAX_CONTEXT_TOKENS", "1000000"))
# 饱和度阈值（超过此比例触发切割）
SATURATION_THRESHOLD = float(os.environ.get("SATURATION_THRESHOLD", "0.50"))


def estimate_tokens(text: str) -> int:
    """使用 cl100k_base 估算文本的 token 数。

    若 tiktoken 不可用，fallback 到字符数 / 3 的粗略估算。
    """
    if not text:
        return 0
    if _enc is not None:
        return len(_enc.encode(text))
    # fallback: 中英文混合约 3 字符 / token
    return len(text) // 3


def write_context_log(stage: str, token_count: int, checkpoint_dir: str = "outputs/checkpoints/"):
    """将某一阶段的 token 消耗写入上下文日志。

    Args:
        stage: 阶段标识，如 "literature.search"
        token_count: estimate_tokens 的输出
        checkpoint_dir: 检查点目录
    """
    log_path = Path(checkpoint_dir) / "_context_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    entry = {"stage": stage, "tokens": token_count, "ts": __import__("datetime").datetime.now().isoformat()}
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_context_log(checkpoint_dir: str = "outputs/checkpoints/") -> list[dict]:
    """读取完整上下文日志。"""
    log_path = Path(checkpoint_dir) / "_context_log.jsonl"
    if not log_path.exists():
        return []
    entries = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def total_consumed_tokens(checkpoint_dir: str = "outputs/checkpoints/") -> int:
    """累计已消耗的 token 总数。"""
    return sum(e["tokens"] for e in read_context_log(checkpoint_dir))


def check_saturation(checkpoint_dir: str = "outputs/checkpoints/") -> tuple[bool, float]:
    """检查上下文饱和度。

    Returns:
        (is_saturated, saturation_percentage)
        is_saturated: True 表示超过 SATURATION_THRESHOLD
        saturation_percentage: 0.0 ~ 1.0
    """
    consumed = total_consumed_tokens(checkpoint_dir)
    pct = consumed / MAX_CONTEXT_TOKENS if MAX_CONTEXT_TOKENS > 0 else 0
    return pct >= SATURATION_THRESHOLD, pct


def auto_split(
    stage: str,
    text: str,
    checkpoint_dir: str = "outputs/checkpoints/"
) -> bool:
    """快捷方法：估算、记录、判断是否要切割。

    Args:
        stage: 阶段标识
        text: 刚产生的文本输出
        checkpoint_dir: 检查点目录

    Returns:
        True 表示需要切割，False 表示继续
    """
    tokens = estimate_tokens(text)
    write_context_log(stage, tokens, checkpoint_dir)
    saturated, pct = check_saturation(checkpoint_dir)
    return saturated


def reset_context_log(checkpoint_dir: str = "outputs/checkpoints/"):
    """清空上下文日志（新会话开始时调用）。"""
    log_path = Path(checkpoint_dir) / "_context_log.jsonl"
    if log_path.exists():
        log_path.unlink()
