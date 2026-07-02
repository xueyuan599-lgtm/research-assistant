"""检查点协议 — 状态序列化与恢复

当上下文饱和度超过阈值时，当前 Agent 将未完成的工作序列化为检查点，
供后续新 Agent 或新会话读取恢复。

用法:
    from checkpoint import write_checkpoint, read_checkpoint, latest_checkpoint

    # 写入检查点
    write_checkpoint(
        task_id="literature-review",
        state={"completed": ["search"], "pending": ["screening", "synthesis"]},
        params={"query": "DID", "max_results": 50},
        intermediate={"search_results": "outputs/literature/search_results.json"},
        note="搜索完成，上下文饱和，切割至筛选阶段"
    )

    # 读取最新检查点
    cp = latest_checkpoint("outputs/checkpoints/")
    if cp:
        print(f"恢复任务 {cp['task_id']}，已完成 {cp['state']['completed']}")
"""

import json
import os
from pathlib import Path
from datetime import datetime

CHECKPOINT_DIR = "outputs/checkpoints/"


def write_checkpoint(
    task_id: str,
    state: dict,
    params: dict | None = None,
    intermediate: dict | None = None,
    note: str = "",
    checkpoint_dir: str = CHECKPOINT_DIR,
) -> str:
    """写入检查点。

    Args:
        task_id: 任务标识，如 'literature-review'
        state: 状态字典，至少包含 completed/pending 列表
               {"completed": [...], "pending": [...]}
        params: 原始参数
        intermediate: 中间结果文件路径索引
        note: 切割原因备注
        checkpoint_dir: 目录

    Returns:
        检查点文件路径
    """
    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{task_id}_{timestamp}.json"
    filepath = path / filename

    checkpoint = {
        "task_id": task_id,
        "timestamp": timestamp,
        "state": state,
        "params": params or {},
        "intermediate": intermediate or {},
        "note": note,
        "schema_version": "1.0",
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)

    # 同时写入 _latest 指针
    latest_path = path / "_latest.json"
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump({"latest": filename, "task_id": task_id}, f)

    return str(filepath)


def read_checkpoint(filepath: str) -> dict | None:
    """读取指定检查点。"""
    path = Path(filepath)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def latest_checkpoint(checkpoint_dir: str = CHECKPOINT_DIR) -> dict | None:
    """读取最新检查点。"""
    latest_path = Path(checkpoint_dir) / "_latest.json"
    if not latest_path.exists():
        return None
    try:
        ref = json.loads(latest_path.read_text(encoding="utf-8"))
        return read_checkpoint(os.path.join(checkpoint_dir, ref["latest"]))
    except (json.JSONDecodeError, KeyError, FileNotFoundError):
        return None


def list_checkpoints(task_id: str | None = None, checkpoint_dir: str = CHECKPOINT_DIR) -> list[dict]:
    """列出所有检查点，可选按 task_id 过滤。"""
    path = Path(checkpoint_dir)
    if not path.exists():
        return []
    checkpoints = []
    for f in sorted(path.glob("*.json")):
        if f.name.startswith("_"):
            continue
        if task_id and task_id not in f.name:
            continue
        cp = read_checkpoint(str(f))
        if cp:
            checkpoints.append(cp)
    return checkpoints


def resume_from_checkpoint(checkpoint_dir: str = CHECKPOINT_DIR) -> dict | None:
    """从最新检查点恢复任务状态。

    返回检查点内容，调用者据此决定下一步调度。
    """
    cp = latest_checkpoint(checkpoint_dir)
    if cp is None:
        return None
    return cp
