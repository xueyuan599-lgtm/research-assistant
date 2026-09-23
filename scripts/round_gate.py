#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""复核轮次闸门 —— 让「再复核一轮」在工具层无法自行开启。

动机：2026 CUMCM C 题在 9-13 一天跑了约 29 轮全文修订（快照 `round33`~`round61`），
当天 `checkpoints/节点状态记录.md` 一字未写（最后修改停在 09-12 19:36）。
失控的机制不是「轮次多」，而是三个条件同时成立：

  ① 判据是「复核到 PASS」，而**全文重扫的搜索空间不缩小** —— 每轮都能找到措辞、
     编号、图表对齐的问题，期望发现数不随轮次衰减。这是个常返过程，无不动点。
  ② 记账是「顺便做的」，在「还有问题要修」的压力下第一个被牺牲 —— 于是轮次不可见。
  ③ 「够不够好了」由正在找问题的那一方判断 —— 它没有能力说够了。

本脚本用两道机械前置把 ① 和 ② 关掉：

  **前置一（治①）**　`begin` 要求账本里存在**已登记的未修 FAIL 项**。
      没有清单就无法开启新一轮 —— 「再扫一遍看看还有没有问题」在工具层被拒绝（退出码 1）。
      这样循环从「常返」变成「单调递减」：清单一空，循环自动结束。

  **前置二（治②）**　轮次计数与状态记录行由**本脚本**产出，不由模型顺手写。
      不跑 `close`，该轮永不闭合，下一轮 `begin` 直接失败（退出码 3）。

③ 无法用脚本关闭 —— 那需要人。脚本能做的只是把人的介入点变得**不可回避**：
轮次触顶时 `begin` 拒绝执行（退出码 2），只能由 `signoff` 显式放行。

用法：
    python scripts/round_gate.py open  --ws <run_dir> --target S3 --id F1 \\
        --where "paper/sections/05-问题二.md:88" --what "公式编号跳号"
    python scripts/round_gate.py begin --ws <run_dir> --target S3
    python scripts/round_gate.py close --ws <run_dir> --target S3 --fixed F1 \\
        --artifact paper/sections/05-问题二.md --tokens ~64k
    python scripts/round_gate.py status --ws <run_dir> --target S3 [--json]
    python scripts/round_gate.py check --ws <run_dir> --target S3    # 自检用，只给退出码

退出码：
    0 正常
    1 无可复核项（循环应结束）
    2 轮次超限（需 signoff 放行，或按规则升级团队）
    3 上一轮未闭合 / 无进行中的轮次
    4 账本或参数错误

> **状态记录表头是本地镜像**：唯一出处为 `knowledge/mcm/templates/node-checklists.md`
> §节点/环节状态记录模板。该文件改动时须同步本脚本的 `RECORD_HEADER`。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# 同一交付物/节点的全文复核上限（含首轮全文）。
# 见 .claude/rules/06-cost-discipline.md「审查轮次封顶」——全局唯一次数口径。
ROUND_MAX = int(os.environ.get("ROUND_MAX", "3"))

# 终稿自收为 1 轮（critic 一次性输出全表 → 修复后只复核 FAIL → 仍不通过即升级团队）。
# 硬编码，不随环境变量放宽：这是「自收更严」，不是可协商项。
# S7 同理：期刊赛道的结构审稿三判据一次跑完记为 1 轮，FAIL 才按判据分轮（S7-1/S7-2/S7-3）。
# 口径唯一源见 knowledge/writing/structure-review-protocol.md §3。
TARGET_MAX_FIXED = {"N7": 1, "S7": 1}

LEDGER_NAME = "复核轮次账.jsonl"
RECORD_NAME = "节点状态记录.md"

# 镜像自 knowledge/mcm/templates/node-checklists.md §节点/环节状态记录模板
RECORD_HEADER = (
    "| 节点/环节 | 产出文件 | 检查核验 | 状态 | 时间 | tokens | 退回对象 |\n"
    "|----------|---------|---------|------|------|--------|---------|\n"
)


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def _stamp() -> str:
    return datetime.now().strftime("%m-%d %H:%M")


def _ledger(ws: Path) -> Path:
    return ws / "checkpoints" / LEDGER_NAME


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    events = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"[4] 账本第 {lineno} 行不是合法 JSON：{exc}")
    return events


def _append(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")


def state(events: list[dict], target: str) -> dict:
    """从账本事件流还原该 target 的当前状态。"""
    rounds_started = 0
    unclosed = False
    signoffs = 0
    open_items: dict[str, dict] = {}
    last_round_items: list[str] = []

    for ev in events:
        if ev.get("target") != target:
            continue
        kind = ev.get("ev")
        if kind == "open":
            open_items[ev["id"]] = {"where": ev.get("where", ""), "what": ev.get("what", "")}
        elif kind == "begin":
            rounds_started += 1
            unclosed = True
            last_round_items = list(ev.get("items", []))
        elif kind == "close":
            unclosed = False
            for fid in ev.get("fixed", []):
                open_items.pop(fid, None)
        elif kind == "signoff":
            signoffs += 1

    base = TARGET_MAX_FIXED.get(target, ROUND_MAX)
    return {
        "target": target,
        "rounds_started": rounds_started,
        "rounds_allowed": base + signoffs,
        "base_max": base,
        "signoffs": signoffs,
        "unclosed": unclosed,
        "current_round": rounds_started,
        "open_items": open_items,
        "last_round_items": last_round_items,
    }


def _print_state(st: dict) -> None:
    print(
        f"目标 {st['target']}：已开 {st['rounds_started']}/{st['rounds_allowed']} 轮"
        + (f"（含 signoff {st['signoffs']} 次放行）" if st["signoffs"] else "")
        + ("，**有一轮未闭合**" if st["unclosed"] else "")
    )
    if st["open_items"]:
        print(f"未修 FAIL 项 {len(st['open_items'])} 条：")
        for fid, item in st["open_items"].items():
            print(f"  [{fid}] {item['where']} — {item['what']}")
    else:
        print("未修 FAIL 项 0 条 —— **已收敛，循环应结束**")


def _write_record_row(ws: Path, st: dict, artifact: str, tokens: str, verdict: str) -> str:
    record = ws / "checkpoints" / RECORD_NAME
    if not record.exists():
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(
            "# 节点状态记录\n\n"
            "> 表头镜像自 `knowledge/mcm/templates/node-checklists.md` §节点/环节状态记录模板，以该文件为准。\n"
            "> 只追加行，不重写全文（`.claude/rules/06-cost-discipline.md`）。\n\n" + RECORD_HEADER,
            encoding="utf-8",
        )
    remaining = len(st["open_items"])
    passed = remaining == 0
    status = "PASS" if passed else "FAIL"
    back_to = "-" if passed else "见待修清单"
    row = (
        f"| {st['target']} | {artifact} | {verdict} | {status} | {_stamp()} "
        f"| {tokens} | {back_to} |\n"
    )
    with record.open("a", encoding="utf-8") as fh:
        fh.write(row)
    return row.strip()


# --------------------------------------------------------------------------- 命令


def cmd_open(args) -> int:
    ws = Path(args.ws)
    path = _ledger(ws)
    events = _load(path)
    if args.id in state(events, args.target)["open_items"]:
        print(f"[4] FAIL 项 {args.id} 已登记且未修", file=sys.stderr)
        return 4
    _append(path, {
        "t": _now(), "ev": "open", "target": args.target,
        "id": args.id, "where": args.where, "what": args.what,
    })
    print(f"已登记 FAIL 项 [{args.id}] {args.where} — {args.what}")
    return 0


def cmd_begin(args) -> int:
    ws = Path(args.ws)
    path = _ledger(ws)
    st = state(_load(path), args.target)

    if st["unclosed"]:
        print(
            f"[3] 第 {st['current_round']} 轮尚未闭合 —— 先跑 close，不得开新一轮。\n"
            f"    （这正是 9-13 那天的失效模式：轮次开了不记账）",
            file=sys.stderr,
        )
        return 3

    if not st["open_items"]:
        print(
            "[1] 账本里没有未修 FAIL 项 —— 循环应结束，不得为「再扫一遍」而开轮。\n"
            "    若确有新发现，先用 `open` 登记成条目，再 begin。",
            file=sys.stderr,
        )
        return 1

    if st["rounds_started"] >= st["rounds_allowed"]:
        print(
            f"[2] 已达复核上限 {st['rounds_allowed']} 轮（同一交付物/节点）。\n"
            f"    按规则：汇总 FAIL 清单 **升级团队人工介入**；\n"
            f"    确需继续须显式放行：round_gate.py signoff --target {args.target} --by <姓名> --reason <理由>",
            file=sys.stderr,
        )
        return 2

    items = sorted(st["open_items"])
    _append(path, {
        "t": _now(), "ev": "begin", "target": args.target,
        "round": st["rounds_started"] + 1, "items": items,
    })
    print(f"第 {st['rounds_started'] + 1}/{st['rounds_allowed']} 轮开始，本轮只复核 {len(items)} 条 FAIL 项：")
    for fid in items:
        print(f"  [{fid}] {st['open_items'][fid]['where']} — {st['open_items'][fid]['what']}")
    return 0


def cmd_close(args) -> int:
    ws = Path(args.ws)
    path = _ledger(ws)
    events = _load(path)
    st = state(events, args.target)

    if not st["unclosed"]:
        print("[3] 当前没有进行中的轮次 —— close 无从对应", file=sys.stderr)
        return 3

    fixed = list(args.fixed or [])
    unknown = [f for f in fixed if f not in st["last_round_items"]]
    if unknown:
        print(
            f"[4] 以下条目不在本轮输入清单里，不得计入修复：{', '.join(unknown)}\n"
            f"    本轮清单：{', '.join(st['last_round_items']) or '（空）'}",
            file=sys.stderr,
        )
        return 4

    # 本轮新发现先登记成 open，再闭合 —— 保证下一轮的输入仍是显式清单。
    for spec in args.new or []:
        parts = spec.split("|")
        if len(parts) != 3:
            print(f"[4] --new 需为 id|位置|问题 三段：{spec}", file=sys.stderr)
            return 4
        fid, where, what = (p.strip() for p in parts)
        if fid in st["open_items"]:
            print(f"[4] 新条目 {fid} 与未修项重名", file=sys.stderr)
            return 4
        _append(path, {
            "t": _now(), "ev": "open", "target": args.target,
            "id": fid, "where": where, "what": what,
        })

    _append(path, {
        "t": _now(), "ev": "close", "target": args.target,
        "round": st["current_round"], "fixed": fixed,
    })

    st2 = state(_load(path), args.target)
    print(f"第 {st['current_round']} 轮闭合：修复 {len(fixed)} 条，剩余 {len(st2['open_items'])} 条未修")

    converged = not st2["open_items"]
    exhausted = st2["rounds_started"] >= st2["rounds_allowed"]
    if converged or exhausted:
        verdict = args.verdict or (
            f"{st2['rounds_started']} 轮收敛" if converged
            else f"{st2['rounds_started']}/{st2['rounds_allowed']} 轮未收敛"
        )
        row = _write_record_row(ws, st2, args.artifact, args.tokens, verdict)
        print(f"状态记录追加行：{row}")
    else:
        print("仍有未修项 —— 可再 begin（受轮次上限约束）")

    return 0


def cmd_signoff(args) -> int:
    ws = Path(args.ws)
    path = _ledger(ws)
    st = state(_load(path), args.target)
    _append(path, {
        "t": _now(), "ev": "signoff", "target": args.target,
        "by": args.by, "round": st["rounds_started"], "reason": args.reason,
    })
    print(
        f"已放行 {args.target} 第 {st['rounds_started'] + 1} 轮（放行人：{args.by}；理由：{args.reason}）"
    )
    return 0


def cmd_status(args) -> int:
    ws = Path(args.ws)
    events = _load(_ledger(ws))
    targets = [args.target] if args.target else sorted(
        {ev["target"] for ev in events if ev.get("target")}
    )
    if not targets:
        print("账本为空")
        return 0
    if args.json:
        print(json.dumps({t: state(events, t) for t in targets}, ensure_ascii=False, indent=2))
        return 0
    for i, t in enumerate(targets):
        if i:
            print()
        _print_state(state(events, t))
    return 0


def cmd_check(args) -> int:
    """自检闸门：只在「已收敛」或「已触顶升级」时放行，否则退非零。"""
    ws = Path(args.ws)
    st = state(_load(_ledger(ws)), args.target)
    if st["unclosed"]:
        print(f"[3] {args.target} 有未闭合的轮次", file=sys.stderr)
        return 3
    if not st["open_items"]:
        print(f"{args.target} 已收敛（{st['rounds_started']} 轮）")
        return 0
    if st["rounds_started"] >= st["rounds_allowed"]:
        print(
            f"{args.target} 已触顶（{st['rounds_started']}/{st['rounds_allowed']} 轮），"
            f"剩 {len(st['open_items'])} 条未修 —— 应升级团队，不再开轮"
        )
        return 0
    print(
        f"[1] {args.target} 既未收敛也未触顶，仍有 {len(st['open_items'])} 条未修 —— 循环未走完",
        file=sys.stderr,
    )
    return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="复核轮次闸门：无 FAIL 清单不得开轮，轮次计数由脚本记账")
    ap.add_argument("--ws", default=".", help="题目工作区，如 outputs/mcm_CUMCM2026_20260918")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("open", help="登记一条 FAIL 项")
    p.add_argument("--target", required=True, help="交付物/环节标识，如 S3 或 N7")
    p.add_argument("--id", required=True, help="条目号，如 F1")
    p.add_argument("--where", required=True, help="位置，如 paper/sections/05-问题二.md:88")
    p.add_argument("--what", required=True, help="问题描述")
    p.set_defaults(func=cmd_open)

    p = sub.add_parser("begin", help="开启一轮复核（需已有未修 FAIL 项且未触顶）")
    p.add_argument("--target", required=True)
    p.set_defaults(func=cmd_begin)

    p = sub.add_parser("close", help="闭合当前轮，并追加状态记录行")
    p.add_argument("--target", required=True)
    p.add_argument("--fixed", nargs="*", default=[], help="本轮修掉的条目号")
    p.add_argument("--new", nargs="*", default=[], help="本轮新发现，格式 id|位置|问题")
    p.add_argument("--artifact", default="-", help="产出文件（写状态记录用）")
    p.add_argument("--tokens", default="-", help="该环节估算 tokens（写状态记录用）")
    p.add_argument("--verdict", help="覆盖「检查核验」列的自动措辞")
    p.set_defaults(func=cmd_close)

    p = sub.add_parser("signoff", help="人工放行一轮（触顶后唯一出路）")
    p.add_argument("--target", required=True)
    p.add_argument("--by", required=True, help="放行人")
    p.add_argument("--reason", required=True, help="放行理由")
    p.set_defaults(func=cmd_signoff)

    p = sub.add_parser("status", help="打印账本状态")
    p.add_argument("--target")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("check", help="自检闸门，只给退出码")
    p.add_argument("--target", required=True)
    p.set_defaults(func=cmd_check)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    sys.exit(main())
