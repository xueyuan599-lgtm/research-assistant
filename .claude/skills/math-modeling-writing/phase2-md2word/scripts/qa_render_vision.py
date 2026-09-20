# -*- coding: utf-8 -*-
"""渲染 docx -> PDF -> PNG，并用 vision 技能逐页 QA。

用法：
  python qa_render_vision.py out.docx [--outdir work/qa_x]
      [--prompt-file qa_prompt.txt] [--model kimi-k2.6]
      [--skip-render] [--skip-vision] [--dpi 110]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


DEFAULT_SOFFICE = r"C:\Program Files\LibreOffice\program\soffice.exe"
DEFAULT_PDFTOPPM = r"C:\Users\lenovo\.cache\codex-runtimes\codex-primary-runtime\dependencies\bin\override\pdftoppm.cmd"
TEXLIVE_PDFTOPPM = r"D:\TEX\texlive\2024\bin\windows\pdftoppm.exe"
TEXLIVE_PDFTOCAIRO = r"D:\TEX\texlive\2024\bin\windows\pdftocairo.exe"
DEFAULT_VISION = Path(r"C:\Users\lenovo\.agents\skills\vision\vision.mjs")
DEFAULT_NODE = r"C:\Program Files\nodejs\node.exe"

DEFAULT_PROMPT_TEMPLATE = (
    "这是同一份国赛数学建模论文渲染出的第{page}页。请检查排版质量：\n"
    "1. 无文字溢出、重叠、裁剪、明显留白异常或图片变形；\n"
    "2. 公式无乱码、无缺字（豆腐块/方框/乱码符号）；公式居中、带编号的编号在右侧；\n"
    "3. 表格为三线表（只有顶线、表头下线、底线），无多余竖线或全框线；\n"
    "4. 标题居中、各级标题层级清晰；\n"
    "5. 图注在图下方、表注在表上方；\n"
    "【图占位】灰色占位框按原样保留属正常，不算缺陷。\n"
    "只输出一行：PASS 或 FAIL：简短原因。禁止输出其他任何内容。"
)


def find_soffice() -> str | None:
    p = Path(DEFAULT_SOFFICE)
    if p.exists():
        return str(p)
    return shutil.which("soffice")


def find_pdftoppm() -> tuple[str, list[str]] | None:
    for p in (Path(TEXLIVE_PDFTOPPM), Path(DEFAULT_PDFTOPPM)):
        if p.exists():
            return str(p), ["-png"]
    for name in ("pdftoppm", "pdftocairo"):
        found = shutil.which(name)
        if found:
            return found, ["-png"]
    if Path(TEXLIVE_PDFTOCAIRO).exists():
        return str(Path(TEXLIVE_PDFTOCAIRO)), ["-png"]
    return None


def render_pdf(docx: Path, out_dir: Path) -> Path:
    soffice = find_soffice()
    if not soffice:
        raise RuntimeError("找不到 LibreOffice soffice，无法渲染")
    profile = out_dir / "lo_profile"
    cmd = [
        soffice, "--headless", "--norestore",
        f"-env:UserInstallation=file:///{profile.as_posix()}",
        "--convert-to", "pdf", "--outdir", str(out_dir), str(docx),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=180)
    pdf = out_dir / (docx.stem + ".pdf")
    if not pdf.exists():
        raise RuntimeError(f"soffice 转换失败：{proc.stderr[-800:]}")
    return pdf


def rasterize(pdf: Path, out_dir: Path, dpi: int) -> list[Path]:
    found = find_pdftoppm()
    if not found:
        raise RuntimeError("找不到 pdftoppm/pdftocairo，无法出 PNG")
    tool, opts = found
    prefix = out_dir / "page"
    args = [*opts, "-r", str(dpi), str(pdf), str(prefix)]
    cmd = [tool, *args]
    if tool.lower().endswith(".cmd"):
        cmd = ["cmd", "/c", tool, *args]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=180)
    pages = sorted(out_dir.glob("page-*.png"))
    if not pages:
        raise RuntimeError(f"PNG 渲染失败：{proc.stderr[-800:]}")
    return pages


def run_vision(pages: list[Path], prompt: str, model: str | None,
               vision_script: Path | None = None, node: str | None = None,
               max_tokens: int = 8192) -> str:
    vs = vision_script or DEFAULT_VISION
    node_bin = node or DEFAULT_NODE
    if not Path(node_bin).exists():
        raise RuntimeError(f"找不到 node：{node_bin}")
    if not vs.exists():
        raise RuntimeError(f"找不到 vision 脚本：{vs}")
    cmd = [node_bin, str(vs), *[str(p) for p in pages], "--prompt", prompt]
    if model:
        cmd += ["--model", model]
    cmd += ["--max-tokens", str(max_tokens)]
    last_err = ""
    for attempt in range(3):
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=600)
        if proc.returncode == 0:
            return proc.stdout
        last_err = proc.stderr[-800:]
        if "429" not in last_err and "overloaded" not in last_err:
            break
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"vision 调用失败：{last_err}")


def parse_single_verdict(text: str) -> tuple[str | None, str]:
    """从单页输出中取最后一次 PASS/FAIL 判定。"""
    verdict = None
    reason = ""
    for line in text.splitlines():
        m = re.match(
            r"\s*(?:判定[:：]\s*)?(PASS|FAIL|通过|未通过)\b(.*)",
            line, re.IGNORECASE,
        )
        if not m:
            continue
        verdict = "PASS" if m.group(1).upper() in ("PASS", "通过") else "FAIL"
        reason = re.sub(r"^[:：\s]+", "", m.group(2)).strip()
    return verdict, reason


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    parser = argparse.ArgumentParser(description="docx 渲染 + vision 逐页 QA")
    parser.add_argument("docx", type=Path)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--model", default="kimi-k2.6",
                        help="视觉模型（默认 kimi-k2.6，结构化 QA 更稳定）")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--vision-retries", type=int, default=3,
                        help="每页未拿到判定时的最大重试次数")
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--skip-vision", action="store_true")
    parser.add_argument("--vision-script", type=Path, default=None)
    parser.add_argument("--node", default=None)
    args = parser.parse_args()

    out_dir = args.outdir or Path(
        tempfile.gettempdir()
    ) / f"md2word_qa_{args.docx.stem}"
    out_dir = out_dir.resolve()  # 相对 outdir → 绝对，否则 lo_profile 的 file:/// URL 解析失败
    out_dir.mkdir(parents=True, exist_ok=True)

    pages: list[Path] = []
    pdf_path: Path | None = None
    if not args.skip_render:
        pdf_path = render_pdf(args.docx.resolve(), out_dir)
        pages = rasterize(pdf_path, out_dir, args.dpi)
    else:
        pages = sorted(out_dir.glob("page-*.png"))
        if not pages:
            print("--skip-render 但找不到已有 page-*.png", file=sys.stderr)
            return 2

    report: dict = {
        "docx": str(args.docx),
        "pdf": str(pdf_path) if pdf_path else None,
        "pages": [str(p) for p in pages],
        "vision": None,
    }

    if args.skip_vision:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"\n已渲染 {len(pages)} 页（跳过 vision）")
        return 0

    base_prompt = DEFAULT_PROMPT_TEMPLATE
    if args.prompt_file:
        base_prompt = args.prompt_file.read_text(encoding="utf-8")

    passes = 0
    fails = 0
    inconclusive = 0
    fail_lines: list[str] = []
    page_results: dict[str, dict] = {}
    for idx, page in enumerate(pages, start=1):
        prompt = base_prompt.format(page=idx)
        verdict = None
        reason = ""
        vision_text = ""
        last_error = ""
        for attempt in range(max(1, args.vision_retries)):
            try:
                vision_text = run_vision(
                    [page], prompt, args.model,
                    vision_script=args.vision_script, node=args.node,
                    max_tokens=args.max_tokens,
                )
            except RuntimeError as exc:
                last_error = str(exc)
                time.sleep(2)
                continue
            verdict, reason = parse_single_verdict(vision_text)
            if verdict is not None:
                break
            time.sleep(2)
        if verdict is None:
            if last_error:
                print(f"第{idx}页 vision 调用失败：{last_error}", file=sys.stderr)
                fail_lines.append(f"第{idx}页：vision 调用失败")
            else:
                fail_lines.append(f"第{idx}页：未输出判定")
            inconclusive += 1
        elif verdict == "PASS":
            passes += 1
        else:
            fails += 1
            fail_lines.append(f"第{idx}页：{reason}")
        page_results[str(idx)] = {"verdict": verdict, "reason": reason,
                                  "output": vision_text}

    report["vision"] = {
        "passes": passes,
        "fails": fails,
        "inconclusive": inconclusive,
        "fail_lines": fail_lines,
        "pages": page_results,
    }
    (out_dir / "qa_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"vision 结果：PASS {passes} / FAIL {fails} / 未判定 {inconclusive}"
          f"（共 {len(pages)} 页）")
    if fails:
        for line in fail_lines:
            print("  " + line)
        return 1
    if inconclusive:
        print("警告：部分页面未输出判定，请人工核对 PNG", file=sys.stderr)
        return 2
    print("渲染 + vision QA 通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
