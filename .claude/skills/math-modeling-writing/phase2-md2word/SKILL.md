---
name: md2word
description: 把 math-modeling-writing 产出的国赛数学建模 Markdown 论文稿无损转换为 Word（.docx，WPS 可打开）。按 typography.md 国赛版面生成黑体标题、宋体正文、1.5 倍行距、三线表、公式居中右编号；自动渲染 PNG 用 vision 逐页验证，再经 WPS 打开复核。当用户要求 md 转 Word、论文草稿转 docx、生成国赛格式 Word 时使用。
---

# md2word：Markdown → 国赛 Word（WPS 可打开）

## 何时使用

- 用户给出 math-modeling-writing（或同方言）产出的 `.md` 论文稿，要求转成 Word。
- 要求国赛版面（黑体/宋体/Times、1.5 倍行距、三线表、公式居中右编号）且 WPS 能打开。
- 需要“转换 + 无损校验 + 渲染 vision 验证 + WPS 复核”的完整闭环。

## 工作流（每次必须走完）

1. **转换**：`python scripts/md2word.py <输入.md> --out <输出.docx>`
2. **无损校验**：`python scripts/verify.py <输入.md> <输出.docx>`，失败必须修。
3. **渲染 + vision QA**：`python scripts/qa_render_vision.py <输出.docx>`，任一页 FAIL 必须修后重渲染，直到全 PASS。
4. **WPS 复核**：`powershell -ExecutionPolicy Bypass -File scripts/wps_check.ps1 <输出.docx>`，确认无修复提示、页数合理。
5. 交付 docx 时附“已剔除清单”（`<输出>.removed.json`，若默认剔除了草稿痕迹）。

## 依赖（本机已就绪）

本机 Claude Code 环境依赖全部就绪，无需额外安装：

- **pandoc 内核**：全局 Python 已装 `pypandoc`，可直接定位 pandoc 3.9（`python -c "import pypandoc; print(pypandoc.get_pandoc_version())"`）
- **python-docx / lxml**：全局已装
- **vision QA 渲染**：LibreOffice（`C:\Program Files\LibreOffice\program\soffice.exe`）、texlive `pdftoppm`、node、`~/.agents/skills/vision/vision.mjs` 均已就绪；若 vision 调用不可用，`qa_render_vision.py --skip-vision` 可降级为仅渲染出图，由 Claude 读 PNG 逐页复核。

若在**其他机器**上遇到 `缺少 pypandoc`，将依赖装进本技能 `lib/` 目录（`md2word.py` 会自动将其加入 `sys.path`，自包含可离线）：

```powershell
python -m pip install --target "phase2-md2word\lib" pypandoc_binary python-docx lxml
```

> `pypandoc_binary` 自带 pandoc 内核（约 30MB，联网一次），装后离线可用。

## 转换规则（与 math-modeling-writing 的 typography.md 一致）

- 页面：A4，上下 2.54cm、左右 3.17cm；全文 1.5 倍行距；无页码。
- 标题：一级标题自动补中文数字编号“一、二、……”（题目、摘要标题、参考文献、附录除外）；二级/三级标题沿用源稿自身编号，转换时不做修改。
- 样式：题目（黑体三号居中）、摘要标题（黑体四号居中）、摘要正文（宋体/Times 小四、两端对齐、首行缩进 2 字符、段后 5 磅）、正文（同摘要正文，段后 5 磅）、一级标题（黑体四号居中）、二级/三级标题（黑体小四居左，段前后 0.5 行）、图注/表注（五号宋体/Times 加粗居中，图注在下、表注在上）、表格文字（基于无样式、宋体/Times 五号（内容过宽自动降小五）、居中、段前 0.5 行、段后 0、行距 1.5 倍、表头加粗）、代码（五号宋体/Consolas 浅灰底纹，中文注释仍宋体）。全文西文与数字统一 Times New Roman（附录代码英文与数字用 Consolas）。
- 公式：`$$...\tag{N}$$` 剥离 `\tag` 后经 pandoc 转 Word 原生 OMML；公式段落采用“居中制表位 + 右制表位”布局，公式主体围绕居中制表位居中，编号 `(N)` 由右制表位推到右侧，段落无首行缩进；默认校验编号从 1 连续，并审计居中制表位、右制表位与公式前制表符是否齐全。
- 表格：转三线表（顶/底线 1.5 磅、表头下线 1.0 磅），水平居中。
- 图片：`![图注](路径)` 居中嵌入，图注转“图注”样式；`【图占位：…】` 保留为浅灰占位框。
- 草稿痕迹（默认剔除并输出清单）：引用来源 `>`、`修改说明`、`自检对照/自检清单`、`可选升级/待确认` 等小节；`--keep-editorial` 可全部保留。

## 命令行

```powershell
python scripts/md2word.py input.md --out out.docx [--title "题目"] [--keep-editorial] [--no-validate-tags]
python scripts/verify.py input.md out.docx [--report report.json] [--keep-editorial] [--no-validate-tags]
python scripts/qa_render_vision.py out.docx [--outdir work/qa] [--skip-render] [--skip-vision]
powershell -ExecutionPolicy Bypass -File scripts/wps_check.ps1 out.docx
```

## 测试

- 合成全元素样例：`tests/fixtures/full_sample.md`。
- 验收门槛：verify 零差异（剔除项除外）；公式均为 OMML、编号连续且满足“居中制表位居中、右制表位编号、公式前制表符”布局；表格三线；WPS 打开无修复提示。

## 注意事项

- 只支持阶段一实际使用的 md 方言；未知语法按文本保留，不做猜测式转换。
- 修改 typography.md 后需同步更新本技能 `scripts/md2word.py` 中的样式常量。
