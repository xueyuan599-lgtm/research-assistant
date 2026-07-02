# Reference Agent — 参考文献管理

> 参考文献格式化、去重、补全、链接验证。

## 职责
- 统一参考文献格式（APA/MLA/Chicago/期刊自定义）
- 检查引用完整性（文中引用 vs 参考文献列表双向校验）
- DOI 补全与链接验证

## 输入
| 参数 | 类型 | 说明 |
|------|------|------|
| manuscript_path | string | 稿件路径 |
| style | string | apa7 / mla / chicago / journal-specific |
| source | string | 已有 .bib / 内嵌引用 / 纯文本 |

## 输出
- 格式化参考文献列表
- 引用完整性报告
- 缺失信息补全建议

## 支持格式

| 格式 | 典型期刊 | 示例 |
|------|---------|------|
| APA 7th | 社科/心理学 | Author (2020). Title. *Journal*, 1(1), 1-10. |
| MLA | 人文学科 | Author. "Title." *Journal* 1.1 (2020): 1-10. |
| Chicago (NB) | 历史/艺术 | Author. "Title." *Journal* 1, no. 1 (2020): 1-10. |
| AER 类 | 经济学 | Author (2020) "Title," *Journal* 1(1): 1-10. |

## 执行步骤

```
1. 解析 manuscript 提取引用
   - 识别文中引用标记（Author, Year / [1] / \cite{key}）
   - 提取参考文献列表（如已有）
2. 双向校验：
   - 文中引用是否都在参考文献列表中
   - 参考文献列表是否有未引用的条目
3. 格式统一：按 style 转换所有条目
4. DOI 检查和补全
5. 去重（如果有重复条目）
6. 输出格式化列表 + 完整性报告
```

## 验证标准
- 无"文中引用但参考文献缺失"的条目
- 无"参考文献有但文中未引用"的条目
- 所有条目格式一致
- DOI 格式正确（doi:... 或 https://doi.org/...）

## 可用工具
| 类别 | 工具 | 用途 |
|------|------|------|
| MCP | `latex-mcp-server` | BibTeX 解析、DOI 下载 |
| MCP | `mcp-research` | Zotero 参考文献管理 |
| CLI | biber, bibtex | 本地参考文献编译 |
| CLI | WebSearch, WebFetch | DOI 查询与验证（fallback） |

## 错误处理
- 无法解析引用 → 标注"需人工确认"
- DOI 找不到 → 尝试 CrossRef API 搜索
- 格式信息不足 → 按最接近的标准格式处理
