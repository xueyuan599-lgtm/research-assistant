# Idea2Top 项目操作网页

> 数学建模半自动项目（Idea2Top）的交互式展示页面。
> 水墨画风格 · 双语支持 · 交互式架构可视化

## 📁 文件结构

```
project-webpage/
├── index.html          # 主页面（单页应用，所有内容在一页内）
├── css/
│   └── style.css       # 水墨画主题样式（配色、布局、动画、响应式）
├── js/
│   ├── i18n.js         # 双语翻译数据（中文/英文，集中管理）
│   └── main.js         # 交互逻辑（语言切换、滚动动画、架构图交互）
└── README.md           # 本文件 — 操作说明
```

## 🚀 打开方式

### 方式一：直接打开（推荐）

双击 `index.html` 在浏览器中打开即可。

支持所有现代浏览器：Chrome / Firefox / Edge / Safari。

### 方式二：本地服务器（功能完整）

使用 Python 启动本地服务器以获得最佳体验：

```bash
# 进入项目目录
cd outputs/project-webpage/

# Python 3
python -m http.server 8080

# 然后浏览器访问 http://localhost:8080
```

## 🌐 功能特性

| 特性 | 说明 |
|------|------|
| 🎨 **水墨画主题** | 宣纸底色、墨色渐变、朱砂印章点缀，中国画风格 |
| 🌍 **双语切换** | 右上角「中 / EN」按钮一键切换，所有内容同步翻译 |
| 🖱️ **交互式架构图** | 点击 Agent 卡片展开详情，查看子 Agent 列表 |
| 📜 **滚动动画** | 页面滚动时内容渐入，流畅的 Reveal 动效 |
| 📱 **响应式设计** | 桌面、平板、手机均可自适应显示 |
| 🔍 **平滑导航** | 顶部导航栏点击平滑滚动到对应区域 |

## 📖 页面导航

1. **首页 (Home)** — 项目大屏展示，带水墨动画背景
2. **项目简介 (About)** — 项目定位、核心数据统计
3. **系统架构 (Architecture)** — 三层 Agent 架构交互图
4. **工作流 (Workflow)** — 从想法到交付的六阶段流水线
5. **智能体 (Agents)** — 30+ 专业 Agent 卡片展示
6. **知识库 (Knowledge)** — 算法与方法的分类索引
7. **使用指南 (Guide)** — 三步快速上手

## 🎨 主题定制

如需修改配色，编辑 `css/style.css` 中的 CSS 变量：

```css
:root {
  --paper: #f7f3ed;        /* 宣纸底色 */
  --ink-deep: #1a1a1a;     /* 深墨色 */
  --seal-red: #c43a31;     /* 朱砂红（印章色） */
  --mountain-green: #5a8a6a; /* 青绿 */
  --indigo-blue: #4a6a8a;  /* 花青 */
}
```

如需新增或修改翻译内容，编辑 `js/i18n.js` 中的 `I18N_DATA` 对象。

## 🌍 添加新语言

1. 在 `js/i18n.js` 的 `I18N_DATA` 中添加新语言键（如 `ja`）
2. 复制中文或英文的结构，填写对应翻译
3. 在 `index.html` 的 `.lang-toggle` 中添加新按钮

## 💻 技术栈

- **纯原生**：无框架依赖，无需构建工具
- **HTML5** + **CSS3** (Flexbox, Grid, Custom Properties, Animations)
- **Vanilla JavaScript** (Intersection Observer, Mutation Observer, i18n)

## 📋 浏览器兼容

| Chrome | Firefox | Safari | Edge |
|--------|---------|--------|------|
| ✅ 最新 | ✅ 最新 | ✅ 最新 | ✅ 最新 |

## 🔗 相关链接

- 项目根目录：`E:\wuyi\数学建模半自动\`
- Agent 定义：`research-assistant/agents/`
- 知识库：`research-assistant/knowledge/`
- 项目规则：`research-assistant/.claude/rules/`

---

*Idea2Top — 从想法到顶刊的自动化旅程*
