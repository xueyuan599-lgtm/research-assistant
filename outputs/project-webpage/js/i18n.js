/**
 * Idea2Top — 双语翻译数据 (ZH / EN)
 * 所有页面文本集中管理，便于维护和扩展。
 */
const I18N_DATA = {
  zh: {
    nav: {
      home: '首页',
      about: '项目简介',
      architecture: '系统架构',
      workflow: '工作流',
      agents: '智能体',
      knowledge: '知识库',
      guide: '使用指南'
    },
    hero: {
      badge: '研究自动化框架',
      title_cn: '研究想法 → 顶刊实现',
      title_en: 'Idea to Top-Journal Implementation',
      subtitle: '将研究方法想法自动转化为顶刊级可复现代码与实验结果的智能框架',
      btn_start: '探索架构',
      btn_guide: '快速上手'
    },
    about: {
      header_zh: '项目简介',
      header_en: 'About',
      desc1: 'Idea2Top 是一个将研究方法想法自动转化为顶刊级可复现实现的智能框架。用户只需用自然语言描述研究想法，系统即可自动完成文献对标、方案设计、代码生成、实验验证和对抗式 QA 的全流程。',
      desc2: '系统采用多智能体（Multi-Agent）架构，由秘书 Agent 统一入口分解任务，Orchestrator 动态调度各领域 Agent 集群，实现从想法到交付的完全自动化流水线。',
      tagline: '从想法的宣纸起笔，到顶刊的朱砂落款——让每一个研究方法创新都被精确实现。',
      stats: {
        agents: { num: '30+', label: '专业智能体', labelEn: 'Specialized Agents' },
        algorithms: { num: '50+', label: '算法知识库', labelEn: 'Algorithms in Repository' },
        workflows: { num: '6', label: '标准流水线', labelEn: 'Core Workflows' },
        domains: { num: '8', label: '领域覆盖', labelEn: 'Domain Coverage' }
      }
    },
    architecture: {
      header_zh: '系统架构',
      header_en: 'Architecture',
      sub_zh: '多智能体分层协作架构',
      sub_en: 'Multi-Agent Hierarchical Architecture',
      intro: '系统采用三层智能体架构：顶层由秘书 Agent 统一接收任务并分解，中层由 Orchestrator 动态编排管线，底层由各领域专业 Agent 集群执行具体任务。',
      secretary: {
        title: 'Secretary',
        zh: '秘书 Agent',
        desc: '任务分解守门人',
        agents: ['任务分解', '用户确认', 'Orchestrator 调度']
      },
      orchestrator: {
        title: 'Orchestrator',
        zh: '总协调人',
        desc: '意图识别 & 管线编排',
        agents: ['路由调度', '上下文管理', '质量门禁']
      },
      literature: { title: 'Literature', zh: '文献智能体', desc: '文献检索与综述', count: '3' },
      topic: { title: 'Topic Analysis', zh: '选题分析', desc: '前沿探测与选题推荐', count: '3' },
      dataviz: { title: 'Data & Viz', zh: '数据处理与可视化', desc: '清洗、建模、出版级图表', count: '4' },
      experiment: { title: 'Experiment', zh: '实验设计', desc: '方案设计、优化、模拟', count: '3' },
      algorithm: { title: 'Algorithm', zh: '算法创造', desc: '形式化→设计→实现→验证', count: '5' },
      paper: { title: 'Paper Format', zh: '论文格式', desc: '模板、参考文献、合规', count: '3' },
      researchqa: { title: 'Research QA', zh: '科研问答', desc: '方法解释、公式推导', count: '3' },
      kaggle: { title: 'Kaggle', zh: '竞赛', desc: '端到端竞赛流水线', count: '7' },
      knowledge: { title: 'Knowledge', zh: '知识库管理', desc: '算法沉淀与检索', count: '1' }
    },
    workflow: {
      header_zh: '工作流',
      header_en: 'Workflow',
      sub_zh: '从想法到交付的完整流水线',
      sub_en: 'End-to-End Pipeline from Idea to Delivery',
      steps: [
        {
          title: '解析与对标',
          titleEn: 'Analysis & Benchmarking',
          desc: '确认用户想法的核心方法，检索该方法在顶刊中的标准实现规范，汇总方法标准供后续使用。',
          agents: ['literature/search-agent', 'literature/screening-agent', 'literature/synthesis-agent']
        },
        {
          title: '方案设计',
          titleEn: 'Design',
          desc: '输出完整数学方案：模型设定、识别策略、估计方法、假设条件。用户确认后进入下一阶段。',
          agents: ['algorithm/formalizer-agent', 'algorithm/designer-agent']
        },
        {
          title: '实现',
          titleEn: 'Implementation',
          desc: '基于设计方案生成顶刊级可运行代码，包含完整的测试与使用示例。',
          agents: ['algorithm/coder-agent']
        },
        {
          title: '实验与验证',
          titleEn: 'Experiment & Validation',
          desc: '运行完整实验，进行物理验证确保代码可运行、结果合理。不信任 AI 口头报告，必须真实运行。',
          agents: ['algorithm/benchmark-agent', 'algorithm/validator-agent']
        },
        {
          title: '对抗式 QA',
          titleEn: 'Adversarial QA',
          desc: '独立的 Critic Agent 对结果进行严格审查。最多 5 轮循环，Critic 输出 APPROVED 则终止。',
          agents: ['algorithm/validator-agent (critic)']
        },
        {
          title: '交付',
          titleEn: 'Delivery',
          desc: '输出完整交付包：代码 + 结果图表 + 方法描述 + 复现说明，经质量门禁评分后交付。',
          agents: ['知识库入库', '质量门禁 ≥90 分']
        }
      ]
    },
    agents: {
      header_zh: '智能体集群',
      header_en: 'Agent Cluster',
      sub_zh: '30+ 专业智能体覆盖科研全流程',
      sub_en: '30+ Specialized Agents Covering the Full Research Pipeline',
      list: [
        {
          icon: '🔐',
          name: 'Secretary Agent',
          zh: '秘书 Agent',
          desc: '所有用户任务的唯一入口。分析任务、输出分解方案、强制等待用户确认，确保所有工作开始前有清晰的规划和范围界定。'
        },
        {
          icon: '🎯',
          name: 'Orchestrator',
          zh: '总协调人',
          desc: '动态意图识别 + Agent 路由 + 管线编排 + 上下文管理。监控上下文饱和度，超 50% 自动切割调度新 Agent。'
        },
        {
          icon: '📚',
          name: 'Literature Agents',
          zh: '文献智能体',
          desc: '文献检索、筛选与综述合成。支持多数据库检索、质量筛选、自动综述生成。',
          tags: ['search', 'screening', 'synthesis']
        },
        {
          icon: '🔍',
          name: 'Topic Analysis Agents',
          zh: '选题分析智能体',
          desc: '研究前沿探测、研究空白识别、选题推荐。使用 bibliometrix 进行文献计量分析，生成选题热力图。',
          tags: ['frontier', 'gap', 'recommendation']
        },
        {
          icon: '📊',
          name: 'Data & Viz Agents',
          zh: '数据处理与可视化',
          desc: '数据清洗与预处理、统计建模与分析、出版级学术可视化、结果解读。支持多种数据格式与图表类型。',
          tags: ['cleaning', 'modeling', 'visualization', 'interpretation']
        },
        {
          icon: '🧪',
          name: 'Experiment Agents',
          zh: '实验设计智能体',
          desc: '实验方案设计、参数优化与敏感性分析、模拟实验运行。支持自动化超参数搜索与结果验证。',
          tags: ['design', 'optimization', 'simulation']
        },
        {
          icon: '⚗️',
          name: 'Algorithm Agents',
          zh: '算法创造智能体',
          desc: '将研究想法转化为新算法：问题形式化 → 算法设计 → 代码实现 → 基准对比 → 验证入库。',
          tags: ['formalizer', 'designer', 'coder', 'benchmark', 'validator']
        },
        {
          icon: '📝',
          name: 'Paper Format Agents',
          zh: '论文格式智能体',
          desc: '模板适配（LaTeX/Word）、参考文献管理、期刊合规检查。支持各期刊模板一键排版。',
          tags: ['template', 'reference', 'compliance']
        },
        {
          icon: '💡',
          name: 'Research QA Agents',
          zh: '科研问答智能体',
          desc: '科研方法解释、公式推导与证明、代码演示。深入解答因果推断、计量方法等专业领域问题。',
          tags: ['explanation', 'formula', 'demo']
        },
        {
          icon: '🏆',
          name: 'Kaggle Agents',
          zh: '竞赛智能体',
          desc: '端到端 Kaggle 竞赛流水线：数据探查 → 基线 → 特征工程 → 精模 → 集成 → 提交 → 复盘。',
          tags: ['explorer', 'baseline', 'feature', 'model', 'ensemble', 'submission']
        },
        {
          icon: '🗄️',
          name: 'Knowledge Agent',
          zh: '知识库管理',
          desc: '算法知识库的创建、检索与维护。所有自创算法经验证后沉淀入库，形成可复用的知识资产。'
        }
      ]
    },
    knowledge: {
      header_zh: '知识库',
      header_en: 'Knowledge Repository',
      sub_zh: '50+ 算法与方法的知识沉淀',
      sub_en: '50+ Algorithms & Methods, Continuously Growing',
      categories: [
        {
          name: '因果推断',
          nameEn: 'Causal Inference',
          items: ['Difference-in-Differences', 'Causal Forest', 'Double Machine Learning', 'Instrumental Variables', 'Conformal Prediction']
        },
        {
          name: '机器学习',
          nameEn: 'Machine Learning',
          items: ['XGBoost / LightGBM / CatBoost', 'Random Forest / SVM', 'Clustering', 'Dimensionality Reduction', 'Causal Representation Learning']
        },
        {
          name: '时序与预测',
          nameEn: 'Time Series & Forecasting',
          items: ['ARIMA / SARIMA', 'Prophet', 'LSTM', 'Time Series Features (tsfresh)', 'Flow Matching']
        },
        {
          name: '深度学习',
          nameEn: 'Deep Learning',
          items: ['Autoencoder', 'FlashAttention', 'Enformer', 'Geneformer', 'AlphaFold3']
        },
        {
          name: '优化与运筹',
          nameEn: 'Optimization & OR',
          items: ['Adaptive ADMM', 'Physarum Network Optimizer', 'Bayesian Robust Optimization', 'Bilevel Optimization']
        },
        {
          name: '贝叶斯与统计',
          nameEn: 'Bayesian & Statistics',
          items: ['Bayesian DRO', 'Conformal Q-Values', 'Knockoffs', 'Meta-Learners (CATE)']
        }
      ]
    },
    guide: {
      header_zh: '使用指南',
      header_en: 'Quick Start',
      sub_zh: '三步上手 Idea2Top',
      sub_en: 'Get started in 3 steps',
      steps: [
        {
          title: '提出研究想法',
          titleEn: 'Propose Your Idea',
          desc: '用自然语言描述你的研究方法想法。例如："我想设计一个异质性处理效应的稳健估计量"或"帮我做这个数据集的因果分析"。',
          hint: '/research 我想设计一个...'
        },
        {
          title: '确认分解方案',
          titleEn: 'Review the Plan',
          desc: '秘书 Agent 会自动分析你的任务并输出分解方案。检查子任务列表、确认工具选择，点击确认后 Orchestrator 开始调度执行。',
          hint: '秘书会输出：工具、配色、规模、交付格式'
        },
        {
          title: '获取顶刊级交付',
          titleEn: 'Get Your Delivery',
          desc: '系统自动完成文献对标、方案设计、代码生成、实验验证和对抗式 QA。最终交付：可运行代码 + 结果图表 + 方法描述 + 复现说明。',
          hint: '质量门禁 ≥ 90 分方可交付'
        }
      ],
      note: '提示：系统支持多种输入类型——研究想法、数据分析请求、竞赛任务、文献综述需求等。只需用自然语言描述，剩下的交给 Agent 集群。'
    },
    footer: {
      brand: '数学建模半自动 · Idea2Top',
      tagline: '将研究方法想法转化为顶刊级可复现实现',
      product: '产品',
      productItems: ['首页', '架构', '工作流', '智能体'],
      resources: '资源',
      resourceItems: ['知识库', '使用指南', '算法索引'],
      copyright: '© 2026 Idea2Top Research Framework',
      built: 'Powered by Claude Multi-Agent Architecture'
    }
  },

  /* ========================================================
     English Translations
     ======================================================== */
  en: {
    nav: {
      home: 'Home',
      about: 'About',
      architecture: 'Architecture',
      workflow: 'Workflow',
      agents: 'Agents',
      knowledge: 'Knowledge',
      guide: 'Guide'
    },
    hero: {
      badge: 'Research Automation Framework',
      title_cn: 'Idea → Top-Journal Implementation',
      title_en: 'From Research Idea to Reproducible Results',
      subtitle: 'An intelligent framework that automatically transforms research methodology ideas into top-journal-grade reproducible code, experiments, and deliverables.',
      btn_start: 'Explore Architecture',
      btn_guide: 'Quick Start'
    },
    about: {
      header_zh: 'About',
      header_en: 'About the Project',
      desc1: 'Idea2Top is an intelligent framework that automatically transforms research methodology ideas into top-journal-grade reproducible implementations. Users describe ideas in natural language, and the system autonomously handles literature benchmarking, solution design, code generation, experimental validation, and adversarial QA.',
      desc2: 'Built on a Multi-Agent architecture, the Secretary Agent serves as the single entry point for task decomposition, while the Orchestrator dynamically schedules specialized domain agent clusters to execute the end-to-end pipeline.',
      tagline: 'From the first brushstroke of an idea to the seal of publication-ready delivery — every research innovation precisely realized.',
      stats: {
        agents: { num: '30+', label: 'Specialized Agents', labelEn: 'Specialized Agents' },
        algorithms: { num: '50+', label: 'Algorithms', labelEn: 'Algorithms in Repository' },
        workflows: { num: '6', label: 'Core Pipelines', labelEn: 'Core Workflows' },
        domains: { num: '8', label: 'Domains Covered', labelEn: 'Domain Coverage' }
      }
    },
    architecture: {
      header_zh: 'Architecture',
      header_en: 'System Architecture',
      sub_zh: 'Multi-Agent Hierarchical Architecture',
      sub_en: 'Multi-Agent Hierarchical Architecture',
      intro: 'The system employs a three-tier agent architecture: the Secretary Agent receives and decomposes all tasks, the Orchestrator dynamically orchestrates the pipeline, and specialized domain agent clusters execute specific tasks at the operational tier.',
      secretary: {
        title: 'Secretary Agent',
        zh: 'Task Gatekeeper',
        desc: 'Entry point & task decomposition',
        agents: ['Task Analysis', 'User Confirmation', 'Orchestrator Dispatch']
      },
      orchestrator: {
        title: 'Orchestrator',
        zh: 'Pipeline Coordinator',
        desc: 'Intent recognition & orchestration',
        agents: ['Agent Routing', 'Context Management', 'Quality Gates']
      },
      literature: { title: 'Literature', zh: 'Literature Agents', desc: 'Search & review', count: '3' },
      topic: { title: 'Topic Analysis', zh: 'Topic Analysis', desc: 'Frontier detection & recommendation', count: '3' },
      dataviz: { title: 'Data & Viz', zh: 'Data Processing & Viz', desc: 'Cleaning, modeling, publication charts', count: '4' },
      experiment: { title: 'Experiment', zh: 'Experiment Design', desc: 'Design, optimization, simulation', count: '3' },
      algorithm: { title: 'Algorithm', zh: 'Algorithm Creation', desc: 'Formalize → design → code → verify', count: '5' },
      paper: { title: 'Paper Format', zh: 'Paper Formatting', desc: 'Templates, references, compliance', count: '3' },
      researchqa: { title: 'Research QA', zh: 'Research Q&A', desc: 'Method explanation, derivation', count: '3' },
      kaggle: { title: 'Kaggle', zh: 'Competition', desc: 'End-to-end competition pipeline', count: '7' },
      knowledge: { title: 'Knowledge', zh: 'Knowledge Management', desc: 'Algorithm storage & retrieval', count: '1' }
    },
    workflow: {
      header_zh: 'Workflow',
      header_en: 'Workflow Pipeline',
      sub_zh: 'End-to-End Pipeline from Idea to Delivery',
      sub_en: 'End-to-End Pipeline from Idea to Delivery',
      steps: [
        {
          title: 'Analysis & Benchmarking',
          titleEn: 'Analysis & Benchmarking',
          desc: 'Identify the core methodology of the user\'s idea, retrieve standard implementation specifications from top journals, and summarize methodological standards.',
          agents: ['literature/search-agent', 'literature/screening-agent', 'literature/synthesis-agent']
        },
        {
          title: 'Solution Design',
          titleEn: 'Solution Design',
          desc: 'Produce a complete mathematical framework: model specification, identification strategy, estimation method, and assumptions. Pause for user confirmation.',
          agents: ['algorithm/formalizer-agent', 'algorithm/designer-agent']
        },
        {
          title: 'Implementation',
          titleEn: 'Implementation',
          desc: 'Generate top-journal-grade runnable code based on the design, with comprehensive tests and usage examples.',
          agents: ['algorithm/coder-agent']
        },
        {
          title: 'Experiment & Validation',
          titleEn: 'Experiment & Validation',
          desc: 'Run full experiments and physically validate that the code runs and results are sound. No AI verbal reports — real execution is mandatory.',
          agents: ['algorithm/benchmark-agent', 'algorithm/validator-agent']
        },
        {
          title: 'Adversarial QA',
          titleEn: 'Adversarial QA',
          desc: 'An independent Critic Agent rigorously reviews outputs. Up to 5 rounds; terminates when the Critic issues APPROVED.',
          agents: ['algorithm/validator-agent (critic)']
        },
        {
          title: 'Delivery',
          titleEn: 'Delivery',
          desc: 'Output complete deliverable package: code + result figures + methodology description + reproducibility notes. Quality gate score ≥ 90 required.',
          agents: ['Knowledge Repository', 'Quality Gate ≥ 90']
        }
      ]
    },
    agents: {
      header_zh: 'Agent Cluster',
      header_en: 'Agent Cluster',
      sub_zh: '30+ Specialized Agents Covering the Full Research Pipeline',
      sub_en: '30+ Specialized Agents Covering the Full Research Pipeline',
      list: [
        {
          icon: '🔐',
          name: 'Secretary Agent',
          zh: 'Task Gatekeeper',
          desc: 'The single entry point for all user tasks. Analyzes requests, produces decomposition plans, and enforces user confirmation before any work begins.'
        },
        {
          icon: '🎯',
          name: 'Orchestrator',
          zh: 'Pipeline Coordinator',
          desc: 'Dynamic intent recognition + agent routing + pipeline orchestration + context management. Monitors context saturation and auto-splits when exceeding 50%.'
        },
        {
          icon: '📚',
          name: 'Literature Agents',
          zh: 'Literature Agents',
          desc: 'Literature search, screening, and synthesis. Supports multi-database search, quality filtering, and automated survey generation.',
          tags: ['search', 'screening', 'synthesis']
        },
        {
          icon: '🔍',
          name: 'Topic Analysis Agents',
          zh: 'Topic Analysis Agents',
          desc: 'Research frontier detection, gap analysis, and topic recommendation. Uses bibliometrix for bibliometric analysis.',
          tags: ['frontier', 'gap', 'recommendation']
        },
        {
          icon: '📊',
          name: 'Data & Viz Agents',
          zh: 'Data & Viz Agents',
          desc: 'Data cleaning, statistical modeling, publication-grade visualization, and result interpretation.',
          tags: ['cleaning', 'modeling', 'visualization', 'interpretation']
        },
        {
          icon: '🧪',
          name: 'Experiment Agents',
          zh: 'Experiment Agents',
          desc: 'Experimental design, parameter optimization & sensitivity analysis, and simulation execution.',
          tags: ['design', 'optimization', 'simulation']
        },
        {
          icon: '⚗️',
          name: 'Algorithm Agents',
          zh: 'Algorithm Creation Agents',
          desc: 'Full pipeline: formalize problem → design algorithm → implement code → benchmark → validate and store.',
          tags: ['formalizer', 'designer', 'coder', 'benchmark', 'validator']
        },
        {
          icon: '📝',
          name: 'Paper Format Agents',
          zh: 'Paper Format Agents',
          desc: 'Template adaptation (LaTeX/Word), reference management, journal compliance checking.',
          tags: ['template', 'reference', 'compliance']
        },
        {
          icon: '💡',
          name: 'Research QA Agents',
          zh: 'Research QA Agents',
          desc: 'Research method explanation, formula derivation & proof, code demonstration.',
          tags: ['explanation', 'formula', 'demo']
        },
        {
          icon: '🏆',
          name: 'Kaggle Agents',
          zh: 'Kaggle Competition Agents',
          desc: 'End-to-end Kaggle pipeline: data exploration → baseline → feature engineering → model → ensemble → submission → post-mortem.',
          tags: ['explorer', 'baseline', 'feature', 'model', 'ensemble', 'submission']
        },
        {
          icon: '🗄️',
          name: 'Knowledge Agent',
          zh: 'Knowledge Manager',
          desc: 'Algorithm knowledge base creation, retrieval, and maintenance. All algorithms are validated before storage.'
        }
      ]
    },
    knowledge: {
      header_zh: 'Knowledge Repository',
      header_en: 'Knowledge Repository',
      sub_zh: '50+ Algorithms & Methods, Continuously Growing',
      sub_en: '50+ Algorithms & Methods, Continuously Growing',
      categories: [
        {
          name: 'Causal Inference',
          nameEn: 'Causal Inference',
          items: ['Difference-in-Differences', 'Causal Forest', 'Double Machine Learning', 'Instrumental Variables', 'Conformal Prediction']
        },
        {
          name: 'Machine Learning',
          nameEn: 'Machine Learning',
          items: ['XGBoost / LightGBM / CatBoost', 'Random Forest / SVM', 'Clustering', 'Dimensionality Reduction', 'Causal Representation Learning']
        },
        {
          name: 'Time Series & Forecasting',
          nameEn: 'Time Series & Forecasting',
          items: ['ARIMA / SARIMA', 'Prophet', 'LSTM', 'Time Series Features (tsfresh)', 'Flow Matching']
        },
        {
          name: 'Deep Learning',
          nameEn: 'Deep Learning',
          items: ['Autoencoder', 'FlashAttention', 'Enformer', 'Geneformer', 'AlphaFold3']
        },
        {
          name: 'Optimization & OR',
          nameEn: 'Optimization & OR',
          items: ['Adaptive ADMM', 'Physarum Network Optimizer', 'Bayesian Robust Optimization', 'Bilevel Optimization']
        },
        {
          name: 'Bayesian & Statistics',
          nameEn: 'Bayesian & Statistics',
          items: ['Bayesian DRO', 'Conformal Q-Values', 'Knockoffs', 'Meta-Learners (CATE)']
        }
      ]
    },
    guide: {
      header_zh: 'Quick Start',
      header_en: 'Quick Start Guide',
      sub_zh: 'Get started in 3 steps',
      sub_en: 'Get started in 3 steps',
      steps: [
        {
          title: 'Propose Your Idea',
          titleEn: 'Propose Your Idea',
          desc: 'Describe your research idea in natural language. For example: "I want to design a robust estimator for heterogeneous treatment effects."',
          hint: '/research I want to design...'
        },
        {
          title: 'Review the Decomposition',
          titleEn: 'Review the Decomposition',
          desc: 'The Secretary Agent automatically analyzes your task and produces a decomposition plan. Check the subtask list, confirm tool selections, and the Orchestrator begins execution.',
          hint: 'Secretary outputs: tools, palette, scale, format'
        },
        {
          title: 'Receive Top-Journal Delivery',
          titleEn: 'Receive Top-Journal Delivery',
          desc: 'The system autonomously completes literature benchmarking, solution design, code generation, experimental validation, and adversarial QA. Final delivery includes runnable code, figures, methodology description, and reproducibility notes.',
          hint: 'Quality gate ≥ 90 required for delivery'
        }
      ],
      note: 'Tip: The system supports multiple input types — research ideas, data analysis requests, competition tasks, literature survey needs, and more. Just describe in natural language, and the agent cluster handles the rest.'
    },
    footer: {
      brand: 'Idea2Top — Semi-Automated Math Modeling',
      tagline: 'Transforming research ideas into top-journal-grade reproducible implementations',
      product: 'Product',
      productItems: ['Home', 'Architecture', 'Workflow', 'Agents'],
      resources: 'Resources',
      resourceItems: ['Knowledge Base', 'Quick Start', 'Algorithm Index'],
      copyright: '© 2026 Idea2Top Research Framework',
      built: 'Powered by Claude Multi-Agent Architecture'
    }
  }
};
