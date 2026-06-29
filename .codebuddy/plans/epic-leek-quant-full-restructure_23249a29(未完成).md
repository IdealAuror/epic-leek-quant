---
name: epic-leek-quant-full-restructure
overview: 对 epic-leek-quant 项目进行六阶段全面重构：项目审计、目录重组、代码规范化、依赖管理、文档体系建立。核心约束：所有 .py 脚本保持聚宽独立运行兼容性（单文件粘贴执行），不引入跨文件 import 依赖。
todos:
  - id: git-branch
    content: 创建独立分支 `refactor/phase5` 并确认干净工作区
    status: pending
  - id: project-audit
    content: 使用 [subagent:code-explorer] 执行全项目审查：扫描重复代码、硬编码路径、超过500行文件、命名违规、缺失类型注解，输出技术债务清单
    status: pending
    dependencies:
      - git-branch
  - id: directory-restructure
    content: 执行目录重组：创建 archive/ 移入空 reference/ 目录，为 joinquant/ 添加 __init__.py，创建 results/P5-F6-MOM-2026Q2-v1/ 目录准备存放跨样本验证 CSV
    status: pending
    dependencies:
      - project-audit
  - id: split-data-layer
    content: 拆分 joinquant/data_layer.py（601行）：将撮合逻辑 extract 为 joinquant/broker.py，data_layer.py 保留数据查询，修正两文件的 import 引用，添加完整的类型注解和 docstring
    status: pending
    dependencies:
      - directory-restructure
  - id: standardize-scripts
    content: 规范化 research/scripts/ 下 10 个脚本：修补 docstring（补全模块级和顶层函数），添加函数签名类型注解，修正命名违规，统一 PEP8 格式，为超过 500 行的脚本添加技术债务标记注释
    status: pending
    dependencies:
      - directory-restructure
  - id: standardize-strategies
    content: 规范化 joinquant/strategies/ 下 7 个策略文件：修补 docstring 和类型注解，修正命名，PEP8 格式化，确保策略文件可 import 更新后的 data_layer/broker
    status: pending
    dependencies:
      - split-data-layer
  - id: deduplicate-utils
    content: 将跨脚本重复的工具逻辑抽取到 joinquant/utils.py：绩效指标计算函数、Winsorize+Z-score 函数、成本模型常量和计算函数。research/scripts/ 下脚本因聚宽约束保持内联，但在文件顶部添加注释标注可替换为 from joinquant.utils import
    status: pending
    dependencies:
      - standardize-scripts
  - id: env-config
    content: 生成 requirements.txt（固定版本号），创建 .env.example 模板，将所有脚本中的硬编码路径 OUT_DIR 替换为基于 pathlib 的相对路径解析
    status: pending
    dependencies:
      - directory-restructure
  - id: rewrite-readme
    content: 重写 README.md：项目简介、完整目录树、环境搭建指南、回测运行命令示例、新策略开发步骤、数据源配置说明
    status: pending
    dependencies:
      - directory-restructure
      - env-config
  - id: update-index
    content: 更新 research/_index.md：刷新目录结构引用，补充本次重构变更日志，确认因子看板路径正确
    status: pending
    dependencies:
      - rewrite-readme
      - standardize-scripts
  - id: deliverables
    content: 使用 [skill:xlsx] 生成文件变更详细日志表格，汇总审查报告核心发现，输出重构前后对比示例
    status: pending
    dependencies:
      - update-index
      - deduplicate-utils
---

## 项目重构目标

对 epic-leek-quant 项目执行 6 阶段系统性重构，提升代码质量、可维护性和工程规范性。

## 核心约束

- **不可改变回测核心逻辑和数学计算结果**
- **research/scripts/ 下脚本必须保持聚宽研究环境粘贴即运行的兼容性**
- 不确定删除的内容移入 `archive/` 目录并说明原因
- 假设在独立分支操作，标注适合独立 commit 的步骤

## 6 阶段概览

1. **项目审查报告**：盘点技术债务、代码坏味道、重复代码
2. **Git 分支管理**：在独立分支上操作
3. **目录重组**：按职责归类文件，修正 import，添加 `__init__.py`
4. **代码规范化**：PEP8 格式化、docstring、类型注解、命名修正、去重、大文件标记
5. **依赖与环境管理**：生成 `requirements.txt`、创建 `.env.example`、路径改为 `pathlib`
6. **文档体系建立**：重写 `README.md`、更新文件路径引用

## 技术背景

### 项目运行环境

- 所有 Python 脚本在**聚宽研究环境**中以单文件粘贴方式运行
- `joinquant/data_layer.py` 和 `joinquant/factor_lib.py` 是本地可复用的基础库
- 策略文件位于 `joinquant/strategies/`，研究脚本位于 `research/scripts/`
- 本地环境无聚宽 SDK，`data_layer.py` 通过 `try/except` 兼容本地导入审核

### 关键限制

- `research/scripts/` 下的 10 个脚本必须保持**自包含**（无法 import 项目内部模块，因为聚宽环境无这些文件）
- `joinquant/strategies/` 下的策略文件若与 `data_layer.py` 同目录部署，则可 import
- 跨脚本的重复代码无法通过 import 消除，但可通过代码内规范化改进

### 当前代码质量问题（审查发现）

| 问题类别 | 具体情况 |
| --- | --- |
| **jqdata 导入样板** | 10 个脚本各重复约 12 行相同的 `from jqdata import *` + try/except 块 |
| **成本模型参数** | `COMMISSION/STAMP_DUTY/SLIPPAGE/BASE_ROUND_TRIP` 在 5+ 个脚本中重复定义 |
| **日频净值计算引擎** | `compute_daily_nav()` 等价逻辑在 P2/P3/P4-v1/P4-v2/P5 五个脚本中独立实现 |
| **绩效指标计算** | `calc_performance_metrics()` 等价函数在 6+ 个脚本中重复 |
| **分段定义** | 7 段市场分段定义在 P4-v2 和 P5 中重复 |
| **Winsorize + Z-score** | 在多个脚本中独立实现 |
| **大文件** | P5-F6-MOM 1318 行、P4-PL-v2 1057 行（远超 500 行阈值） |
| **无类型注解** | 所有函数签名无类型提示 |
| **无 PEP8 格式化** | 多处缩进/空行不一致 |
| **无 `__init__.py`** | `joinquant/` 及所有子目录无包初始化文件 |
| **硬编码路径** | `OUT_DIR` 使用字符串路径如 `'results/P1-F2-EP-2026Q2-v1'` |
| **无依赖文件** | 无 `requirements.txt`、无 `.env.example` |


## 重构策略

### 目录重组方案

```
epic-leek-quant/
├── README.md                    # [重写] 项目入口文档
├── README-zh.md                 # [保留] 中文版介绍
├── epic-leek-value-investing.md # [保留] 原文理论
├── .gitignore
├── .env.example                 # [NEW] 环境变量模板
├── requirements.txt             # [NEW] Python 依赖清单
├── LICENSE
├── docs/
│   ├── PROJECT-PLAN.md
│   ├── agent-workflow.md
│   ├── manual-investment-guide.md
│   ├── phase-0-status.md
│   ├── plan-review.md
│   ├── prompts/                 # 9 个提示词 + START-HERE
│   └── task-state/              # 2 个状态文件
├── joinquant/
│   ├── __init__.py              # [NEW] 包初始化
│   ├── data_layer.py            # [MODIFY] 添加类型注解/docstring（不改逻辑）
│   ├── factor_lib.py            # [MODIFY] 添加类型注解/docstring（不改逻辑）
│   └── strategies/              # 7 个策略文件（保持聚宽兼容）
├── research/
│   ├── _index.md                # [MODIFY] 更新路径引用
│   ├── theory-framework.md
│   ├── thinking-prompt.md
│   ├── decisions/               # 6 个决策文档
│   ├── reports/                 # 10 个报告
│   ├── scripts/                 # 10 个脚本（保持自包含）
│   └── specs/                   # 5 个 spec 文档
├── results/
│   └── README.md
└── archive/                     # [NEW] 归档不确定删除的内容
    └── reference/               # [MOVE] 原空目录，保留结构以备后用
```

### 代码规范化策略

由于 `research/scripts/` 脚本必须自包含，**不拆分这些文件**（拆分后聚宽环境无法运行）。整改重点：

- 在每个现有文件内添加 docstring、类型注解、PEP8 格式化
- 超过 500 行的文件标记技术债务（`# TODO: refactor - exceeds 500 lines`），但不强制拆分
- `joinquant/data_layer.py`（601 行）可以拆分：将撮合逻辑独立为 `joinquant/broker.py`
- 变量/函数 `snake_case`、常量 `UPPER_CASE`、类 `PascalCase` 修正

### 依赖管理策略

- 扫描所有 `import` 语句，提取 `numpy`、`pandas`、`scipy`、`statsmodels` 等
- 固定聚宽兼容版本（基于聚宽当前环境版本）
- `.env.example` 模板：聚宽账号（注释说明，实际值不提交）

## Agent Extensions

### SubAgent

- **code-explorer**
- Purpose: 在阶段一执行全项目代码审查，扫描重复代码模式、硬编码路径、敏感信息、行数统计
- Expected outcome: 生成完整的技术债务清单，包括：重复代码块位置映射、超过 500 行的文件列表、所有 import 语句汇总、所有硬编码路径列表

### Skill

- **pdf**
- Purpose: 如需要将审查报告输出为 PDF 格式交付物
- Expected outcome: 生成格式化的 PDF 审查报告

- **xlsx**
- Purpose: 生成文件变更日志表格（CSV/XLSX 格式），列出每个文件的操作类型和理由
- Expected outcome: 结构化的变更日志表格，清晰可追溯