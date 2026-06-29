---
name: epic-leek-quant-full-restructure
overview: 对 epic-leek-quant 项目进行全面重构：项目审计、目录重组、代码规范化（仅文件内改格式/加注解/不改结构）、依赖管理、文档建立。零破坏性：所有 .py 文件保持单文件完整性和聚宽粘贴即运行兼容性。
todos:
  - id: git-branch
    content: 创建独立分支 `refactor/phase6` 并确认工作区干净
    status: completed
  - id: project-audit
    content: 使用 [subagent:code-explorer] 执行全项目审查：扫描重复代码模式、硬编码路径、超过500行文件、命名违规、缺失类型注解，输出技术债务清单
    status: completed
    dependencies:
      - git-branch
  - id: directory-restructure
    content: 执行目录重组：①创建 archive/ 移入 reference/ ②docs/ 下新建 guides/ 移入 START-HERE-claude-code.md ③为 joinquant/ 添加 __init__.py ④创建 results/P5-F6-MOM-2026Q2-v1/ ⑤docs/prompts/ 文件名统一编号格式
    status: completed
    dependencies:
      - project-audit
  - id: reorganize-docs
    content: 整理 docs/ .md 文档：①START-HERE 从 prompts/ 移入 guides/ ②更新 CURRENT-STATE.md 到 Phase 5 关闭态 ③清理过时的 phase 启动提示词引用 ④更新所有文档内路径引用
    status: completed
    dependencies:
      - directory-restructure
  - id: current-strategy-doc
    content: 新建 docs/CURRENT-STRATEGY.md：项目最终策略总览（因子构成/F2+F5+F6-40d/收益指标/回撤/各阶段实验结论速查表），作为新加入者的第一份阅读材料
    status: completed
    dependencies:
      - reorganize-docs
  - id: standardize-joinquant
    content: 规范化 joinquant/ 下 2 个库文件：data_layer.py 和 factor_lib.py 内部添加完整 docstring、函数签名类型注解、PEP8 格式、命名修正（不拆分不改变文件结构）
    status: completed
    dependencies:
      - directory-restructure
  - id: standardize-strategies
    content: 规范化 joinquant/strategies/ 下 7 个策略文件内部：补 docstring、类型注解、PEP8、命名修正
    status: completed
    dependencies:
      - standardize-joinquant
  - id: standardize-scripts
    content: 规范化 research/scripts/ 下 10 个脚本内部：补全模块级和函数级 docstring、添加函数签名类型注解、修正命名违规、统一 PEP8 格式；超过 500 行文件加注释标注聚宽单文件约束
    status: completed
    dependencies:
      - standardize-joinquant
  - id: env-config
    content: 生成 requirements.txt（固定版本号），创建 .env.example 模板，将脚本中硬编码 OUT_DIR 字符串路径改为 pathlib.Path 动态解析
    status: completed
    dependencies:
      - directory-restructure
  - id: rewrite-readme
    content: 重写 README.md：项目简介、完整目录树、环境搭建指南、回测运行命令、新策略开发步骤、数据源说明；更新 _index.md 路径引用和变更日志
    status: completed
    dependencies:
      - current-strategy-doc
      - env-config
  - id: deliverables
    content: 使用 [skill:xlsx] 生成文件变更日志表格，汇总审查报告核心发现，输出重构前后对比示例
    status: completed
    dependencies:
      - rewrite-readme
      - standardize-scripts
---

## 项目重构目标

对 epic-leek-quant 项目执行六阶段系统性重构，提升代码质量和工程规范性。

## 核心约束

- **所有 .py 文件保持单文件，不拆分、不重组文件结构**
- 回测核心逻辑和数学计算结果不可改变
- research/scripts/ 下脚本必须在聚宽研究环境以单文件粘贴运行
- 代码改动限定在文件内部：加 docstring、类型注解、PEP8 格式、修命名
- 不确定删除的内容移入 `archive/` 目录

## 六阶段内容

1. **Git 分支管理**：在独立分支上操作
2. **项目审查报告**：盘点技术债务、代码坏味道、重复代码
3. **目录重组**：归档空目录、docs/ 文档归类整理、添加 `__init__.py`、创建结果子目录
4. **代码内部规范化**：PEP8/docstring/type hints/命名修正（不改变文件数量和结构）
5. **依赖与环境管理**：生成 `requirements.txt`、创建 `.env.example`、路径 `pathlib` 化
6. **文档体系建立**：重写 `README.md`、更新 `_index.md`

## 技术背景

### 运行环境约束

- 所有 `research/scripts/` 下脚本在聚宽研究环境以单文件粘贴运行
- `joinquant/data_layer.py` 和 `factor_lib.py` 是本地基础库，通过 `try/except` 兼容无聚宽 SDK 的本地环境
- 脚本间无 import 依赖（聚宽沙盒内无法 import 项目内模块）
- 脚本内无硬编码 API 密钥（聚宽平台注入认证）

### 重构策略

#### 目录重组

```
epic-leek-quant/
├── README.md                    # [REWRITE] 项目入口
├── README-zh.md                 # [KEEP] 中文版
├── epic-leek-value-investing.md # [KEEP] 原文理论
├── .gitignore
├── .env.example                 # [NEW] 环境变量模板
├── requirements.txt             # [NEW] 依赖清单
├── LICENSE
├── archive/                     # [NEW] 归档目录
│   └── reference/               # [MOVE] 原空目录
├── docs/
│   ├── PROJECT-PLAN.md          # [KEEP] 单一事实来源
│   ├── CURRENT-STRATEGY.md      # [NEW] 当前最佳策略总览（最终产出）
│   ├── plan-review.md           # [KEEP] 审查报告
│   ├── agent-workflow.md        # [KEEP] 工作流定义
│   ├── manual-investment-guide.md # [KEEP] 个人投资指南
│   ├── phase-0-status.md        # [KEEP] Phase 0 完成报告（历史参考）
│   ├── prompts/                 # 7 个提示词模板（00-07）
│   ├── task-state/              # 2 个状态文件
│   ├── guides/                  # [NEW] 操作指南
│   │   └── START-HERE-claude-code.md  # [MOVE] 从 prompts/ 移入
│   └── archive/                 # [NEW] 过时文档归档
│       └── (预留)
├── joinquant/
│   ├── __init__.py              # [NEW] 包初始化
│   ├── data_layer.py            # [MODIFY] 加注解/类型/PEP8
│   ├── factor_lib.py            # [MODIFY] 加注解/类型/PEP8
│   └── strategies/              # 7 个策略文件（内部规范化）
├── research/
│   ├── _index.md                # [MODIFY] 刷新引用
│   ├── theory-framework.md
│   ├── thinking-prompt.md
│   ├── decisions/ (6个)
│   ├── reports/ (10个)
│   ├── scripts/ (10个，内部规范化)
│   └── specs/ (5个)
└── results/
    ├── README.md
    └── P5-F6-MOM-2026Q2-v1/     # [NEW] 跨样本验证 CSV
```

#### 代码内部规范化（不改结构）

- 所有函数添加三引号 docstring（功能/参数/返回/异常）
- 所有函数签名添加类型注解（Type Hints）
- 变量 `snake_case`、类 `PascalCase`、常量 `UPPER_CASE`
- PEP8：缩进 4 空格、行长 ≤ 120、空行规范、import 排序
- 超过 500 行文件在顶部添加注释 `# NOTE: 此文件 xxx 行，因聚宽单文件约束不可拆分`

#### 依赖与环境

- 扫描所有 `import` 生成 `requirements.txt`，固定聚宽兼容版本
- `.env.example` 模板：注释说明聚宽账号配置项
- `OUT_DIR` 等硬编码路径改为 `pathlib.Path` 动态解析

## 使用的扩展

### SubAgent

- **code-explorer**
- 用途：阶段二执行全项目审查，扫描重复代码、硬编码路径、超过500行文件、命名违规
- 预期产出：完整技术债务清单（重复代码位置映射、大文件列表、import 汇总、路径列表）

### Skill

- **xlsx**
- 用途：生成文件变更日志表格（CSV 格式）
- 预期产出：结构化变更日志，列出每个文件的操作类型和理由