# epic-leek-quant 统一执行计划（PROJECT-PLAN）

> 本文是项目的**单一事实来源（single source of truth）**。三份原文档（`epic-leek-value-investing.md`、`docs/agent-workflow.md`、`research/thinking-prompt.md`）作为背景资料保留，**当任何文档与本计划冲突时，以本计划口径为准**。
>
> 冲突调和依据：见 `docs/plan-review.md` 第二节"口径冲突清单"。调和原则——**以 `thinking-prompt.md` 的严格学术标准为准绳，以 theory 文档的乐观数字为"待验证假设"。**

---

## 一、核心架构决策（统一口径）

> 这一组决策直接对应 `plan-review.md` 第二节的冲突清单（C1–C13）与第三节代码缺陷（B1–B10）。每条标注依据。

### 1.1 回测参数统一口径

| 议题 | 统一口径 | 依据 |
|------|---------|------|
| **换仓频率** | **季度调仓**：每季报披露截止后首个交易日（一季报 4/30、半年报 8/31、三季报 10/31、年报次年 4/30 之后）。月频仅作敏感性对照，不作主结论。 | C1 / B9：分红税月频 20%→季频 10%；对齐披露节奏减少未来函数风险；降换手磨损 |
| **佣金** | 万 2.5，双边 | C2 |
| **印花税** | 千 1，仅卖出 | C3 |
| **最低佣金** | 5 元/笔 | C4 |
| **滑点** | 0.3%（中等保守） | C5 |
| **冲击成本** | **分级估算**：先按"持仓市值/日均成交额"分档（如 <5% / 5–20% / >20%），后续 Phase 4 升级 Almgren-Chriss | C6 |
| **基准** | 主：沪深 300 全收益（`000300.XSHG`）；辅：中证 800 全收益。**强制报告跟踪误差（TE）与信息比率（IR）** | C7 |
| **退市股** | **必须纳入**：退市按实际回报计入，无法卖出时按跌停板排队模拟 | C8 / B3 |
| **涨跌停 / T+1 / 停牌** | 必须实现限价单排队撮合器：涨停排队买入、跌停排队卖出；T+1 下**先卖后买**；停牌冻结至复牌首日按实际成交价计入 | C9 / B4 |
| **PIT 数据** | spec 显式要求校验"换仓日实际可得数据"；代码审核门禁强检 `get_fundamentals(date=)` 的 PIT 语义与财报修正覆盖 | C10 / B8 |
| **中性化** | 单因子 IC 报告**同时给出行业中性化 IC 与市值中性化 IC**；Phase 2 用 Fama-MacBeth 筛独立截面定价能力（重点排除国企=low-beta/size 代理） | C11 |

### 1.2 因子计算统一口径

| 议题 | 统一口径 | 依据 |
|------|---------|------|
| **二值因子（0/1）处理** | **直接等权求和**，不做 Z-score（Z-score 对二值因子是冗余线性变换，不改排序） | C12 / B2 |
| **连续因子处理** | 先做 1%/99% winsorize，再做 Z-score 标准化，最后按方案合成 | C12 |
| **`debt_to_assets` bug** | 必须修复：`median_debt = (df['total_liability'] / df['total_assets']).median()`，先算比率再取中位数 | B1 |
| **ST 判定** | 改用 `get_extras('is_st', ...)` 或聚宽 ST 标记，禁用 `display_name.startswith('ST')` 遍历全市场 | B5 |
| **`datetime` 导入** | 补 `import datetime`，或统一用 `pd.Timestamp` | B6 |

### 1.3 修复后的因子计算参考片段

> 仅作口径说明与代码审核对照参考，**非项目源码**。完整策略在 Phase 0 脚手架内实现。

```python
import datetime
import jqdata
import pandas as pd
import numpy as np

def calculate_all_factors(df):
    """
    五大因子计算（修复 B1/B2）。
    二值因子直接给 0/1，连续因子保留原值供后续 winsorize+Zscore。
    退市/涨跌停/停牌处理见撮合层，不在此函数。
    """
    # 因子一：EV 为负（净现金 > 总市值+总负债）
    df['ev'] = df['market_cap'] + df['total_liability'] - df['cash_equivalents']
    df['factor_ev_negative'] = (df['ev'] < 0).astype(int)            # 二值

    # 因子二：盈利收益率（连续）+ 低 PE 阈值（二值）
    df['earnings_yield'] = df['net_profit'] / df['market_cap']        # 连续
    df['factor_low_pe'] = (df['pe_ttm'] < 10).astype(int)             # 二值

    # 因子三：高股息（连续）+ 是否连续分红（二值，需外部判断）
    df['div_yield_continuous'] = df['div_yield']                      # 连续

    # 因子四：国企背景（二值）
    state_owners = ['国务院国有资产监督管理委员会',
                    '地方国有资产监督管理委员会', '地方政府']
    df['factor_state_owned'] = df['actual_controller'].isin(state_owners).astype(int)

    # 因子五：财务质量 —— 修复 B1：先算比率，再取中位数
    df['debt_to_assets'] = df['total_liability'] / df['total_assets']
    median_debt = df['debt_to_assets'].median()                       # 修复 B1
    df['factor_low_leverage'] = (df['debt_to_assets'] < median_debt).astype(int)
    df['factor_positive_cf'] = (df['operating_cash_flow'] > 0).astype(int)

    return df

def composite_score(df):
    """
    合成：二值因子等权求和 + 连续因子 winsorize/Zscore。
    注意：真实回测中分档/排序应使用连续因子原始值，
          winsorize 与 Zscore 仅用于把连续因子拉到与二值同尺度。
    """
    binary_cols = ['factor_ev_negative', 'factor_low_pe',
                   'factor_state_owned', 'factor_low_leverage',
                   'factor_positive_cf']
    continuous_cols = ['earnings_yield', 'div_yield_continuous']

    score = df[binary_cols].sum(axis=1)

    for col in continuous_cols:
        s = df[col]
        lo, hi = s.quantile(0.01), s.quantile(0.99)                  # 1%/99% winsorize
        s = s.clip(lo, hi)
        std = s.std()
        score = score + ((s - s.mean()) / std if std > 0 else 0)

    df['composite_score'] = score
    return df
```

### 1.4 基调声明

- theory 文档中的 "年化 16–20%""超额 409%""显著且持续" 等数字，**一律视为待验证的乐观假设**，在 spec、报告、提示词中必须标注"待验证假设"，不得作为结论或通过标准。
- 任何结论必须由本计划定义的 IC 体系（NW-t、IC_IR、Q1 多头绝对收益、Q1-Q5 单调性、跨样本一致性）支撑。

---

## 二、目录结构（Phase 0 产出）

```
epic-leek-quant/
├── epic-leek-value-investing.md        # 原文，不改
├── README.md                            # Phase 0 补全
├── .gitignore                           # Phase 0 新建
├── docs/
│   ├── agent-workflow.md               # 原文，不改
│   ├── plan-review.md                  # 审查报告
│   ├── PROJECT-PLAN.md                 # 本文件
│   └── prompts/                        # DeepSeek 提示词
│       ├── 00-system-role.md
│       ├── 01-research-spec-design.md
│       ├── 02-flash-execution.md
│       ├── 03-code-review-gate.md
│       └── 04-result-adjudication.md
├── joinquant/                           # Phase 0：聚宽策略与数据接口封装（P0）
│   ├── data_layer.py                   # PIT 查询、退市/停牌标记、限价单撮合
│   ├── factor_lib.py                   # 五大因子计算（修复后口径）
│   └── strategies/                     # 每个因子/变体一个策略文件
├── research/
│   ├── thinking-prompt.md              # 原文，不改
│   ├── _index.md                       # 研究记录索引/看板/迭代追溯
│   ├── specs/                          # 每轮 research-spec
│   ├── reports/                        # 每轮分析报告
│   └── decisions/                      # 因子去留/权重/参数决策记录
└── results/                            # 回测原始输出（净值 CSV、IC 表），gitignore 大文件
```

`.gitignore` 要点：`results/` 下大 CSV/日志按大小忽略；聚宽本地不跑的 `.py` 仍入库；密钥/凭据一律忽略。

---

## 三、项目阶段（Phase 0–4）

> 每阶段含：**目标 / 输入 / 产出物 / 通过 Gate / 负责模型 / 防 data-mining 规则**。
> 模型分工：DeepSeek V4 Pro = 高级模型（spec 设计 + 代码审核 + Gate 判定）；Flash = 执行模型（写代码、跑回测、批量扫描）。

### Phase 0：脚手架（Scaffolding）

| 项 | 内容 |
|----|------|
| **目标** | 搭建目录、数据接口封装层、模板，使后续 Phase 1 的 spec 能被稳定执行 |
| **输入** | 本计划、`plan-review.md` |
| **产出物** | ① 目录结构与 `.gitignore`；② `joinquant/data_layer.py`：统一 PIT 查询 + 退市/停牌标记 + 限价单撮合器 + 先卖后买调仓顺序；③ `joinquant/factor_lib.py`：修复后因子计算；④ `research/_index.md` 索引与命名规范；⑤ spec/报告模板（见 `prompts/01`） |
| **通过 Gate** | 用一只已知股票人工核验 data_layer 的 PIT 查询返回值与财报披露日一致；撮合器在涨停/跌停/停牌三类场景单测通过 |
| **负责模型** | 高级模型设计接口契约，Flash 实现 |
| **防 data-mining** | 本阶段不产生任何因子结论 |

### Phase 1：单因子验证（Single-Factor Validation）

| 项 | 内容 |
|----|------|
| **目标** | 逐一验证五大因子在 A 股的截面选股能力，从 EV<0 起步 |
| **输入** | Phase 0 脚手架；每因子一份 research-spec |
| **样本** | 沪深 300 / 中证 500 / 全 A 三样本（跨样本一致性是 Gate 1 必检项） |
| **因子顺序** | EV<0 → 盈利收益率（PE）→ 股东回报（分红+回购）→ 国企背景 → 财务质量 |
| **产出物** | 每因子：执行代码 + 三张表（IC 描述统计、Q1-Q5 分组、分市值档 IC）+ 净值曲线 + 偏差记录 + 分析报告（`research/reports/`） |
| **通过 Gate（Gate 1）** | ① Rank IC_IR > 0.3（**使用 Newey-West 标准误**，滞后 4 期）；② Q1 多头组年化收益 > 基准年化 + 2%；③ Q1-Q5 分组单调；④ 三样本（沪深300/中证500/全A）表现方向一致；⑤ 行业中性化 IC 与市值中性化 IC 均显著 |
| **负责模型** | 高级模型写 spec + 代码审核 + Gate 判定；Flash 执行 |
| **防 data-mining** | 预注册通过标准不可事后调整；连续两轮 IC_IR<0.2 的因子方向放弃并记录原因；分析前先读 `research/_index.md` 历史结论 |

### Phase 2：多因子合成（Multi-Factor Synthesis）

| 项 | 内容 |
|----|------|
| **目标** | 基于 Phase 1 通过的因子，构建多因子模型 |
| **输入** | Phase 1 通过的因子及其结构化报告 |
| **步骤** | ① 等权打分 Baseline；② 因子相关性矩阵（Spearman Rank Corr > 0.6 视为重复暴露，去重或合并）；③ Fama-MacBeth 检验各因子独立截面定价能力（**重点排除国企=low-beta/size 代理、回购=无效**）；④ 基于 Fama-MacBeth 筛后因子做 Risk Parity（等风险贡献）合成 |
| **产出物** | 合成方案 spec + 回测结果 + 过拟合检验报告 + 归因报告（return-based，拆到 size/value/momentum/low-vol/quality） |
| **通过 Gate（Gate 2+3）** | ① 合成组合超额收益不被已知风险因子完全解释（残差 alpha 显著>0）；② 行业中性化后超额收益不显著下降（证明存在行业内选股能力，而非仅行业配置）；③ 因子暴露稳定无系统性漂移 |
| **负责模型** | 高级模型全责（权重方案、参数、Fama-MacBeth 解读）；Flash 跑回测 |
| **防 data-mining** | 权重方案在 spec 中预注册；Fama-MacBeth 用于"筛选"而非"组合构建"；不通过强制诊断"假设错误 vs 实现错误" |

### Phase 3：压力测试与稳健性（Stress Testing & Robustness）

| 项 | 内容 |
|----|------|
| **目标** | 验证策略在极端状态与跨样本下的稳健性 |
| **输入** | Phase 2 通过的合成策略 |
| **测试项** | ① Purged Walk-Forward CV（含 purging + embargoing）；② 按**制度断点**分段：注册制 2019.06、量化 DMA 踩踏 2024.02、新国九条 2024.04、MSCI 纳入 2017-18、杠杆牛 2014-15；③ 按**宏观状态**分段：通胀上行+紧货币（最差）、Goldilocks（最佳）、流动性危机（同跌）；④ Deflated Sharpe Ratio（多重比较修正） |
| **产出物** | 分段表现表 + 参数敏感性表（持仓数 10/20/30、PE 阈值 8/10/12、换仓频率月/季/半年）+ DSR 报告 |
| **通过 Gate（Gate 4）** | ① 各制度断点/宏观状态下表现方向不反转；② 参数敏感性：最优参数非边际极端值；③ DSR 通过多重比较修正 |
| **负责模型** | 高级模型设计测试方案 + 判定；Flash 批量跑 |
| **防 data-mining** | 高频换手（月换手>50%）的 alpha 必须证明能覆盖交易成本；CPCV 列为可选项，工程成本过高时不作硬性 Gate |

### Phase 4：实盘约束校准（Live-Trading Prep）

| 项 | 内容 |
|----|------|
| **目标** | 校准为可实盘执行的策略 |
| **输入** | Phase 3 通过的策略 |
| **校准项** | ① 成本后净收益（毛 vs 净，区分报告）；② 容量估算：千万/5000 万/1 亿资金下的冲击成本（升级 Almgren-Chriss）；③ 因子拥挤度监测（全市场因子 Z-score 均值上行、纳入 vs 剔除股的不对称行为）；④ 换手-alpha 闭环（边际信息收益 = 边际交易成本）；⑤ 实操约束落地（流动性日均成交额阈值、市值下限注册制后上调、个股 10% 上限、行业相对风险预算 ±10%） |
| **产出物** | 实盘就绪策略 + 容量报告 + 拥挤度监控仪表盘 + 失效预案（IC 归零/反转/相关性跳跃/拥挤触发应对） |
| **通过 Gate（Gate 5）** | ① 扣除保守交易成本后仍有正期望；② 容量匹配目标资金规模；③ 拥挤度未进入历史 90 分位警戒 |
| **负责模型** | 高级模型判定；Flash 跑容量/拥挤度扫描 |
| **防 data-mining** | 不新增因子；本阶段只做约束校准与监控搭建 |

---

## 四、迭代终止与确认偏误抑制规则（落地 agent-workflow）

1. **代码审核是结果分析的前置条件**——看结果前必须完成 `prompts/03` 门禁，不可跳过。
2. **通过标准在 spec 中预注册**——结果出来后不可调整。
3. **不通过时强制诊断**"假设错误 vs 实现错误"——区分真负与假负。
4. **连续两轮 IC_IR<0.2 → 放弃**该因子方向；继续迭代前必须在 `research/_index.md` 记录"为什么这个方向仍合理"。
5. **所有结论必须有数据支持**——禁用"根据经验"作为判断依据。
6. **分析前先读** `research/_index.md` 历史结论，保持一致性，避免重复尝试已放弃方向。
7. **所有澄清与迭代轮次必须落盘**到 `research/_index.md`（见该文件命名规范）。

---

## 五、Definition of Done（每阶段完成定义）

| 阶段 | DoD |
|------|-----|
| Phase 0 | 目录/data_layer/factor_lib/模板齐全；data_layer PIT 单测 + 三类撮合单测通过；`_index.md` 命名规范就位 |
| Phase 1 | 五大因子各有 spec+代码+审核记录+三表+净值+报告；通过因子在 `_index.md` 标"通过"，未通过标"放弃/原因" |
| Phase 2 | 合成策略通过 Gate 2+3；Fama-MacBeth 筛选记录完整；归因报告产出 |
| Phase 3 | 分段/敏感性/DSR 三项齐全；Gate 4 判定有书面依据 |
| Phase 4 | 成本后净收益为正；容量报告 + 拥挤度监控 + 失效预案齐全；Gate 5 判定通过 |

---

## 六、与原文档的关系

- 本计划**不修改**三份原文档，它们作为背景与理念来源保留。
- 当 agent-workflow、thinking-prompt、theory 与本计划冲突时，**以本计划为准**。
- 提示词模板（`docs/prompts/*`）是本计划的执行抓手，必须与本计划口径严格一致。
- 任何对本计划核心口径（第一节）的修订，必须同步更新 `plan-review.md` 的冲突清单状态，并在 `research/_index.md` 记录修订原因。
