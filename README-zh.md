<p align="center"><a href="./README.md">English</a> | <a href="./README-zh.md">中文</a></p>

<p align="center">
  <h1 align="center">🥬 epic-leek-quant</h1>
  <p align="center">
    <em>把深度价值"捡烟蒂"投资理念量化落地 A 股市场</em>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="许可证">
    <img src="https://img.shields.io/badge/phase-0%20Scaffolding%20✅-brightgreen" alt="阶段">
    <img src="https://img.shields.io/badge/platform-JoinQuant-blue" alt="平台">
    <img src="https://img.shields.io/badge/language-Python%203.13-blue" alt="语言">
    <img src="https://img.shields.io/badge/status-Research%20Only-lightgrey" alt="状态">
  </p>
</p>

---

研究仓库，把 B 站财经 UP 主**"史诗级韭菜"**的格系"捡烟蒂"深度价值投资理念，转化为可在[聚宽](https://www.joinquant.com)平台上回测、归因、实盘校准的多因子选股框架。

**核心理念**：*利用市场系统性情绪偏差（neglect / pessimism），在控制破产风险的前提下，捕获跨周期的估值回归收益。*

每个因子都经过纪律化管线验证：预注册 spec → Flash 执行 → 独立代码审核 → 结果判定。原文档的乐观数字一律视为**待验证假设**，不作为结论。

## 五大因子

<table>
<tr>
<td width="420">

#### F1 — EV 为负（净现金 > 总市值 + 总负债）

**原则：** 财务稳健——市场给经营资产贴了负价格。

**因子：** `factor_ev_negative`（二值）

**流派：** 格雷厄姆 Net-Net（NCAV）策略的极端子集。

</td>
<td>

| 字段 | 来源 |
|-------|--------|
| `market_cap` | `valuation.market_cap` |
| `total_liability` | `balance.total_liability` |
| `cash_equivalents` | `balance.cash_equivalents` |

**关键风险：** 2019 注册制前的壳价值污染

</td>
</tr>
</table>

<details>
<summary><strong>理论与 A 股适配</strong></summary>

- Graham NCAV：市值 <（流动资产 − 总负债）× 2/3
- Mohanty & Oxman (2026)：美股 1969–2019 NCAV alpha 13.9%/年，控制 FF5 + 流动性后仍显著
- A 股风险：2016 借壳新规后小市值 EV<0 收益从 28% 骤降至 −5%；2019.06 注册制进一步削弱壳价值
- Phase 1 必须以 2019.06 为断点分段 IC，并跑"壳价值净化版"变体

</details>

<table>
<tr>
<td width="420">

#### F2 — 高盈利收益率（PE < 10）

**原则：** 经典低估值价值因子。

**因子：** `factor_low_pe`（二值）+ `earnings_yield`（连续）

**流派：** Basu (1977)；Fama-French HML 极端尾部。

</td>
<td>

| 字段 | 来源 |
|-------|--------|
| `pe_ttm` | `valuation.pe_ttm` |
| `net_profit` | `income.net_profit` |
| `market_cap` | `valuation.market_cap` |

**A 股实证：** EP 比 BM 更稳定（Hu et al. 2019）

</td>
</tr>
</table>

<details>
<summary><strong>理论与 A 股适配</strong></summary>

- Basu (1977)：低 PE 股票获得超额收益
- Hu, Chen, Shao & Wang (2019)：A 股 EP（盈利收益率）比 BM（账面市值比）更稳定；BM 在 2011 后衰减
- 顾明等 (2024)：EP 全样本月均 0.59%（t=3.58***），2011–2020 子期仍 0.55%**
- 适配：用 EP 而非 BM；考虑行业分档 PE 阈值（消费 10–14x / 制造 6–9x / 公用 8–10x）；同时跑 FCF/Y 交叉验证

</details>

<table>
<tr>
<td width="420">

#### F3 — 持续股东回报（分红 + 回购）

**原则：** 长期分红是利润真实性的最好证明。

**因子：** `div_yield_continuous`（连续）+ 连续分红标志（二值）

**流派：** 股息率因子；中证红利指数族。

</td>
<td>

| 字段 | 来源 |
|-------|--------|
| `div_yield` | `valuation.div_yield` |
| `dividend_payable` | `balance.dividend_payable` |

**A 股注意：** 回购因子截面区分度近零

</td>
</tr>
</table>

<details>
<summary><strong>理论与 A 股适配</strong></summary>

- 分红策略有效的根本前提是填权
- 银行业研思录 (2026)：分红前存在显著抢权效应（约 6 天内相对沪深 300 +0.50%）
- 中证红利 2025 调样：调入 ROE 12.93% vs 调出 5.33%——质量过滤是隐性 alpha
- A 股回购规模相比美股可忽略；回购降级为辅助信号，不单独成因子
- 必须同时报告"裸股息率 IC"与"叠加利润稳定性过滤后的 IC"

</details>

<table>
<tr>
<td width="420">

#### F4 — 国企背景

**原则：** 融资便利、政策支持、极端风险兜底的制度性优势。

**因子：** `factor_state_owned`（二值）

**流派：** A 股"中特估"研究。

</td>
<td>

| 字段 | 来源 |
|-------|--------|
| `actual_controller` | `valuation.actual_controller` |

**核心质疑：** 极可能是 size / low-beta 代理变量

</td>
</tr>
</table>

<details>
<summary><strong>理论与 A 股适配</strong></summary>

- 国企折价可能是代理问题（低效/腐败）的合理定价，非定价错误
- 顾明等 (2024)：A 股低波动与规模因子显著，国企天然低 beta 大市值
- 张跃（西南财大）：壳价值溢价注册制前仍显著，注册制后削弱
- Phase 1 必须 Fama-MacBeth 控制 size + low-vol；以 2019.06 为断点分段 IC
- 若控制后失效 → 放弃或降级

</details>

<table>
<tr>
<td width="420">

#### F5 — 财务质量（低杠杆 + 正经营现金流）

**原则：** 下行保护，而非选股 alpha。

**因子：** `factor_low_leverage`（二值）+ `factor_positive_cf`（二值）

**流派：** 质量因子族；A 股 ROE 最稳定。

</td>
<td>

| 字段 | 来源 |
|-------|--------|
| `total_liability` | `balance.total_liability` |
| `total_assets` | `balance.total_assets` |
| `operating_cash_flow` | `cash_flow.operating_cash_flow` |

**Bug 修复：** B1 — 先算比率，再取中位数

</td>
</tr>
</table>

<details>
<summary><strong>理论与 A 股适配</strong></summary>

- 顾明等 (2024)：ROE 是 A 股最稳定的质量因子（0.52%/月，t=3.41***）；2011 后升至 0.73%***
- ACC / NOA / INV（美股有效）在 A 股失效——不能照搬美股质量因子
- 核心价值在极端尾部的条件 VaR（2015Q3 / 2018Q4 / 2024Q1），而非 IC
- 低杠杆 + 正 OCF 与 low-vol 重叠 → Phase 2 需控制 low-vol

</details>

## 阶段路线图

| 阶段 | 目标 | 状态 | Gate |
|-------|------|--------|------|
| **0** 脚手架 | 目录 / 数据接口 / 因子库 / 模板 | ✅ 完成 | — |
| **1** 单因子验证 | 5 大因子逐一 IC 验证，从 EV<0 起步 | ⏳ 待启动 | Gate 1 |
| **2** 多因子合成 | Fama-MacBeth 筛选 + Risk Parity | ⏸ 阻塞 | Gate 2+3 |
| **3** 压力测试 | Purged WF-CV / 制度断点 / DSR | ⏸ 阻塞 | Gate 4 |
| **4** 实盘校准 | 成本净收益 / 容量 / 拥挤度 | ⏸ 阻塞 | Gate 5 |

每阶段必须通过对应 Gate 才能进入下一阶段。定义见 `docs/PROJECT-PLAN.md` 第三节。

## 关键口径

> 与原文档冲突时以此为准。完整版见 `docs/PROJECT-PLAN.md` 第一节。

| 议题 | 口径 |
|------|------|
| 换仓频率 | **季度调仓**（披露截止后首个交易日：4/30、8/31、10/31、次年 4/30） |
| 佣金 / 印花税 / 最低 | 万 2.5 双边 / 千 1 仅卖 / 5 元每笔 |
| 滑点 / 冲击成本 | 0.3% / 分级估算（Phase 4 升级 Almgren-Chriss） |
| 基准 | 主沪深 300 全收益 + 辅中证 800 全收益；强制 TE/IR |
| 退市 / 涨跌停 / T+1 / 停牌 | 必须纳入；限价单排队撮合；先卖后买；停牌冻结至复牌首日 |
| PIT 数据 | `get_fundamentals(date=)` PIT 语义；代码审核门禁必检 |
| 中性化 | 行业 + 市值中性化 IC；Phase 2 用 Fama-MacBeth 控制市值 |
| 二值因子 | 等权求和不做 Z-score；连续因子 1%/99% winsorize + Z-score |

## 快速开始

```bash
git clone https://github.com/IdealAuror/epic-leek-quant.git
cd epic-leek-quant
```

**阅读顺序：**
1. `README.md`（英文）/ `README-zh.md`（本文件）
2. `docs/PROJECT-PLAN.md` — 单一事实来源
3. `docs/plan-review.md` — 口径冲突清单（C1–C13、B1–B10）
4. `research/thinking-prompt.md` — 学术标准
5. `research/theory-framework.md` — 互联网检索整理的理论框架与 A 股适配
6. `epic-leek-value-investing.md` — 原文理论（乐观数字需标注"待验证"）

**在聚宽运行：** 所有数据访问必须经 `joinquant/data_layer.py`，禁止直接调 `get_fundamentals`。

## 目录结构

```
epic-leek-quant/
├── README.md                            ← 英文
├── README-zh.md                         ← 中文（本文件）
├── LICENSE                              ← Apache 2.0
├── .gitignore
├── epic-leek-value-investing.md         ← 原文理论（不改）
├── docs/
│   ├── PROJECT-PLAN.md                  ← ★ 单一事实来源
│   ├── plan-review.md                   ← 冲突清单 C1–C13 / B1–B10
│   ├── phase-0-status.md                ← Phase 0 DoD 核对
│   ├── agent-workflow.md                ← Agent 工作流程（不改）
│   └── prompts/                         ← DeepSeek 提示词模板
│       ├── 00-system-role.md
│       ├── 01-research-spec-design.md
│       ├── 02-flash-execution.md
│       ├── 03-code-review-gate.md
│       └── 04-result-adjudication.md
├── joinquant/
│   ├── data_layer.py                    ← PIT / 退市 / 限价单 / 先卖后买
│   ├── factor_lib.py                    ← 五大因子计算（Phase 1）
│   └── strategies/                      ← 每个因子/变体一个文件（Phase 1+）
├── research/
│   ├── thinking-prompt.md               ← 分析框架（不改）
│   ├── theory-framework.md              ← 互联网检索理论 + A 股适配
│   ├── _index.md                        ← ★ 研究看板 / H1–H15 假设
│   ├── specs/                           ← 每轮 research-spec
│   ├── reports/                         ← 每轮分析报告
│   └── decisions/                       ← 因子去留决策
└── results/                             ← 回测输出（大文件 gitignore）
```

---

<p align="center">
  <sub>基于 <a href="LICENSE">Apache 2.0</a> 协议。仅作研究用途——<strong>不构成投资建议</strong>。过往业绩不代表未来收益。</sub>
</p>
