[English](README.md) | [中文](README-zh.md)

# 🌱 cst-quant · 低估分散量化

> A 股深度价值量化策略，借鉴格系"捡烟蒂"投资理念。
> 三个透明因子 — 低估、低波、趋势动量 — 等权合成，
> 50 只等权重持仓，季度调仓，始终满仓。不择时、不看宏观、不依赖黑箱模型。

**核心成果** — 2014-01 ~ 2026-06（12.5年）：累计收益 **+776.27%**（基准 +121.24%），
年化 **+19.08%**，夏普 **0.67**，最大回撤 −37.18%。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform: JoinQuant](https://img.shields.io/badge/平台-聚宽-blue)](#)
[![Backtest: 2013–2026](https://img.shields.io/badge/回测-2014.01–2026.06-orange)](#)
[![Sharpe: 0.67](https://img.shields.io/badge/夏普-0.67-brightgreen)](#)
[![Annual Return: 19%](https://img.shields.io/badge/年化-19.08%25-brightgreen)](#)
[![Status: Production](https://img.shields.io/badge/状态-生产就绪-success)](#)

---

## 当前策略

| 因子 | 权重 | 说明 |
|------|------|------|
| **F2 EP** | 1/3 | 盈利收益率 = 净利润 / 总市值 |
| **F5 LowVol** | 1/3 | −40日收益率标准差 |
| **F6 MOM-40d** | 1/3 | 61-21 动量（40天趋势窗口） |

| 指标 | 回测值 |
|------|--------|
| 回测区间 | 2014-01-01 ~ 2026-06-28（12.5年） |
| 累计收益 | **+776.27%**（基准 +121.24%） |
| 最大回撤 | −37.18% |
| Sharpe / Alpha / Beta | 0.67 / 0.13 / 0.77 |

> 跨样本验证（全A / 沪深300 / 中证500）2026-06-29 **OVERALL: PASS**。
> 策略代码：[`results/P5-F2F5F6-40d-final-strategy.py`](results/P5-F2F5F6-40d-final-strategy.py)

综合得分（三因子等权 Z-score）：
```
综合分 = ⅓·Z(EP) + ⅓·Z(−波动) + ⅓·Z(动量)  →  取前 50
```

---

## 项目结构

```
cst-quant/
├── README.md
├── README-zh.md
├── epic-leek-value-investing.md
├── requirements.txt
├── LICENSE
├── archive/                           # 历史归档
├── docs/
│   ├── CURRENT-STRATEGY.md            # 最终策略总览
│   ├── PROJECT-PLAN.md                # 执行计划
│   ├── manual-investment-guide.md     # 手动实盘指南
│   ├── prompts/                       # 研究流水线 (00-04)
│   └── task-state/                    # 任务状态与调试笔记
├── research/
│   ├── _index.md                      # 研究看板
│   ├── scripts/                       # 研究脚本（聚宽粘贴运行）
│   ├── reports/                       # 分析报告
│   ├── decisions/                     # 决策记录
│   └── specs/                         # 研究规格
└── results/                           # 最终交付物
    ├── P5-F2F5F6-40d-final-strategy.py    # 生产策略 (VOL=40, 777%/S=0.67)
    ├── manual-investment-guide.html        # 投资操作手册（网页版）
    ├── manual-investment-guide.md          # 投资操作手册（MD版）
    └── P5-F6-MOM-2026Q2-v1/               # 跨样本验证数据
```

---

## 快速开始

### 聚宽平台运行

**策略环境**（正式回测，真实成本）：
```python
# 粘贴 results/P5-F2F5F6-40d-final-strategy.py → 运行
```

**研究环境**（验证分析，跨样本对比）：
```python
# 粘贴 research/scripts/P5-F6-MOM-2026Q2-v1-segmented-standalone.py → Run
```

> 脚本均为单文件自包含，无需 import 本地模块。

### 本地环境
```bash
git clone https://github.com/IdealAuror/Cheap-Stable-Trending-quant.git
cd Cheap-Stable-Trending-quant
pip install -r requirements.txt
```

---

## 实验全貌

| Phase | 因子 / 主题 | 结论 |
|-------|------------|------|
| P1-F1 | EV<0（净现金） | ❌ Size proxy，放弃 |
| P1-F2 | EP（盈利收益率） | ✅ 核心 alpha |
| P1-F3 | 股息率 | 🟡 IC 通过，断点后失效 |
| P1-F4 | ROE（质量） | ❌ 负 alpha，放弃 |
| P1-F5 | LowVol | ⚠️ 仅作辅助因子 |
| P2 | 多因子合成 | 🟡 F2 唯一 alpha，F5 风险调节 |
| P3 | 压力测试 | 🟡 3/4 通过 |
| P4 | 实盘校准 V1/V2 | ❌ M2/M5 未达标 |
| **P5** | **F6-MOM 动量** | **✅ 三样本 PASS** |

---

## 文档

| 文档 | 读者 |
|------|------|
| [`results/manual-investment-guide.html`](results/manual-investment-guide.html) | 投资者 — 网页版手册（推荐） |
| [`docs/CURRENT-STRATEGY.md`](docs/CURRENT-STRATEGY.md) | 所有人 — 策略总览 |
| [`research/_index.md`](research/_index.md) | 研究者 — 因子看板 |
| [`results/P5-F2F5F6-40d-final-strategy.py`](results/P5-F2F5F6-40d-final-strategy.py) | 开发者 — 生产代码 |

## License

MIT · 仅作研究用途 · 不构成投资建议 · 历史回测不代表未来收益
