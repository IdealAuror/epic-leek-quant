[English](README.md) | [中文](README-zh.md)

# 🌱 cst-quant · Cheap · Stable · Trending

> A-share deep-value quantitative strategy, inspired by Graham-style "cigar butt" investing.
> Three transparent factors — cheap valuation, low volatility, trend momentum — equally weighted,
> 50 stocks, quarterly rebalancing, always fully invested. No market timing, no macro calls,
> no black boxes.

**Key Result** — 2014-01 ~ 2026-06 (12.5 yrs): **+776.27%** cumulative return vs benchmark +121.24%,
annualized **+19.08%**, Sharpe **0.67**, max drawdown −37.18%.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform: JoinQuant](https://img.shields.io/badge/Platform-JoinQuant-blue)](#)
[![Backtest: 2014–2026](https://img.shields.io/badge/Backtest-2014.01–2026.06-orange)](#)
[![Sharpe: 0.67](https://img.shields.io/badge/Sharpe-0.67-brightgreen)](#)
[![Annual Return: 19%](https://img.shields.io/badge/Annual_Return-19.08%25-brightgreen)](#)
[![Status: Production](https://img.shields.io/badge/Status-Production-success)](#)

---

## Live Strategy

| Factor | Weight | Description |
|--------|--------|-------------|
| **F2 EP** | 1/3 | Earnings yield = net profit / market cap |
| **F5 LowVol** | 1/3 | −40d daily return std dev |
| **F6 MOM-40d** | 1/3 | 61-21 momentum (40-day trend window) |

| Metric | Value |
|--------|-------|
| Period | 2014-01-01 ~ 2026-06-28 (12.5 yrs) |
| Cumulative Return | **+776.27%** (benchmark +121.24%) |
| Max Drawdown | −37.18% |
| Sharpe / Alpha / Beta | 0.67 / 0.13 / 0.77 |

> Cross-sample validation (AllA / CSI300 / CSI500) — 2026-06-29 — **OVERALL: PASS**.
> Code: [`results/P5-F2F5F6-40d-final-strategy.py`](results/P5-F2F5F6-40d-final-strategy.py)

Composite score (equal-weight Z-score):
```
Score = ⅓·Z(EP) + ⅓·Z(−Vol) + ⅓·Z(MOM)  →  top 50
```

---

## Project Structure

```
cst-quant/
├── README.md
├── README-zh.md
├── epic-leek-value-investing.md
├── requirements.txt
├── LICENSE
├── archive/                           # Deprecated scripts
├── docs/
│   ├── CURRENT-STRATEGY.md            # Final strategy overview
│   ├── PROJECT-PLAN.md                # Execution plan
│   ├── manual-investment-guide.md     # Manual investing guide
│   ├── prompts/                       # Research workflow prompts (00-04)
│   └── task-state/                    # Task state & debug notes
├── research/
│   ├── _index.md                      # Research dashboard
│   ├── scripts/                       # Research scripts (JoinQuant paste-to-run)
│   ├── reports/                       # Analysis reports
│   ├── decisions/                     # Decision records
│   └── specs/                         # Research specs
└── results/                           # Final deliverables
    ├── P5-F2F5F6-40d-final-strategy.py    # Production strategy (VOL=40, 777%/S=0.67)
    ├── manual-investment-guide.html        # Investing manual (HTML)
    ├── manual-investment-guide.md          # Investing manual (Markdown)
    └── P5-F6-MOM-2026Q2-v1/               # Cross-sample validation
```

---

## Quick Start

### On JoinQuant (run backtests)

**Strategy environment** (production, real costs):
```python
# Paste results/P5-F2F5F6-40d-final-strategy.py → Run
```

**Research environment** (validation, cross-sample):
```python
# Paste research/scripts/P5-F6-MOM-2026Q2-v1-segmented-standalone.py → Run
```

> All scripts are self-contained — no local imports needed.

### Locally (code review / IDE)
```bash
git clone https://github.com/IdealAuror/Cheap-Stable-Trending-quant.git
cd Cheap-Stable-Trending-quant
pip install -r requirements.txt
```

---

## Experiment Summary

| Phase | Factor / Topic | Verdict |
|-------|---------------|---------|
| P1-F1 | EV<0 (net cash) | ❌ Size proxy, abandoned |
| P1-F2 | EP (earnings yield) | ✅ Core alpha |
| P1-F3 | Dividend yield | 🟡 IC passed, post-break failure |
| P1-F4 | ROE (quality) | ❌ Negative alpha, abandoned |
| P1-F5 | LowVol | ⚠️ Auxiliary factor only |
| P2 | Multi-factor synthesis | 🟡 F2 sole alpha, F5 risk adjuster |
| P3 | Stress testing | 🟡 3/4 robustness checks passed |
| P4 | Live calibration V1/V2 | ❌ M2/M5 failed Gate 5 |
| **P5** | **F6-MOM momentum** | **✅ Cross-sample PASS** |

---

## Docs

| Document | Audience |
|----------|----------|
| [`results/manual-investment-guide.html`](results/manual-investment-guide.html) | Investors — HTML manual (recommended) |
| [`docs/CURRENT-STRATEGY.md`](docs/CURRENT-STRATEGY.md) | Everyone — strategy overview |
| [`research/_index.md`](research/_index.md) | Researchers — factor dashboard |
| [`results/P5-F2F5F6-40d-final-strategy.py`](results/P5-F2F5F6-40d-final-strategy.py) | Developers — production code |

## License

MIT · Research only · Not investment advice · Past performance ≠ future results
