<p align="center"><a href="./README.md">English</a> | <a href="./README-zh.md">中文</a></p>

<p align="center">
  <h1 align="center">🥬 epic-leek-quant</h1>
  <p align="center">
    <em>Quantifying deep-value "cigar-butt" investing for the A-share market</em>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="License">
    <img src="https://img.shields.io/badge/phase-0%20Scaffolding%20✅-brightgreen" alt="Phase">
    <img src="https://img.shields.io/badge/platform-JoinQuant-blue" alt="Platform">
    <img src="https://img.shields.io/badge/language-Python%203.13-blue" alt="Language">
    <img src="https://img.shields.io/badge/status-Research%20Only-lightgrey" alt="Status">
  </p>
</p>

---

A research repository that converts the deep-value "cigar-butt" investing philosophy of Bilibili finance creator **"Epic Leek" (史诗级韭菜)** into a backtestable, attributable, live-tradable multi-factor stock-selection framework on the [JoinQuant](https://www.joinquant.com) platform.

**Core idea**: *Capture cross-cycle mean-reversion returns by exploiting systematic market sentiment biases (neglect / pessimism), while controlling for bankruptcy risk.*

Every factor is validated through a disciplined pipeline: pre-registered spec → Flash execution → independent code review → result adjudication. Optimistic figures from the original source are treated as **hypotheses to be validated**, not conclusions.

## Five Factors

<table>
<tr>
<td width="420">

#### F1 — Negative EV (Net Cash > Market Cap + Liabilities)

**Principle:** Financial soundness — market gives negative price to operating assets.

**Factor:** `factor_ev_negative` (binary)

**Lineage:** Extreme subset of Graham's Net-Net (NCAV) strategy.

</td>
<td>

| Field | Source |
|-------|--------|
| `market_cap` | `valuation.market_cap` |
| `total_liability` | `balance.total_liability` |
| `cash_equivalents` | `balance.cash_equivalents` |

**Key risk:** shell-value pollution pre-2019 (registration reform)

</td>
</tr>
</table>

<details>
<summary><strong>Theory & A-share adaptation</strong></summary>

- Graham NCAV: market cap < (current assets − total liabilities) × 2/3
- Mohanty & Oxman (2026): NCAV on US stocks 1969–2019, alpha 13.9%/year surviving FF5 + liquidity controls
- A-share risk: 2016 backdoor-listing rule cut small-cap EV<0 returns from 28% to −5%; 2019.06 registration reform further depressed shell value
- Phase 1 must split IC at 2019.06 and run a "shell-value-purged" variant

</details>

<table>
<tr>
<td width="420">

#### F2 — High Earnings Yield (PE < 10)

**Principle:** Classic low-valuation value factor.

**Factor:** `factor_low_pe` (binary) + `earnings_yield` (continuous)

**Lineage:** Basu (1977); Fama-French HML extreme tail.

</td>
<td>

| Field | Source |
|-------|--------|
| `pe_ttm` | `valuation.pe_ttm` |
| `net_profit` | `income.net_profit` |
| `market_cap` | `valuation.market_cap` |

**A-share evidence:** EP more stable than BM (Hu et al. 2019)

</td>
</tr>
</table>

<details>
<summary><strong>Theory & A-share adaptation</strong></summary>

- Basu (1977): low PE stocks earn excess returns
- Hu, Chen, Shao & Wang (2019): in A-shares, EP (earnings yield) is more stable than BM (book-to-market); BM weakened post-2011
- Gu et al. (2024): EP monthly mean 0.59% (t=3.58***) across full sample; remains 0.55%** in 2011–2020
- Adaptation: use EP not BM; consider industry-tiered PE thresholds (consumer 10–14x / manufacturing 6–9x / utility 8–10x); run FCF/Y cross-validation

</details>

<table>
<tr>
<td width="420">

#### F3 — Sustained Shareholder Return (Dividend + Buyback)

**Principle:** Long-term dividend is the best proof of profit reality.

**Factor:** `div_yield_continuous` (continuous) + consecutive-dividend flag (binary)

**Lineage:** Dividend yield factor; CSI Dividend index family.

</td>
<td>

| Field | Source |
|-------|--------|
| `div_yield` | `valuation.div_yield` |
| `dividend_payable` | `balance.dividend_payable` |

**A-share caveat:** buyback has near-zero cross-sectional variance

</td>
</tr>
</table>

<details>
<summary><strong>Theory & A-share adaptation</strong></summary>

- Dividend strategy works only if ex-right gap is filled (填权)
- Bank-sector study (2026): significant pre-right scramble (+0.50% vs CSI300 in ~6 days)
- CSI Dividend 2025 reconstitution: incoming names ROE 12.93% vs outgoing 5.33% — quality filter is hidden alpha
- A-share buyback scale negligible vs US; demote buyback to auxiliary signal, not standalone factor
- Must report "raw dividend IC" vs "profit-stability-filtered dividend IC"

</details>

<table>
<tr>
<td width="420">

#### F4 — State-Owned Background

**Principle:** Institutional advantage in financing, policy support, extreme-risk backstop.

**Factor:** `factor_state_owned` (binary)

**Lineage:** A-share "China Special Valuation" (中特估) research.

</td>
<td>

| Field | Source |
|-------|--------|
| `actual_controller` | `valuation.actual_controller` |

**Core doubt:** likely a size / low-beta proxy

</td>
</tr>
</table>

<details>
<summary><strong>Theory & A-share adaptation</strong></summary>

- SOE discount may be rational pricing for agency problems (inefficiency), not mispricing
- Gu et al. (2024): A-share low-vol and size factors are strong; SOEs are naturally low-beta large-cap
- Zhang Yue (SWUFE): shell-value premium still significant pre-registration, weakened post
- Phase 1 must run Fama-MacBeth controlling size + low-vol; split IC at 2019.06
- If control kills the factor → abandon or demote

</details>

<table>
<tr>
<td width="420">

#### F5 — Financial Quality (Low Leverage + Positive OCF)

**Principle:** Downside protection, not stock-picking alpha.

**Factor:** `factor_low_leverage` (binary) + `factor_positive_cf` (binary)

**Lineage:** Quality factor family; A-share ROE is most stable.

</td>
<td>

| Field | Source |
|-------|--------|
| `total_liability` | `balance.total_liability` |
| `total_assets` | `balance.total_assets` |
| `operating_cash_flow` | `cash_flow.operating_cash_flow` |

**Bug fix:** B1 — compute ratio first, then median

</td>
</tr>
</table>

<details>
<summary><strong>Theory & A-share adaptation</strong></summary>

- Gu et al. (2024): ROE is the most stable quality factor in A-shares (0.52%/mo, t=3.41***); rises to 0.73%*** post-2011
- ACC / NOA / INV (effective in US) FAIL in A-shares — do not copy US quality factors blindly
- Core value is conditional VaR in tail scenarios (2015Q3 / 2018Q4 / 2024Q1), not IC
- Low-leverage + positive-OCF overlaps with low-vol → must control low-vol in Phase 2

</details>

## Phase Roadmap

| Phase | Goal | Status | Gate |
|-------|------|--------|------|
| **0** Scaffolding | Directory / data interface / factor lib / templates | ✅ Done | — |
| **1** Single-factor validation | IC validation for each of 5 factors, from EV<0 | ⏳ Pending | Gate 1 |
| **2** Multi-factor synthesis | Fama-MacBeth screening + Risk Parity | ⏸ Blocked | Gate 2+3 |
| **3** Stress testing | Purged WF-CV / regime breaks / DSR | ⏸ Blocked | Gate 4 |
| **4** Live-trading prep | Net-of-cost return / capacity / crowding | ⏸ Blocked | Gate 5 |

Each phase must pass its Gate before advancing. Definitions: `docs/PROJECT-PLAN.md` §3.

## Key Calibration

> Overrides original docs on conflict. Full version: `docs/PROJECT-PLAN.md` §1.

| Topic | Calibration |
|-------|-------------|
| Rebalance | **Quarterly** (after disclosure deadlines: 4/30, 8/31, 10/31, next-year 4/30) |
| Commission / stamp / min | 0.025% both sides / 0.1% sell-only / ¥5 per trade |
| Slippage / impact | 0.3% / tiered (Almgren-Chriss in Phase 4) |
| Benchmark | CSI 300 TR (primary) + CSI 800 TR (secondary); TE/IR mandatory |
| Delisting / limits / T+1 / suspension | Must include; queue matching; sell-first; freeze until resumption |
| PIT data | `get_fundamentals(date=)` PIT semantics; code-review gate mandatory |
| Neutralization | Industry + size neutralized IC; Fama-MacBeth in Phase 2 |
| Binary factors | Equal-weight sum, no Z-score; continuous: 1%/99% winsorize + Z-score |

## Getting Started

```bash
git clone https://github.com/IdealAuror/epic-leek-quant.git
cd epic-leek-quant
```

**Reading order:**
1. `README.md` (this file)
2. `docs/PROJECT-PLAN.md` — single source of truth
3. `docs/plan-review.md` — calibration conflict list (C1–C13, B1–B10)
4. `research/thinking-prompt.md` — academic standard
5. `research/theory-framework.md` — web-researched theory + A-share adaptation
6. `epic-leek-value-investing.md` — original theory (optimistic figures need "to-be-validated" tag)

**Run in JoinQuant:** all data access must go through `joinquant/data_layer.py` — direct `get_fundamentals` is forbidden.

## Repository Layout

```
epic-leek-quant/
├── README.md                            ← English (this file)
├── README-zh.md                         ← 中文
├── LICENSE                              ← Apache 2.0
├── .gitignore
├── epic-leek-value-investing.md         ← original theory (unchanged)
├── docs/
│   ├── PROJECT-PLAN.md                  ← ★ single source of truth
│   ├── plan-review.md                   ← conflict list C1–C13 / B1–B10
│   ├── phase-0-status.md                ← Phase 0 DoD checklist
│   ├── agent-workflow.md                ← agent workflow (unchanged)
│   └── prompts/                         ← DeepSeek prompt templates
│       ├── 00-system-role.md
│       ├── 01-research-spec-design.md
│       ├── 02-flash-execution.md
│       ├── 03-code-review-gate.md
│       └── 04-result-adjudication.md
├── joinquant/
│   ├── data_layer.py                    ← PIT / delisting / limit-order / sell-first
│   ├── factor_lib.py                    ← five-factor calc (Phase 1)
│   └── strategies/                      ← one file per factor/variant (Phase 1+)
├── research/
│   ├── thinking-prompt.md               ← academic framework (unchanged)
│   ├── theory-framework.md              ← web-researched theory + A-share
│   ├── _index.md                        ← ★ research dashboard / H1–H15 hypotheses
│   ├── specs/                           ← per-round research-spec
│   ├── reports/                         ← per-round analysis reports
│   └── decisions/                       ← factor keep/drop decisions
└── results/                             ← backtest outputs (gitignored large files)
```

---

<p align="center">
  <sub>Licensed under <a href="LICENSE">Apache 2.0</a>. Research only — <strong>not investment advice</strong>. Past performance does not guarantee future returns.</sub>
</p>
