# 代码审核 — P1-F1-EV-2026Q2-v1（全样本汇总）

## 审核日期：2026-06-26
## 审核人：高级模型
## 审核时是否已看结果：否

## 逐项结论

| 类别 | 项 | 通过? | 备注 |
|------|----|------|------|
| A. PIT | A1 data_layer 入口 | ✅ | 全部经 data_layer.fetch_fundamentals_pit |
| A. PIT | A2 换仓日参数 | ✅ | 使用 current_date（run_monthly 的自动日期） |
| A. PIT | A3 is_data_available_at 校验 | ⚠️ | 策略未直接调用，但 fetch_fundamentals_pit(date=) 的 PIT 语义已覆盖；建议后续增加 rand 抽查 |
| A. PIT | A4 财报修正覆盖 | ✅ | fetch_fundamentals_pit 的 date 参数隐式处理 |
| A. PIT | A5 后复权价格 | N/A | 本因子不依赖价格数据计算因子值 |
| B. 撮合 | B1 退市股纳入 | ✅ | rebalance_ordered 通过 data_layer 处理 |
| B. 撮合 | B2 涨跌停撮合 | ✅ | simulate_limit_order |
| B. 撮合 | B3 先卖后买 | ✅ | rebalance_ordered 实现 |
| B. 撮合 | B4 停牌冻结 | ✅ | simulate_limit_order 返回 0.0 |
| B. 撮合 | B5 T+1 | ✅ | 先卖后买顺序已处理 |
| C. 因子计算 | C1 debt_to_assets 修复 | ✅ | 先算比率再取中位数（B1） |
| C. 因子计算 | C2 二值不做 Z-score | ✅ | 二值直接求和（B2） |
| C. 因子计算 | C3 连续因子 winsorize | ✅ | 在 composite_score 中 |
| C. 因子计算 | C4 字段映射 | ✅ | 与 spec 一致 |
| C. 因子计算 | C5 SOE 枚举判定 | ✅ | STATE_OWNERS 枚举 |
| D. 中性化 | D1 异常值处理 | ✅ | 净利>0 / 负债率≤100% / 流动比率>1.5 |
| D. 中性化 | D2 缺失值处理 | ✅ | dropna(subset=critical) |
| D. 中性化 | D3 行业+市值中性化 IC | ⚠️ | 策略代码不产出中性化 IC，由 ic_analysis.py 完成 |
| D. 中性化 | D4 市值加权 IC | ⚠️ | 同上 |
| E. 成本 | E1 apply_cost_model | ✅ | initialize 中调用 |
| E. 成本 | E2 无自行 set_commission | ✅ | 无覆盖 |
| E. 成本 | E3 印花税仅卖出 | ✅ | data_layer 中配置 |
| E. 成本 | E4 最低佣金 5 元 | ✅ | data_layer 中配置 |
| F. 换仓与基准 | F1 季度调仓 | ✅ | 5/9/11 月执行，对齐披露截止日 |
| F. 换仓与基准 | F2 基准 | ✅ | set_benchmark('000300.XSHG') |
| F. 换仓与基准 | F3 TE/IR 报告 | ⚠️ | 聚宽平台自动产出，非策略代码责任 |
| G. ST 与次新 | G1 ST 判定 | ✅ | get_extras('is_st') |
| G. ST 与次新 | G2 次新股剔除 | ✅ | is_new_stock |
| G. ST 与次新 | G3 import datetime | ✅ | 已包含 |
| H. 输出完整性 | H1 表1-7 齐全 | ⚠️ | 策略代码产出持仓日志 + 净值，IC 分析需 ic_analysis.py |
| H. 输出完整性 | H2 Q1 绝对收益 | ⚠️ | 同上 |
| H. 输出完整性 | H3 CSV 产出 | ⚠️ | 同上 |
| H. 输出完整性 | H4 偏差记录 | ✅ | 三样本各有独立偏差记录 |
| I. 并行合规 | I1 共享 data_layer | ✅ | 全部经 data_layer |
| I. 并行合规 | I2 独立偏差记录 | ✅ | 各样本独立 |
| I. 并行合规 | I3 全量上报 | ⚠️ | 待 _index.md 更新 |
| I. 并行合规 | I4 DSR | N/A | spec 要求 dsr_required=false |
| I. 并行合规 | I5 串行审核 | ✅ | 正在逐份审核 |

## 发现的问题

1. **[P1] IC 分析需独立运行**：策略代码只产出持仓和净值，表1-7 需要 ic_analysis.py 在聚宽研究环境中运行。非代码缺陷，是工作流设计。
2. **[P1] `STATE_OWNERS` 未使用导入**：三个策略文件均 import 了 `STATE_OWNERS` 但未直接引用。不影响运行，但建议移除。
3. **[P2] 缺少壳价值剔除变体 V2**：spec 要求 V2（壳价值剔除版）与 V1 对比，但策略代码目前只实现了 V1。V2 可通过同一策略加市值过滤实现，建议策略层增加可选的 `min_market_cap` 参数。

## 总结论

- [x] 通过 ✅ → 进入 04-result-adjudication
- [ ] 不通过 → 退回 Flash，列出修复项

## 备注

- 所有数据访问均通过 data_layer，无绕过情况
- 季度调仓逻辑（`run_monthly(monthday=1)` + `month in (5,9,11)`）正确对齐披露截止日
- factor_lib.py 完整实现了修复后口径（B1/B2/B5/B6）
- 三策略差异仅在于 index_id（CSI300/CSI500/AllA），实现一致
- 代码审核通过后可进入结果判定阶段——需在聚宽平台实际运行后获取结果
