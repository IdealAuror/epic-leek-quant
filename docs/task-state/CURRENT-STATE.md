# epic-leek-quant 任务状态看板

> **用途**：跨会话/跨设备同步项目执行进度。每次会话开始先读本文件，结束前更新。
> **最后更新**：2026-06-26 22:16
> **当前阶段**：Phase 1 — F1 EV<0 因子验证
> **单一事实来源**：`docs/PROJECT-PLAN.md`（口径冲突以此为准）；本文件仅记录执行流程进度。

---

## 一、总体进度地图

| 阶段 | 状态 | 说明 |
|------|------|------|
| Phase 0 脚手架 | ✅ 完成 | 目录/模板/data_layer 就位。缺口：撮合层独立单测（聚宽环境补齐） |
| Phase 1 — F1 EV<0 | 🟡 执行中 | spec locked、代码生成、审核通过、待聚宽回测结果 |
| Phase 1 — F2 PE | ⏳ 阻塞于 F1 | — |
| Phase 1 — F3 股东回报 | ⏳ 阻塞于 F2 | — |
| Phase 1 — F4 国企背景 | ⏳ 阻塞于 F3 | — |
| Phase 1 — F5 财务质量 | ⏳ 阻塞于 F4 | — |
| Phase 1 — MF 多因子合成 | ⏳ 阻塞于 Phase 1 | — |
| Phase 2/3/4 | ⏳ 未启动 | 压测/跨市场/实盘 |

---

## 二、F1 EV<0（P1-F1-EV-2026Q2-v1）分步进度

> 工作流：spec 设计 → Flash 执行 → 代码审核 → 结果判定(Gate 1) → _index 落盘

| # | 步骤 | 状态 | 交付物 | 备注 |
|---|------|------|--------|------|
| 1 | spec 设计与 locked | ✅ | `research/specs/P1-F1-EV-2026Q2-v1.md` | _index 变更日志确认 locked |
| 2 | factor_lib.py 实现 | ✅ | `joinquant/factor_lib.py` | calculate_all_factors + composite_score |
| 3 | 策略代码生成（三样本） | ✅ | `joinquant/strategies/P1-F1-EV-2026Q2-v1-{AllA,CSI300,CSI500}-standalone.py` | 三样本结构一致 |
| 4 | IC 分析脚本生成 | ✅ | `research/scripts/P1-F1-EV-2026Q2-v1-ic_analysis.py` | 表1-7 由其在聚宽研究环境产出 |
| 5 | 执行偏差记录（三样本） | ✅ | `research/reports/P1-F1-EV-2026Q2-v1-{AllA,CSI300,CSI500}-deviation.md` | 标注 time_deposits_note 不可得等偏差 |
| 6 | _index.md 更新 | ✅ | `research/_index.md` | F1=🟡进行中，迭代轮次=1 |
| 7 | 代码审核（03 门禁） | ✅ 通过 | `research/reports/P1-F1-EV-2026Q2-v1-code-review.md` | 审核时未看结果；3 项待办见下 |
| 8 | bug 修复（order_target_percent NameError） | ✅ | 三策略文件新增交易 API 兼容垫片 | jqboson 引擎未注入交易函数，已加降级链 |
| 9 | **聚宽平台回测执行** | ✅ 三样本完成 | `results/P1-F1-EV-2026Q2-v1/` | AllA/CSI300/CSI500 均跑通；待 IC 分析 |
| 10 | IC 分析执行（聚宽研究环境） | ⏳ 待执行 | 表1-7 + CSV | 依赖回测净值产出 |
| 11 | 结果判定（Gate 1 七条） | ⏳ 待执行 | `research/reports/P1-F1-EV-2026Q2-v1-adjudication.md` | 需先有回测结果 |
| 12 | _index.md 终态落盘 | ⏳ 待执行 | F1 状态→✅/❌/⚠️ | Gate 1 判定后 |

**当前卡点**：步骤 9 — 需在聚宽平台运行修复后的三样本策略，导出净值 CSV。

---

## 三、代码审核遗留待办（来自 code-review.md）

| 优先级 | 问题 | 状态 |
|--------|------|------|
| P1 | IC 分析需在聚宽研究环境独立运行 ic_analysis.py（非代码缺陷，工作流设计） | 待执行 |
| P1 | 三策略 import 了 `STATE_OWNERS` 但未直接引用（不影响运行） | 待清理 |
| P2 | 缺壳价值剔除变体 V2（spec 要求 V1/V2 对比，策略目前只实现 V1） | 待补 `min_market_cap` 参数 |

---

## 四、下一步行动（按优先级）

1. **在聚宽平台运行修复后的策略**（三样本可并行）：
   - 粘贴 `P1-F1-EV-2026Q2-v1-CSI300-standalone.py` 等到聚宽回测
   - 上次 NameError 已修复（新增 `_safe_order_target_percent` 降级链）
   - 导出 Q1 多头组累计净值 CSV 到 `results/P1-F1-EV-2026Q2-v1/`
2. **在聚宽研究环境运行** `research/scripts/P1-F1-EV-2026Q2-v1-ic_analysis.py` 产出表1-7
3. **结果判定**：按 `docs/prompts/04` + Gate 1 七条标准判定，落盘 adjudication 报告
4. **清理 P1/P2 待办**（可在回测期间并行处理）

---

## 五、已知偏差与风险（来自 deviation 报告）

- **附注定期存款不可得**：聚宽 balance 表无 `time_deposits_note`，`cash_available_ext` 仅取 `monetary_funds + financial_assets_held_for_trading`，净现金可能高估 EV（低估现金）。偏差已记录。
- **有息负债为估算值**：聚宽无直接 `interest_bearing_debt` 字段，用 total_liability 扣除非有息流动负债估算。
- **全 A 样本耗时**：AllA 股票池大，回测耗时较长。

---

## 六、会话变更日志

> 每次会话结束追加一行。格式：日期 | 会话主题 | 变更摘要

| 日期 | 会话主题 | 变更摘要 |
|------|---------|---------|
| 2026-06-26 | Phase 0 脚手架 | 目录/模板/data_layer.py/_index.md 创建；factor_lib 推迟 |
| 2026-06-26 | F1 spec 设计 + Flash 执行 | spec locked；factor_lib.py 实现；三策略+ic_analysis 生成；偏差记录；代码审核通过 |
| 2026-06-26 | 修复 order_target_percent NameError | 三策略文件新增聚宽交易 API 兼容垫片（降级链 order_target_percent→order_target_value→order_target→order） |
| 2026-06-26 | 创建任务状态看板 | 新建 `docs/task-state/CURRENT-STATE.md`，跨会话同步进度 |
| 2026-06-26 | AllA 首跑成功 | P1-F1-EV-2026Q2-v1-AllA 回测跑通：策略211.58%/基准127.64%/Sharpe0.24/回撤65.94%。证明 NameError 修复有效；待 CSI300/CSI500 + IC 分析 |
| 2026-06-26 | CSI300 跑通 | 策略116.77%/基准115.45%/Alpha0.00/Sharpe0.12/回撤45.36%。近乎跑平基准，与 AllA 方向分化，疑因子在大盘股失效 |
| 2026-06-26 | CSI500 跑通 | 策略123.94%/基准133.42%/Alpha-0.00/Sharpe0.12/回撤62.87%。跑输基准。三样本综合：因子仅 AllA 有效，疑壳价值污染驱动 |
| 2026-06-26 | ic_analysis standalone 重写+调试 | 经 7 轮调试修复聚宽环境兼容问题（字段名/单位/panel/index类型），快速验证 CSI300 3 个调仓日跑通表1-7。待全量运行 |
| 2026-06-26 | 聚宽调试经验落盘 | 新建 docs/task-state/JOINQUANT-DEBUG-NOTES.md，记录字段名差异/单位陷阱/panel行为/研究vs回测环境区分等踩坑经验 |
| 2026-06-26 | ic_analysis.py 重写完成 | 修复二值因子无法Q1-Q5缺陷（改用连续ev排序，Q1=净现金最多=多头）；补全表3/4/5/7+V2壳价值剔除；符号约定IC>0=有效。待聚宽研究环境运行 |

---

## 七、关键纪律提醒（每次会话自查）

1. 原文乐观数字（年化16-20%/超额409%）一律视为待验证假设，不作结论
2. 预注册通过标准（Gate 1 七条）锁定后不调整
3. 代码审核必须先于结果分析（看任何数字前完成 03 门禁）—— **F1 已完成审核**
4. 连续两轮 IC_IR<0.2 放弃该因子方向
5. 全量上报（含失败切片），禁止择优
6. 所有结论有数据支持，禁用"根据经验"
7. 分析前先读 `_index.md` 历史结论 + 本文件
8. 不绕过 data_layer 直接调 get_fundamentals
