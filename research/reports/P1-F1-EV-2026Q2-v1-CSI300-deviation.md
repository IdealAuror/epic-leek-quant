# 执行偏差记录 — P1-F1-EV-2026Q2-v1-CSI300

## 1. spec 歧义处理
| # | spec 原文 | 我的处理 | 理由 |
|---|----------|---------|------|
| 1 | 有息负债 = total_liability - 非有息流动负债 | 按 spec 用 accounts_payable + advances_from_customers + wages_payable + taxes_payable + other_current_liabilities 估算 | 聚宽无直接 interest_bearing_debt 字段 |
| 2 | cash_available_ext = monetary_funds + financial_assets_held_for_trading + time_deposits_note | 仅取前两项，附注定期存款字段在聚宽中不可直接通过 balance 表获得 | 聚宽 balance 表无 time_deposits_note |

## 2. 数据缺失
| 字段 | spec 要求 | 聚宽实际 | 处理 |
|------|----------|---------|------|
| balance.time_deposits_note | 附注定期存款 | 无此字段 | 跳过，偏差记录标注"未挖附注" |
| actual_controller | 实际控制人 | 需通过 finance.STK_COMPANY_INFO 查询 | 新增 fetch_actual_controller 函数 |

## 3. 执行异常
- 无（代码尚未在聚宽环境执行）

## 4. 未完成的输出项
- spec 要求表1-7（IC 分析）：策略代码不产出 IC 分析，需独立研究环境脚本完成
- CSV 净值序列：策略在聚宽运行后导出

## 5. IC 分析脚本 spec 歧义处理（2026-06-26 补充）

| # | spec 原文 | 实际处理 | 理由 |
|---|----------|---------|------|
| 3 | factors.type=binary（factor_ev_negative 0/1）+ analysis.grouping=Q1-Q5 | 二值因子无法做 Q1-Q5 五分组，改用连续 ev 升序分位：Q1=最低ev(净现金最多)=多头组；二值 IC 单独报告 | 二值变量 pd.qcut 会报错或退化分组，连续 ev 排序是统计上唯一可行解 |
| 4 | analysis.significance 未指定 IC 符号约定 | 定义 signal=-ev（高=净现金多=好），使 IC>0 表示因子有效，对齐 Gate 1 规则1「IC_IR>0.3」 | 避免 IC 方向混淆导致 Gate 1 误判 |
| 5 | variants.V2 壳价值剔除版 | 实现为 V2 变体独立跑，阈值 pre-2019.06=20亿 / post=30亿，与 V1 全量对比 | theory-framework §5.2 |
| 6 | analysis.neutralize=[industry, size] | 表4 行业中性化（OLS 残差，jq_l1 分类）+ 表5 市值中性化（对 log(market_cap) OLS 残差） | Gate 1 规则5 |
| 7 | analysis.ic_decay=[lag0,lag1,lag3,lag6] | 本版暂未实现 IC 衰减分析 | 非 Gate 1 七条必需项，可在判定后补 |

## 6. IC 分析脚本字段口径变更（2026-06-26 补充）

聚宽真实字段与 spec/factor_lib 假设的差异，IC 脚本 standalone 版已适配：

| 项 | spec/factor_lib 假设 | 聚宽真实 | IC 脚本处理 |
|----|---------------------|---------|------------|
| PE 字段 | `valuation.pe_ttm` | `valuation.pe_ratio`（TTM） | 改用 pe_ratio（本因子不依赖 PE，无影响） |
| 有息负债 | total_liability 扣减 5 项无息流动负债 | 聚宽无 accounts_payable 等细分字段 | 改用 `short_term_loans + long_term_loans`（直接有息负债项，更精确） |
| 货币资金 | `monetary_funds` | 聚宽无此独立字段 | cash_equivalents 已包含货币资金，直接用 |
| 交易性金融资产 | `financial_assets_held_for_trading` | 聚宽无此字段 | cash_available_ext 暂用 cash_equivalents，偏差"未挖附注" |
| 经营现金流 | `operating_cash_flow` | `net_operate_cash_flow` | 改用 net_operate_cash_flow |
| 流动负债 | `total_current_liabilities`（复数） | `total_current_liability`（单数） | 改用单数 |

**影响评估**：有息负债口径从"估算"改为"直接借款项"更精确，可能使 EV 计算更准；
其余变更不影响因子核心逻辑。
