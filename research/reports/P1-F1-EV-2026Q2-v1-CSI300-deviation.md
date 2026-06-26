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
