# 聚宽（JoinQuant）平台调试经验与踩坑记录

> 2026-06-26 调试 F1 EV<0 IC 分析脚本时积累，供后续因子开发复用。
> 涉及：策略回测环境 vs 研究环境、字段名差异、单位陷阱、API 行为。

---

## 一、两个运行环境，绝不能混用

| 环境 | 入口 | 用途 | 入口函数 |
|------|------|------|---------|
| **策略回测** | 聚宽 → 我的策略 → 新建策略 | 跑策略净值 | `initialize`/`before_trading`/`run_monthly` 等钩子 |
| **研究环境** | 聚宽 → 研究（Jupyter Notebook） | 跑 IC 分析/统计 | 直接执行，无钩子 |

**踩坑**：把 IC 分析脚本粘到策略回测环境 → 没有钩子触发交易 → 净值一条直线（资金不动）。
**结论**：策略代码去回测环境，IC/统计分析去研究环境。

---

## 二、交易函数注入差异（jqboson 引擎）

**现象**：策略回测环境中 `order_target_percent` 未注入到用户代码全局命名空间，裸调用报 `NameError`。
**根因**：jqboson 引擎注入了 `log`/`get_fundamentals`/`set_benchmark` 等数据 API，但交易函数注入范围有差异。
**解决**：加兼容垫片 `_safe_order_target_percent`，降级链：
`order_target_percent` → `order_target_value` → `order_target` → `order`（手动算差额整手）。
见 `joinquant/strategies/P1-F1-EV-2026Q2-v1-*-standalone.py`。

---

## 三、财务字段名：文档/猜测与真实 API 严重不一致

**这是最大的坑**。`get_fundamentals(query(...))` 的字段名在不同 API 版本/文档间存在单复数、命名风格差异。属性访问（`balance.xxx`）一旦字段名不对就 `AttributeError`。

### 诊断方法（强烈推荐先跑）

运行前先用 `get_fundamentals(query(balance), date=...)` 查一行真实数据，打印 `list(df.columns)` 看真实列名。不要信任文档，只信任运行时返回。

### 确认的真实字段名（2026-06 核实，BalanceSheet 表）

| 含义 | 错误猜测（文档/factor_lib） | 聚宽真实 |
|------|---------------------------|---------|
| 短期借款 | `short_term_loans` / `short_term_loan` | **`shortterm_loan`**（无下划线！） |
| 长期借款 | `long_term_loans` / `long_term_loan` | **`longterm_loan`**（无下划线！） |
| 流动负债合计 | `total_current_liabilities`（复数） | **`total_current_liability`**（单数） |
| 货币资金 | `monetary_funds` | **`cash_equivalents`**（无独立 monetary_funds） |
| 交易性金融资产 | `financial_assets_held_for_trading` | **不存在**，用 cash_equivalents 近似 |
| 应付账款 | `accounts_payable` | ✅ 存在 |
| 应交税费 | `taxes_payable` | **`taxs_payable`**（拼写！少 e） |
| 预收账款 | `advances_from_customers` | **`advance_peceipts`**（拼写！receipts→peceipts） |
| 应付职工薪酬 | `wages_payable` | **`salaries_payable`** |

### valuation 表

| 含义 | 错误猜测 | 真实 |
|------|---------|------|
| 市盈率 TTM | `pe_ttm` | **`pe_ratio`** |
| 市盈率静态 | — | `pe_ratio_lyr` |
| 市净率 | — | `pb_ratio` |
| 总市值 | `market_cap` | ✅ |
| 代码 | `code` | ✅ |

### cash_flow 表

| 含义 | 错误猜测 | 真实 |
|------|---------|------|
| 经营现金流净额 | `operating_cash_flow` | **`net_operate_cash_flow`** |

### 解决方案：动态字段探测

不要硬编码字段名。用 `query(Table)` 全表查询 + `_get_col(df, *候选名)` 多候选匹配：
```python
def _get_col(df, *candidates):
    for c in candidates:
        if c in df.columns:
            return df[c]
    return None
```
见 `research/scripts/P1-F1-EV-2026Q2-v1-ic_analysis-standalone.py` 的 `_fetch_fundamentals_pit` + `_calculate_all_factors`。

---

## 四、单位陷阱：valuation 与 balance 单位不同（致命）

**这是最隐蔽的坑**，会导致因子值完全失真但不报错。

| 数据源 | 字段 | 单位 |
|--------|------|------|
| valuation | `market_cap` | **亿元** |
| balance | `cash_equivalents`/`total_assets`/`total_liability` 等 | **元** |
| income | `net_profit` | **元** |
| cash_flow | `net_operate_cash_flow` | **元** |

**踩坑**：`ev = market_cap + 有息负债 - cash` 直接相减 → market_cap（亿）和 cash（元）差 1e8 倍 → EV 永远为巨大正/负数，因子失真。
**解决**：统一为元，`market_cap * 1e8`。

诊断验证：茅台（600519）`market_cap=24608.9`（亿元，即 2.46 万亿），`cash_equivalents=5.07e10`（元，即 507 亿）——数量级对得上。

---

## 五、get_price 的 panel 参数与返回格式

### panel=True vs panel=False

| 参数 | 返回 | 适用 |
|------|------|------|
| `panel=True` | Panel/dict（`px['close']` 是 index=date, columns=code 的 DataFrame） | 旧版默认 |
| `panel=False` | 长表 DataFrame（index=行号, columns=time/code/close） | 新版可能默认 |

**踩坑**：`get_price(codes, ..., panel=False)` 返回长表，日期在 **`time` 列**而非 index。直接 `pivot_table(index=df.index, ...)` 会用整数行号作 index → `pd.to_datetime` 误判为 Unix 纳秒时间戳 → 全变成 1970 年。
**解决**：先 `df.set_index('time')` 再 pivot，或 pivot 后 `close.index = pd.to_datetime(close.index)`。

### get_price(None, ...) 全市场取价

研究环境可能返回空或超时。建议按股票池批量取：`get_price(codes_list, ...)`。

---

## 六、get_trade_days 返回类型

返回 `datetime.date` 对象，**不是** `pd.Timestamp`。
**踩坑**：`td.date()` 报 `AttributeError: 'datetime.date' object has no attribute 'date'`。
**解决**：`td = pd.Timestamp(tds[0])` 统一转换。

---

## 七、datetime.date vs pd.Timestamp 比较报错

**现象**：`ic_series[ic_series.index < pd.Timestamp(BREAKPOINT)]` 报 `TypeError: Cannot compare type 'Timestamp' with type 'date'`。
**根因**：`ic_series.index` 是 `datetime.date`，`BREAKPOINT` 转 Timestamp 后类型不匹配。
**解决**：先 `ic_series.index = pd.to_datetime(ic_series.index)` 统一为 Timestamp。

---

## 八、性能优化：批量价格缓存

**问题**：每个调仓日单独 `get_price` 4000+ 股票，36 横截面 × 6 组 = 216 次网络请求，聚宽 API 限速导致 1-2 小时。
**解决**：按月批量取价并缓存（`_load_month_prices`），月内多个调仓日共用缓存。
见脚本 `_PRICE_CACHE` 字典。

**资源限制**：聚宽研究环境 CPU 1 核、磁盘 2G、带宽低。全 A 样本（4000+ 股）建议单独跑，先跑指数成分股（300/500）验证。

---

## 九、ModuleNotFoundError：研究环境无自定义模块

**现象**：`from data_layer import ...` 报 `ModuleNotFoundError: 导入错误，未找到系统库或自定义库 data_layer`。
**根因**：研究环境只加载 Notebook 内代码，不自动加载研究文件目录的其他 .py（除非上传到正确路径）。
**解决**：standalone 脚本——把 data_layer/factor_lib 的必要逻辑内联，不依赖外部 import。

---

## 十、调试方法论（复用）

1. **先诊断再写主逻辑**：写 `diagnose.py` 探测真实字段名、API 返回格式、单位
2. **分步 debug 输出**：在关键函数加 `debug` 参数，打印每步 shape/非空数/index 范围
3. **QUICK_TEST 开关**：先跑 3 个调仓日验证数据链，通了再跑全量
4. **不信任文档，只信任运行时返回**：聚宽文档字段名与实际 API 有出入，以 `list(df.columns)` 为准
5. **单位对齐检查**：跨表计算（valuation × balance）前，先用单只知名股票（如茅台）验证数量级

---

## 附：本项目相关文件

- 策略回测代码：`joinquant/strategies/P1-F1-EV-2026Q2-v1-*-standalone.py`
- IC 分析脚本（standalone）：`research/scripts/P1-F1-EV-2026Q2-v1-ic_analysis-standalone.py`
- 诊断脚本：`research/scripts/diagnose.py`
- 偏差记录：`research/reports/P1-F1-EV-2026Q2-v1-CSI300-deviation.md`
- 任务状态：`docs/task-state/CURRENT-STATE.md`
