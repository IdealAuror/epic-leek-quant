"""
joinquant/data_layer.py
=======================

聚宽数据接口封装层（Phase 0 脚手架）。

本模块是 **所有策略访问聚宽数据的唯一入口**。Phase 1 起的策略代码禁止直接调用
`get_fundamentals` / `get_price` / `get_extras`，必须经本模块的封装函数访问数据。
这样做有两个目的：

1. **统一 PIT 语义**（对应 plan-review.md B8 / C10）——把"换仓日实际可得数据"
   的校验逻辑集中到一处，避免每个策略重复实现且各写各的 bug。
2. **统一撮合口径**（对应 C8 / C9 / B3 / B4）——退市股、涨跌停、T+1、停牌的
   处理全部收敛到 `rebalance_ordered` 与 `simulate_limit_order` 两个函数，让
   Phase 1 的因子验证只需要写因子计算本身。

> **运行环境**：本文件设计在聚宽研究环境 / 策略环境运行（依赖 `jqdata`）。
> 本地仓库保存是为了代码审核与版本管理；本地直接 `python data_layer.py` 不会
> 真正联网取数，末尾的 `__main__` 仅做语法/导入冒烟测试。

口径来源：`docs/PROJECT-PLAN.md` 第一节。冲突时以 PROJECT-PLAN 为准。
"""

from __future__ import annotations

import datetime as dt
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

# 聚宽平台会自动注入 jqdata 全局对象与若干顶层函数（get_fundamentals、
# get_index_stocks、get_extras、get_price、get_all_securities、get_security_info
# 等）。这里用 try/except 兼容本地导入审核（无 jqdata 时不报错）。
try:
    import jqdata  # noqa: F401
    _HAS_JQDATA = True
except Exception:  # pragma: no cover - 本地审核环境
    _HAS_JQDATA = False


# ============================================================
# 一、PIT（Point-in-Time）财务数据查询
# ============================================================

# 聚宽财报披露截止日（季报披露截止后的首个可交易日用于调仓）。
# 对应 PROJECT-PLAN.md §1.1 的"季度调仓"口径与 thinking-prompt.md §2.1 的
# "换仓频率与披露节奏对齐"要求。
DISCLOSURE_DEADLINES = {
    # 报告期 -> 法定披露截止（月/日）
    'Q1': (4, 30),   # 一季报
    'Q2': (8, 31),   # 半年报（中报）
    'Q3': (10, 31),  # 三季报
    'Q4': (4, 30),   # 年报（次年）
}


def fetch_fundamentals_pit(
    date: str | dt.datetime | pd.Timestamp,
    fields: Sequence[str],
    stock_list: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """PIT 财务数据查询。

    封装聚宽 `get_fundamentals(q, date=date)`，强制要求显式 `date` 参数。
    调用方必须传入"换仓日"——即财报披露截止后的首个交易日。本函数**不做**
    "向前找最近披露日"的隐式回退，因为那种回退是未来函数的高发区；如果换仓日
    当天某只股票没有可得的财报数据，对应行会在返回结果中缺失，由调用方决定
    如何处理（剔除或标记）。

    Parameters
    ----------
    date : 换仓日。聚宽内部会取该日 *已披露* 的最新财报。
    fields : 需要查询的字段列表，形如 ['valuation.market_cap', 'balance.total_liability']。
    stock_list : 股票池。None 表示不限制。

    Returns
    -------
    DataFrame，聚宽 get_fundamentals 的原始返回，列名为 `table.column` 形式。

    PIT 校验
    --------
    调用方在拿到结果后，应进一步用 `is_data_available_at` 校验关键股票的
    财报披露日确实早于换仓日（防止财报修正覆盖历史导致的未来函数）。
    """
    if not _HAS_JQDATA:
        raise RuntimeError(
            "fetch_fundamentals_pit 需要在聚宽环境中运行；本地仅用于代码审核。"
        )

    if isinstance(date, (dt.datetime, pd.Timestamp)):
        date = pd.Timestamp(date).strftime('%Y-%m-%d')

    # 用 eval 构造 query 字段——聚宽的 query() 接受形如 valuation.market_cap 的对象，
    # 但我们这里用字符串列表更便于审计。实际调用转成 query 对象。
    # 注意：聚宽策略环境中 valuation/balance/income/cash_flow 是注入的全局对象，
    # 因此这里用 locals()/globals() 的方式解析。
    import inspect
    frame = inspect.currentframe()
    try:
        globs = frame.f_back.f_globals  # 调用方的全局，应包含聚宽注入对象
    finally:
        del frame

    q_obj = _build_query(fields, stock_list, globs)
    return get_fundamentals(q_obj, date=date)  # type: ignore[name-defined]


def _build_query(fields, stock_list, globs):
    """根据字符串字段列表构造聚宽 query 对象。"""
    # 聚宽 query 接口：query(table1.col1, table2.col2).filter(...)
    # 这里通过 getattr 解析 'valuation.market_cap' -> valuation 对象的 market_cap 属性
    field_objs = []
    for f in fields:
        table_name, col_name = f.split('.', 1)
        table_obj = globs[table_name]
        field_objs.append(getattr(table_obj, col_name))
    q = globs['query'](*field_objs)
    if stock_list is not None and len(stock_list) > 0:
        q = q.filter(globs['valuation'].code.in_(list(stock_list)))
    return q


def is_data_available_at(
    code: str,
    report_period: str,
    query_date: str | dt.datetime | pd.Timestamp,
) -> bool:
    """校验某只股票的某期财报在 `query_date` 当天是否已经披露。

    用于 PIT 强校验（对应 C10 / B8）。report_period 形如 '2024-12-31'（财报期，
    非披露日）。聚宽提供 `finance.runs.STK_INCOME_STATEMENT.parent_announce_date`
    等披露日字段，但实现较繁琐；Phase 0 先用"披露截止日"作近似：

    - 一季报（3-31）→ 4-30 后可得
    - 半年报（6-30）→ 8-31 后可得
    - 三季报（9-30）→ 10-31 后可得
    - 年报（12-31）→ 次年 4-30 后可得

    Phase 1 代码审核门禁（prompts/03）会强制要求调用方对关键股票做此校验。
    """
    rp = pd.Timestamp(report_period)
    qd = pd.Timestamp(query_date)
    month = rp.month
    if month == 3:
        deadline_m, deadline_d = DISCLOSURE_DEADLINES['Q1']
        deadline_year = rp.year
    elif month == 6:
        deadline_m, deadline_d = DISCLOSURE_DEADLINES['Q2']
        deadline_year = rp.year
    elif month == 9:
        deadline_m, deadline_d = DISCLOSURE_DEADLINES['Q3']
        deadline_year = rp.year
    elif month == 12:
        deadline_m, deadline_d = DISCLOSURE_DEADLINES['Q4']
        deadline_year = rp.year + 1  # 年报次年披露
    else:
        return False  # 非标准报告期

    deadline = pd.Timestamp(year=deadline_year, month=deadline_m, day=deadline_d)
    return qd >= deadline


# ============================================================
# 二、股票池构建
# ============================================================

def get_stock_pool(
    index_id: Optional[str] = '000300.XSHG',
    date: Optional[str | dt.datetime | pd.Timestamp] = None,
    exclude_st: bool = True,
    min_listed_days: int = 180,
) -> list[str]:
    """构建股票池。

    口径（PROJECT-PLAN.md §1.1）：
    - 取指数成分股（index_id=None 时取全 A）；
    - 剔除 ST / *ST（用 `get_extras('is_st', ...)`，对应 B5）；
    - 剔除上市不满 `min_listed_days` 天的次新股；
    - 退市股由 `get_all_securities(date=date)` 自然过滤（仅含当日仍上市的），
      但回测期内的退市股会在撮合层显式处理，见 `get_delisted_calendar`。

    Parameters
    ----------
    index_id : 指数代码。默认沪深 300。可换 '000905.XSHG'（中证 500）或 None（全 A）。
    date : 调仓日。None 表示当前。
    exclude_st : 是否剔除 ST。
    min_listed_days : 上市天数下限。
    """
    if not _HAS_JQDATA:
        raise RuntimeError("get_stock_pool 需要在聚宽环境中运行。")

    if date is None:
        date_str = dt.datetime.now().strftime('%Y-%m-%d')
    else:
        date_str = pd.Timestamp(date).strftime('%Y-%m-%d')

    if index_id is None:
        sec_df = get_all_securities(['stock'], date=date_str)  # type: ignore[name-defined]
        stocks = list(sec_df.index)
    else:
        stocks = get_index_stocks(index_id, date=date_str)  # type: ignore[name-defined]

    # ST 剔除：用 get_extras 取当日 is_st 标记，避免遍历 display_name（B5）
    if exclude_st and len(stocks) > 0:
        st_df = get_extras('is_st', stocks,  # type: ignore[name-defined]
                           end_date=date_str, count=1)
        if st_df is not None and not st_df.empty:
            st_today = st_df.iloc[-1]
            stocks = [s for s in stocks
                      if s in st_today.index and not st_today[s]]

    # 次新股剔除
    stocks = [s for s in stocks
              if not is_new_stock(s, date_str, min_listed_days)]

    return stocks


def is_new_stock(code: str, date: str | dt.datetime | pd.Timestamp,
                 days: int = 180) -> bool:
    """判断 `code` 在 `date` 当天是否上市不满 `days` 天（对应 B6：补 import）。"""
    if not _HAS_JQDATA:
        return False  # 本地审核环境
    info = get_security_info(code)  # type: ignore[name-defined]
    if info is None:
        return True
    cur = pd.Timestamp(date)
    start = pd.Timestamp(info.start_date)
    return (cur - start).days < days


def get_st_stocks(stocks: Sequence[str], date: str) -> list[str]:
    """辅助函数：返回 `stocks` 中在 `date` 当天被标记为 ST 的子集。"""
    if not _HAS_JQDATA or len(stocks) == 0:
        return []
    df = get_extras('is_st', list(stocks), end_date=date, count=1)  # type: ignore[name-defined]
    if df is None or df.empty:
        return []
    row = df.iloc[-1]
    return [s for s in stocks if s in row.index and row[s]]


def fetch_actual_controller(
    stock_list: Sequence[str],
    date: str | dt.datetime | pd.Timestamp,
) -> dict[str, str]:
    """查询实际控制人映射 {code -> actual_controller}。

    通过聚宽 finance.STK_COMPANY_INFO 表查询。返回 dict 而非 DataFrame，
    方便 merge 到因子 DataFrame。
    """
    if not _HAS_JQDATA or len(stock_list) == 0:
        return {}
    from jqdata import finance
    q = query(  # type: ignore[name-defined]
        finance.STK_COMPANY_INFO.actual_controller,
        finance.STK_COMPANY_INFO.code,
    ).filter(finance.STK_COMPANY_INFO.code.in_(list(stock_list)))
    date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
    df = get_fundamentals(q, date=date_str)  # type: ignore[name-defined]
    if df is None or df.empty:
        return {}
    return dict(zip(df['code'], df['actual_controller']))


# ============================================================
# 三、退市 / 停牌日历
# ============================================================

def get_delisted_calendar(start_date: str, end_date: str) -> pd.DataFrame:
    """退市日历：返回 [start_date, end_date] 区间内退市的股票。

    对应 C8 / B3：退市股必须纳入回测，不能静默剔除。

    Returns
    -------
    DataFrame，index 为股票代码，列：
        - delist_date : 退市日
        - last_close  : 退市前最后交易日收盘价
    """
    if not _HAS_JQDATA:
        return pd.DataFrame(columns=['delist_date', 'last_close'])

    # 聚宽 get_all_securities 返回当前在市的所有股票；通过对比 [start, end] 区间
    # 两端的上市股票集合，差集即为区间内退市的股票。
    sec_start = get_all_securities(['stock'], date=start_date)  # type: ignore[name-defined]
    sec_end = get_all_securities(['stock'], date=end_date)  # type: ignore[name-defined]
    delisted_codes = list(set(sec_start.index) - set(sec_end.index))

    if not delisted_codes:
        return pd.DataFrame(columns=['delist_date', 'last_close'])

    # 取每只退市股在 [start, end] 区间的最后价格作为退市日参考
    rows = []
    for code in delisted_codes:
        try:
            price = get_price(code,  # type: ignore[name-defined]
                              start_date=start_date, end_date=end_date,
                              fields=['close'], skip_paused=False)
            if price is None or price.empty:
                continue
            last_row = price.iloc[-1]
            rows.append({'code': code,
                         'delist_date': price.index[-1].strftime('%Y-%m-%d'),
                         'last_close': float(last_row['close'])})
        except Exception:
            continue
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.set_index('code')


def get_suspended_calendar(stocks: Sequence[str],
                           start_date: str, end_date: str) -> pd.DataFrame:
    """停牌日历：返回每只股票在 [start, end] 内每个交易日是否停牌。

    对应 C9：停牌期间持仓冻结，复牌首日按实际成交价计入。
    用 `get_price(..., skip_paused=False)` 的 `paused` 字段。
    """
    if not _HAS_JQDATA or len(stocks) == 0:
        return pd.DataFrame()
    df = get_price(list(stocks),  # type: ignore[name-defined]
                   start_date=start_date, end_date=end_date,
                   fields=['paused'], skip_paused=False, panel=False)
    # 长格式：date / code / paused (bool)
    return df


def is_suspended(code: str, date: str) -> bool:
    """单只股票单日是否停牌。"""
    if not _HAS_JQDATA:
        return False
    df = get_price(code, end_date=date, count=1,  # type: ignore[name-defined]
                   fields=['paused'], skip_paused=False)
    if df is None or df.empty:
        return False
    return bool(df.iloc[-1]['paused'])


# ============================================================
# 四、涨跌停撮合器（限价单排队模拟）
# ============================================================

def get_limit_prices(stocks: Sequence[str], date: str) -> pd.DataFrame:
    """获取 `stocks` 在 `date` 当天的涨停价 / 跌停价。

    聚宽 `get_price` 不直接给涨跌停价，需通过 `get_current_data`（仅当日实时）
    或基于前收盘价 ×(1±10%) 计算（主板/创业板 10%，科创板/创业板注册制 20%，
    ST 股 5%）。本函数用前收盘价 + 板块规则计算，回测友好。
    """
    if not _HAS_JQDATA or len(stocks) == 0:
        return pd.DataFrame(columns=['high_limit', 'low_limit'])

    rows = []
    for code in stocks:
        prev_close = _get_prev_close(code, date)
        if prev_close is None or prev_close <= 0:
            continue
        limit_pct = _get_limit_pct(code)
        high = round(prev_close * (1 + limit_pct), 2)
        low = round(prev_close * (1 - limit_pct), 2)
        rows.append({'code': code, 'high_limit': high, 'low_limit': low})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.set_index('code')


def _get_prev_close(code: str, date: str) -> Optional[float]:
    df = get_price(code, end_date=date, count=2,  # type: ignore[name-defined]
                   fields=['close'], skip_paused=False)
    if df is None or len(df) < 2:
        return None
    return float(df.iloc[-2]['close'])


def _get_limit_pct(code: str) -> float:
    """根据股票代码前缀返回涨跌停幅度。"""
    sym = code.split('.')[0]
    if sym.startswith('688'):      # 科创板
        return 0.20
    if sym.startswith('300') or sym.startswith('301'):  # 创业板注册制
        return 0.20
    if sym.startswith(('ST', '*ST')):  # ST 股（实际应查名称，此处简化）
        return 0.05
    # 北交所 30%，本项目主要覆盖沪深主板，按 10% 处理
    return 0.10


def simulate_limit_order(
    code: str,
    side: str,
    date: str,
) -> float:
    """模拟限价单排队撮合，返回实际成交比例（0~1）。

    对应 C9 / B4。简化口径（Phase 0 默认）：

    - **买入**：若当日开盘价 < 涨停价 → 视为可成交，返回 1.0；
                  若开盘价 == 涨停价（一字涨停）→ 返回 0.0（排队未果）；
                  若盘中曾打开涨停（high > low == high_limit 且 high > high_limit
                  实际不会发生，等价于 high > low）→ 返回 1.0。
    - **卖出**：若开盘价 > 跌停价 → 1.0；开盘价 == 跌停价 → 0.0；
                  盘中打开跌停 → 1.0。

    严格版（分钟级排队模拟）作为 Phase 1 升级选项，见模块末尾注释。
    停牌当日：返回 0.0（撮合冻结）。
    """
    if not _HAS_JQDATA:
        return 1.0  # 本地审核：乐观假设

    if is_suspended(code, date):
        return 0.0

    df = get_price(code, end_date=date, count=1,  # type: ignore[name-defined]
                   fields=['open', 'high', 'low', 'close'], skip_paused=False)
    if df is None or df.empty:
        return 0.0
    open_p = float(df.iloc[-1]['open'])
    high_p = float(df.iloc[-1]['high'])
    low_p = float(df.iloc[-1]['low'])

    prev_close = _get_prev_close(code, date)
    if prev_close is None or prev_close <= 0:
        return 0.0
    limit_pct = _get_limit_pct(code)
    high_limit = round(prev_close * (1 + limit_pct), 2)
    low_limit = round(prev_close * (1 - limit_pct), 2)

    if side == 'buy':
        if open_p < high_limit:
            return 1.0
        # 一字板：开盘即涨停
        if open_p >= high_limit and high_p > low_p:
            # 盘中曾打开（high > low 说明有非涨停价成交）
            return 1.0
        return 0.0  # 一字涨停，排队未果
    elif side == 'sell':
        if open_p > low_limit:
            return 1.0
        if open_p <= low_limit and high_p > low_p:
            return 1.0
        return 0.0  # 一字跌停，无法卖出
    else:
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")


# ============================================================
# 五、调仓：先卖后买（T+1）
# ============================================================

def rebalance_ordered(
    context,
    target_weights: Mapping[str, float],
    log_info: bool = True,
) -> dict:
    """按"先卖后买"顺序执行调仓（对应 C9 / B4 / T+1）。

    流程：
    1. 对当前持仓中不在 `target_weights` 或目标权重为 0 的股票，先尝试卖出
       （经 `simulate_limit_order` 判断是否可成交）；
    2. 卖出后释放资金；
    3. 再对 `target_weights` 中的目标股票，按目标权重买入（同样经撮合判断）；
    4. 因涨跌停未成交的，记录到返回值的 `unfilled` 字段，**不**自动重试
       （Phase 0 简化口径；Phase 4 实盘校准可加"次日补单"逻辑）。

    Parameters
    ----------
    context : 聚宽 context 对象。
    target_weights : {code: weight}，weight 为 0~1 之间的目标仓位占比。
    log_info : 是否打印调仓日志。

    Returns
    -------
    dict:
        - sold : list of (code, attempted_weight, filled_ratio)
        - bought : list of (code, target_weight, filled_ratio)
        - unfilled : list of (code, side, reason)
    """
    if not _HAS_JQDATA:
        raise RuntimeError("rebalance_ordered 需要在聚宽环境中运行。")

    current_date = context.current_dt.strftime('%Y-%m-%d')
    result = {'sold': [], 'bought': [], 'unfilled': []}

    # ---------- 1. 先卖 ----------
    positions = context.portfolio.positions
    for code, pos in positions.items():
        if pos.total_amount <= 0:
            continue
        target = target_weights.get(code, 0.0)
        if target > 0:
            continue  # 继续持有
        # 卖出
        filled = simulate_limit_order(code, 'sell', current_date)
        if filled > 0:
            order_target_percent(code, 0)  # type: ignore[name-defined]
            result['sold'].append((code, pos.value / context.portfolio.total_value, filled))
            if log_info:
                log.info('卖出 %s' % code)  # type: ignore[name-defined]
        else:
            result['unfilled'].append((code, 'sell', '一字跌停或停牌'))
            if log_info:
                log.info('卖出失败 %s（跌停/停牌）' % code)  # type: ignore[name-defined]

    # ---------- 2. 后买 ----------
    for code, weight in target_weights.items():
        if weight <= 0:
            continue
        filled = simulate_limit_order(code, 'buy', current_date)
        if filled > 0:
            order_target_percent(code, weight)  # type: ignore[name-defined]
            result['bought'].append((code, weight, filled))
            if log_info:
                log.info('买入 %s, 权重 %.2f%%' % (code, weight * 100))  # type: ignore[name-defined]
        else:
            result['unfilled'].append((code, 'buy', '一字涨停或停牌'))
            if log_info:
                log.info('买入失败 %s（涨停/停牌）' % code)  # type: ignore[name-defined]

    return result


# ============================================================
# 六、成本模型（佣金 / 印花税 / 最低佣金 / 滑点）
# ============================================================

# 对应 PROJECT-PLAN.md §1.1：万 2.5 双边 + 千 1 印花税（仅卖） + 5 元最低 + 0.3% 滑点。
COMMISSION_BUY = 0.00025      # 万 2.5
COMMISSION_SELL = 0.00025     # 万 2.5
STAMP_TAX_SELL = 0.001        # 千 1，仅卖出
MIN_COMMISSION = 5.0          # 元/笔
SLIPPAGE = 0.003              # 0.3%


def apply_cost_model(context) -> None:
    """在 `initialize` 中调用，设定统一成本模型。

    将聚宽 set_commission / set_slippage 的口径对齐到 PROJECT-PLAN.md §1.1，
    避免每个策略重复设定且各写各的值（对应 B7）。
    """
    if not _HAS_JQDATA:
        return
    # 聚宽 set_order_cost 接口更细粒度，可拆分佣金与印花税
    set_order_cost(  # type: ignore[name-defined]
        OrderCost(  # type: ignore[name-defined]
            open_tax=0,
            close_tax=STAMP_TAX_SELL,           # 印花税仅卖出
            open_commission=COMMISSION_BUY,
            close_commission=COMMISSION_SELL,
            close_today_commission=0,
            min_commission=MIN_COMMISSION,
        ),
        type='stock'
    )
    set_slippage(PriceSlippage(SLIPPAGE))  # type: ignore[name-defined]


# ============================================================
# 七、本地冒烟测试（不依赖聚宽）
# ============================================================

def _local_smoke_test() -> None:
    """本地导入与纯逻辑冒烟测试，不调用任何聚宽 API。"""
    # PIT 截止日逻辑
    assert is_data_available_at('600000', '2024-03-31', '2024-05-01') is True
    assert is_data_available_at('600000', '2024-03-31', '2024-04-29') is False
    assert is_data_available_at('600000', '2023-12-31', '2024-04-30') is True
    assert is_data_available_at('600000', '2023-12-31', '2024-04-29') is False
    # 涨跌停幅度
    assert _get_limit_pct('688001.SH') == 0.20
    assert _get_limit_pct('300001.SZ') == 0.20
    assert _get_limit_pct('000001.SZ') == 0.10
    print('[data_layer] 本地冒烟测试通过。')


if __name__ == '__main__':
    _local_smoke_test()


# ============================================================
# 严格版撮合升级路径（Phase 1 可选，Phase 4 实盘前必做）
# ============================================================
#
# 当前的 `simulate_limit_order` 是日级简化版：基于 [open, high, low, close]
# 判断"是否能成交"。它有两个已知偏差：
#
# 1. **部分成交未建模**：实际排队中可能只成交一部分。简化版要么全成交（1.0）
#    要么全不成交（0.0）。
# 2. **排队优先级未建模**：一字涨停时，按委托时间排队，散户策略实际很难买到。
#    简化版直接给 0.0 是保守的，但与真实情况有差距。
#
# 严格版需要：
#   - 分钟级 `get_price` 数据；
#   - 模拟开盘集合竞价 + 连续竞价的"打开涨停瞬间"撮合；
#   - 成交量加权的部分成交比例（持仓市值 / 当日成交量）。
#
# Phase 4 实盘校准时必须升级到此版本，否则容量估算与净收益不可信。
