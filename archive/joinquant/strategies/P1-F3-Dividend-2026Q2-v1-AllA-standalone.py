"""
P1-F3-Dividend-2026Q2-v1-AllA — 股息率因子策略（全A样本，standalone）
====================================================================
聚宽策略编辑器直接粘贴运行，无外部依赖。

spec_id: P1-F3-Dividend-2026Q2-v1-AllA
样本: 全A股（剔除ST/次新股/金融股）
基准: 000985.XSHG（中证全指）
换仓: 季度（5/9/11 月首个交易日）
持仓: 50 只，IC加权（非等权）
成本: 万2.5双边 + 千1印花税(仅卖) + 5元最低 + 0.3%滑点

因子定义（Phase 1.5，IC 验证通过）:
  dividend_yield = 过去400天累计每股现金分红 / 换仓日收盘价
  分红来源: finance.STK_XR_XD 表（bonus_ratio_rmb 每10股派息，/10得每股）
  收盘价: get_price 不复权

选股方式:
  - 按 dividend_yield 降序排序（高股息率优先）
  - 取前 50 只
  - 权重按 dividend_yield 横截面 z-score 归一化（IC加权）

Gate 1 结论（2026-06-27，详见 research/decisions/P1-F1-EV-2026Q2-v1-decision.md）:
  - 市值中性化后 IC t=2.74（通过门槛）
  - 行业中性化后 IC t=2.29（通过门槛）
  - 七条 3/7 过，形式与 F2-EP 当时相似
  - Q 倒挂但 Q5 夏普最高（风险管理特征）
  - 断点后失效（后段 t=0.41），注册制后信号衰减

数据链:
  - finance.STK_XR_XD 除权除息表（分红明细）
  - get_price 不复权收盘价
  - 金融股剔除保留（银行天然高股息，不剔会主导信号）

口径来源: research/decisions/P1-F1-EV-2026Q2-v1-decision.md Phase 1.5 终判
"""

import datetime
import numpy as np
import pandas as pd

# 聚宽策略环境自动注入（无需 import）:
#   jqdata, get_fundamentals, query, valuation, balance, income, cash_flow,
#   get_index_stocks, get_all_securities, get_extras, get_security_info, get_price,
#   get_industry, set_benchmark, set_order_cost, OrderCost, set_slippage, PriceSlippage,
#   order_target_percent, log, g, run_monthly, finance


# ============================================================
# 零、聚宽交易 API 兼容垫片（应对不同引擎注入差异）
# ============================================================

def _resolve_jq_func(name):
    """安全解析聚宽注入的全局函数，找不到返回 None。"""
    obj = globals().get(name)
    if obj is not None:
        return obj
    try:
        import builtins
        return getattr(builtins, name, None)
    except Exception:
        return None


def _get_current_price(code, date_str):
    """通过 get_price 获取最新收盘价，供降级计算目标股数。"""
    df = get_price(code, end_date=date_str, count=1,
                   fields=['close'], skip_paused=False)
    if df is None or df.empty:
        return None
    return float(df.iloc[-1]['close'])


def _safe_order_target_percent(context, code, weight):
    """降级链下单封装，兼容聚宽各引擎交易函数注入差异。"""
    total_value = context.portfolio.total_value

    fn = _resolve_jq_func('order_target_percent')
    if fn is not None:
        return fn(code, weight)

    fn = _resolve_jq_func('order_target_value')
    if fn is not None:
        return fn(code, total_value * weight)

    current_date = context.current_dt.strftime('%Y-%m-%d')
    price = _get_current_price(code, current_date)
    positions = context.portfolio.positions
    pos = positions.get(code)
    current_amount = pos.total_amount if pos is not None else 0

    fn = _resolve_jq_func('order_target')
    if fn is not None:
        if price is None or price <= 0:
            raise RuntimeError('无法获取 %s 价格以计算目标股数' % code)
        target_shares = int(total_value * weight / price / 100) * 100
        return fn(code, target_shares)

    fn = _resolve_jq_func('order')
    if fn is not None:
        if price is None or price <= 0:
            raise RuntimeError('无法获取 %s 价格以计算差额股数' % code)
        target_shares = int(total_value * weight / price / 100) * 100
        delta = target_shares - current_amount
        if delta != 0:
            return fn(code, delta)
        return None

    raise RuntimeError(
        '聚宽交易函数 order_target_percent/order_target_value/'
        'order_target/order 均未注入，请确认在聚宽回测环境中运行'
    )


# ============================================================
# 一、股票池与 ST/次新股剔除
# ============================================================

def get_stock_pool(index_id, date_str, min_listed_days=180):
    """构建股票池：指数成分股 - ST - 次新股 - 金融股。"""
    if index_id is None:
        sec_df = get_all_securities(['stock'], date=date_str)
        stocks = list(sec_df.index)
    else:
        stocks = get_index_stocks(index_id, date=date_str)
    if len(stocks) > 0:
        st_df = get_extras('is_st', stocks, end_date=date_str, count=1)
        if st_df is not None and not st_df.empty:
            st_today = st_df.iloc[-1]
            stocks = [s for s in stocks if s in st_today.index and not st_today[s]]
    stocks = [s for s in stocks if not is_new_stock(s, date_str, min_listed_days)]
    if stocks:
        stocks = exclude_finance_stocks(stocks, date_str)
    return stocks


def exclude_finance_stocks(stocks, date_str):
    """剔除金融行业股票（银行/非银金融），sw_l1 严格相等匹配。"""
    if not stocks:
        return stocks
    try:
        ind = get_industry(stocks, date=date_str)
    except Exception:
        return stocks
    if not ind:
        return stocks
    FINANCE_NAMES = {'银行I', '非银金融I'}
    finance_codes = set()
    for code, schemes in ind.items():
        if not isinstance(schemes, dict):
            continue
        sw_l1 = schemes.get('sw_l1')
        if not isinstance(sw_l1, dict):
            continue
        name = str(sw_l1.get('industry_name', '') or '')
        if name in FINANCE_NAMES:
            finance_codes.add(code)
    return [s for s in stocks if s not in finance_codes]


def is_new_stock(code, date_str, days=180):
    """判断上市是否不满 days 天。"""
    info = get_security_info(code)
    if info is None:
        return True
    cur = pd.Timestamp(date_str)
    start = pd.Timestamp(info.start_date)
    return (cur - start).days < days


# ============================================================
# 二、限价单撮合器（涨跌停/停牌）
# ============================================================

def is_suspended(code, date_str):
    """是否停牌。"""
    df = get_price(code, end_date=date_str, count=1,
                   fields=['paused'], skip_paused=False)
    if df is None or df.empty:
        return False
    return bool(df.iloc[-1]['paused'])


def _get_limit_pct(code):
    """涨跌停幅度。"""
    sym = code.split('.')[0]
    if sym.startswith('688'):
        return 0.20
    if sym.startswith('300') or sym.startswith('301'):
        return 0.20
    return 0.10


def simulate_limit_order(code, side, date_str):
    """模拟限价单排队，返回成交比例 0.0 或 1.0。"""
    if is_suspended(code, date_str):
        return 0.0
    df = get_price(code, end_date=date_str, count=1,
                   fields=['open', 'high', 'low'], skip_paused=False)
    if df is None or df.empty:
        return 0.0
    open_p = float(df.iloc[-1]['open'])
    high_p = float(df.iloc[-1]['high'])
    low_p = float(df.iloc[-1]['low'])
    df_px = get_price(code, end_date=date_str, count=2,
                      fields=['close'], skip_paused=False)
    if df_px is None or len(df_px) < 2:
        return 0.0
    prev_close = float(df_px.iloc[-2]['close'])
    if prev_close <= 0:
        return 0.0
    limit_pct = _get_limit_pct(code)
    high_limit = round(prev_close * (1 + limit_pct), 2)
    low_limit = round(prev_close * (1 - limit_pct), 2)

    if side == 'buy':
        if open_p < high_limit:
            return 1.0
        if high_p > low_p:
            return 1.0
        return 0.0
    elif side == 'sell':
        if open_p > low_limit:
            return 1.0
        if high_p > low_p:
            return 1.0
        return 0.0
    return 0.0


# ============================================================
# 三、先卖后买调仓（T+1）
# ============================================================

def rebalance_ordered(context, target_weights):
    """先卖后买，涨跌停/停牌未成交记入 unfilled。"""
    current_date = context.current_dt.strftime('%Y-%m-%d')
    result = {'sold': [], 'bought': [], 'unfilled': []}

    for code, pos in context.portfolio.positions.items():
        if pos.total_amount <= 0:
            continue
        if target_weights.get(code, 0.0) > 0:
            continue
        if simulate_limit_order(code, 'sell', current_date) > 0:
            _safe_order_target_percent(context, code, 0)
            result['sold'].append(code)
            log.info('卖出 %s' % code)
        else:
            result['unfilled'].append((code, 'sell'))
            log.info('卖出失败 %s（跌停/停牌）' % code)

    for code, weight in target_weights.items():
        if weight <= 0:
            continue
        if simulate_limit_order(code, 'buy', current_date) > 0:
            _safe_order_target_percent(context, code, weight)
            result['bought'].append(code)
            log.info('买入 %s, 权重 %.2f%%' % (code, weight * 100))
        else:
            result['unfilled'].append((code, 'buy'))
            log.info('买入失败 %s（涨停/停牌）' % code)
    return result


# ============================================================
# 四、成本模型
# ============================================================

def apply_cost_model():
    """万2.5双边 + 千1印花税(仅卖) + 5元最低 + 0.3%滑点。"""
    try:
        set_order_cost(
            OrderCost(
                open_tax=0,
                close_tax=0.001,
                open_commission=0.00025,
                close_commission=0.00025,
                close_today_commission=0,
                min_commission=5.0,
            ),
            type='stock',
        )
    except Exception:
        pass
    try:
        set_slippage(PriceSlippage(0.003))
    except NameError:
        try:
            set_slippage(FixedSlippage(0.003))
        except NameError:
            pass


# ============================================================
# 五、F3 因子计算（股息率，Phase 1.5 IC 验证通过）
# ============================================================

DIV_LOOKBACK_DAYS = 400  # 过去400天的分红记录
DIV_BATCH_SIZE = 300     # 分批查询大小


def fetch_dividend_data(date_str, stocks):
    """用 finance.STK_XR_XD 查询过去 DIV_LOOKBACK_DAYS 天已实施的现金分红。

    返回 {code: 累计每股税前现金分红}。
    bonus_ratio_rmb 是每10股派息，除以10得每股。
    """
    if not stocks:
        return {}

    d = pd.Timestamp(date_str)
    start = (d - pd.Timedelta(days=DIV_LOOKBACK_DAYS)).strftime('%Y-%m-%d')
    end = date_str

    all_recs = []
    for i in range(0, len(stocks), DIV_BATCH_SIZE):
        batch = stocks[i:i + DIV_BATCH_SIZE]
        try:
            q = query(
                finance.STK_XR_XD.code,
                finance.STK_XR_XD.bonus_ratio_rmb,
                finance.STK_XR_XD.a_xr_date,
                finance.STK_XR_XD.plan_progress,
            ).filter(
                finance.STK_XR_XD.code.in_(list(batch)),
                finance.STK_XR_XD.a_xr_date >= start,
                finance.STK_XR_XD.a_xr_date <= end,
            )
            df_batch = finance.run_query(q)
            if df_batch is not None and not df_batch.empty:
                all_recs.append(df_batch)
        except Exception:
            pass

    if not all_recs:
        return {}

    df = pd.concat(all_recs, ignore_index=True)

    # 只取已实施的
    if 'plan_progress' in df.columns:
        df = df[df['plan_progress'].astype(str) == '实施方案']

    # bonus_ratio_rmb > 0
    if 'bonus_ratio_rmb' in df.columns:
        df['bonus_ratio_rmb'] = pd.to_numeric(df['bonus_ratio_rmb'], errors='coerce')
        df = df[df['bonus_ratio_rmb'] > 0]
    else:
        return {}

    if df.empty:
        return {}

    # 按 code 累加每股分红（每10股 → 每股）
    div_per_share = df.groupby('code')['bonus_ratio_rmb'].sum() / 10.0

    stocks_set = set(stocks)
    result = {}
    for code, val in div_per_share.items():
        if code in stocks_set:
            result[code] = float(val)
    return result


def get_close_prices(date_str, stocks):
    """获取换仓日收盘价（不复权，用于股息率计算）。"""
    if not stocks:
        return {}
    try:
        df = get_price(list(stocks), end_date=date_str, count=1,
                       fields=['close'], skip_paused=False,
                       panel=False, fq=None)
        if df is None or df.empty:
            return {}
        if 'time' in df.columns:
            df = df.set_index('time')
        elif 'date' in df.columns:
            df = df.set_index('date')
        if 'code' in df.columns:
            close = df.pivot_table(index=df.index,
                                   columns='code', values='close')
        else:
            close = df
        if close is None or close.empty:
            return {}
        last_row = close.iloc[-1]
        return {code: float(last_row[code]) for code in close.columns
                if not np.isnan(last_row[code]) and last_row[code] > 0}
    except Exception:
        return {}


# ============================================================
# 六、策略主体
# ============================================================

INDEX_ID = None  # None 表示全 A
BENCHMARK = '000985.XSHG'  # 中证全指


def initialize(context):
    set_benchmark(BENCHMARK)
    apply_cost_model()
    g.stock_num = 50  # 持仓数量
    g.index_id = INDEX_ID
    run_monthly(factor_rebalance, monthday=1)


def factor_rebalance(context):
    """季频因子选股：5/9/11 月，计算 IC 加权目标权重。"""
    current_date = context.current_dt
    if current_date.month not in (5, 9, 11):
        return
    date_str = current_date.strftime('%Y-%m-%d')

    # 1. 股票池
    stocks = get_stock_pool(g.index_id, date_str)
    if len(stocks) == 0:
        log.info('[%s] 股票池为空' % date_str)
        return

    # 2. 分红数据
    div_map = fetch_dividend_data(date_str, stocks)
    if not div_map:
        log.info('[%s] 无分红数据' % date_str)
        return

    # 3. 收盘价（不复权）
    close_map = get_close_prices(date_str, stocks)
    if not close_map:
        log.info('[%s] 无收盘价数据' % date_str)
        return

    # 4. 构建因子 DataFrame
    codes = [c for c in stocks if c in div_map and c in close_map]
    if len(codes) < 10:
        log.info('[%s] 有效股票不足 (%d)' % (date_str, len(codes)))
        return

    df = pd.DataFrame({
        'div_per_share': {c: div_map[c] for c in codes},
        'close_price': {c: close_map[c] for c in codes},
    })
    df['dividend_yield'] = df['div_per_share'] / df['close_price']
    df = df[df['dividend_yield'] > 0].dropna(subset=['dividend_yield'])

    if df.empty:
        log.info('[%s] 无符合条件的 Dividend 股票' % date_str)
        return

    # 5. IC加权选股：按 dividend_yield 降序取前 N，权重按 z-score 归一化
    df = df.sort_values('dividend_yield', ascending=False).head(g.stock_num)

    dy_vals = df['dividend_yield'].values
    z = (dy_vals - dy_vals.mean()) / (dy_vals.std() if dy_vals.std() > 0 else 1)
    weights = np.where(z > 0, z, 0)
    if weights.sum() == 0:
        weights = np.ones(len(df))
    weights = weights / weights.sum()
    g.target_weights = {code: float(w) for code, w in zip(df.index, weights)}

    log.info('[%s] 调仓：买入 %d 只股票 (IC加权)' % (date_str, len(df)))
    for code, w in list(g.target_weights.items())[:5]:
        log.info('  买入 %s  dy=%.4f  w=%.2f%%' % (
            code, df.loc[code, 'dividend_yield'], w * 100))
    if len(g.target_weights) > 5:
        log.info('  ... 共 %d 只' % len(g.target_weights))

    rebalance_ordered(context, g.target_weights)


def before_trading_start(context):
    pass
