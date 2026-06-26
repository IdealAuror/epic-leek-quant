"""
P1-F1-EV-2026Q2-v1-CSI300 — EV<0 因子策略（沪深300样本，standalone）
====================================================================
聚宽策略编辑器直接粘贴运行，无外部依赖。

spec_id: P1-F1-EV-2026Q2-v1-CSI300
样本: 沪深300成分股
基准: 000300.XSHG（沪深300全收益）
换仓: 季度（5/9/11 月首个交易日，对齐财报披露截止 4/30、8/31、10/31）
持仓: 20 只等权
成本: 万2.5双边 + 千1印花税(仅卖) + 5元最低 + 0.3%滑点

因子定义（Phase 1 简化口径，原文档 §3.1）:
  EV = market_cap + total_liability - cash_equivalents
  factor_ev_negative = 1 if EV < 0 else 0

过滤条件:
  - factor_ev_negative == 1
  - net_profit > 0
  - debt_to_assets <= 1.0

口径来源: docs/PROJECT-PLAN.md §1, research/theory-framework.md §5.1
"""

import datetime
import numpy as np
import pandas as pd

# 聚宽策略环境自动注入（无需 import）:
#   jqdata, get_fundamentals, query, valuation, balance, income, cash_flow,
#   get_index_stocks, get_all_securities, get_extras, get_security_info, get_price,
#   set_benchmark, set_order_cost, OrderCost, set_slippage, PriceSlippage,
#   order_target_percent, log, g, run_monthly


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
    """降级链下单封装，兼容聚宽各引擎交易函数注入差异。

    Level 1: order_target_percent（原生目标占比）
    Level 2: order_target_value（目标市值 = total_value × weight）
    Level 3: order_target（目标股数，整手处理）
    Level 4: order（与当前持仓差额下单）
    全不可用: 抛 RuntimeError（而非 NameError）
    """
    total_value = context.portfolio.total_value

    # Level 1: 原生 order_target_percent
    fn = _resolve_jq_func('order_target_percent')
    if fn is not None:
        return fn(code, weight)

    # Level 2: order_target_value（目标市值）
    fn = _resolve_jq_func('order_target_value')
    if fn is not None:
        return fn(code, total_value * weight)

    # 取当前持仓与价格，用于 Level 3/4
    current_date = context.current_dt.strftime('%Y-%m-%d')
    price = _get_current_price(code, current_date)
    positions = context.portfolio.positions
    pos = positions.get(code)
    current_amount = pos.total_amount if pos is not None else 0

    # Level 3: order_target（目标股数，整手处理）
    fn = _resolve_jq_func('order_target')
    if fn is not None:
        if price is None or price <= 0:
            raise RuntimeError('无法获取 %s 价格以计算目标股数' % code)
        target_shares = int(total_value * weight / price / 100) * 100
        return fn(code, target_shares)

    # Level 4: order（差额下单，整手处理）
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
    """构建股票池：指数成分股 - ST - 次新股。"""
    if index_id is None:
        sec_df = get_all_securities(['stock'], date=date_str)
        stocks = list(sec_df.index)
    else:
        stocks = get_index_stocks(index_id, date=date_str)
    # ST 剔除
    if len(stocks) > 0:
        st_df = get_extras('is_st', stocks, end_date=date_str, count=1)
        if st_df is not None and not st_df.empty:
            st_today = st_df.iloc[-1]
            stocks = [s for s in stocks if s in st_today.index and not st_today[s]]
    # 次新股剔除
    stocks = [s for s in stocks if not is_new_stock(s, date_str, min_listed_days)]
    return stocks


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
# 五、F1 因子计算（EV<0，Phase 1 简化口径）
# ============================================================

def calculate_ev_factor(df):
    """计算 EV<0 因子及过滤指标。

    Phase 1 简化口径（原文档 §3.1 + PROJECT-PLAN §1.3）:
      EV = market_cap + total_liability - cash_equivalents

    输入 df 必须含列（聚宽确定存在）:
      market_cap, total_liability, cash_equivalents, total_assets, net_profit
    """
    df['ev'] = df['market_cap'] + df['total_liability'] - df['cash_equivalents']
    df['factor_ev_negative'] = (df['ev'] < 0).astype(int)
    df['debt_to_assets'] = df['total_liability'] / df['total_assets'].replace(0, np.nan)
    return df


# ============================================================
# 六、策略主体
# ============================================================

INDEX_ID = '000300.XSHG'
BENCHMARK = '000300.XSHG'


def initialize(context):
    set_benchmark(BENCHMARK)
    apply_cost_model()
    g.stock_num = 20
    g.index_id = INDEX_ID
    run_monthly(rebalance, monthday=1)


def rebalance(context):
    current_date = context.current_dt
    if current_date.month not in (5, 9, 11):
        return
    date_str = current_date.strftime('%Y-%m-%d')

    # 1. 股票池
    stocks = get_stock_pool(g.index_id, date_str)
    if len(stocks) == 0:
        log.info('[%s] 股票池为空' % date_str)
        return

    # 2. 财务数据（仅聚宽确定存在的字段）
    q = query(
        valuation.code,
        valuation.market_cap,
        balance.total_liability,
        balance.cash_equivalents,
        balance.total_assets,
        income.net_profit,
    ).filter(valuation.code.in_(stocks))
    df = get_fundamentals(q, date=date_str)

    if df is None or df.empty:
        log.info('[%s] 无财务数据' % date_str)
        return

    df = df.set_index('code')

    # 3. 关键字段缺失剔除
    critical = ['market_cap', 'total_liability', 'cash_equivalents',
                'total_assets', 'net_profit']
    df = df.dropna(subset=critical)

    # 4. 因子计算
    df = calculate_ev_factor(df)

    # 5. 过滤：EV<0 + 净利润>0 + 资产负债率<=1
    mask = df['factor_ev_negative'] == 1
    mask &= df['net_profit'] > 0
    mask &= df['debt_to_assets'] <= 1.0
    df = df[mask]

    if df.empty:
        log.info('[%s] 无符合条件的 EV<0 股票' % date_str)
        return

    # 6. 取前 N 只等权
    df = df.head(g.stock_num)
    target_weights = {code: 1.0 / len(df) for code in df.index}

    log.info('[%s] 调仓：买入 %d 只股票' % (date_str, len(df)))
    for code in df.index:
        log.info('  买入 %s  ev=%.2f dta=%.2f' % (
            code, df.loc[code, 'ev'], df.loc[code, 'debt_to_assets'],
        ))

    rebalance_ordered(context, target_weights)


def before_trading_start(context):
    pass
