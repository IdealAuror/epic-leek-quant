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

因子定义:
  interest_bearing_debt = total_liability
                          - accounts_payable - advance_payment
                          - wages_payable - taxes_payable - other_payable
  EV = market_cap + interest_bearing_debt - cash_equivalents
  factor_ev_negative = 1 if EV < 0 else 0

过滤条件:
  - factor_ev_negative == 1
  - net_profit > 0
  - debt_to_assets <= 1.0
  - current_ratio > 1.5

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
    # 前收盘
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
        if high_p > low_p:  # 盘中打开涨停
            return 1.0
        return 0.0  # 一字涨停
    elif side == 'sell':
        if open_p > low_limit:
            return 1.0
        if high_p > low_p:  # 盘中打开跌停
            return 1.0
        return 0.0  # 一字跌停
    return 0.0


# ============================================================
# 三、先卖后买调仓（T+1）
# ============================================================

def rebalance_ordered(context, target_weights):
    """先卖后买，涨跌停/停牌未成交记入 unfilled。"""
    current_date = context.current_dt.strftime('%Y-%m-%d')
    result = {'sold': [], 'bought': [], 'unfilled': []}

    # 先卖
    for code, pos in context.portfolio.positions.items():
        if pos.total_amount <= 0:
            continue
        if target_weights.get(code, 0.0) > 0:
            continue
        if simulate_limit_order(code, 'sell', current_date) > 0:
            order_target_percent(code, 0)
            result['sold'].append(code)
            log.info('卖出 %s' % code)
        else:
            result['unfilled'].append((code, 'sell'))
            log.info('卖出失败 %s（跌停/停牌）' % code)

    # 后买
    for code, weight in target_weights.items():
        if weight <= 0:
            continue
        if simulate_limit_order(code, 'buy', current_date) > 0:
            order_target_percent(code, weight)
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
# 五、F1 因子计算（EV<0）
# ============================================================

def calculate_ev_factor(df):
    """计算 EV<0 因子及相关过滤指标。

    输入 df 必须含列（聚宽真实字段名）:
      market_cap, total_liability, cash_equivalents, total_assets,
      total_current_assets, total_current_liability,
      accounts_payable, advance_payment, wages_payable,
      taxes_payable, other_payable, net_profit, net_operate_cash_flow
    """
    # 有息负债 = 总负债 - 无息经营性负债
    df['interest_bearing_debt'] = (
        df['total_liability']
        - df['accounts_payable']
        - df['advance_payment']
        - df['wages_payable']
        - df['taxes_payable']
        - df['other_payable']
    )
    # EV = 市值 + 有息负债 - 现金及等价物
    df['ev'] = df['market_cap'] + df['interest_bearing_debt'] - df['cash_equivalents']
    df['factor_ev_negative'] = (df['ev'] < 0).astype(int)

    # 过滤指标
    df['debt_to_assets'] = df['total_liability'] / df['total_assets'].replace(0, np.nan)
    df['current_ratio'] = (
        df['total_current_assets'] / df['total_current_liability'].replace(0, np.nan)
    )
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
    # 季度调仓：对齐财报披露截止（4/30、8/31、10/31）后的首个交易日
    # 即 5/9/11 月首个交易日
    if current_date.month not in (5, 9, 11):
        return
    date_str = current_date.strftime('%Y-%m-%d')

    # 1. 股票池
    stocks = get_stock_pool(g.index_id, date_str)
    if len(stocks) == 0:
        log.info('[%s] 股票池为空' % date_str)
        return

    # 2. 财务数据（聚宽真实字段名）
    q = query(
        valuation.code,
        valuation.market_cap,
        balance.total_liability,
        balance.cash_equivalents,
        balance.total_assets,
        balance.total_current_assets,
        balance.total_current_liability,
        balance.accounts_payable,
        balance.advance_payment,
        balance.wages_payable,
        balance.taxes_payable,
        balance.other_payable,
        income.net_profit,
        cash_flow.net_operate_cash_flow,
    ).filter(valuation.code.in_(stocks))
    df = get_fundamentals(q, date=date_str)

    if df is None or df.empty:
        log.info('[%s] 无财务数据' % date_str)
        return

    # 用 code 作为 index
    df = df.set_index('code')

    # 3. 关键字段缺失剔除
    critical = ['market_cap', 'total_liability', 'cash_equivalents',
                'total_assets', 'net_profit']
    df = df.dropna(subset=critical)

    # 4. 因子计算
    df = calculate_ev_factor(df)

    # 5. 过滤：EV<0 + 净利润>0 + 资产负债率<=1 + 流动比率>1.5
    mask = df['factor_ev_negative'] == 1
    mask &= df['net_profit'] > 0
    mask &= df['debt_to_assets'] <= 1.0
    mask &= df['current_ratio'] > 1.5
    df = df[mask]

    if df.empty:
        log.info('[%s] 无符合条件的 EV<0 股票' % date_str)
        return

    # 6. 取前 N 只等权
    df = df.head(g.stock_num)
    target_weights = {code: 1.0 / len(df) for code in df.index}

    log.info('[%s] 调仓：买入 %d 只股票' % (date_str, len(df)))
    for code in df.index:
        log.info('  买入 %s  ev=%.2f cr=%.2f dta=%.2f' % (
            code, df.loc[code, 'ev'], df.loc[code, 'current_ratio'],
            df.loc[code, 'debt_to_assets'],
        ))

    # 7. 先卖后买
    rebalance_ordered(context, target_weights)


def before_trading_start(context):
    pass
