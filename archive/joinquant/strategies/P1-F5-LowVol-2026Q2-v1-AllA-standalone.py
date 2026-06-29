"""
P1-F5-LowVol-2026Q2-v1-AllA — 低波动率因子策略（全A样本，standalone）
====================================================================
聚宽策略编辑器直接粘贴运行，无外部依赖。

spec_id: P1-F5-LowVol-2026Q2-v1-AllA
样本: 全A股（剔除ST/次新股/金融股）
基准: 000985.XSHG（中证全指）
换仓: 季度（5/9/11 月首个交易日）
持仓: 50 只，IC加权（非等权）
成本: 万2.5双边 + 千1印花税(仅卖) + 5元最低 + 0.3%滑点

因子定义（Phase 1.4，IC 验证通过）:
  vol_60d = std(过去60交易日日收益率)
  signal = -vol_60d（取负，低波动=高信号=IC>0有效）

选股方式:
  - 按 signal 降序排序（低波动优先）
  - 取前 50 只
  - 权重按 signal 横截面 z-score 归一化（IC加权）

Gate 1 结论（2026-06-27，详见 research/decisions/P1-F1-EV-2026Q2-v1-decision.md）:
  - 市值中性化后 IC t=3.17（大幅过门槛）
  - 行业中性化后 IC t=3.70（大幅过门槛）
  - 七条 6/7 过，形式表现优于 F2-EP
  - Q 分组倒 U 型（非单调），故用全截面IC加权而非Q5多头
  - 定位：风险管理因子（降波动/提夏普），非收益主力

数据链简化:
  - 仅需 get_price 算波动率，无需查 balance/income/cash_flow
  - 无财务字段名探测陷阱
  - 金融股剔除保留（银行天然低波动，不剔会主导信号）

口径来源: research/decisions/P1-F1-EV-2026Q2-v1-decision.md Phase 1.4 终判
"""

import datetime
import numpy as np
import pandas as pd

# 聚宽策略环境自动注入（无需 import）:
#   jqdata, get_fundamentals, query, valuation, balance, income, cash_flow,
#   get_index_stocks, get_all_securities, get_extras, get_security_info, get_price,
#   get_industry, set_benchmark, set_order_cost, OrderCost, set_slippage, PriceSlippage,
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
    """构建股票池：指数成分股 - ST - 次新股 - 金融股。"""
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
    # 金融股剔除（sw_l1 严格相等，银行I/非银金融I）
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
# 五、F5 因子计算（低波动率，Phase 1.4 IC 验证通过）
# ============================================================

VOL_LOOKBACK = 60       # 主信号：60交易日（约3个月）
VOL_MIN_OBS_RATIO = 0.5  # 有效观测数下限 = lookback * 0.5


def calc_realized_volatility(date_str, stocks, lookback_days=60):
    """计算实现波动率：过去 lookback_days 交易日日收益率标准差。

    使用 get_price(count=lookback_days+1) 取 trailing 日收盘价，
    pct_change 算日收益率后取 std。count+1 因 pct_change 丢首行。

    返回 {code: vol}，vol = std(daily returns)。
    有效观测 < lookback_days * VOL_MIN_OBS_RATIO 的股票不返回（新股/长期停牌）。
    """
    if not stocks:
        return {}
    try:
        df = get_price(stocks, end_date=date_str, count=lookback_days + 1,
                       fields=['close'], skip_paused=False,
                       panel=False, fq='post')
        if df is None or df.empty:
            return {}
        if 'time' in df.columns:
            df = df.set_index('time')
        elif 'date' in df.columns:
            df = df.set_index('date')
        if 'code' in df.columns:
            close = df.pivot_table(index=df.index, columns='code', values='close')
        else:
            close = df
        close.index = pd.to_datetime(close.index)
    except Exception:
        return {}

    if close is None or close.empty:
        return {}

    rets = close.pct_change()
    vol = rets.std(skipna=True)
    valid_counts = rets.count()
    min_obs = int(lookback_days * VOL_MIN_OBS_RATIO)

    result = {}
    for code in stocks:
        if code not in vol.index:
            continue
        cnt = valid_counts.get(code, 0)
        if cnt < min_obs:
            continue
        v = vol[code]
        if not np.isnan(v) and v > 0:
            result[code] = float(v)
    return result


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

    # 2. 波动率因子（仅需 get_price，无需查财务表）
    vol_map = calc_realized_volatility(date_str, stocks, VOL_LOOKBACK)
    if not vol_map:
        log.info('[%s] 无波动率数据' % date_str)
        return

    # 3. 构建因子 DataFrame
    df = pd.DataFrame({'vol_60d': pd.Series(vol_map)})
    df['signal'] = -df['vol_60d']  # 低波动=高信号
    df = df.dropna(subset=['signal'])

    if df.empty:
        log.info('[%s] 无符合条件的 LowVol 股票' % date_str)
        return

    # 4. IC加权选股：按 signal 降序取前 N，权重按 z-score 归一化
    df = df.sort_values('signal', ascending=False).head(g.stock_num)

    sig_vals = df['signal'].values
    z = (sig_vals - sig_vals.mean()) / (sig_vals.std() if sig_vals.std() > 0 else 1)
    weights = np.where(z > 0, z, 0)
    if weights.sum() == 0:
        weights = np.ones(len(df))
    weights = weights / weights.sum()
    g.target_weights = {code: float(w) for code, w in zip(df.index, weights)}

    log.info('[%s] 调仓：买入 %d 只股票 (IC加权)' % (date_str, len(df)))
    for code, w in list(g.target_weights.items())[:5]:
        log.info('  买入 %s  vol=%.4f  w=%.2f%%' % (
            code, df.loc[code, 'vol_60d'], w * 100))
    if len(g.target_weights) > 5:
        log.info('  ... 共 %d 只' % len(g.target_weights))

    rebalance_ordered(context, g.target_weights)


def before_trading_start(context):
    pass
