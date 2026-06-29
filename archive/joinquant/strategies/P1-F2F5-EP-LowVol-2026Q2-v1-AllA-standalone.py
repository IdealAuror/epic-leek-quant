"""
P1-F2F5-EP-LowVol-2026Q2-v1-AllA — EP+LowVol 双因子组合策略（全A样本，standalone）
================================================================================
聚宽策略编辑器直接粘贴运行，无外部依赖。

spec_id: P1-F2F5-EP-LowVol-2026Q2-v1-AllA
样本: 全A股（剔除ST/次新股/金融股）
基准: 000985.XSHG（中证全指）
换仓: 季度（5/9/11 月首个交易日）
持仓: 50 只，IC加权（非等权）
成本: 万2.5双边 + 千1印花税(仅卖) + 5元最低 + 0.3%滑点

因子定义:
  F2-EP:  ep_spot = net_profit / (market_cap * 1e8)（市盈率倒数，越高=越便宜）
  F5-LV:  vol_60d = std(过去60交易日日收益率)
          vol_signal = -vol_60d（低波动=高信号）

合成方式（等权 z-score，无参数避免过拟合）:
  ep_z  = z_score(ep_spot)
  vol_z = z_score(vol_signal)
  combined = 0.5 * ep_z + 0.5 * vol_z

选股方式:
  - 按 combined 降序排序
  - 取前 50 只
  - 权重按 combined 横截面 z-score 归一化（IC加权）

过滤条件:
  - net_profit > 0
  - debt_to_assets <= 1.0
  - ep_spot 有效且 > 0
  - vol_60d 有效且 > 0
  - 剔除金融股（银行/非银金融，sw_l1 严格相等）

设计理由:
  - F2-EP 是收益主力（年化 13.1%），但回撤 51.6% 太大
  - F5-LowVol 是风险管理因子（Q5 波动降 30%，夏普提升 50%）
  - 两者选股池不同（EP 选便宜股，LowVol 选稳健股），低相关
  - 等权 z-score 合成是最简方案：无参数、不引入过拟合、不偏向任一因子
  - 终极目标：验证 F5 能否为 F2 降回撤

Gate 1 结论（2026-06-27，详见 research/decisions/P1-F1-EV-2026Q2-v1-decision.md）:
  - F2-EP:  市值中性化 t=3.41
  - F5-LV:   市值中性化 t=3.17，行业中性化 t=3.70，断点后增强
  - 两因子都通过决定性门槛，低相关，适合组合

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
# 五、因子计算（F2-EP + F5-LowVol，Phase 1.2/1.4 IC 验证通过）
# ============================================================

VOL_LOOKBACK = 60       # F5 主信号：60交易日（约3个月）
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


def calculate_combined_factors(df, vol_map):
    """计算 EP + LowVol 组合因子。

    F2-EP:  ep_spot = net_profit / (market_cap * 1e8)
            即市盈率倒数（盈利收益率），越高=越便宜=好
    F5-LV:  vol_signal = -vol_60d（低波动=高信号）

    合成（等权 z-score，无参数）:
      ep_z  = z_score(ep_spot)
      vol_z = z_score(vol_signal)
      combined = 0.5 * ep_z + 0.5 * vol_z

    输入 df 必须含列:
      market_cap（亿元）, net_profit（元）, total_liability（元）, total_assets（元）
    vol_map: {code: vol_60d} 字典
    """
    mcap_yuan = df['market_cap'] * 1e8  # 亿元 -> 元
    df['ep_spot'] = df['net_profit'] / mcap_yuan.replace(0, np.nan)
    df['debt_to_assets'] = df['total_liability'] / df['total_assets'].replace(0, np.nan)

    # F5 波动率信号
    df['vol_60d'] = df.index.map(lambda c: vol_map.get(c, np.nan))
    df['vol_signal'] = -df['vol_60d']

    return df


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
    """季频双因子选股：5/9/11 月，EP+LowVol 等权 z-score 合成，IC 加权选股。"""
    current_date = context.current_dt
    if current_date.month not in (5, 9, 11):
        return
    date_str = current_date.strftime('%Y-%m-%d')

    # 1. 股票池
    stocks = get_stock_pool(g.index_id, date_str)
    if len(stocks) == 0:
        log.info('[%s] 股票池为空' % date_str)
        return

    # 2. F2 财务数据（EP 需要 net_profit/market_cap/debt_to_assets）
    q = query(
        valuation.code,
        valuation.market_cap,
        balance.total_liability,
        balance.total_assets,
        income.net_profit,
    ).filter(valuation.code.in_(stocks))
    df = get_fundamentals(q, date=date_str)

    if df is None or df.empty:
        log.info('[%s] 无财务数据' % date_str)
        return

    df = df.set_index('code')

    # 3. 关键字段缺失剔除
    critical = ['market_cap', 'total_liability', 'total_assets', 'net_profit']
    df = df.dropna(subset=critical)

    # 4. F5 波动率因子
    vol_map = calc_realized_volatility(date_str, list(df.index), VOL_LOOKBACK)

    # 5. 因子计算（EP + LowVol）
    df = calculate_combined_factors(df, vol_map)

    # 6. 过滤：F2 基本面 + F5 波动率有效
    mask = df['net_profit'] > 0
    mask &= df['debt_to_assets'] <= 1.0
    mask &= df['ep_spot'].notna() & (df['ep_spot'] > 0)
    mask &= df['vol_60d'].notna() & (df['vol_60d'] > 0)
    df = df[mask]

    if df.empty:
        log.info('[%s] 无符合条件的组合股票' % date_str)
        return

    # 7. 等权 z-score 合成（无参数，避免过拟合）
    ep_std = df['ep_spot'].std()
    vol_std = df['vol_signal'].std()
    df['ep_z'] = (df['ep_spot'] - df['ep_spot'].mean()) / (ep_std if ep_std > 0 else 1)
    df['vol_z'] = (df['vol_signal'] - df['vol_signal'].mean()) / (vol_std if vol_std > 0 else 1)
    df['combined'] = 0.5 * df['ep_z'] + 0.5 * df['vol_z']

    # 8. IC加权选股：按 combined 降序取前 N，权重按 z-score 归一化
    df = df.sort_values('combined', ascending=False).head(g.stock_num)

    comb_vals = df['combined'].values
    z = (comb_vals - comb_vals.mean()) / (comb_vals.std() if comb_vals.std() > 0 else 1)
    weights = np.where(z > 0, z, 0)
    if weights.sum() == 0:
        weights = np.ones(len(df))
    weights = weights / weights.sum()
    g.target_weights = {code: float(w) for code, w in zip(df.index, weights)}

    log.info('[%s] 调仓：买入 %d 只股票 (EP+LowVol 等权z-score合成)' % (
        date_str, len(df)))
    for code, w in list(g.target_weights.items())[:5]:
        log.info('  买入 %s  ep=%.4f vol=%.4f w=%.2f%%' % (
            code, df.loc[code, 'ep_spot'], df.loc[code, 'vol_60d'], w * 100))
    if len(g.target_weights) > 5:
        log.info('  ... 共 %d 只' % len(g.target_weights))

    rebalance_ordered(context, g.target_weights)


def before_trading_start(context):
    pass
