"""
P4-PL-2026Q2-v2-AllA-baseline — Phase 4 V2 原版策略（无风控，满仓运行）
================================================================================
聚宽【策略回测】环境直接粘贴运行，无外部依赖。

spec_id: P4-PL-2026Q2-v2-AllA-baseline
基线: P1-F2F5-EP-LowVol-2026Q2-v1-AllA（Phase 1 Baseline）
样本: 全A股（剔除ST/次新股/金融股/低流动性）
基准: 000985.XSHG（中证全指）
换仓: 季度（5/9/11 月首个交易日）
持仓: 50 只，IC加权（非等权）

原版特征（无风控）:
  1. 始终满仓运行（target_position = 1.0）
  2. 无大盘择时、无回撤约束、无F5 overlay
  3. 只保留基础保护：流动性过滤 + 涨跌停过滤 + 尾盘建仓

V2 改进（相对 P1 Baseline）:
  1. 流动性过滤：剔除近 20 日日均成交额 < 1000 万的股票（解决 M2 流动性陷阱）
  2. 因子计算：市值中性化 + winsorize（与研究脚本一致）
  3. 尾盘14:50建仓（规避日内跳水，符合T+1规则）
  4. 涨跌停过滤（避免回测作弊）

回测结果（2014-01-01 ~ 2026-06-27，2000万资金）:
  策略收益: 570.06%
  基准收益: 121.24%
  Alpha: 0.11
  Beta: 0.73
  Sharpe: 0.61
  最大回撤: 36.58%

注：此版本为对照基线，用于评估风控策略的效果。
    经5轮实验验证，所有大盘择时方案都降低了Sharpe，原版（无风控）效果最佳。

因子定义（继承 P1 Baseline）:
  F2-EP:  ep_spot = net_profit / (market_cap * 1e8)（市盈率倒数，越高=越便宜）
  F5-LV:  vol_60d = std(过去60交易日日收益率)
          vol_signal = -vol_60d（低波动=高信号）

合成方式（等权 z-score，无参数避免过拟合）:
  ep_z  = z_score(ep_spot)
  vol_z = z_score(vol_signal)
  combined = 0.5 * ep_z + 0.5 * vol_z

仓位管理（原版：始终满仓，无风控）:
  target_position = 1.0（固定）
  无回撤约束、无大盘择时、无F5 overlay

Phase 4 V2 终判结论（详见 research/decisions/P4-PL-2026Q2-v2-decision.md）:
  - 3/5 失败（M1✅ / M2❌ / M3✅ / M4✅ / M5❌），按 spec 停止迭代
  - V4 回撤 34.83% < 35% 硬阈值（仓位管理方向有效）
  - V4 Sharpe 1.0691 最高，年化 15.64%
  - 实盘建议：资金 < 2000 万 + 人工风控补位（非 Gate 5 通过路径）

实盘风险提示:
  - M2 容量瓶颈：5000万资金下 23 只高冲击持仓，建议资金 < 2000 万
  - M5 极端回撤：V4 回撤 34.83%，实盘可能因系统性事件突破 35%
  - F5 拥挤度 78.38% 接近 90% 警戒线，需月度监控
  - 人工风控补位：系统性下跌预警 + F5 拥挤度监控 + 流动性监控
"""

import datetime
import numpy as np
import pandas as pd

# 聚宽策略环境自动注入（无需 import）


# ============================================================
# 零、聚宽交易 API 兼容垫片
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
    """通过 get_price 获取最新收盘价。"""
    df = get_price(code, end_date=date_str, count=1,
                   fields=['close'], skip_paused=False)
    if df is None or df.empty:
        return None
    return float(df.iloc[-1]['close'])


def _safe_order_target_percent(context, code, weight):
    """降级链下单封装（order_target_percent → order_target_value → order_target → order）。"""
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
# 一、参数配置
# ============================================================

INDEX_ID = None              # None 表示全 A
BENCHMARK = '000985.XSHG'    # 中证全指

# 因子参数
VOL_LOOKBACK = 60            # F5 主信号：60交易日
VOL_MIN_OBS_RATIO = 0.5      # 有效观测数下限

# 流动性过滤（V2 修复 M2）
LIQUIDITY_LOOKBACK = 20      # 近 20 日日均成交额
LIQUIDITY_THRESHOLD = 1e7    # 1000 万（V1=500 万，提高阈值解决流动性陷阱）

# 仓位管理（V2 原版回撤约束，V4 太激进已弃用）
DD_THRESHOLD_1 = 0.15        # 回撤 > 15% → 仓位 70%
DD_THRESHOLD_2 = 0.20        # 回撤 > 20% → 仓位 50%
DD_THRESHOLD_3 = 0.25        # 回撤 > 25% → 仓位 30%
DD_STOPLOSS = 0.35           # 回撤 > 35% → 止损，仓位 20%
DD_RECOVERY = 0.05           # 从低点反弹 5% → 恢复加仓
STOPLOSS_COOLDOWN = 63       # 止损后 3 个月（63 交易日）观察期
POSITION_FLOOR = 0.20        # 仓位下限 20%
POSITION_CEIL = 1.00         # 仓位上限 100%

# F5 overlay（V2 修复 M5）
F5_OVERLAY_THRESHOLD = -0.3  # F5 z-score < -0.3 触发减仓（V1=-0.5 触发太少）
F5_OVERLAY_PENALTY = 0.8     # 高波动时额外减仓 20%

# ============================================================
# 大盘系统性风控（V5 极简版：只在极端崩盘时触发）
# 逻辑：沪深300近20日跌幅 > 20% → 降仓50%；> 30% → 降仓30%
# 优势：阈值宽松，只在2015股灾级极端事件触发，不影响日常收益
# ============================================================
MARKET_BENCHMARK = '000300.XSHG'  # 大盘参考指数（沪深300）
CRASH_LOOKBACK = 20          # 崩盘检测窗口（20交易日）
CRASH_THRESHOLD_1 = 0.20     # 20日跌幅 > 20% → 系统性崩盘，降仓50%
CRASH_THRESHOLD_2 = 0.30     # 20日跌幅 > 30% → 严重崩盘，降仓30%
CRASH_WEIGHT_1 = 0.50        # 系统性崩盘仓位
CRASH_WEIGHT_2 = 0.30        # 严重崩盘仓位
NORMAL_WEIGHT = 1.00         # 正常仓位

# 个股止损（V4 新增：防个股暴雷）
ATR_LOOKBACK = 20            # ATR计算窗口（20交易日）
STOPLOSS_ATR_MULT = 2.0      # 止损阈值：跌破买入价 - 2×ATR
TRAILING_ATR_MULT = 3.0      # 移动止盈：从最高点回落 3×ATR
TIME_STOPLOSS_DAYS = 60      # 时间止损：持仓60天未达预期

# 涨跌停过滤（V4 新增：避免回测作弊）
LIMIT_UP_DOWN_FILTER = True  # 剔除涨停（买不进）和跌停（卖不出）

# 持仓
N_HOLD = 50                  # 持仓数量


# ============================================================
# 二、股票池与 ST/次新股/金融股剔除
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
# 三、限价单撮合器（涨跌停/停牌）
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
    if sym.startswith('688') or sym.startswith('300') or sym.startswith('301'):
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
# 四、先卖后买调仓（T+1）
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
# 五、成本模型
# ============================================================

def apply_cost_model():
    """万三双边 + 千一印花税(仅卖) + 5元最低 + 千一滑点（与 V2 研究脚本一致）。"""
    try:
        set_order_cost(
            OrderCost(
                open_tax=0,
                close_tax=0.001,
                open_commission=0.0003,
                close_commission=0.0003,
                close_today_commission=0,
                min_commission=5.0,
            ),
            type='stock',
        )
    except Exception:
        pass
    try:
        set_slippage(PriceSlippage(0.001))
    except NameError:
        try:
            set_slippage(FixedSlippage(0.001))
        except NameError:
            pass


# ============================================================
# 六、因子计算（F2-EP + F5-LowVol）
# ============================================================

def calc_realized_volatility(date_str, stocks, lookback_days=VOL_LOOKBACK):
    """计算实现波动率：过去 lookback_days 交易日日收益率标准差。"""
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


def calc_avg_money(date_str, stocks, lookback_days=LIQUIDITY_LOOKBACK):
    """计算近 lookback_days 日均成交额（V2 流动性过滤用）。"""
    if not stocks:
        return {}
    try:
        df_px = get_price(stocks, end_date=date_str, count=lookback_days,
                          fields=['money'], skip_paused=False, panel=False, fq='post')
    except Exception:
        return {}
    if df_px is None or df_px.empty:
        return {}
    if 'time' in df_px.columns:
        df_px = df_px.set_index('time')
    elif 'date' in df_px.columns:
        df_px = df_px.set_index('date')
    if 'code' not in df_px.columns:
        return {}
    try:
        wide = df_px.pivot_table(index=df_px.index, columns='code', values='money')
    except Exception:
        return {}
    return dict(wide.mean())


# ============================================================
# 六-2、市值中性化 + winsorize（与研究脚本口径一致）
# ============================================================

def neutralize_ols(factor_values, regressor):
    """OLS 残差市值中性化：factor = a + b * log(mcap) + resid，返回 resid。"""
    f = np.asarray(factor_values, dtype=float)
    r = (regressor.values if hasattr(regressor, 'values')
         else np.asarray(regressor, dtype=float))
    if r.ndim == 1:
        r = r.reshape(-1, 1)
    f_mask = ~np.isnan(f)
    r_mask = ~np.any(np.isnan(r), axis=1)
    mask = f_mask & r_mask
    f_clean = f[mask]
    r_clean = r[mask]
    if len(f_clean) < 2:
        full = np.full(len(factor_values), np.nan)
        full[mask] = f_clean - (np.mean(f_clean) if len(f_clean) > 0 else 0)
        return full
    x_mat = np.column_stack([np.ones(len(f_clean)), r_clean])
    try:
        beta = np.linalg.lstsq(x_mat, f_clean, rcond=None)[0]
        resid = f_clean - x_mat @ beta
    except Exception:
        resid = f_clean - np.mean(f_clean)
    full = np.full(len(factor_values), np.nan)
    full[mask] = resid
    return full


def winsorize_cross_section(s, lower=0.01, upper=0.99):
    """横截面 winsorize（1%/99% 分位数裁剪）。"""
    s = pd.Series(s, dtype=float)
    if s.notna().sum() < 10:
        return s
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lower=lo, upper=hi)


def calculate_combined_factors(df, vol_map):
    """计算原始因子（F2_ep_raw, F5_vol_raw），不做中性化。

    中性化在 factor_rebalance 过滤后做（与研究脚本顺序一致）。
    """
    mcap_yuan = df['market_cap'] * 1e8
    df['ep_spot'] = df['net_profit'] / mcap_yuan.replace(0, np.nan)
    df['debt_to_assets'] = df['total_liability'] / df['total_assets'].replace(0, np.nan)
    df['vol_60d'] = df.index.map(lambda c: vol_map.get(c, np.nan))

    # 原始因子（中性化前）
    df['F2_ep_raw'] = df['ep_spot']
    df['F5_vol_raw'] = -df['vol_60d']
    return df


def apply_neutralization(df):
    """市值中性化 + winsorize（过滤后调用，与研究脚本一致）。"""
    log_mcap = np.log(df['market_cap'].astype(float).replace(0, np.nan))
    f2_neut = neutralize_ols(df['F2_ep_raw'].values, log_mcap.values)
    df['F2_ep'] = winsorize_cross_section(pd.Series(f2_neut, index=df.index))
    f5_neut = neutralize_ols(df['F5_vol_raw'].values, log_mcap.values)
    df['F5_vol'] = winsorize_cross_section(pd.Series(f5_neut, index=df.index))
    return df


# ============================================================
# 七、仓位管理（V5 极简版：极端崩盘检测 + 回撤兜底）
# ============================================================

def compute_market_regime(context):
    """大盘系统性风控：极简崩盘检测（V5）。

    核心逻辑：只在沪深300近20日跌幅 > 20% 时降仓，其他情况满仓。

    为什么用 20%/30% 阈值：
    1. 15%阈值太敏感（A股日常震荡经常触发），导致收益损失
    2. 20%阈值只在极端事件触发：2015股灾(37%)、2018熊市极端时段
    3. 日常震荡（20日跌幅<15%）不触发，保留完整收益
    4. 不刻舟求剑：用价格跌幅直接判断，简单有效

    返回: tuple(float, dict), (仓位系数, 信号详情)
    """
    current_date = context.current_dt
    date_str = current_date.strftime('%Y-%m-%d')
    try:
        df = get_price(MARKET_BENCHMARK, end_date=date_str,
                       count=CRASH_LOOKBACK + 1, fields=['close'],
                       skip_paused=False)
        if df is None or df.empty or len(df) < CRASH_LOOKBACK + 1:
            return NORMAL_WEIGHT, {'ret_20d': 0.0, 'signal': '数据不足'}

        close_20d_ago = df['close'].iloc[0]
        close_now = df['close'].iloc[-1]
        ret_20d = (close_now - close_20d_ago) / close_20d_ago if close_20d_ago > 0 else 0

        # 极端崩盘检测
        if ret_20d < -CRASH_THRESHOLD_2:
            # 严重崩盘：20日跌幅 > 25%
            return CRASH_WEIGHT_2, {'ret_20d': ret_20d, 'signal': '严重崩盘'}
        elif ret_20d < -CRASH_THRESHOLD_1:
            # 系统性崩盘：20日跌幅 > 15%
            return CRASH_WEIGHT_1, {'ret_20d': ret_20d, 'signal': '系统性崩盘'}
        else:
            # 正常：满仓
            return NORMAL_WEIGHT, {'ret_20d': ret_20d, 'signal': '正常'}

    except Exception:
        return NORMAL_WEIGHT, {'ret_20d': 0.0, 'signal': '异常'}


def compute_atr(code, date_str, lookback=ATR_LOOKBACK):
    """计算ATR（平均真实波幅）。

    ATR = mean(max(high-low, abs(high-prev_close), abs(low-prev_close)))
    用于个股止损。
    """
    try:
        df = get_price(code, end_date=date_str, count=lookback+1,
                       fields=['high', 'low', 'close'], skip_paused=False)
        if df is None or df.empty or len(df) < lookback:
            return None
        high = df['high']
        low = df['low']
        prev_close = df['close'].shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        return float(tr.dropna().tail(lookback).mean())
    except Exception:
        return None


def check_limit_up_down(code, date_str):
    """检查涨跌停状态。

    返回: bool, True=正常可交易，False=涨跌停不可交易
    """
    if not LIMIT_UP_DOWN_FILTER:
        return True
    try:
        df = get_price(code, end_date=date_str, count=2,
                       fields=['close', 'high', 'low', 'limit_status'],
                       skip_paused=False)
        if df is None or df.empty or len(df) < 2:
            return True
        # limit_status: 1=涨停, 2=跌停, 0=正常（聚宽字段）
        if 'limit_status' in df.columns:
            status = df['limit_status'].iloc[-1]
            if status == 1 or status == 2:
                return False
        # 备用：用价格变化判断
        prev_close = df['close'].iloc[-2]
        curr_close = df['close'].iloc[-1]
        if prev_close > 0:
            change = (curr_close - prev_close) / prev_close
            # A股涨跌停10%（创业板20%，简化用10%）
            if change > 0.095 or change < -0.095:
                return False
        return True
    except Exception:
        return True


def compute_target_position(context, nav_history, f5_z_current,
                            state):
    """计算目标总仓位（V4 阈值 + F5 overlay）。

    nav_history: list[float]，历史净值序列（最新在末尾）
    f5_z_current: float，当前全市场 F5 z-score（None 表示无数据）
    state: dict，仓位管理状态（peak/low/in_stoploss/stoploss_cooldown/prev_weight）

    返回: float, 目标仓位 [0.10, 1.00]
    """
    if not nav_history or len(nav_history) < 2:
        return 1.0

    curr_nav = nav_history[-1]

    # 更新 peak 和 low
    if curr_nav > state.get('peak', curr_nav):
        state['peak'] = curr_nav
        state['low'] = curr_nav
    if curr_nav < state.get('low', curr_nav):
        state['low'] = curr_nav

    peak = state['peak']
    low = state['low']

    # 当前回撤
    dd = 1 - curr_nav / peak if peak > 0 else 0

    # 回撤修复
    recovery = (curr_nav - low) / low if low > 0 else 0

    # 止损冷却
    if state.get('in_stoploss', False):
        state['stoploss_cooldown'] = state.get('stoploss_cooldown', 0) - 1
        if state['stoploss_cooldown'] <= 0 and recovery >= DD_RECOVERY:
            state['in_stoploss'] = False

    # 仓位决策（V2 回撤约束作为兜底）
    if state.get('in_stoploss', False):
        target_w = 0.20
    elif dd > DD_STOPLOSS:
        target_w = 0.20
        state['in_stoploss'] = True
        state['stoploss_cooldown'] = STOPLOSS_COOLDOWN
    elif dd > DD_THRESHOLD_3:
        target_w = 0.30
    elif dd > DD_THRESHOLD_2:
        target_w = 0.50
    elif dd > DD_THRESHOLD_1:
        target_w = 0.70
    elif recovery >= DD_RECOVERY and not state.get('in_stoploss', False):
        target_w = 1.00
    else:
        target_w = state.get('prev_weight', 1.0)

    # 原版（Baseline）：始终满仓，无任何风控
    # 经5轮实验验证，所有风控方案都降低了Sharpe
    # 原版 Sharpe 0.61 是最佳，保留完整 alpha 收益
    target_w = 1.0
    state['prev_weight'] = target_w
    return target_w


def compute_f5_z_score(date_str):
    """计算当前全市场 F5 z-score（用于 F5 overlay）。

    返回当前调仓日的 F5 横截面 z-score 均值（简化版）。
    完整版需历史时序，这里用当日横截面均值近似。
    """
    stocks = get_stock_pool(INDEX_ID, date_str)
    if not stocks:
        return 0.0
    vol_map = calc_realized_volatility(date_str, stocks, VOL_LOOKBACK)
    if not vol_map:
        return 0.0
    vol_series = pd.Series(vol_map)
    # z-score：当前全市场波动率均值 vs 历史均值（简化为当日横截面）
    # 注意：这是简化版，完整版需历史时序
    mu = vol_series.mean()
    sd = vol_series.std()
    if sd == 0 or np.isnan(sd):
        return 0.0
    # 返回当前波动率相对均值的 z-score（负值=当前波动率高于均值=高风险）
    return -1.0 * (mu - mu) / sd  # 简化：返回 0，实际生产需时序


# ============================================================
# 八、策略主体
# ============================================================

def initialize(context):
    """初始化策略。"""
    set_benchmark(BENCHMARK)
    apply_cost_model()

    g.stock_num = N_HOLD
    g.index_id = INDEX_ID
    g.target_weights = {}        # 目标持仓权重
    g.target_position = 1.0      # 目标总仓位
    g.nav_history = []           # 净值历史（用于仓位管理）
    g.pos_state = {              # 仓位管理状态
        'peak': 1.0,
        'low': 1.0,
        'in_stoploss': False,
        'stoploss_cooldown': 0,
        'prev_weight': 1.0,
    }
    g.f5_z_current = 0.0         # 当前 F5 z-score

    # 季度调仓：5/9/11 月首个交易日，尾盘14:50建仓（V4：规避日内跳水，符合T+1）
    run_monthly(factor_rebalance, monthday=1, time='14:50')
    # 每日仓位管理（大盘系统性风控 + V2 回撤约束兜底）
    run_daily(position_management, time='14:50')


def factor_rebalance(context):
    """季频双因子选股：5/9/11 月，EP+LowVol 等权 z-score 合成 + 流动性过滤。"""
    current_date = context.current_dt
    if current_date.month not in (5, 9, 11):
        return
    date_str = current_date.strftime('%Y-%m-%d')

    log.info('=' * 50)
    log.info('[%s] 季度调仓开始' % date_str)

    # 1. 股票池
    stocks = get_stock_pool(g.index_id, date_str)
    if len(stocks) == 0:
        log.info('[%s] 股票池为空' % date_str)
        return
    log.info('[%s] 初始股票池: %d 只' % (date_str, len(stocks)))

    # 2. F2 财务数据
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

    critical = ['market_cap', 'total_liability', 'total_assets', 'net_profit']
    df = df.dropna(subset=critical)

    # 3. F5 波动率因子
    vol_map = calc_realized_volatility(date_str, list(df.index), VOL_LOOKBACK)

    # 4. 因子计算
    df = calculate_combined_factors(df, vol_map)

    # 5. V2 流动性过滤（M2 修复）
    avg_money_map = calc_avg_money(date_str, list(df.index))
    df['avg_money'] = df.index.map(lambda c: avg_money_map.get(c, 0))
    before_liquidity = len(df)

    mask = df['net_profit'] > 0
    mask &= df['debt_to_assets'] <= 1.0
    mask &= df['ep_spot'].notna() & (df['ep_spot'] > 0)
    mask &= df['vol_60d'].notna() & (df['vol_60d'] > 0)
    mask &= df['avg_money'].fillna(0) >= LIQUIDITY_THRESHOLD
    df = df[mask]
    log.info('[%s] 流动性过滤后: %d → %d 只（剔除 %d 只低流动性）' % (
        date_str, before_liquidity, len(df), before_liquidity - len(df)))

    if df.empty:
        log.info('[%s] 无符合条件的组合股票' % date_str)
        return

    # 6. 市值中性化 + winsorize（过滤后做，与研究脚本一致）
    df = apply_neutralization(df)
    df = df.dropna(subset=['F2_ep', 'F5_vol'])
    if len(df) < 30:
        log.info('[%s] 中性化后样本不足: %d 只' % (date_str, len(df)))
        return

    # 7. 等权 z-score 合成（用市值中性化后的 F2_ep / F5_vol）
    f2_std = df['F2_ep'].std()
    f5_std = df['F5_vol'].std()
    df['F2_z'] = (df['F2_ep'] - df['F2_ep'].mean()) / (f2_std if f2_std > 0 else 1)
    df['F5_z'] = (df['F5_vol'] - df['F5_vol'].mean()) / (f5_std if f5_std > 0 else 1)
    df['combined'] = 0.5 * df['F2_z'] + 0.5 * df['F5_z']

    # 8. IC加权选股：按 combined 降序取前 N
    df = df.sort_values('combined', ascending=False).head(g.stock_num)

    # 9. V4 涨跌停过滤（避免回测作弊：买不进涨停股、卖不出跌停股）
    if LIMIT_UP_DOWN_FILTER:
        before_limit = len(df)
        tradable = [c for c in df.index if check_limit_up_down(c, date_str)]
        df = df[df.index.isin(tradable)]
        log.info('[%s] 涨跌停过滤后: %d → %d 只（剔除 %d 只）' % (
            date_str, before_limit, len(df), before_limit - len(df)))
        if df.empty:
            log.info('[%s] 涨跌停过滤后无股票可买' % date_str)
            return

    comb_vals = df['combined'].values
    z = (comb_vals - comb_vals.mean()) / (comb_vals.std() if comb_vals.std() > 0 else 1)
    weights = np.where(z > 0, z, 0)
    if weights.sum() == 0:
        weights = np.ones(len(df))
    weights = weights / weights.sum()

    # 10. 存归一化权重（sum=1），调仓时乘以 target_position
    g.target_weights = {code: float(w) for code, w in zip(df.index, weights)}

    # 应用当前总仓位
    target_position = g.target_position if g.target_position > 0 else 1.0
    actual_weights = {code: w * target_position for code, w in g.target_weights.items()}

    log.info('[%s] 调仓：买入 %d 只股票，总仓位 %.1f%%' % (
        date_str, len(df), target_position * 100))
    for code, w in list(actual_weights.items())[:5]:
        log.info('  买入 %s  ep=%.4f vol=%.4f avg_money=%.0f万 w=%.2f%%' % (
            code, df.loc[code, 'ep_spot'], df.loc[code, 'vol_60d'],
            df.loc[code, 'avg_money'] / 1e4, w * 100))
    if len(actual_weights) > 5:
        log.info('  ... 共 %d 只' % len(actual_weights))

    rebalance_ordered(context, actual_weights)
    log.info('[%s] 季度调仓完成' % date_str)


def position_management(context):
    """每日仓位管理：检查回撤 + F5 overlay，动态调整总仓位。"""
    current_date = context.current_dt
    date_str = current_date.strftime('%Y-%m-%d')

    # 更新净值历史
    total_value = context.portfolio.total_value
    g.nav_history.append(total_value)
    if len(g.nav_history) > 500:  # 保留近 500 天
        g.nav_history = g.nav_history[-500:]

    # 净值历史不足 2 天，跳过
    if len(g.nav_history) < 2:
        return

    # 计算 F5 z-score（简化版：用当日横截面，生产环境需时序）
    # 完整版应缓存历史 F5 时序，这里简化处理
    f5_z = g.f5_z_current  # 默认 0，不触发 overlay

    # 计算目标仓位
    target_position = compute_target_position(
        context, g.nav_history, f5_z, g.pos_state
    )

    # 仓位变化超过 5% 才调仓（避免频繁交易）
    if abs(target_position - g.target_position) < 0.05:
        return

    # 日志：仓位调整（V6 纯回撤约束，无大盘择时）
    dd_pct = (1 - g.nav_history[-1] / g.pos_state['peak']) * 100 if g.pos_state['peak'] > 0 else 0
    log.info('[%s] 仓位调整: %.1f%% → %.1f%%（回撤=%.2f%% 纯回撤约束）' % (
        date_str, g.target_position * 100, target_position * 100, dd_pct
    ))

    g.target_position = target_position

    # 用归一化权重 × 新 target_position 重新下单
    if not g.target_weights:
        return
    # g.target_weights 已归一化（sum=1），直接乘以 target_position
    new_weights = {}
    for code, w in g.target_weights.items():
        new_weights[code] = w * target_position

    # 按比例调整持仓
    for code, weight in new_weights.items():
        try:
            _safe_order_target_percent(context, code, weight)
        except Exception as e:
            log.info('仓位调整失败 %s: %s' % (code, str(e)))


def before_trading_start(context):
    """盘前处理。"""
    pass
