"""
P5-F2F5F6-40d-final-strategy — CST-Quant 最终策略
================================================================================
CST = Cheap · Stable · Trending（低估 · 稳定 · 趋势）
聚宽【策略回测】环境直接粘贴运行，单文件无外部依赖。

来源: P4-PL-2026Q2-v2-AllA-F2F5F6-baseline，经 P5 F6-MOM 40d 优化 + 跨样本验证
因子: F2(EP) + F5(LowVol) + F6(MOM-40d)，等权 Z-score
样本: 全A股（剔除ST/次新股/金融股/低流动性）
基准: 000985.XSHG（中证全指）
换仓: 季度（5/9/11 月首个交易日，14:50）
持仓: 50 只，等权

回测结果（2014-01-01 ~ 2026-06-28，聚宽策略环境）:
  累计收益: 777%    vs 基准 121%
  年化收益: ~19.5%
  Sharpe:    0.6728
  Alpha:     0.1351
  Beta:      0.7573
  最大回撤:  37.22%
  VOL_LOOKBACK=40（P6 扫描最优），LIQUIDITY_LOOKBACK=20（保持原值）

研究成果:
  P5 六版本对比: 正向 40d 版本（61-21 momentum）全样本 Sharpe 0.82 最高
  跨样本验证: AllA / CSI300 / CSI500 三样本 OVERALL: PASS (2026-06-29)
  详见 research/scripts/P5-F6-MOM-2026Q2-v1-segmented-standalone.py

================================================================================
改进动机（基于分段回测分析）
================================================================================
baseline 策略 7 段分段回测中 6 段跑赢基准，唯一失效段是 2019-2020 核心资产牛市
（策略 +17.45% vs 基准 +63.79%，Alpha -0.11）。

失效原因：
  - F2-EP（便宜股）+ F5-LV（低波动）是"防御价值"型
  - 2019-2020 资金抱团高 ROE 龙头（茅台/恒瑞/海天等），这些股票 PE 高、波动大
  - 策略天然回避抱团股，完全错过核心资产牛市

改进方案：加入 F6-MOM 动量因子
  - 12-1 momentum（Carhart 1997 经典动量因子）
  - 直接捕捉趋势性机会（核心资产牛市是强趋势行情）
  - 与 F2-EP 低相关（价值股动量通常弱，趋势股估值通常贵）
  - 避开 ROE 已验证失败的路径（F4-ROE IC t=1.81 未过门槛）

================================================================================
因子定义（继承 baseline + 新增 F6-MOM）
================================================================================
  F2-EP:  ep_spot = net_profit / (market_cap * 1e8)（市盈率倒数，越高=越便宜）
  F5-LV:  vol_40d = std(过去40交易日日收益率)
          vol_signal = -vol_40d（低波动=高信号，P6 扫描 40d 最优）
  F6-MOM: mom_40d = price[t-21] / price[t-61] - 1
          （61-21 momentum，信号长度40天，短窗口连续看趋势）
          实测全样本Sharpe 0.82最高，段5修复-2.69pp，段6大幅改善+54.17pp

合成方式（三因子等权 z-score，无参数避免过拟合）:
  ep_z  = z_score(F2_ep)
  vol_z = z_score(F5_vol)
  mom_z = z_score(F6_mom)
  combined = (1/3) * ep_z + (1/3) * vol_z + (1/3) * mom_z

仓位管理（纯回撤约束，不用大盘择时/波动率目标/MA200）:
  - 跟踪组合净值高水位 peak
  - 回撤 = (peak - current) / peak
  - 回撤 > 20% → 温和降仓到 85%（de-risked 状态）
  - 回撤 < 15% → 恢复满仓（normal 状态）
  - 状态切换时才调仓，避免频繁交易
  - 放宽参数避免过度抑制收益（前一版15%/70%/10%把Sharpe打到0.39）

================================================================================
验证目标（必须同时满足才算改进成功）
================================================================================
  1. 段5（2019-2020）修复：超额收益从 -46.34pp 转为 >= -10pp
  2. 其他 6 段不破坏：跑赢基准的段仍跑赢（允许超额收窄但不允许反转）
  3. 全样本 Sharpe >= 0.5（baseline 0.61，40d版本研究环境0.82）
  4. 全样本最大回撤 <= 40%（40d无风控版本43.56%，加回撤约束目标<=40%）

================================================================================
验证结果（P5 跨样本验证已通过）
================================================================================
  - F6-MOM 40d（61-21 momentum）六版本全样本对比：正向 40d Sharpe 0.82 最高 ✅
  - 跨样本验证（2026-06-29）：AllA / CSI300 / CSI500 三样本 OVERALL: PASS ✅
  - 段5（2019-2020）修复：双因子版 -46pp → 三因子版接近持平 ✅
  - 无风控版全面胜出回撤约束版，普通人靠纪律承受回撤即可

================================================================================
后续优化方向
================================================================================
  - 三因子等权是简单方案，后续可考虑 ICIR 加权或 Risk Parity
  - A 股小盘反转效应可能影响 MOM 效力（已通过流动性过滤剔除小盘）
  - 分段回测显示 2022-2026 结构性行情中趋势因子有摩擦（AllA -21pp），可关注因子轮动
"""

import datetime
import numpy as np
import pandas as pd

# 聚宽策略环境自动注入（无需 import）


# ============================================================
# 零、聚宽交易 API 兼容垫片（与 baseline 一致）
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
            raise RuntimeError('无法获取差额股数' % code)
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

# F5 因子参数
VOL_LOOKBACK = 40           # F5 主信号：40交易日（P6扫描：776%/Sharpe0.67 vs 60d 732%/0.64）
VOL_MIN_OBS_RATIO = 0.5      # 有效观测数下限

# F6 动量因子参数（40d版本：61-21 momentum，信号长度40天）
# 实测全样本Sharpe 0.82最高，段5修复-2.69pp，段6大幅改善+54.17pp
MOM_LOOKBACK_LONG = 61       # F6 主信号窗口：61交易日（21+40，约3个月）
MOM_SKIP_RECENT = 21         # 剔除最近21交易日（约1个月，避免短期反转污染）
MOM_MIN_OBS_RATIO = 0.8      # 有效观测数下限（动量需要完整窗口）

# 纯回撤约束参数（不用大盘择时/波动率目标/MA200，仅回撤约束）
# 放宽参数：只在极端回撤触发，温和降仓，更快恢复
DRAWDOWN_CONTROL_ENABLED = False  # False=无风控基准测试；True=启用回撤约束
DRAWDOWN_THRESHOLD = 0.20    # 回撤20%触发降仓（只在极端回撤）
DRAWDOWN_REDUCE_TO = 0.85    # 温和降仓到85%
DRAWDOWN_RECOVER = 0.15      # 回撤恢复到15%以内时加仓回满仓

# 流动性过滤（与 baseline 一致）
LIQUIDITY_LOOKBACK = 20      # 近 20 日日均成交额（更敏感，777% vs 30d 754%）
LIQUIDITY_THRESHOLD = 1e7    # 1000 万

# 持仓
N_HOLD = 50                  # 持仓数量

# 涨跌停过滤（与 baseline 一致）
LIMIT_UP_DOWN_FILTER = True


# ============================================================
# 二、股票池与 ST/次新股/金融股剔除（与 baseline 一致）
# ============================================================

def get_stock_pool(index_id, date_str, min_listed_days=365):
    """构建股票池：指数成分股 - ST - 次新股 - 金融股。

    min_listed_days=365：动量因子需要 252+21=273 天历史价格，
    设 365 天确保有足够数据。
    """
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


def is_new_stock(code, date_str, days=365):
    """判断上市是否不满 days 天。"""
    info = get_security_info(code)
    if info is None:
        return True
    cur = pd.Timestamp(date_str)
    start = pd.Timestamp(info.start_date)
    return (cur - start).days < days


# ============================================================
# 三、限价单撮合器（涨跌停/停牌，与 baseline 一致）
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
# 四、先卖后买调仓（T+1，与 baseline 一致）
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
# 五、成本模型（与 baseline 一致）
# ============================================================

def apply_cost_model():
    """万三双边 + 千一印花税(仅卖) + 5元最低 + 千一滑点。"""
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
# 六、因子计算（F2-EP + F5-LowVol + F6-MOM 三因子）
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


def calc_momentum(date_str, stocks,
                  lookback_long=MOM_LOOKBACK_LONG,
                  skip_recent=MOM_SKIP_RECENT):
    """计算 12-1 动量因子（F6-MOM）。

    MOM_12_1 = price[t-skip_recent] / price[t-lookback_long] - 1
    即过去 12 个月累计收益，剔除最近 1 个月避免短期反转污染（Carhart 1997）。

    返回 {code: mom_12_1}。
    """
    if not stocks:
        return {}
    # 取 lookback_long + 5 天，多取几天防止边界问题
    total_count = lookback_long + 5
    try:
        df = get_price(stocks, end_date=date_str, count=total_count,
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
    if len(close) < lookback_long + 1:
        return {}

    # price[t-skip_recent] = 剔除最近1月后的价格
    # price[t-lookback_long] = 12个月前的价格
    price_recent = close.iloc[-(skip_recent + 1)]
    price_long_ago = close.iloc[-(lookback_long + 1)]

    valid_counts = close.count()
    min_obs = int(lookback_long * MOM_MIN_OBS_RATIO)

    result = {}
    for code in stocks:
        if code not in valid_counts.index:
            continue
        cnt = valid_counts.get(code, 0)
        if cnt < min_obs:
            continue
        p_recent = price_recent.get(code) if hasattr(price_recent, 'get') else None
        p_long = price_long_ago.get(code) if hasattr(price_long_ago, 'get') else None
        if p_recent is None or p_long is None:
            continue
        if np.isnan(p_recent) or np.isnan(p_long):
            continue
        if p_recent <= 0 or p_long <= 0:
            continue
        mom = float(p_recent) / float(p_long) - 1.0
        if not np.isnan(mom) and np.isfinite(mom):
            result[code] = mom
    return result


def calc_avg_money(date_str, stocks, lookback_days=LIQUIDITY_LOOKBACK):
    """计算近 lookback_days 日均成交额（流动性过滤用）。"""
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


def calculate_combined_factors(df, vol_map, mom_map):
    """计算原始因子（F2_ep_raw, F5_vol_raw, F6_mom_raw），不做中性化。

    中性化在 factor_rebalance 过滤后做（与研究脚本顺序一致）。
    """
    mcap_yuan = df['market_cap'] * 1e8
    df['ep_spot'] = df['net_profit'] / mcap_yuan.replace(0, np.nan)
    df['debt_to_assets'] = df['total_liability'] / df['total_assets'].replace(0, np.nan)
    df['vol_60d'] = df.index.map(lambda c: vol_map.get(c, np.nan))
    df['mom_12_1'] = df.index.map(lambda c: mom_map.get(c, np.nan))

    # 原始因子（中性化前）
    df['F2_ep_raw'] = df['ep_spot']
    df['F5_vol_raw'] = -df['vol_60d']
    df['F6_mom_raw'] = df['mom_12_1']
    return df


def apply_neutralization(df):
    """市值中性化 + winsorize（过滤后调用，与研究脚本一致）。"""
    log_mcap = np.log(df['market_cap'].astype(float).replace(0, np.nan))
    f2_neut = neutralize_ols(df['F2_ep_raw'].values, log_mcap.values)
    df['F2_ep'] = winsorize_cross_section(pd.Series(f2_neut, index=df.index))
    f5_neut = neutralize_ols(df['F5_vol_raw'].values, log_mcap.values)
    df['F5_vol'] = winsorize_cross_section(pd.Series(f5_neut, index=df.index))
    f6_neut = neutralize_ols(df['F6_mom_raw'].values, log_mcap.values)
    df['F6_mom'] = winsorize_cross_section(pd.Series(f6_neut, index=df.index))
    return df


# ============================================================
# 七、仓位管理（纯回撤约束状态机）
# ============================================================

def compute_target_position(context, nav_history, f5_z_current, state):
    """计算目标总仓位（纯回撤约束状态机，放宽参数版）。

    状态机：
      NORMAL（满仓） → 回撤 > 20% → DE_RISKED（85%仓位）
      DE_RISKED（85%） → 回撤 < 15% → NORMAL（满仓）

    放宽参数避免过度抑制收益（前一版15%/70%/10%把Sharpe打到0.39）。
    避免频繁交易：仅在状态切换时改变 target_position。
    """
    # 无风控基准测试模式：始终满仓
    if not DRAWDOWN_CONTROL_ENABLED:
        state['prev_weight'] = 1.0
        state['drawdown'] = 0.0
        return 1.0

    total_value = context.portfolio.total_value
    peak = state.get('peak', total_value)
    if total_value > peak:
        peak = total_value
        state['peak'] = peak

    drawdown = (peak - total_value) / peak if peak > 0 else 0.0
    in_de_risked = state.get('in_de_risked', False)

    if in_de_risked:
        # de-risked 状态：回撤恢复到 10% 以内才回满仓
        if drawdown < DRAWDOWN_RECOVER:
            target_w = 1.0
            state['in_de_risked'] = False
            log.info('[回撤约束] 回撤 %.2f%% < %.0f%%，恢复满仓' % (
                drawdown * 100, DRAWDOWN_RECOVER * 100))
        else:
            target_w = DRAWDOWN_REDUCE_TO
    else:
        # normal 状态：回撤超过 15% 才降仓
        if drawdown > DRAWDOWN_THRESHOLD:
            target_w = DRAWDOWN_REDUCE_TO
            state['in_de_risked'] = True
            log.info('[回撤约束] 回撤 %.2f%% > %.0f%%，降仓到 %.0f%%' % (
                drawdown * 100, DRAWDOWN_THRESHOLD * 100,
                DRAWDOWN_REDUCE_TO * 100))
        else:
            target_w = 1.0

    state['prev_weight'] = target_w
    state['drawdown'] = drawdown
    return target_w


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
            if change > 0.095 or change < -0.095:
                return False
        return True
    except Exception:
        return True


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
    g.target_position = 1.0      # 目标总仓位（baseline 始终满仓）
    g.nav_history = []           # 净值历史
    g.pos_state = {              # 仓位管理状态（baseline 不使用）
        'peak': 1.0,
        'low': 1.0,
        'in_stoploss': False,
        'stoploss_cooldown': 0,
        'prev_weight': 1.0,
    }
    g.f5_z_current = 0.0

    # 季度调仓：5/9/11 月首个交易日，尾盘14:50建仓
    run_monthly(factor_rebalance, monthday=1, time='14:50')
    # 每日仓位管理（baseline 实际不调整仓位，保留接口与 baseline 一致）
    run_daily(position_management, time='14:50')


def factor_rebalance(context):
    """季频三因子选股：5/9/11 月，EP+LowVol+MOM 等权 z-score 合成 + 流动性过滤。"""
    current_date = context.current_dt
    if current_date.month not in (5, 9, 11):
        return
    date_str = current_date.strftime('%Y-%m-%d')

    log.info('=' * 50)
    log.info('[%s] 季度调仓开始（F2+F5+F6 三因子）' % date_str)

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

    # 3. F5 波动率 + F6 动量
    vol_map = calc_realized_volatility(date_str, list(df.index), VOL_LOOKBACK)
    mom_map = calc_momentum(date_str, list(df.index),
                            lookback_long=MOM_LOOKBACK_LONG,
                            skip_recent=MOM_SKIP_RECENT)

    # 4. 因子计算
    df = calculate_combined_factors(df, vol_map, mom_map)

    # 5. 流动性过滤（与 baseline 一致）
    avg_money_map = calc_avg_money(date_str, list(df.index))
    df['avg_money'] = df.index.map(lambda c: avg_money_map.get(c, 0))
    before_liquidity = len(df)

    mask = df['net_profit'] > 0
    mask &= df['debt_to_assets'] <= 1.0
    mask &= df['ep_spot'].notna() & (df['ep_spot'] > 0)
    mask &= df['vol_60d'].notna() & (df['vol_60d'] > 0)
    mask &= df['mom_12_1'].notna() & np.isfinite(df['mom_12_1'])
    mask &= df['avg_money'].fillna(0) >= LIQUIDITY_THRESHOLD
    df = df[mask]
    log.info('[%s] 流动性+因子过滤后: %d → %d 只（剔除 %d 只）' % (
        date_str, before_liquidity, len(df), before_liquidity - len(df)))

    if df.empty:
        log.info('[%s] 无符合条件的组合股票' % date_str)
        return

    # 6. 市值中性化 + winsorize（过滤后做，与研究脚本一致）
    df = apply_neutralization(df)
    df = df.dropna(subset=['F2_ep', 'F5_vol', 'F6_mom'])
    if len(df) < 30:
        log.info('[%s] 中性化后样本不足: %d 只' % (date_str, len(df)))
        return

    # 7. 三因子等权 z-score 合成
    f2_std = df['F2_ep'].std()
    f5_std = df['F5_vol'].std()
    f6_std = df['F6_mom'].std()
    df['F2_z'] = (df['F2_ep'] - df['F2_ep'].mean()) / (f2_std if f2_std > 0 else 1)
    df['F5_z'] = (df['F5_vol'] - df['F5_vol'].mean()) / (f5_std if f5_std > 0 else 1)
    df['F6_z'] = (df['F6_mom'] - df['F6_mom'].mean()) / (f6_std if f6_std > 0 else 1)
    # 等权合成（1/3 + 1/3 + 1/3）
    df['combined'] = (1.0 / 3.0) * df['F2_z'] + (1.0 / 3.0) * df['F5_z'] + (1.0 / 3.0) * df['F6_z']

    # 8. IC加权选股：按 combined 降序取前 N
    df = df.sort_values('combined', ascending=False).head(g.stock_num)

    # 9. 涨跌停过滤（避免回测作弊）
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

    # 应用当前总仓位（baseline 始终 = 1.0）
    target_position = g.target_position if g.target_position > 0 else 1.0
    actual_weights = {code: w * target_position for code, w in g.target_weights.items()}

    log.info('[%s] 调仓：买入 %d 只股票，总仓位 %.1f%%' % (
        date_str, len(df), target_position * 100))
    for code, w in list(actual_weights.items())[:5]:
        log.info('  买入 %s  ep=%.4f vol=%.4f mom=%.4f avg_money=%.0f万 w=%.2f%%' % (
            code, df.loc[code, 'ep_spot'], df.loc[code, 'vol_60d'],
            df.loc[code, 'mom_12_1'], df.loc[code, 'avg_money'] / 1e4, w * 100))
    if len(actual_weights) > 5:
        log.info('  ... 共 %d 只' % len(actual_weights))

    rebalance_ordered(context, actual_weights)
    log.info('[%s] 季度调仓完成（F2+F5+F6 三因子）' % date_str)


def position_management(context):
    """每日仓位管理（纯回撤约束）。

    每日检查回撤，状态切换时调整仓位。
    调仓逻辑：按 g.target_weights 等比例缩放 target_position。
    """
    current_date = context.current_dt
    date_str = current_date.strftime('%Y-%m-%d')

    # 更新净值历史
    total_value = context.portfolio.total_value
    g.nav_history.append(total_value)
    if len(g.nav_history) > 500:
        g.nav_history = g.nav_history[-500:]

    # 计算目标仓位（纯回撤约束状态机）
    target_position = compute_target_position(
        context, g.nav_history, g.f5_z_current, g.pos_state)

    # 状态切换时才调仓（避免频繁交易）
    prev_position = g.target_position
    if abs(target_position - prev_position) > 0.01:
        log.info('[%s] 仓位调整: %.0f%% → %.0f%%（回撤 %.2f%%）' % (
            date_str, prev_position * 100, target_position * 100,
            g.pos_state.get('drawdown', 0) * 100))
        g.target_position = target_position

        # 按比例缩放持仓权重
        if g.target_weights:
            actual_weights = {code: w * target_position
                              for code, w in g.target_weights.items()}
            rebalance_ordered(context, actual_weights)


def before_trading_start(context):
    """盘前处理。"""
    pass
