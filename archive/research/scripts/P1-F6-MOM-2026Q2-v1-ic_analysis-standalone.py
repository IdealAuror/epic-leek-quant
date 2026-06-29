"""
P1-F6-MOM-2026Q2-v1 IC 分析脚本（Phase 1.5 standalone 版）
=================================================

Phase 1.5：F6 动量（Momentum）因子 IC 验证。

背景：
- Phase 4 V2 分段回测发现策略唯一失效段是 2019-2020 核心资产牛市
- 失效原因：F2-EP（便宜股）+ F5-LV（低波动）是"防御价值"型，无法捕捉
  资金抱团高估值龙头的趋势性机会
- F4-ROE 已验证失败（IC t=1.81 未过门槛，与 EP 高相关）
- 需要加入能捕捉趋势性机会的因子，且与 F2-EP 低相关

因子选择依据：
- 12-1 momentum（Carhart 1997 经典四因子动量因子）
- 学术依据：Jegadeesh & Titman 1993，过去 12 个月收益排序能预测未来收益
- A 股适配性：中大盘有动量效应（小盘反转效应，已通过流动性过滤剔除小盘）
- 与 F2-EP 低相关：价值股动量通常弱，趋势股估值通常贵 → 组合分散化

因子定义：
- 主信号 MOM_12_1 = price[t-21] / price[t-252] - 1
  （12 个月累计收益，剔除最近 1 个月避免短期反转污染）
- 对照信号 MOM_6_1 = price[t-21] / price[t-126] - 1
  （6 个月累计收益，剔除最近 1 个月，验证窗口敏感性）

为什么剔除最近 1 个月：
- A 股短期反转效应显著（1 个月内涨跌幅对未来收益有负向预测力）
- 不剔除会导致动量信号被反转效应污染，IC 失真
- Carhart (1997) 标准做法

Size 共线风险（决定性测试）：
- 高动量股通常是大盘股（资金抱团），MOM 与 ln(market_cap) 可能相关
- 市值中性化后 IC NW-t >= 2 是决定性门槛
- 保留金融股剔除（银行股动量信号失真）

验证门槛（与 F2/F4/F5 一致）：市值中性化后 IC NW-t >= 2

【符号约定】
- signal = MOM_12_1（越高=动量越强=好，IC>0 有效）
- Q1-Q5 按 signal 升序：Q5 = 最高动量 = 多头组
"""

import datetime
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.sandwich_covariance import cov_hac

# ============================================================
# 聚宽对象解析（兼容研究环境多种注入方式）
# ============================================================
from jqdata import *  # noqa: F403,F401

try:
    from jqdata import (  # noqa: F401
        valuation, query,
        get_fundamentals, get_price, get_all_securities,
        get_index_stocks, get_extras, get_security_info,
        get_trade_days, get_industry,
    )
except Exception:
    pass


warnings.filterwarnings('ignore')


# ============================================================
# 参数
# ============================================================
START_DATE = '2014-01-01'
END_DATE = '2026-06-30'
INDEX_IDS = {
    'AllA': None,
}

# QUICK_TEST=True 先验证数据链（3个调仓日），确认后改 False 全量
QUICK_TEST = False

# 动量窗口
MOM_LOOKBACK_LONG = 252      # 主信号长窗口：252 交易日（约 12 个月）
MOM_LOOKBACK_SHORT = 126     # 对照信号长窗口：126 交易日（约 6 个月）
MOM_LOOKBACK_40D = 61        # 40d版本窗口：61 交易日（21+40，约3个月，Phase 5最终方案）
MOM_SKIP_RECENT = 21         # 剔除最近 21 交易日（约 1 个月，避免短期反转污染）
MOM_MIN_OBS_RATIO = 0.8      # 有效观测数下限（动量需要完整窗口，提高阈值）

BREAKPOINT = datetime.date(2019, 6, 1)
OUT_DIR = 'results/P1-F6-MOM-2026Q2-v1'

# 金融股剔除（申万一级 sw_l1，严格相等匹配）
_EXCLUDE_FINANCE = True


# ============================================================
# 内联 data_layer 必要函数（standalone，无需 import data_layer）
# ============================================================

def _get_stock_pool(index_id, date_str, min_listed_days=365):
    """构建股票池：成分股（或全A） -> 剔ST -> 剔次新股 -> 剔金融股。

    min_listed_days=365：动量因子需要 252+21=273 天历史价格，
    设 365 天确保有足够数据。
    """
    if index_id is None:
        sec_df = get_all_securities(['stock'], date=date_str)
        stocks = list(sec_df.index)
    else:
        stocks = get_index_stocks(index_id, date=date_str)

    # ST 剔除
    if stocks:
        st_df = get_extras('is_st', stocks, end_date=date_str, count=1)
        if st_df is not None and not st_df.empty:
            st_today = st_df.iloc[-1]
            stocks = [s for s in stocks
                      if s in st_today.index and not st_today[s]]

    # 次新股剔除（动量需要足够历史价格，365天与 252+21 匹配）
    result = []
    cur = pd.Timestamp(date_str)
    for s in stocks:
        try:
            info = get_security_info(s)
        except Exception:
            continue
        if info is None:
            continue
        start = pd.Timestamp(info.start_date)
        if (cur - start).days >= min_listed_days:
            result.append(s)
    stocks = result

    # 金融股剔除（银行股动量信号失真，不剔会主导信号）
    if _EXCLUDE_FINANCE and stocks:
        before = len(stocks)
        stocks = _exclude_finance_stocks(stocks, date_str, debug=True)
        excluded = before - len(stocks)
        print('    [debug] %s: 金融股剔除 %d 只，剩余 %d 只' % (
            date_str, excluded, len(stocks)))
    return stocks


# 金融行业关键词（仅匹配 sw_l1 申万一级，严格相等避免类型陷阱）
_FINANCE_NAMES = {'银行I', '非银金融I'}


def _exclude_finance_stocks(stocks, date_str, debug=False):
    """从 stocks 中剔除金融行业股票。仅用 sw_l1（申万一级）严格相等匹配。"""
    if not stocks:
        return stocks
    try:
        ind_raw = get_industry(stocks, date=date_str)
    except Exception as e:
        if debug:
            print('    [debug] %s: get_industry 调用失败(%s)，金融股未剔除' % (
                date_str, e))
        return stocks
    if not ind_raw:
        if debug:
            print('    [debug] %s: get_industry 返回空，金融股未剔除' % date_str)
        return stocks

    finance_codes = set()
    for code, schemes in ind_raw.items():
        if not isinstance(schemes, dict):
            continue
        sw_l1 = schemes.get('sw_l1')
        if not isinstance(sw_l1, dict):
            continue
        name = str(sw_l1.get('industry_name', '') or '')
        if name in _FINANCE_NAMES:
            finance_codes.add(code)
    return [s for s in stocks if s not in finance_codes]


def _fetch_market_cap(date_str, stocks):
    """PIT 市值查询（动量因子只需 market_cap 做中性化）。

    valuation.market_cap 单位是【亿元】。
    返回以 code 为 index 的 DataFrame，含 market_cap 列。
    """
    try:
        q_val = query(valuation.code, valuation.market_cap)
        df_val = get_fundamentals(q_val, date=date_str)
        if df_val is None or df_val.empty:
            return None
        if 'code' in df_val.columns:
            df_val = df_val.set_index('code')
        if stocks:
            df_val = df_val[df_val.index.isin(list(stocks))]
        return df_val if not df_val.empty else None
    except Exception:
        return None


def _get_col(df, *candidates):
    """从 df 中按候选列名找第一个存在的列，返回 Series；找不到返回 None。"""
    for c in candidates:
        if c in df.columns:
            return df[c]
    return None


# ============================================================
# 内联 factor_lib（动量因子计算）
# ============================================================

def _calc_momentum(date_str, stocks, lookback_long=MOM_LOOKBACK_LONG,
                   lookback_short=MOM_LOOKBACK_SHORT,
                   lookback_40d=MOM_LOOKBACK_40D,
                   skip_recent=MOM_SKIP_RECENT):
    """计算动量因子：12-1 / 6-1 / 40d 三个版本。

    主信号 MOM_12_1  = price[t-skip_recent] / price[t-lookback_long]  - 1
    对照信号 MOM_6_1  = price[t-skip_recent] / price[t-lookback_short] - 1
    Phase5信号 MOM_40d = price[t-skip_recent] / price[t-lookback_40d]  - 1

    即过去累计收益，剔除最近 1 个月（21 交易日）。
    剔除最近 1 个月是为避免短期反转效应污染动量信号（Carhart 1997）。

    返回 (mom_long_map, mom_short_map, mom_40d_map)，{code: mom}。
    有效观测数 < lookback * MOM_MIN_OBS_RATIO 的股票不返回（新股/长期停牌）。
    """
    if not stocks:
        return {}, {}, {}
    # 取 lookback_long + 1 天（多取一天防止边界）
    total_count = lookback_long + 5
    try:
        df = get_price(stocks, end_date=date_str, count=total_count,
                       fields=['close'], skip_paused=False,
                       panel=False, fq='post')
        if df is None or df.empty:
            return {}, {}, {}
        # panel=False 长表：日期在 'time' 列
        if 'time' in df.columns:
            df = df.set_index('time')
        elif 'date' in df.columns:
            df = df.set_index('date')
        # pivot 成宽表
        if 'code' in df.columns:
            close = df.pivot_table(index=df.index, columns='code', values='close')
        else:
            close = df
        close.index = pd.to_datetime(close.index)
    except Exception:
        return {}, {}, {}

    if close is None or close.empty:
        return {}, {}, {}

    # 检查有效观测数
    valid_counts = close.count()
    min_obs_long = int(lookback_long * MOM_MIN_OBS_RATIO)
    min_obs_short = int(lookback_short * MOM_MIN_OBS_RATIO)
    min_obs_40d = int(lookback_40d * MOM_MIN_OBS_RATIO)

    mom_long_map = {}
    mom_short_map = {}
    mom_40d_map = {}

    if len(close) < lookback_long + 1:
        # 历史不足，返回空
        return {}, {}, {}

    # price[t-skip_recent] = skip_recent+1 天前的收盘价
    # price[t-lookback_long] = lookback_long+1 天前的收盘价
    # 注意 iloc[-1] 是最近一天，往前推
    price_recent = close.iloc[-(skip_recent + 1)]   # 剔除最近 1 月后的价格
    price_long_ago = close.iloc[-(lookback_long + 1)]  # 12 个月前的价格
    price_short_ago = close.iloc[-(lookback_short + 1)]  # 6 个月前的价格
    price_40d_ago = close.iloc[-(lookback_40d + 1)]  # 40d版本的价格

    for code in stocks:
        if code not in valid_counts.index:
            continue
        cnt = valid_counts.get(code, 0)
        # 价格必须非空且大于0
        p_recent = price_recent.get(code) if hasattr(price_recent, 'get') else None
        p_long = price_long_ago.get(code) if hasattr(price_long_ago, 'get') else None
        p_short = price_short_ago.get(code) if hasattr(price_short_ago, 'get') else None
        p_40d = price_40d_ago.get(code) if hasattr(price_40d_ago, 'get') else None

        if p_recent is None or p_long is None or p_short is None or p_40d is None:
            continue
        if (np.isnan(p_recent) or np.isnan(p_long) or np.isnan(p_short)
                or np.isnan(p_40d)):
            continue
        if p_recent <= 0 or p_long <= 0 or p_short <= 0 or p_40d <= 0:
            continue

        # 长窗口动量（需要足够观测数）
        if cnt >= min_obs_long:
            mom_long = float(p_recent) / float(p_long) - 1.0
            if not np.isnan(mom_long) and np.isfinite(mom_long):
                mom_long_map[code] = mom_long

        # 短窗口动量
        if cnt >= min_obs_short:
            mom_short = float(p_recent) / float(p_short) - 1.0
            if not np.isnan(mom_short) and np.isfinite(mom_short):
                mom_short_map[code] = mom_short

        # 40d版本动量（Phase 5最终方案）
        if cnt >= min_obs_40d:
            mom_40d = float(p_recent) / float(p_40d) - 1.0
            if not np.isnan(mom_40d) and np.isfinite(mom_40d):
                mom_40d_map[code] = mom_40d

    return mom_long_map, mom_short_map, mom_40d_map


def _calculate_all_factors(df, mom_long_map=None, mom_short_map=None,
                          mom_40d_map=None):
    """F6 MOM 因子计算。

    signal = MOM_12_1（12-1 momentum，越高=动量越强=IC>0有效）
    对照：signal_alt = MOM_6_1（6-1 momentum，验证窗口敏感性）
    Phase5：signal_40d = MOM_40d（40d momentum，Phase 5最终方案）
    """
    mcap = _get_col(df, 'market_cap')
    if mcap is None:
        return None
    df['market_cap'] = mcap.astype(float)

    # 主信号
    if mom_long_map:
        df['mom_12_1'] = df.index.map(lambda c: mom_long_map.get(c, np.nan))
    else:
        df['mom_12_1'] = np.nan

    # 对照信号
    if mom_short_map:
        df['mom_6_1'] = df.index.map(lambda c: mom_short_map.get(c, np.nan))
    else:
        df['mom_6_1'] = np.nan

    # Phase 5信号（40d版本）
    if mom_40d_map:
        df['mom_40d'] = df.index.map(lambda c: mom_40d_map.get(c, np.nan))
    else:
        df['mom_40d'] = np.nan

    # 信号 = MOM_12_1（动量越强，信号越大）
    df['signal'] = df['mom_12_1']
    df['signal_alt'] = df['mom_6_1']
    df['signal_40d'] = df['mom_40d']

    return df


# ============================================================
# 工具函数（复用 F2-EP/F5-LowVol）
# ============================================================

def get_rebalance_dates(start, end):
    """季度调仓日：每年 5/9/11 月首个交易日（对齐策略 run_monthly）。"""
    months = [5, 9, 11]
    dates = []
    year = int(start[:4])
    end_year = int(end[:4])
    while year <= end_year:
        for m in months:
            dt_str = '%s-%02d-01' % (year, m)
            if start <= dt_str <= end:
                tds = get_trade_days(start_date=dt_str, count=1)
                if len(tds) > 0:
                    td = pd.Timestamp(tds[0])
                    if start <= td.strftime('%Y-%m-%d') <= end:
                        dates.append(td.date())
        year += 1
    return sorted(set(dates))


def newey_west_t(series, lags=4):
    """Newey-West 调整 t 统计量（IC 均值 / NW 标准误）。"""
    y = np.asarray(series, dtype=float)
    y = y[~np.isnan(y)]
    if len(y) < 2:
        return 0.0, 0.0, 0.0
    beta = np.mean(y)
    resid = y - beta
    x = np.ones((len(y), 1))
    try:
        nw_cov = cov_hac(np.column_stack([resid, x - x.mean()]),
                         nlags=lags, use_correction=True)
        se = float(np.sqrt(nw_cov[0, 0] / len(y)))
    except Exception:
        se = float(np.std(y, ddof=1) / np.sqrt(len(y)))
    t_stat = beta / se if se > 0 else 0.0
    return float(beta), se, float(t_stat)


def get_industry_map(stocks, date_str):
    """获取 {code: 行业名}。用 jq_l1，缺失回退到 zjw/sw_l1。"""
    if not stocks:
        return {}
    try:
        ind = get_industry(stocks, date=date_str)
    except Exception:
        return {}
    if not ind:
        return {}
    out = {}
    for code, schemes in ind.items():
        name = None
        if isinstance(schemes, dict):
            for key in ('jq_l1', 'zjw', 'sw_l1'):
                if key in schemes and isinstance(schemes[key], dict):
                    name = schemes[key].get('industry_name')
                    if name:
                        break
        out[code] = name if name else '未知'
    return out


def neutralize_ols(factor_values, regressor):
    """OLS 中性化：对 regressor 回归 factor_values，返回残差。"""
    f = np.asarray(factor_values, dtype=float)
    r = (regressor.values if hasattr(regressor, 'values')
         else np.asarray(regressor, dtype=float))
    if r.ndim == 1:
        r = r.reshape(-1, 1)
    # 同时排除 f 和 r 中的 NaN
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


# ============================================================
# 批量价格缓存（大幅加速 forward_month_return）
# ============================================================
_PRICE_CACHE = {}  # {(year_month, frozenset_codes_hash): DataFrame}


def _load_month_prices(year_month, codes):
    """加载某年某月指定 codes 的收盘价，缓存。"""
    cache_key = (year_month, hash(frozenset(codes)))
    if cache_key in _PRICE_CACHE:
        return _PRICE_CACHE[cache_key]

    y, m = int(year_month[:4]), int(year_month[5:7])
    start = datetime.date(y, m, 1)
    end = (start.replace(day=28) + datetime.timedelta(days=7))
    codes_list = list(codes)
    close = pd.DataFrame()
    try:
        df = get_price(codes_list,
                       start_date=start.strftime('%Y-%m-%d'),
                       end_date=end.strftime('%Y-%m-%d'),
                       fields=['close'], skip_paused=False,
                       panel=False, fq='post')
        if df is not None and not df.empty:
            if 'time' in df.columns:
                df = df.set_index('time')
            elif 'date' in df.columns:
                df = df.set_index('date')
            if 'code' in df.columns:
                close = df.pivot_table(index=df.index,
                                       columns='code', values='close')
            else:
                close = df
            close.index = pd.to_datetime(close.index)
    except Exception:
        close = pd.DataFrame()

    _PRICE_CACHE[cache_key] = close
    return close


def forward_month_return(codes, date_str, debug=False):
    """计算 codes 在 date_str 之后约一个月的累计收益（批量缓存加速）。"""
    d = pd.Timestamp(date_str)
    ym0 = '%04d-%02d' % (d.year, d.month)
    if d.month == 12:
        ym1 = '%04d-01' % (d.year + 1)
    else:
        ym1 = '%04d-%02d' % (d.year, d.month + 1)

    px0 = _load_month_prices(ym0, codes)
    px1 = _load_month_prices(ym1, codes)
    if debug:
        print('    [debug] %s: px0=%s, px1=%s' % (
            date_str,
            px0.shape if px0 is not None else None,
            px1.shape if px1 is not None else None))

    if px0 is None or px0.empty or px1 is None or px1.empty:
        return None

    try:
        px0.index = pd.to_datetime(px0.index)
        px1.index = pd.to_datetime(px1.index)
    except Exception:
        return None

    d_ts = pd.Timestamp(date_str)
    valid0 = px0.index[px0.index >= d_ts]
    if len(valid0) == 0:
        return None
    close_start = px0.loc[valid0[0]]
    close_end = px1.iloc[-1]

    codes_set = set(codes)
    cs = close_start[close_start.index.isin(codes_set)]
    ce = close_end[close_end.index.isin(codes_set)]
    if cs.empty or ce.empty:
        return None

    common = cs.index.intersection(ce.index)
    cs = cs.loc[common]
    ce = ce.loc[common]
    valid = cs > 0
    cs = cs[valid]
    ce = ce[valid]
    if cs.empty:
        return None
    ret = ce / cs - 1
    return ret


# ============================================================
# 单截面处理
# ============================================================

def build_cross_section(date, index_id, debug=False):
    """构建单个调仓日的横截面 DataFrame。None 表示数据不足。"""
    date_str = date.strftime('%Y-%m-%d')
    stocks = _get_stock_pool(index_id, date_str)
    if not stocks:
        if debug:
            print('    [debug] %s: 股票池为空' % date_str)
        return None
    if debug:
        print('    [debug] %s: 股票池 %d 只' % (date_str, len(stocks)))

    # 市值（中性化用）
    df = _fetch_market_cap(date_str, stocks)
    if df is None or df.empty:
        if debug:
            print('    [debug] %s: market_cap 查询为空' % date_str)
        return None
    if debug:
        print('    [debug] %s: market_cap %d 只' % (date_str, len(df)))

    # 动量因子
    mom_long_map, mom_short_map, mom_40d_map = _calc_momentum(
        date_str, list(df.index),
        lookback_long=MOM_LOOKBACK_LONG,
        lookback_short=MOM_LOOKBACK_SHORT,
        lookback_40d=MOM_LOOKBACK_40D,
        skip_recent=MOM_SKIP_RECENT,
    )
    if debug:
        print('    [debug] %s: MOM_12_1 取到 %d/%d 只, MOM_6_1 取到 %d/%d 只, MOM_40d 取到 %d/%d 只' % (
            date_str,
            len(mom_long_map), len(df.index),
            len(mom_short_map), len(df.index),
            len(mom_40d_map), len(df.index)))

    df = _calculate_all_factors(df, mom_long_map=mom_long_map,
                                mom_short_map=mom_short_map,
                                mom_40d_map=mom_40d_map)
    if df is None or df.empty:
        if debug:
            print('    [debug] %s: _calculate_all_factors 返回空' % date_str)
        return None
    if debug:
        ml = df['mom_12_1'].notna().sum() if 'mom_12_1' in df.columns else 0
        ms = df['mom_6_1'].notna().sum() if 'mom_6_1' in df.columns else 0
        m4 = df['mom_40d'].notna().sum() if 'mom_40d' in df.columns else 0
        print('    [debug] %s: 因子计算后 %d 只, mom_12_1非空=%d, mom_6_1非空=%d, mom_40d非空=%d' % (
            date_str, len(df), ml, ms, m4))

    # 基础过滤：必须有动量信号
    if 'signal' not in df.columns or df['signal'].isna().all():
        if debug:
            print('    [debug] %s: signal 列不存在或全空' % date_str)
        return None
    mask = df['signal'].notna()
    df = df[mask].copy()
    if df.empty:
        if debug:
            print('    [debug] %s: 过滤后为空' % date_str)
        return None
    if debug:
        print('    [debug] %s: 过滤后 %d 只' % (date_str, len(df)))

    df['signal_raw'] = df['signal']

    # 行业
    ind_map = get_industry_map(list(df.index), date_str)
    df['industry'] = df.index.map(lambda c: ind_map.get(c, '未知'))

    # 前向收益
    codes = list(df.index)
    fwd = forward_month_return(codes, date_str, debug=debug)
    if fwd is None:
        if debug:
            print('    [debug] %s: forward_month_return 返回 None' % date_str)
        return None
    df['fwd_return'] = df.index.map(fwd)
    drop_cols = ['fwd_return', 'signal']
    df = df.dropna(subset=drop_cols)
    if len(df) < 10:
        if debug:
            print('    [debug] %s: dropna 后不足 10 只 (%d)' % (date_str, len(df)))
        return None
    if debug:
        print('    [debug] %s: 最终横截面 %d 只' % (date_str, len(df)))
    return df


# ============================================================
# 主分析
# ============================================================

def analyze_sample(sample_name, index_id):
    """对单样本计算完整 IC 表。"""
    tag = '%s-V1' % sample_name
    print('\n%s\n===== %s =====\n%s' % ('=' * 60, tag, '=' * 60))

    rebal_dates = get_rebalance_dates(START_DATE, END_DATE)
    if QUICK_TEST:  # 快速模式：只取前 3 个调仓日
        rebal_dates = rebal_dates[:3]
        print('  [快速模式] 只跑前 3 个调仓日: %s' % rebal_dates)
    records = []
    ic_list, ic_alt_list, ic_40d_list, dates_list = [], [], [], []
    q_rets = {q: [] for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']}

    for i, date in enumerate(rebal_dates):
        df = build_cross_section(date, index_id, debug=(i == 0))
        if df is None:
            continue
        if i == 0:  # 首日调试
            print('  [调试] %s: 横截面 %d 只股票 (主信号: MOM_12_1)' % (date, len(df)))
            if 'mom_12_1' in df.columns:
                rv = df['mom_12_1'].dropna()
                if len(rv) > 0:
                    print('         mom_12_1: %.4f ~ %.4f (中位 %.4f, 非空 %d/%d)' % (
                        rv.min(), rv.max(), rv.median(),
                        len(rv), len(df)))
            if 'mom_6_1' in df.columns:
                rs = df['mom_6_1'].dropna()
                if len(rs) > 0:
                    print('         mom_6_1:  %.4f ~ %.4f (中位 %.4f)' % (
                        rs.min(), rs.max(), rs.median()))
            if 'mom_40d' in df.columns:
                r4 = df['mom_40d'].dropna()
                if len(r4) > 0:
                    print('         mom_40d:   %.4f ~ %.4f (中位 %.4f, 非空 %d/%d)' % (
                        r4.min(), r4.max(), r4.median(),
                        len(r4), len(df)))
            print('         fwd_return 非空: %d' % df['fwd_return'].notna().sum())

        ic = stats.spearmanr(df['signal'], df['fwd_return'])[0]
        if np.isnan(ic):
            continue
        ic_list.append(ic)
        # 对照：6-1 momentum 的 IC
        if 'signal_alt' in df.columns:
            ic_alt = stats.spearmanr(df['signal_alt'], df['fwd_return'])[0]
            ic_alt_list.append(0.0 if np.isnan(ic_alt) else ic_alt)
        else:
            ic_alt_list.append(0.0)
        # Phase 5：40d momentum 的 IC
        if 'signal_40d' in df.columns:
            ic_40d = stats.spearmanr(df['signal_40d'], df['fwd_return'])[0]
            ic_40d_list.append(0.0 if np.isnan(ic_40d) else ic_40d)
        else:
            ic_40d_list.append(0.0)
        dates_list.append(date)

        # Q1-Q5：按 MOM_12_1 升序（Q5=最高动量=多头组）
        q_col = 'signal'
        try:
            df['group'] = pd.qcut(df[q_col].rank(method='first'), 5,
                                  labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        except Exception:
            df['group'] = pd.qcut(df[q_col], 5,
                                  labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                                  duplicates='drop')
        for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
            grp = df[df['group'] == q]
            q_rets[q].append(float(grp['fwd_return'].mean())
                             if len(grp) > 0 else np.nan)

        # 中性化 IC
        log_mcap = pd.Series(np.log(df['market_cap'].values), index=df.index)
        sig_neu_mcap = neutralize_ols(df['signal'].values, log_mcap.values)
        ic_mcap = stats.spearmanr(sig_neu_mcap, df['fwd_return'].values)[0]
        ind_dummies = pd.get_dummies(df['industry'], drop_first=True)
        sig_neu_ind = neutralize_ols(df['signal'].values, ind_dummies.values)
        ic_ind = stats.spearmanr(sig_neu_ind, df['fwd_return'].values)[0]

        # 6-1 动量中性化对照
        if 'signal_alt' in df.columns:
            alt_neu_mcap = neutralize_ols(df['signal_alt'].values, log_mcap.values)
            ic_mcap_alt = stats.spearmanr(alt_neu_mcap, df['fwd_return'].values)[0]
            alt_neu_ind = neutralize_ols(df['signal_alt'].values, ind_dummies.values)
            ic_ind_alt = stats.spearmanr(alt_neu_ind, df['fwd_return'].values)[0]
        else:
            ic_mcap_alt = np.nan
            ic_ind_alt = np.nan

        # 40d 动量中性化（Phase 5最终方案）
        if 'signal_40d' in df.columns:
            m40d_neu_mcap = neutralize_ols(df['signal_40d'].values, log_mcap.values)
            ic_mcap_40d = stats.spearmanr(m40d_neu_mcap, df['fwd_return'].values)[0]
            m40d_neu_ind = neutralize_ols(df['signal_40d'].values, ind_dummies.values)
            ic_ind_40d = stats.spearmanr(m40d_neu_ind, df['fwd_return'].values)[0]
        else:
            ic_mcap_40d = np.nan
            ic_ind_40d = np.nan

        # 分市值档 IC
        try:
            df['cap_tier'] = pd.qcut(df['market_cap'].rank(method='first'),
                                     3, labels=['小', '中', '大'])
        except Exception:
            df['cap_tier'] = '中'
        ic_by_tier = {}
        for tier in ['小', '中', '大']:
            sub = df[df['cap_tier'] == tier]
            if len(sub) >= 5:
                ic_by_tier[tier] = stats.spearmanr(
                    sub['signal'], sub['fwd_return'])[0]
            else:
                ic_by_tier[tier] = np.nan

        records.append({
            'date': date,
            'n': len(df),
            'rank_ic': ic,
            'rank_ic_alt': ic_alt_list[-1],
            'rank_ic_40d': ic_40d_list[-1],
            'ic_mcap_neutral': (np.nan if np.isnan(ic_mcap) else ic_mcap),
            'ic_ind_neutral': (np.nan if np.isnan(ic_ind) else ic_ind),
            'ic_mcap_alt': (np.nan if np.isnan(ic_mcap_alt) else ic_mcap_alt),
            'ic_ind_alt': (np.nan if np.isnan(ic_ind_alt) else ic_ind_alt),
            'ic_mcap_40d': (np.nan if np.isnan(ic_mcap_40d) else ic_mcap_40d),
            'ic_ind_40d': (np.nan if np.isnan(ic_ind_40d) else ic_ind_40d),
            'ic_small_cap': ic_by_tier['小'],
            'ic_mid_cap': ic_by_tier['中'],
            'ic_large_cap': ic_by_tier['大'],
            'median_mcap': float(df['market_cap'].median()),
        })

        if (i + 1) % 6 == 0:
            print('  [%s] 进度 %d/%d 日处理完成' % (tag, i + 1, len(rebal_dates)))

    if not ic_list:
        print('[%s] 无有效数据' % tag)
        return None

    rec_df = pd.DataFrame(records)
    ic_series = pd.Series(ic_list, index=dates_list)

    # 表1
    ic_mean, _, ic_t = newey_west_t(ic_series, lags=4)
    ic_std = float(ic_series.std(ddof=1))
    ic_ir = ic_mean / ic_std if ic_std > 0 else 0.0
    win_rate = float((ic_series > 0).mean())
    pos_pct = float((ic_series > 0).sum() / len(ic_series))
    neg_pct = 1 - pos_pct

    print('\n【表1】月度 Rank IC 描述统计')
    print('  主信号MOM_12_1: mean=%.6f std=%.6f NW-t=%.4f IC_IR=%.4f 胜率=%.2f%% 正/负=%.2f%%/%.2f%%' % (
        ic_mean, ic_std, ic_t, ic_ir, win_rate * 100, pos_pct * 100, neg_pct * 100))
    # 对照 6-1 动量
    alt_mean, alt_std, alt_t = newey_west_t(pd.Series(ic_alt_list), lags=4)
    alt_ir = alt_mean / alt_std if alt_std > 0 else 0
    print('  对照MOM_6_1:    mean=%.6f std=%.6f NW-t=%.4f IC_IR=%.4f' % (
        alt_mean, alt_std, alt_t, alt_ir))
    # Phase 5: 40d 动量
    m40d_mean, m40d_std, m40d_t = newey_west_t(pd.Series(ic_40d_list), lags=4)
    m40d_ir = m40d_mean / m40d_std if m40d_std > 0 else 0
    m40d_win = float((pd.Series(ic_40d_list) > 0).mean())
    print('  Phase5 MOM_40d:  mean=%.6f std=%.6f NW-t=%.4f IC_IR=%.4f 胜率=%.2f%%' % (
        m40d_mean, m40d_std, m40d_t, m40d_ir, m40d_win * 100))

    # 表2
    print('\n【表2】Q1-Q5 分组（Q5=最高MOM_12_1=多头组）')
    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        r = q_rets[q]
        mean_m = float(np.nanmean(r))
        std_m = float(np.nanstd(r))
        ann_ret = (1 + mean_m) ** 12 - 1
        ann_vol = std_m * np.sqrt(12)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
        print('  %s: 月均=%.6f 年化=%.4f 年化波动=%.4f 夏普=%.4f' % (
            q, mean_m, ann_ret, ann_vol, sharpe))

    # 表3
    print('\n【表3】分市值档 Rank IC（横截面三分位：小/中/大）')
    for tier, col in [('小', 'ic_small_cap'), ('中', 'ic_mid_cap'),
                      ('大', 'ic_large_cap')]:
        s = rec_df[col].dropna()
        if len(s) > 1:
            m, _, t = newey_west_t(s, lags=4)
            print('  %s盘: mean=%.6f NW-t=%.4f N=%d' % (tier, m, t, len(s)))
        else:
            print('  %s盘: 样本不足' % tier)

    # 表4
    print('\n【表4】分行业 Rank IC（行业中性化后残差 IC）')
    s = rec_df['ic_ind_neutral'].dropna()
    if len(s) > 1:
        m, _, t = newey_west_t(s, lags=4)
        print('  行业中性化 IC: mean=%.6f NW-t=%.4f IC_IR=%.4f N=%d' % (
            m, t, m / float(s.std(ddof=1)), len(s)))
    else:
        print('  样本不足')
    # 6-1 动量行业中性化对照
    s_alt = rec_df['ic_ind_alt'].dropna()
    if len(s_alt) > 1:
        m_alt, _, t_alt = newey_west_t(s_alt, lags=4)
        print('  MOM_6_1 行业中性化: mean=%.6f NW-t=%.4f N=%d' % (
            m_alt, t_alt, len(s_alt)))
    # 40d 动量行业中性化
    s_40d = rec_df['ic_ind_40d'].dropna()
    if len(s_40d) > 1:
        m_40d, _, t_40d = newey_west_t(s_40d, lags=4)
        print('  MOM_40d 行业中性化: mean=%.6f NW-t=%.4f N=%d' % (
            m_40d, t_40d, len(s_40d)))

    # 表5
    print('\n【表5】市值中性化后 Rank IC（决定性门槛 NW-t>=2）')
    s = rec_df['ic_mcap_neutral'].dropna()
    if len(s) > 1:
        m, _, t = newey_west_t(s, lags=4)
        print('  MOM_12_1 市值中性化: mean=%.6f NW-t=%.4f IC_IR=%.4f N=%d' % (
            m, t, m / float(s.std(ddof=1)), len(s)))
    else:
        print('  样本不足')
    # 6-1 动量市值中性化对照
    s2 = rec_df['ic_mcap_alt'].dropna()
    if len(s2) > 1:
        m2, _, t2 = newey_west_t(s2, lags=4)
        print('  MOM_6_1  市值中性化: mean=%.6f NW-t=%.4f IC_IR=%.4f N=%d' % (
            m2, t2, m2 / float(s2.std(ddof=1)), len(s2)))
    # 40d 动量市值中性化（Phase 5最终方案）
    s3 = rec_df['ic_mcap_40d'].dropna()
    if len(s3) > 1:
        m3, _, t3 = newey_west_t(s3, lags=4)
        ir3 = m3 / float(s3.std(ddof=1)) if float(s3.std(ddof=1)) > 0 else 0
        print('  MOM_40d  市值中性化: mean=%.6f NW-t=%.4f IC_IR=%.4f N=%d  [Phase5最终方案]' % (
            m3, t3, ir3, len(s3)))

    # 表6
    print('\n【表6】2019.06 断点前后分段 IC')
    ic_series.index = pd.to_datetime(ic_series.index)
    bp_ts = pd.Timestamp(BREAKPOINT)
    pre_ic = ic_series[ic_series.index < bp_ts]
    post_ic = ic_series[ic_series.index >= bp_ts]
    pre_mean = post_mean = np.nan
    if len(pre_ic) > 1:
        pm, _, pt = newey_west_t(pre_ic, lags=4)
        pre_mean = pm
        print('  前段(%s~%s): mean=%.6f NW-t=%.4f N=%d' % (
            pre_ic.index[0].date(), pre_ic.index[-1].date(),
            pm, pt, len(pre_ic)))
    if len(post_ic) > 1:
        qm, _, qt = newey_west_t(post_ic, lags=4)
        post_mean = qm
        print('  后段(%s~%s): mean=%.6f NW-t=%.4f N=%d' % (
            post_ic.index[0].date(), post_ic.index[-1].date(),
            qm, qt, len(post_ic)))
    # 40d 动量断点前后
    ic_40d_series = pd.Series(ic_40d_list, index=dates_list)
    ic_40d_series.index = pd.to_datetime(ic_40d_series.index)
    pre_40d = ic_40d_series[ic_40d_series.index < bp_ts]
    post_40d = ic_40d_series[ic_40d_series.index >= bp_ts]
    if len(pre_40d) > 1:
        pm, _, pt = newey_west_t(pre_40d, lags=4)
        print('  MOM_40d前段(%s~%s): mean=%.6f NW-t=%.4f N=%d' % (
            pre_40d.index[0].date(), pre_40d.index[-1].date(),
            pm, pt, len(pre_40d)))
    if len(post_40d) > 1:
        qm, _, qt = newey_west_t(post_40d, lags=4)
        print('  MOM_40d后段(%s~%s): mean=%.6f NW-t=%.4f N=%d' % (
            post_40d.index[0].date(), post_40d.index[-1].date(),
            qm, qt, len(post_40d)))

    # CSV
    os.makedirs(OUT_DIR, exist_ok=True)
    out_ic = pd.DataFrame({
        'date': dates_list,
        'rank_ic_12_1': ic_list,
        'rank_ic_6_1': ic_alt_list,
        'rank_ic_40d': ic_40d_list,
        'ic_mcap_neutral_12_1': rec_df['ic_mcap_neutral'].values,
        'ic_ind_neutral_12_1': rec_df['ic_ind_neutral'].values,
        'ic_mcap_neutral_6_1': rec_df['ic_mcap_alt'].values,
        'ic_ind_neutral_6_1': rec_df['ic_ind_alt'].values,
        'ic_mcap_neutral_40d': rec_df['ic_mcap_40d'].values,
        'ic_ind_neutral_40d': rec_df['ic_ind_40d'].values,
        'n': rec_df['n'].values,
    })
    out_ic.to_csv('%s/%s_V1_ic_monthly.csv' % (OUT_DIR, sample_name),
                  index=False)
    out_q = pd.DataFrame({'date': dates_list})
    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        out_q[q] = q_rets[q]
    out_q.to_csv('%s/%s_V1_quantile.csv' % (OUT_DIR, sample_name),
                 index=False)
    print('\n  CSV 已保存到 %s/%s_V1_*.csv' % (OUT_DIR, sample_name))

    return {
        'tag': tag,
        'ic_ir': ic_ir,
        'ic_mean': ic_mean,
        'ic_t': ic_t,
        'q1_monthly_mean': float(np.nanmean(q_rets['Q1'])),
        'q5_monthly_mean': float(np.nanmean(q_rets['Q5'])),
        'pre_ic_mean': pre_mean,
        'post_ic_mean': post_mean,
    }


# ============================================================
# 运行入口
# ============================================================

def run():
    """单样本（AllA）全量跑 IC 分析。"""
    if QUICK_TEST:
        print('>>> 快速验证模式：只跑 2014 年 3 个调仓日')
    samples = {'CSI300': '000300.XSHG'} if QUICK_TEST else INDEX_IDS
    summary = []
    for name, idx in samples.items():
        r = analyze_sample(name, idx)
        if r:
            summary.append(r)

    print('\n\n' + '=' * 60)
    print('===== 全样本汇总 =====')
    print('=' * 60)
    print('%-16s%10s%12s%10s%10s%10s%10s%10s' % (
        'tag', 'IC_IR', 'IC_mean', 'NW-t', 'Q1月均', 'Q5月均',
        '前段IC', '后段IC'))
    for r in summary:
        print('%-16s%10.4f%12.6f%10.4f%10.6f%10.6f%10.6f%10.6f' % (
            r['tag'], r['ic_ir'], r['ic_mean'], r['ic_t'],
            r['q1_monthly_mean'], r['q5_monthly_mean'],
            r['pre_ic_mean'], r['post_ic_mean']))

    if summary:
        pd.DataFrame(summary).to_csv('%s/_summary.csv' % OUT_DIR, index=False)
        print('\n汇总已保存到 %s/_summary.csv' % OUT_DIR)


if __name__ == '__main__':
    run()
