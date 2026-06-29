"""
P1-F5-LowVol-2026Q2-v1 IC 分析脚本（Phase 1.4 standalone 版）
=================================================

Phase 1.4：F5 低波动率（LowVol）因子 IC 验证。

背景：
- F2-EP 单因子回测年化13.1%，但回撤51.6%（单因子固有特征）
- F4-ROE 因子 IC 市值中性化 t=1.81 未过门槛，回测负alpha，方向放弃
- 降回撤正解是多因子组合（非择时，非叠加质量因子）
- 低波动率因子（LowVol）与 EP 低相关：EP选便宜股，LowVol选稳健股
- A股低波动溢价比美股更显著，下行保护属性强

因子定义：
- 主信号 vol_60d = -std(过去60交易日日收益率)，取负使低波动=高信号=IC>0有效
- 对照信号 vol_120d = -std(过去120交易日日收益率)，验证窗口敏感性

Size 共线风险（决定性测试）：
- 小盘股波动大，LowVol信号与ln(market_cap)强相关
- 市值中性化后IC NW-t>=2是决定性门槛，未过则是size proxy
- 银行股天然低波动：保留金融股剔除，避免信号被银行主导

数据链简化（相比F2-EP/F4-ROE）：
- 仅需 valuation 表（market_cap列，中性化用），无需balance/income/cash_flow
- 无财务字段名探测陷阱，无跨期财报查询
- 波动率直接从get_price取历史收盘价计算

验证门槛（不变）：市值中性化后 IC NW-t >= 2

【符号约定】
- signal = -vol_60d（越高=越低波动=好，IC>0有效）
- Q1-Q5按signal升序：Q5=最低波动=多头组
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

# QUICK_TEST=True 先验证数据链（3个调仓日），确认后改False全量
QUICK_TEST = False

# 波动率窗口
VOL_LOOKBACK = 60       # 主信号：60交易日（约3个月）
VOL_LOOKBACK_ALT = 120  # 对照信号：120交易日（约6个月）
VOL_MIN_OBS_RATIO = 0.5  # 有效观测数下限 = lookback * 0.5

BREAKPOINT = datetime.date(2019, 6, 1)
OUT_DIR = 'results/P1-F5-LowVol-2026Q2-v1'

# 金融股剔除（申万一级 sw_l1，严格相等匹配）
_EXCLUDE_FINANCE = True


# ============================================================
# 内联 data_layer 必要函数（standalone，无需 import data_layer）
# ============================================================

def _get_stock_pool(index_id, date_str, min_listed_days=180):
    """构建股票池：成分股（或全A） -> 剔ST -> 剔次新股 -> 剔金融股。"""
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

    # 次新股剔除（波动率需要足够历史价格，180天与lookback_120d匹配）
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

    # 金融股剔除（银行天然低波动，不剔会主导信号）
    if _EXCLUDE_FINANCE and stocks:
        before = len(stocks)
        stocks = _exclude_finance_stocks(stocks, date_str, debug=True)
        excluded = before - len(stocks)
        print('    [debug] %s: 金融股剔除 %d 只，剩余 %d 只' % (
            date_str, excluded, len(stocks)))
    return stocks


# 金融行业关键词（仅匹配 sw_l1 申万一级，严格相等避免类型陷阱）
# 探测确认：sw_l1 金融股名为 "银行I" / "非银金融I"
# 注意：不可用 `in`（包含匹配），实测会误伤全样本；改用严格相等
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
    """PIT 市值查询（简化版，低波动率因子只需 market_cap）。

    valuation.market_cap 单位是【亿元】（诊断确认）。
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
# 内联 factor_lib（低波动率因子计算）
# ============================================================

def _calc_realized_volatility(date_str, stocks, lookback_days=60):
    """计算实现波动率：过去 lookback_days 交易日日收益率标准差。

    使用 get_price(count=lookback_days+1) 取 trailing 日收盘价，
    pct_change 算日收益率后取 std。count+1 因 pct_change 丢首行。

    返回 {code: vol}，vol = std(daily returns)。
    有效观测 < lookback_days * VOL_MIN_OBS_RATIO 的股票不返回（新股/长期停牌）。
    """
    if not stocks:
        return {}
    try:
        # count+1 因 pct_change 丢首行
        df = get_price(stocks, end_date=date_str, count=lookback_days + 1,
                       fields=['close'], skip_paused=False,
                       panel=False, fq='post')
        if df is None or df.empty:
            return {}
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
        return {}

    if close is None or close.empty:
        return {}

    # 日收益率
    rets = close.pct_change()
    # 每只股票的 std（跳过NaN）
    vol = rets.std(skipna=True)
    # 有效观测数
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


def _calculate_all_factors(df, vol_map=None, vol_alt_map=None):
    """F5 LowVol 因子计算。

    signal = -vol_60d（低波动 = 高信号 = IC>0有效）
    对照：signal_alt = -vol_120d（验证窗口敏感性）
    """
    mcap = _get_col(df, 'market_cap')
    if mcap is None:
        return None
    df['market_cap'] = mcap.astype(float)

    # 主信号
    if vol_map:
        df['vol_60d'] = df.index.map(lambda c: vol_map.get(c, np.nan))
    else:
        df['vol_60d'] = np.nan

    # 对照信号
    if vol_alt_map:
        df['vol_120d'] = df.index.map(lambda c: vol_alt_map.get(c, np.nan))
    else:
        df['vol_120d'] = np.nan

    # 信号 = -vol（取负，低波动=高信号）
    df['signal'] = -df['vol_60d']
    df['signal_alt'] = -df['vol_120d']

    return df


# ============================================================
# 工具函数（复用 F2-EP）
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
# 批量价格缓存（复用 F2-EP，用于 forward_month_return）
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

    if debug:
        print('    [debug] %s: px0.index %s~%s, px1.index %s~%s' % (
            date_str, px0.index.min(), px0.index.max(),
            px1.index.min(), px1.index.max()))

    d_ts = pd.Timestamp(date_str)
    valid0 = px0.index[px0.index >= d_ts]
    if debug:
        print('    [debug] %s: d_ts=%s, valid0=%d' % (
            date_str, d_ts, len(valid0)))
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

    df = _fetch_market_cap(date_str, stocks)
    if df is None or df.empty:
        if debug:
            print('    [debug] %s: market_cap 查询为空' % date_str)
        return None
    if debug:
        print('    [debug] %s: market_cap %d 只' % (date_str, len(df)))

    # 计算波动率
    vol_map = _calc_realized_volatility(date_str, list(df.index), VOL_LOOKBACK)
    vol_alt_map = _calc_realized_volatility(date_str, list(df.index), VOL_LOOKBACK_ALT)
    if debug:
        print('    [debug] %s: vol_60d 取到 %d/%d, vol_120d 取到 %d/%d' % (
            date_str, len(vol_map), len(df.index),
            len(vol_alt_map), len(df.index)))

    df = _calculate_all_factors(df, vol_map=vol_map, vol_alt_map=vol_alt_map)
    if df is None or df.empty:
        if debug:
            print('    [debug] %s: _calculate_all_factors 返回空' % date_str)
        return None
    if debug:
        v60 = df['vol_60d'].notna().sum() if 'vol_60d' in df.columns else 0
        v120 = df['vol_120d'].notna().sum() if 'vol_120d' in df.columns else 0
        print('    [debug] %s: 因子计算后 %d 只, vol_60d非空=%d, vol_120d非空=%d' % (
            date_str, len(df), v60, v120))

    # 过滤：必须有 vol_60d
    if 'vol_60d' not in df.columns or df['vol_60d'].isna().all():
        if debug:
            print('    [debug] %s: vol_60d 列不存在或全空' % date_str)
        return None
    mask = df['vol_60d'].notna()
    df = df[mask].copy()
    if df.empty:
        if debug:
            print('    [debug] %s: 过滤后为空' % date_str)
        return None
    if debug:
        print('    [debug] %s: 过滤后 %d 只' % (date_str, len(df)))

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
    df = df.dropna(subset=['fwd_return', 'signal'])
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
    """对单样本计算完整 IC 表（Gate 1 七条）。"""
    tag = '%s-V1' % sample_name
    print('\n%s\n===== %s =====\n%s' % ('=' * 60, tag, '=' * 60))

    rebal_dates = get_rebalance_dates(START_DATE, END_DATE)
    if QUICK_TEST:
        rebal_dates = rebal_dates[:3]
        print('  [快速模式] 只跑前 3 个调仓日: %s' % rebal_dates)
    records = []
    ic_list, ic_alt_list, dates_list = [], [], []
    q_rets = {q: [] for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']}

    for i, date in enumerate(rebal_dates):
        df = build_cross_section(date, index_id, debug=(i == 0))
        if df is None:
            continue
        if i == 0:
            print('  [调试] %s: 横截面 %d 只股票 (主信号: -vol_60d)' % (
                date, len(df)))
            if 'vol_60d' in df.columns:
                v = df['vol_60d'].dropna()
                if len(v) > 0:
                    print('         vol_60d: %.6f ~ %.6f (中位 %.6f, 非空 %d/%d)' % (
                        v.min(), v.max(), v.median(), len(v), len(df)))
            if 'vol_120d' in df.columns:
                v2 = df['vol_120d'].dropna()
                if len(v2) > 0:
                    print('         vol_120d: %.6f ~ %.6f (中位 %.6f)' % (
                        v2.min(), v2.max(), v2.median()))
            print('         fwd_return 非空: %d' % df['fwd_return'].notna().sum())

        ic = stats.spearmanr(df['signal'], df['fwd_return'])[0]
        if np.isnan(ic):
            continue
        ic_list.append(ic)
        # 对照：120日波动率
        ic_alt = stats.spearmanr(df['signal_alt'], df['fwd_return'])[0]
        ic_alt_list.append(0.0 if np.isnan(ic_alt) else ic_alt)
        dates_list.append(date)

        # Q1-Q5：按 signal(-vol_60d) 升序（Q5=最低波动=多头）
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

        # 对照信号中性化
        alt_neu_mcap = neutralize_ols(df['signal_alt'].values, log_mcap.values)
        ic_mcap_alt = stats.spearmanr(alt_neu_mcap, df['fwd_return'].values)[0]
        alt_neu_ind = neutralize_ols(df['signal_alt'].values, ind_dummies.values)
        ic_ind_alt = stats.spearmanr(alt_neu_ind, df['fwd_return'].values)[0]

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
            'ic_mcap_neutral': (np.nan if np.isnan(ic_mcap) else ic_mcap),
            'ic_ind_neutral': (np.nan if np.isnan(ic_ind) else ic_ind),
            'ic_mcap_alt': (np.nan if np.isnan(ic_mcap_alt) else ic_mcap_alt),
            'ic_ind_alt': (np.nan if np.isnan(ic_ind_alt) else ic_ind_alt),
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
    print('  主信号-vol_60d: mean=%.6f std=%.6f NW-t=%.4f IC_IR=%.4f 胜率=%.2f%% 正/负=%.2f%%/%.2f%%' % (
        ic_mean, ic_std, ic_t, ic_ir, win_rate * 100, pos_pct * 100, neg_pct * 100))
    # 对照 120日
    alt_mean, alt_std, alt_t = newey_west_t(pd.Series(ic_alt_list), lags=4)
    alt_ir = alt_mean / alt_std if alt_std > 0 else 0
    print('  对照-vol_120d: mean=%.6f std=%.6f NW-t=%.4f IC_IR=%.4f' % (
        alt_mean, alt_std, alt_t, alt_ir))

    # 表2
    print('\n【表2】Q1-Q5 分组（Q5=最低波动=多头组）')
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
        print('  vol_60d 行业中性化: mean=%.6f NW-t=%.4f IC_IR=%.4f N=%d' % (
            m, t, m / float(s.std(ddof=1)), len(s)))
    else:
        print('  样本不足')
    s3 = rec_df['ic_ind_alt'].dropna()
    if len(s3) > 1:
        m3, _, t3 = newey_west_t(s3, lags=4)
        print('  vol_120d 行业中性化: mean=%.6f NW-t=%.4f IC_IR=%.4f' % (
            m3, t3, m3 / float(s3.std(ddof=1))))

    # 表5（决定性门槛）
    print('\n【表5】市值中性化后 Rank IC（决定性门槛 t>=2）')
    s = rec_df['ic_mcap_neutral'].dropna()
    if len(s) > 1:
        m, _, t = newey_west_t(s, lags=4)
        flag = 'PASS' if t >= 2.0 else 'FAIL'
        print('  vol_60d 市值中性化: mean=%.6f NW-t=%.4f IC_IR=%.4f N=%d [%s]' % (
            m, t, m / float(s.std(ddof=1)), len(s), flag))
    else:
        print('  样本不足')
    s2 = rec_df['ic_mcap_alt'].dropna()
    if len(s2) > 1:
        m2, _, t2 = newey_west_t(s2, lags=4)
        flag2 = 'PASS' if t2 >= 2.0 else 'FAIL'
        print('  vol_120d 市值中性化: mean=%.6f NW-t=%.4f IC_IR=%.4f [%s]' % (
            m2, t2, m2 / float(s2.std(ddof=1)), flag2))

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
            pre_ic.index[0].date(), pre_ic.index[-1].date(), pm, pt, len(pre_ic)))
    if len(post_ic) > 1:
        qm, _, qt = newey_west_t(post_ic, lags=4)
        post_mean = qm
        print('  后段(%s~%s): mean=%.6f NW-t=%.4f N=%d' % (
            post_ic.index[0].date(), post_ic.index[-1].date(), qm, qt, len(post_ic)))

    # Gate 1 七条汇总
    print('\n【Gate 1 七条判定】')
    print('  1. IC方向(>0):     mean=%.6f %s' % (
        ic_mean, 'OK' if ic_mean > 0 else 'FAIL'))
    # Q单调性
    q_means = [float(np.nanmean(q_rets[q])) for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']]
    q_mono = all(q_means[i] <= q_means[i + 1] for i in range(4))
    q5_gt_q1 = q_means[4] > q_means[0]
    print('  2. Q5多头>Q1:      Q1=%.4f Q5=%.4f %s' % (
        q_means[0], q_means[4], 'OK' if q5_gt_q1 else 'FAIL(倒挂)'))
    print('  3. 全样本NW-t>=2:  t=%.4f %s' % (
        ic_t, 'OK' if ic_t >= 2.0 else 'FAIL'))
    # 分市值档
    tier_ok = True
    for tier, col in [('小', 'ic_small_cap'), ('中', 'ic_mid_cap'),
                      ('大', 'ic_large_cap')]:
        s_t = rec_df[col].dropna()
        if len(s_t) > 1:
            _, _, t_t = newey_west_t(s_t, lags=4)
            if t_t < 1.0:
                tier_ok = False
    print('  4. 分市值档稳定:   %s' % ('OK' if tier_ok else 'FAIL'))
    # 行业中性化
    s_ind = rec_df['ic_ind_neutral'].dropna()
    ind_t = 0.0
    if len(s_ind) > 1:
        _, _, ind_t = newey_west_t(s_ind, lags=4)
    print('  5. 行业中性t>=2:   t=%.4f %s' % (
        ind_t, 'OK' if ind_t >= 2.0 else 'FAIL'))
    # 断点后
    post_ok = False
    if not np.isnan(post_mean):
        s_post = post_ic
        if len(s_post) > 1:
            _, _, post_t = newey_west_t(s_post, lags=4)
            post_ok = post_t >= 1.5
    print('  6. 断点后增强:     %s' % ('OK' if post_ok else 'FAIL'))
    # 多空夏普
    q1_list = q_rets['Q1']
    q5_list = q_rets['Q5']
    ls_monthly = [q5_list[i] - q1_list[i]
                  if not (np.isnan(q5_list[i]) or np.isnan(q1_list[i]))
                  else np.nan
                  for i in range(len(q1_list))]
    ls_arr = np.array([x for x in ls_monthly if not np.isnan(x)])
    if len(ls_arr) > 1:
        ls_sharpe = float(np.mean(ls_arr) / np.std(ls_arr, ddof=1) * np.sqrt(12))
    else:
        ls_sharpe = 0.0
    print('  7. 多空夏普:       %.4f %s' % (
        ls_sharpe, 'OK' if ls_sharpe > 0.3 else 'FAIL'))

    # CSV
    os.makedirs(OUT_DIR, exist_ok=True)
    out_ic = pd.DataFrame({
        'date': dates_list,
        'rank_ic': ic_list,
        'rank_ic_alt': ic_alt_list,
        'ic_mcap_neutral': rec_df['ic_mcap_neutral'].values,
        'ic_ind_neutral': rec_df['ic_ind_neutral'].values,
        'ic_mcap_alt': rec_df['ic_mcap_alt'].values,
        'ic_ind_alt': rec_df['ic_ind_alt'].values,
        'n': rec_df['n'].values,
    })
    out_ic.to_csv('%s/%s_V1_ic_monthly.csv' % (OUT_DIR, sample_name), index=False)
    out_q = pd.DataFrame({'date': dates_list})
    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        out_q[q] = q_rets[q]
    out_q.to_csv('%s/%s_V1_quantile.csv' % (OUT_DIR, sample_name), index=False)
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
    """单样本 IC 分析。全量上报，禁止择优。"""
    if QUICK_TEST:
        print('>>> 快速验证模式：只跑 2014 年 3 个调仓日 + 只 AllA')
    samples = {'AllA': None}
    summary = []
    for name, idx in samples.items():
        r = analyze_sample(name, idx)
        if r:
            summary.append(r)

    if summary:
        print('\n\n' + '=' * 60)
        print('===== 汇总 =====')
        print('=' * 60)
        print('%-16s%10s%12s%10s%10s%10s%10s%10s' % (
            'tag', 'IC_IR', 'IC_mean', 'NW-t', 'Q1月均', 'Q5月均', '前段IC', '后段IC'))
        for r in summary:
            print('%-16s%10.4f%12.6f%10.4f%10.6f%10.6f%10.6f%10.6f' % (
                r['tag'], r['ic_ir'], r['ic_mean'], r['ic_t'],
                r['q1_monthly_mean'], r['q5_monthly_mean'],
                r['pre_ic_mean'], r['post_ic_mean']))
        pd.DataFrame(summary).to_csv('%s/_summary.csv' % OUT_DIR, index=False)
        print('\n汇总已保存到 %s/_summary.csv' % OUT_DIR)


if __name__ == '__main__':
    run()
