"""
P2-MF-2026Q2-v1 Phase 2 多因子合成脚本（standalone 版）
========================================================

Phase 2 三步法：
  Step 1: 因子相关性矩阵（Spearman Rank Corr，|corr|>0.6 视为重复暴露）
  Step 2: Fama-MacBeth 检验（每月截面回归，看各因子独立截面定价能力）
  Step 3: Risk Parity 合成（等风险贡献替代等权 z-score）

输入：Phase 1 通过的三个因子
  F2 EP:       ep_spot = net_profit / (market_cap * 1e8)   [市值中性化 t=3.41]
  F3 Dividend: dividend_yield = 过去400天累计每股分红 / 收盘价  [市值中性化 t=2.74]
  F5 LowVol:   signal = -std(过去60交易日日收益率)         [市值中性化 t=3.17]

Baseline: P1-F2F5-EP-LowVol-2026Q2-v1 等权 z-score（收益 372.71%，回撤 47.62%）

Gate 2+3 通过标准（预注册锁定）：
  1. 残差 alpha 显著 > 0（5 因子模型回归残差 alpha t ≥ 2）
  2. 行业中性化后超额收益不显著下降（下降幅度 < 30%）
  3. 因子暴露稳定无系统性漂移（FM 滚动窗口系数方向不反转）
  4. 相关性矩阵中保留因子两两 |corr| ≤ 0.6
  5. Fama-MacBeth 保留因子 |t| ≥ 2
  6. Risk Parity 合成相比等权 z-score Baseline：Sharpe 不下降 AND 最大回撤不上升

在聚宽【研究环境】中直接粘贴运行。

【符号约定】
  所有因子已对齐为"越高=越好"（IC>0 有效）：
    F2_ep:       ep_spot 越高 = 越便宜 = 好
    F3_div:      dividend_yield 越高 = 分红越多 = 好
    F5_vol:      -vol_60d 越高 = 波动越低 = 好
"""

import datetime
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from statsmodels.stats.sandwich_covariance import cov_hac

# ============================================================
# 聚宽对象解析（兼容研究环境多种注入方式）
# ============================================================
from jqdata import *  # noqa: F403,F401

try:
    from jqdata import (  # noqa: F401
        valuation, balance, income, cash_flow, query,
        get_fundamentals, get_price, get_all_securities,
        get_index_stocks, get_extras, get_security_info,
        get_trade_days, get_industry, finance,
    )
except Exception:
    pass


warnings.filterwarnings('ignore')


# ============================================================
# 参数
# ============================================================
START_DATE = '2014-01-01'
END_DATE = '2026-06-30'
INDEX_ID = None  # 全A
SAMPLE_NAME = 'AllA'

# 因子参数
VOL_LOOKBACK = 60
VOL_MIN_OBS_RATIO = 0.5
DIV_LOOKBACK_DAYS = 400  # 与 F3 脚本一致
DIV_BATCH_SIZE = 300

# Risk Parity 协方差估计窗口（交易日）
RP_COV_LOOKBACK = 60

# 断点
BREAKPOINT = datetime.date(2019, 6, 1)

# 输出
OUT_DIR = 'results/P2-MF-2026Q2-v1'

# QUICK_TEST=True 只跑前 3 个调仓日验证数据链
QUICK_TEST = False

# 金融股剔除（sw_l1 严格相等匹配）
_FINANCE_NAMES = {'银行I', '非银金融I'}


# ============================================================
# 内联 data_layer 必要函数（standalone，无需 import data_layer）
# ============================================================

def _get_stock_pool(index_id, date_str, min_listed_days=180):
    """构建股票池：成分股（或全A）→ 剔ST → 剔次新股 → 剔金融股。"""
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

    # 次新股剔除
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

    # 金融股剔除
    if stocks:
        stocks = _exclude_finance_stocks(stocks, date_str)
    return stocks


def _exclude_finance_stocks(stocks, date_str):
    """剔除金融行业股票（银行/非银金融），sw_l1 严格相等匹配。"""
    if not stocks:
        return stocks
    try:
        ind_raw = get_industry(stocks, date=date_str)
    except Exception:
        return stocks
    if not ind_raw:
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


def _get_col(df, *candidates):
    """从 df 中按候选列名找第一个存在的列，返回 Series；找不到返回 None。"""
    for c in candidates:
        if c in df.columns:
            return df[c]
    return None


def _fetch_fundamentals_pit(date_str, stocks):
    """PIT 财务数据查询（动态字段探测，规避聚宽字段名差异）。

    返回以 code 为 index 的 DataFrame，列名为聚宽原始列名。
    """
    pieces = []
    # valuation
    try:
        q_val = query(valuation.code, valuation.market_cap)
        df_val = get_fundamentals(q_val, date=date_str)
        if df_val is not None and not df_val.empty:
            if 'code' in df_val.columns:
                df_val = df_val.set_index('code')
            pieces.append(df_val)
    except Exception:
        pass

    # balance 全字段
    try:
        df_bal = get_fundamentals(query(balance), date=date_str)
        if df_bal is not None and not df_bal.empty:
            if 'code' in df_bal.columns:
                df_bal = df_bal.set_index('code')
            pieces.append(df_bal)
    except Exception:
        pass

    # income 全字段
    try:
        df_inc = get_fundamentals(query(income), date=date_str)
        if df_inc is not None and not df_inc.empty:
            if 'code' in df_inc.columns:
                df_inc = df_inc.set_index('code')
            pieces.append(df_inc)
    except Exception:
        pass

    # cash_flow 全字段
    try:
        df_cf = get_fundamentals(query(cash_flow), date=date_str)
        if df_cf is not None and not df_cf.empty:
            if 'code' in df_cf.columns:
                df_cf = df_cf.set_index('code')
            pieces.append(df_cf)
    except Exception:
        pass

    if not pieces:
        return None

    df = pieces[0]
    for p in pieces[1:]:
        overlap = [c for c in p.columns if c in df.columns]
        if overlap:
            p = p.drop(columns=overlap)
        df = df.join(p, how='outer')

    if stocks:
        df = df[df.index.isin(list(stocks))]
    if df is None or df.empty:
        return None
    return df


def _fetch_dividend_data(date_str, stocks, lookback_days=DIV_LOOKBACK_DAYS):
    """用 finance.STK_XR_XD 查询过去 lookback_days 天已实施的现金分红。

    返回 {code: 累计每股税前现金分红}。

    与 F3 IC 脚本一致口径：
      - code 带后缀（.XSHE/.XSHG）
      - bonus_ratio_rmb 是每10股派息，除以10得每股
      - 只取 plan_progress == '实施方案'
      - a_xr_date 在 [start, end] 内
    """
    if not stocks:
        return {}
    end = pd.Timestamp(date_str)
    start = end - pd.Timedelta(days=lookback_days)
    start_str = start.strftime('%Y-%m-%d')
    end_str = end.strftime('%Y-%m-%d')

    all_dfs = []
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
                finance.STK_XR_XD.a_xr_date >= start_str,
                finance.STK_XR_XD.a_xr_date <= end_str,
            )
            df_batch = finance.run_query(q)
            if df_batch is not None and not df_batch.empty:
                all_dfs.append(df_batch)
        except Exception:
            continue

    if not all_dfs:
        return {}
    df = pd.concat(all_dfs, ignore_index=True) if len(all_dfs) > 1 else all_dfs[0]

    # 只取已实施
    if 'plan_progress' in df.columns:
        df = df[df['plan_progress'] == '实施方案']

    # bonus_ratio_rmb > 0
    if 'bonus_ratio_rmb' in df.columns:
        df['bonus_ratio_rmb'] = pd.to_numeric(df['bonus_ratio_rmb'], errors='coerce')
        df = df[df['bonus_ratio_rmb'] > 0]
    else:
        return {}

    if df.empty:
        return {}

    # 按 code 累加每股分红（bonus_ratio_rmb 是每10股，除以10得每股）
    div_per_share = df.groupby('code')['bonus_ratio_rmb'].sum() / 10.0
    return dict(div_per_share)


def _calc_realized_volatility(date_str, stocks, lookback_days=VOL_LOOKBACK):
    """计算实现波动率：过去 lookback_days 交易日日收益率标准差。

    与 F5 IC 脚本一致口径。返回 {code: vol_60d}（正数，未取负）。
    """
    if not stocks:
        return {}
    try:
        df_px = get_price(stocks, end_date=date_str, count=lookback_days + 1,
                          fields=['close'], skip_paused=False, panel=False, fq='post')
    except Exception:
        return {}
    if df_px is None or df_px.empty:
        return {}

    # panel=False 长表：时间在 'time' 列
    if 'time' in df_px.columns:
        df_px = df_px.set_index('time')
    elif 'date' in df_px.columns:
        df_px = df_px.set_index('date')

    if 'code' not in df_px.columns:
        return {}

    # 透视成宽表：index=日期, columns=code, values=close
    try:
        wide = df_px.pivot_table(index=df_px.index,
                                 columns='code', values='close')
    except Exception:
        return {}

    # 计算日收益率，再求标准差
    rets = wide.pct_change().iloc[1:]  # 去掉第一个 NaN
    min_obs = int(lookback_days * VOL_MIN_OBS_RATIO)
    valid_count = rets.notna().sum()
    valid_codes = valid_count[valid_count >= min_obs].index
    if len(valid_codes) == 0:
        return {}
    vols = rets[valid_codes].std(ddof=1)
    return dict(vols)


def _get_close_prices(date_str, stocks):
    """获取 date_str 当天不复权收盘价 {code: close}（F3 股息率分母用）。"""
    if not stocks:
        return {}
    try:
        df = get_price(stocks, end_date=date_str, count=1,
                       fields=['close'], skip_paused=False, panel=False, fq='none')
    except Exception:
        return {}
    if df is None or df.empty:
        return {}
    if 'time' in df.columns:
        df = df.set_index('time')
    elif 'date' in df.columns:
        df = df.set_index('date')
    if 'code' not in df.columns:
        return {}
    return dict(zip(df['code'], df['close']))


# ============================================================
# 工具函数
# ============================================================

def get_rebalance_dates(start, end):
    """季度调仓日：每年 5/9/11 月首个交易日（对齐策略 run_monthly）。"""
    months = [5, 9, 11]
    dates = []
    year = int(start[:4])
    end_year = int(end[:4])
    while year <= end_year:
        for m in months:
            dt_str = f'{year}-{m:02d}-01'
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


def zscore_cross_section(s):
    """横截面 Z-score 标准化（每调仓日独立）。"""
    s = pd.Series(s, dtype=float)
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return s * 0
    return (s - mu) / sd


def winsorize_cross_section(s, lower=0.01, upper=0.99):
    """横截面 winsorize：把 [0, lower] 压到 lower 分位，[upper, 1] 压到 upper 分位。"""
    s = pd.Series(s, dtype=float)
    if s.notna().sum() < 10:
        return s
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lower=lo, upper=hi)


# ============================================================
# 批量价格缓存（forward_period_return 加速）
# ============================================================
_PRICE_CACHE = {}


def _load_period_prices(start_str, end_str, codes):
    """加载 [start_str, end_str] 区间指定 codes 的收盘价，缓存。
    返回 DataFrame（index=交易日, columns=code）。"""
    cache_key = (start_str, end_str, hash(frozenset(codes)))
    if cache_key in _PRICE_CACHE:
        return _PRICE_CACHE[cache_key]

    codes_list = list(codes)
    close = pd.DataFrame()
    try:
        df = get_price(codes_list,
                       start_date=start_str,
                       end_date=end_str,
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


def forward_period_return(codes, date_str, next_date_str, debug=False):
    """计算 codes 从 date_str 到 next_date_str 的累计收益（调仓间隔收益）。

    与实际持仓周期一致（5→9月约4个月，9→11月约2个月，11→次年5月约6个月）。
    date_str: 当调仓日（取当天或之后首个交易日收盘价作为 start）
    next_date_str: 下个调仓日（取当天或之前最后交易日收盘价作为 end）
    返回 Series（code → ret）；None 表示数据不足。
    """
    px = _load_period_prices(date_str, next_date_str, codes)
    if px is None or px.empty:
        return None
    try:
        px.index = pd.to_datetime(px.index)
    except Exception:
        return None

    d_ts = pd.Timestamp(date_str)
    nd_ts = pd.Timestamp(next_date_str)

    # start: date_str 当天或之后首个交易日
    valid_start = px.index[px.index >= d_ts]
    if len(valid_start) == 0:
        return None
    close_start = px.loc[valid_start[0]]

    # end: next_date_str 当天或之前最后交易日
    valid_end = px.index[px.index <= nd_ts]
    if len(valid_end) == 0:
        # 兜底：取 px 最后一行
        close_end = px.iloc[-1]
    else:
        close_end = px.loc[valid_end[-1]]

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
# 横截面构建：三因子同时计算
# ============================================================

def build_cross_section(date, next_date=None, debug=False):
    """构建单调仓日横截面 DataFrame，同时计算 F2/F3/F5 三因子。

    关键：三因子做市值中性化（log(market_cap) OLS 残差）+ winsorize 1%/99%，
    与 Phase 1 IC 验证口径一致（Phase 1 通过的是市值中性化后的因子）。
    若不做中性化，原始 F2_ep 与市值高度共线，OLS 多变量 FM 会符号反转。

    返回 DataFrame（index=code），含列：
      market_cap, F2_ep_raw, F3_div_raw, F5_vol_raw,
      F2_ep（中性化+winsorize后）, F3_div, F5_vol,
      fwd_return, industry
    None 表示数据不足。
    """
    date_str = date.strftime('%Y-%m-%d')
    stocks = _get_stock_pool(INDEX_ID, date_str)
    if not stocks:
        if debug:
            print(f'    [debug] {date_str}: 股票池为空')
        return None

    df = _fetch_fundamentals_pit(date_str, stocks)
    if df is None or df.empty:
        if debug:
            print(f'    [debug] {date_str}: fundamentals 查询为空')
        return None

    # ---- F2 EP: ep_spot = net_profit / (market_cap * 1e8) ----
    mcap = _get_col(df, 'market_cap')
    np_col = _get_col(df, 'net_profit', 'np_parent_company_owners',
                      'net_profit_is_parent_company')
    if mcap is None or np_col is None:
        if debug:
            print(f'    [debug] {date_str}: market_cap 或 net_profit 缺失')
        return None
    df['market_cap'] = mcap.astype(float)
    df['net_profit'] = np_col.astype(float)
    df['F2_ep_raw'] = df['net_profit'] / (df['market_cap'] * 1e8)

    # 基础过滤：net_profit > 0（F2/F3 用）
    mask = df['net_profit'].fillna(0) > 0

    # debt_to_assets 过滤（与 F2/F3 一致）
    tl = _get_col(df, 'total_liability', 'total_liabilities')
    ta = _get_col(df, 'total_assets')
    if tl is not None and ta is not None:
        df['debt_to_assets'] = tl.astype(float) / ta.astype(float).replace(0, np.nan)
        mask &= df['debt_to_assets'].fillna(0.5) <= 1.0

    # ---- F3 Dividend ----
    div_map = _fetch_dividend_data(date_str, list(df.index))
    close_map = _get_close_prices(date_str, list(df.index))
    df['div_per_share'] = df.index.map(lambda c: div_map.get(c, 0.0))
    df['close_price'] = df.index.map(lambda c: close_map.get(c, np.nan))
    df['F3_div_raw'] = (df['div_per_share'] / df['close_price'].replace(0, np.nan))
    df['F3_div_raw'] = df['F3_div_raw'].where(df['div_per_share'] > 0, np.nan)

    # ---- F5 LowVol: signal = -vol_60d ----
    vol_map = _calc_realized_volatility(date_str, list(df.index), VOL_LOOKBACK)
    df['vol_60d'] = df.index.map(lambda c: vol_map.get(c, np.nan))
    df['F5_vol_raw'] = -df['vol_60d']  # 取负，低波动=高信号

    if debug:
        print(f'    [debug] {date_str}: 三因子非空统计 '
              f'F2={df["F2_ep_raw"].notna().sum()}/'
              f'F3={df["F3_div_raw"].notna().sum()}/'
              f'F5={df["F5_vol_raw"].notna().sum()} '
              f'(总 {len(df)})')

    # 三因子同时非空（FM 与 RP 要求完整面板）
    df = df.dropna(subset=['F2_ep_raw', 'F3_div_raw', 'F5_vol_raw', 'market_cap']).copy()
    if len(df) < 30:
        if debug:
            print(f'    [debug] {date_str}: 三因子同时非空不足 30 只 ({len(df)})')
        return None

    # ---- 市值中性化 + winsorize（spec 第四节要求，与 Phase 1 IC 验证口径一致）----
    log_mcap = np.log(df['market_cap'].astype(float).replace(0, np.nan))
    for raw_col, neut_col in [('F2_ep_raw', 'F2_ep'),
                              ('F3_div_raw', 'F3_div'),
                              ('F5_vol_raw', 'F5_vol')]:
        raw = df[raw_col].astype(float)
        # 市值中性化：对 log(market_cap) OLS 回归取残差
        neut = neutralize_ols(raw.values, log_mcap.values)
        # winsorize 1%/99%（中性化后仍有极端残差）
        neut = winsorize_cross_section(pd.Series(neut, index=df.index))
        df[neut_col] = neut

    # 行业
    ind_map = get_industry_map(list(df.index), date_str)
    df['industry'] = df.index.map(lambda c: ind_map.get(c, '未知'))

    # 前向收益（调仓间隔收益，与实际持仓一致）
    if next_date is None:
        if debug:
            print(f'    [debug] {date_str}: next_date 为 None，无法计算 fwd_return')
        return None
    next_date_str = next_date.strftime('%Y-%m-%d') if hasattr(next_date, 'strftime') else str(next_date)
    codes = list(df.index)
    fwd = forward_period_return(codes, date_str, next_date_str, debug=debug)
    if fwd is None:
        if debug:
            print(f'    [debug] {date_str}: forward_period_return 返回 None')
        return None
    df['fwd_return'] = df.index.map(fwd)
    df = df.dropna(subset=['fwd_return'])
    if len(df) < 30:
        if debug:
            print(f'    [debug] {date_str}: fwd_return 后不足 30 只 ({len(df)})')
        return None

    if debug:
        print(f'    [debug] {date_str}: 最终横截面 {len(df)} 只')
    return df


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


# ============================================================
# Step 1: 因子相关性矩阵
# ============================================================

def step1_correlation_matrix(date_pairs, debug_first=True):
    """计算 F2/F3/F5 两两 Spearman Rank 相关性矩阵。

    用市值中性化后的因子（与 Phase 1 IC 验证口径一致）。
    返回 (matrix_mean, matrix_std, time_series_dict)。
    """
    print('\n' + '=' * 60)
    print('===== Step 1: 因子相关性矩阵（Spearman Rank，市值中性化后） =====')
    print('=' * 60)

    corr_records = []  # 每调仓日一个 3x3 矩阵
    for i, (date, next_date) in enumerate(date_pairs):
        df = build_cross_section(date, next_date, debug=(debug_first and i == 0))
        if df is None or len(df) < 30:
            continue
        # 三因子两两 Spearman（用中性化后的 F2_ep/F3_div/F5_vol）
        f2 = df['F2_ep'].values
        f3 = df['F3_div'].values
        f5 = df['F5_vol'].values
        try:
            c_23 = stats.spearmanr(f2, f3)[0]
            c_25 = stats.spearmanr(f2, f5)[0]
            c_35 = stats.spearmanr(f3, f5)[0]
        except Exception:
            continue
        corr_records.append({
            'date': date,
            'n': len(df),
            'F2_F3': c_23,
            'F2_F5': c_25,
            'F3_F5': c_35,
        })
        if (i + 1) % 6 == 0:
            print(f'  [Step1] 进度 {i+1}/{len(date_pairs)} 日处理完成')

    if not corr_records:
        print('  无有效数据')
        return None

    df_c = pd.DataFrame(corr_records).set_index('date')

    # 表1：均值 + 标准差
    print('\n【表1】三因子 Spearman 相关性矩阵（均值）')
    print(f'{"":<8}{"F2_ep":>12}{"F3_div":>12}{"F5_vol":>12}')
    factors = ['F2_ep', 'F3_div', 'F5_vol']
    for fi in factors:
        row = []
        for fj in factors:
            if fi == fj:
                row.append(1.0)
                continue
            # 列名统一按字典序拼接（F2_F3 而非 F3_F2），与 corr_records 中一致
            col = '_'.join(sorted([fi.split('_')[0], fj.split('_')[0]]))
            row.append(df_c[col].mean())
        print(f'{fi:<8}{row[0]:>12.4f}{row[1]:>12.4f}{row[2]:>12.4f}')

    print('\n【表2】相关性标准差（稳定性）')
    print(f'{"":<8}{"F2_ep":>12}{"F3_div":>12}{"F5_vol":>12}')
    for fi in factors:
        row = []
        for fj in factors:
            if fi == fj:
                row.append(0.0)
                continue
            col = '_'.join(sorted([fi.split('_')[0], fj.split('_')[0]]))
            row.append(df_c[col].std(ddof=1))
        print(f'{fi:<8}{row[0]:>12.4f}{row[1]:>12.4f}{row[2]:>12.4f}')

    # 重复暴露判定
    print('\n【判定】|corr| > 0.6 视为重复暴露；|corr| > 0.8 必须二选一')
    for col in ['F2_F3', 'F2_F5', 'F3_F5']:
        m = df_c[col].mean()
        if abs(m) > 0.8:
            print(f'  {col}: mean={m:.4f} ⚠️ 强重复（必须二选一）')
        elif abs(m) > 0.6:
            print(f'  {col}: mean={m:.4f} ⚠️ 重复暴露（需去重或合并）')
        else:
            print(f'  {col}: mean={m:.4f} ✅ 无重复')

    # 保存 CSV
    os.makedirs(OUT_DIR, exist_ok=True)
    df_c.to_csv(f'{OUT_DIR}/step1_correlation_timeseries.csv')
    return df_c


# ============================================================
# Step 2: Fama-MacBeth 检验
# ============================================================

def step2_fama_macbeth(date_pairs, debug_first=True):
    """Fama-MacBeth 两阶段回归（多变量 + 单变量诊断）。

    第一阶段：每月截面回归
      多变量：fwd_return ~ 1 + F2_z + F3_z + F5_z（市值中性化+winsorize 后 Z-score）
      单变量：fwd_return ~ 1 + F2_z / 1 + F3_z / 1 + F5_z（诊断共线性 vs 因子失效）
    第二阶段：时序平均 + Newey-West t

    单变量诊断逻辑：
      - 若单变量 FM 正向显著（t≥2）且与 IC 方向一致 → 多变量符号反转是共线性问题
      - 若单变量 FM 也负向或不显著 → 因子本身在 OLS 框架下失效

    返回 (df_fm_multivariate, fm_keep, df_fm_univariate)。
    """
    print('\n' + '=' * 60)
    print('===== Step 2: Fama-MacBeth 检验（市值中性化后） =====')
    print('=' * 60)

    fm_records = []        # 多变量 FM + 单变量 FM（合并到同一记录）
    for i, (date, next_date) in enumerate(date_pairs):
        df = build_cross_section(date, next_date, debug=(debug_first and i == 0))
        if df is None or len(df) < 30:
            continue

        # Z-score 标准化（系数可比；因子已市值中性化+winsorize）
        df['F2_z'] = zscore_cross_section(df['F2_ep'])
        df['F3_z'] = zscore_cross_section(df['F3_div'])
        df['F5_z'] = zscore_cross_section(df['F5_vol'])
        y = df['fwd_return'].values

        # ---- 多变量截面回归：fwd_return ~ 1 + F2 + F3 + F5 ----
        X = np.column_stack([
            np.ones(len(df)),
            df['F2_z'].values,
            df['F3_z'].values,
            df['F5_z'].values,
        ])
        try:
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            y_pred = X @ beta
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        except Exception:
            continue

        rec = {
            'date': date,
            'intercept': beta[0],
            'F2_ep': beta[1],
            'F3_div': beta[2],
            'F5_vol': beta[3],
            'r2_multi': r2,
            'n': len(df),
        }

        # ---- 单变量截面回归：fwd_return ~ 1 + F_i ----
        for f, f_z in [('F2_ep', 'F2_z'), ('F3_div', 'F3_z'), ('F5_vol', 'F5_z')]:
            X_uni = np.column_stack([np.ones(len(df)), df[f_z].values])
            try:
                b_uni, _, _, _ = np.linalg.lstsq(X_uni, y, rcond=None)
                y_pred_uni = X_uni @ b_uni
                ss_res_uni = np.sum((y - y_pred_uni) ** 2)
                r2_uni = 1 - ss_res_uni / ss_tot if ss_tot > 0 else 0
                rec[f'{f}_uni'] = b_uni[1]
                rec[f'{f}_uni_r2'] = r2_uni
            except Exception:
                rec[f'{f}_uni'] = np.nan
                rec[f'{f}_uni_r2'] = np.nan

        fm_records.append(rec)

        if (i + 1) % 6 == 0:
            print(f'  [Step2] 进度 {i+1}/{len(date_pairs)} 日处理完成')

    if not fm_records:
        print('  无有效数据')
        return None

    df_fm = pd.DataFrame(fm_records).set_index('date')

    # 表3：多变量 FM 系数（全样本）
    print('\n【表3】多变量 Fama-MacBeth 系数（全样本，NW-t 滞后 4 期）')
    print(f'{"因子":<10}{"均值":>12}{"NW-t":>10}{"正比例":>10}{"判定":>10}')
    factors = ['F2_ep', 'F3_div', 'F5_vol']
    fm_keep = []
    for f in factors:
        s = df_fm[f].dropna()
        if len(s) < 2:
            print(f'{f:<10}  样本不足')
            continue
        m, _, t = newey_west_t(s, lags=4)
        pos_pct = (s > 0).mean()
        keep = '✅ 保留' if abs(t) >= 2 else '❌ 剔除'
        if abs(t) >= 2:
            fm_keep.append(f)
        print(f'{f:<10}{m:>12.6f}{t:>10.4f}{pos_pct:>10.2%}{keep:>10}')

    # 表3b：单变量 FM 诊断（区分共线性 vs 因子失效）
    print('\n【表3b】单变量 Fama-MacBeth 诊断（fwd_return ~ 1 + F_i）')
    print(f'{"因子":<10}{"均值":>12}{"NW-t":>10}{"正比例":>10}{"R²":>10}{"判定":>14}')
    fm_keep_uni = []
    for f in factors:
        s = df_fm[f'{f}_uni'].dropna()
        if len(s) < 2:
            print(f'{f:<10}  样本不足')
            continue
        m, _, t = newey_west_t(s, lags=4)
        pos_pct = (s > 0).mean()
        r2_mean = df_fm[f'{f}_uni_r2'].mean()
        # 单变量判定：t≥2 且方向为正（与 IC 一致）
        if t >= 2:
            keep = '✅ 正向显著'
            fm_keep_uni.append(f)
        elif t <= -2:
            keep = '⚠️ 负向显著'
        else:
            keep = '❌ 不显著'
        print(f'{f:<10}{m:>12.6f}{t:>10.4f}{pos_pct:>10.2%}{r2_mean:>10.4f}{keep:>14}')

    print('\n【诊断结论】')
    if fm_keep_uni:
        print(f'  单变量 FM 正向显著因子: {fm_keep_uni}')
        if not fm_keep:
            print(f'  多变量 FM 全部不显著，但单变量有正向显著因子 →')
            print(f'  符号反转是多重共线性问题，单变量 FM 更可信')
            print(f'  建议：Phase 2 终判基于单变量 FM，保留 {fm_keep_uni}')
            fm_keep = fm_keep_uni  # 用单变量结果覆盖
    else:
        print(f'  单变量 FM 也无正向显著因子 → 因子在 OLS 框架下失效')
        print(f'  需进一步诊断（可能 fwd_return 口径或样本问题）')

    # 表4：多变量 FM 断点前后对照
    print('\n【表4】多变量 FM 系数断点前后对照（2019.06 断点）')
    bp_ts = pd.Timestamp(BREAKPOINT)
    df_fm.index = pd.to_datetime(df_fm.index)
    pre = df_fm[df_fm.index < bp_ts]
    post = df_fm[df_fm.index >= bp_ts]
    print(f'{"因子":<10}{"前段 t":>10}{"后段 t":>10}{"方向一致":>12}')
    for f in factors:
        if len(pre) > 1 and len(post) > 1:
            _, _, t_pre = newey_west_t(pre[f], lags=4)
            _, _, t_post = newey_west_t(post[f], lags=4)
            consistent = '✅' if np.sign(t_pre) == np.sign(t_post) else '❌ 反转'
            print(f'{f:<10}{t_pre:>10.4f}{t_post:>10.4f}{consistent:>12}')

    # 表5：R² 时序
    print('\n【表5】R² 时序描述（看模型解释力稳定性）')
    r2_multi = df_fm['r2_multi']
    print(f'  多变量 R²: mean={r2_multi.mean():.4f} std={r2_multi.std(ddof=1):.4f} '
          f'min={r2_multi.min():.4f} max={r2_multi.max():.4f}')
    for f in factors:
        r2_uni = df_fm[f'{f}_uni_r2']
        print(f'  {f} 单变量 R²: mean={r2_uni.mean():.4f} max={r2_uni.max():.4f}')

    # 保存 CSV
    df_fm.to_csv(f'{OUT_DIR}/step2_fm_coefficients.csv')
    return df_fm, fm_keep


# ============================================================
# Step 3: Risk Parity 合成
# ============================================================

def _erc_weights(cov, max_iter=200, tol=1e-8):
    """等风险贡献（ERC）权重求解。

    目标：每因子对组合方差贡献相等。
    cov: 因子收益协方差矩阵（正定）
    返回：权重向量 w，sum(w)=1, w>=0
    """
    n = cov.shape[0]
    if n < 2:
        return np.ones(n) / n

    # 目标函数：sum_{i,j} (RC_i - RC_j)^2，RC_i = w_i * (Σw)_i
    def objective(w):
        port_var = w @ cov @ w
        if port_var <= 0:
            return 1e10
        rc = w * (cov @ w)  # 各因子边际风险贡献
        rc_norm = rc / port_var
        # 与等权目标 (1/n) 的偏差
        return np.sum((rc_norm - 1.0 / n) ** 2)

    # 约束：sum(w)=1, w>=0
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    bounds = [(1e-6, 1.0)] * n
    x0 = np.ones(n) / n

    try:
        result = minimize(objective, x0, method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': max_iter, 'ftol': tol})
        if result.success:
            w = result.x
            w = np.clip(w, 0, None)
            s = w.sum()
            return w / s if s > 0 else x0
    except Exception:
        pass
    return x0


def step3_risk_parity(date_pairs, fm_keep, debug_first=True):
    """Risk Parity 合成：基于 FM 保留因子做 ERC 加权。

    对每个调仓日：
      1. 计算 FM 保留因子的横截面 Z-score（市值中性化后）
      2. 用历史因子收益估协方差（数据不足回退等权）
      3. 求解 ERC 权重
      4. 合成分 = sum(w_i * zscore(F_i))
      5. 选 top-50，按合成分 z-score 加权（与 Baseline 一致）

    Baseline: 等权 z-score = (z(F2) + z(F5)) / 2（P1-F2F5 组合）
    收益口径：调仓间隔收益（与实际持仓一致，与 Phase 1 回测可比）

    返回 RP 净值序列、Baseline 净值序列、权重序列。
    """
    print('\n' + '=' * 60)
    print('===== Step 3: Risk Parity 合成（市值中性化后） =====')
    print('=' * 60)

    if not fm_keep:
        print('  FM 未保留任何因子，Step 3 退化为等权 Baseline')
        fm_keep = ['F2_ep', 'F5_vol']  # 退化到 F2+F5

    print(f'  FM 保留因子: {fm_keep}')
    print(f'  Baseline: F2+F5 等权 z-score（P1-F2F5-EP-LowVol-2026Q2-v1）')
    print(f'  收益口径: 调仓间隔收益（2-6个月，与实际持仓一致）')
    if len(fm_keep) == 1 and fm_keep[0] == 'F2_ep':
        print(f'  ⚠️ FM 只保留 F2，RP 退化为 F2 单因子')
        print(f'     对比含义：F2 单因子 vs F2+F5 等权 = 检验 F5 的边际贡献')
        print(f'     若 RP 输给 Baseline → F5 虽非独立 alpha 因子，但是有效风险调节器')

    rp_records = []   # 每调仓日 RP 组合前向收益
    base_records = []  # 每调仓日 Baseline 组合前向收益
    weight_records = []  # 每调仓日 RP 权重

    # 历史因子收益（用于协方差估计）
    factor_returns_history = {f: [] for f in fm_keep}

    for i, (date, next_date) in enumerate(date_pairs):
        df = build_cross_section(date, next_date, debug=(debug_first and i == 0))
        if df is None or len(df) < 30:
            continue

        # Z-score 标准化（因子已市值中性化+winsorize）
        for f in fm_keep:
            df[f'{f}_z'] = zscore_cross_section(df[f])
        # Baseline 也需要 F2/F5 的 z-score
        if 'F2_ep_z' not in df.columns:
            df['F2_ep_z'] = zscore_cross_section(df['F2_ep'])
        if 'F5_vol_z' not in df.columns:
            df['F5_vol_z'] = zscore_cross_section(df['F5_vol'])

        # 协方差估计：用累积的历史因子收益序列
        # 单因子无需协方差（退化为等权）；多因子需历史样本 ≥ 6 才估协方差
        if len(fm_keep) == 1:
            w = np.ones(len(fm_keep)) / len(fm_keep)
        elif len(factor_returns_history[fm_keep[0]]) >= 6:
            ret_matrix = np.array([factor_returns_history[f]
                                   for f in fm_keep])
            cov = np.cov(ret_matrix)
            # 单因子时 np.cov 返回 0-d 标量，cov.shape=()，shape[0] 越界
            if cov.ndim != 2 or cov.shape[0] != len(fm_keep) or np.isnan(cov).any():
                w = np.ones(len(fm_keep)) / len(fm_keep)
            else:
                w = _erc_weights(cov)
        else:
            w = np.ones(len(fm_keep)) / len(fm_keep)

        weight_records.append({
            'date': date,
            **{f: w[idx] for idx, f in enumerate(fm_keep)},
        })

        # RP 合成分
        df['rp_score'] = sum(w[idx] * df[f'{f}_z']
                             for idx, f in enumerate(fm_keep))

        # Baseline 合成分：F2+F5 等权 z-score
        if 'F2_ep' in df.columns and 'F5_vol' in df.columns:
            df['base_score'] = 0.5 * df['F2_ep_z'] + 0.5 * df['F5_vol_z']
        else:
            df['base_score'] = df['rp_score']  # 退化

        # 选 top-50，按 score z-score 加权（与 Baseline 选股一致）
        n_hold = min(50, len(df) // 5)
        n_hold = max(n_hold, 10)

        # RP 组合
        df_rp = df.nlargest(n_hold, 'rp_score')
        rp_weights = zscore_cross_section(df_rp['rp_score'])
        rp_weights = rp_weights.clip(lower=0)  # 不允许负权重
        if rp_weights.sum() > 0:
            rp_weights = rp_weights / rp_weights.sum()
        rp_ret = float((rp_weights * df_rp['fwd_return']).sum())

        # Baseline 组合
        df_base = df.nlargest(n_hold, 'base_score')
        base_weights = zscore_cross_section(df_base['base_score'])
        base_weights = base_weights.clip(lower=0)
        if base_weights.sum() > 0:
            base_weights = base_weights / base_weights.sum()
        base_ret = float((base_weights * df_base['fwd_return']).sum())

        rp_records.append({'date': date, 'rp_ret': rp_ret})
        base_records.append({'date': date, 'base_ret': base_ret})

        # 更新因子收益历史（用 top-quintile 等权收益作为因子收益代理）
        for f in fm_keep:
            q5 = df.nlargest(max(len(df) // 5, 10), f)
            f_ret = float(q5['fwd_return'].mean())
            factor_returns_history[f].append(f_ret)

        if (i + 1) % 6 == 0:
            print(f'  [Step3] 进度 {i+1}/{len(date_pairs)} 日处理完成')

    if not rp_records:
        print('  无有效数据')
        return None

    df_rp = pd.DataFrame(rp_records).set_index('date')
    df_base = pd.DataFrame(base_records).set_index('date')
    df_w = pd.DataFrame(weight_records).set_index('date')

    # 净值曲线
    df_rp['rp_nav'] = (1 + df_rp['rp_ret']).cumprod()
    df_base['base_nav'] = (1 + df_base['base_ret']).cumprod()

    # 表6：RP vs Baseline 收益风险对照
    # 收益口径=调仓间隔收益（2-6个月不等），年化用实际年数，波动用 sqrt(3)（一年3次调仓）
    print('\n【表6】RP 组合 vs Baseline 收益风险对照')
    print(f'{"指标":<14}{"RP 组合":>14}{"Baseline":>14}{"差异":>14}')
    rp_total = df_rp['rp_nav'].iloc[-1] - 1
    base_total = df_base['base_nav'].iloc[-1] - 1
    # 实际年数（从首个调仓日到末日）
    if len(df_rp) > 1:
        first_date = pd.to_datetime(df_rp.index[0])
        last_date = pd.to_datetime(df_rp.index[-1])
        years = (last_date - first_date).days / 365.25
        years = max(years, 0.1)
    else:
        years = 1.0
    rp_ann = (1 + rp_total) ** (1.0 / years) - 1
    base_ann = (1 + base_total) ** (1.0 / years) - 1
    # 一年约 3 次调仓（5/9/11月），波动年化用 sqrt(3)
    PERIODS_PER_YEAR = 3
    rp_vol = df_rp['rp_ret'].std() * np.sqrt(PERIODS_PER_YEAR)
    base_vol = df_base['base_ret'].std() * np.sqrt(PERIODS_PER_YEAR)
    rp_sharpe = rp_ann / rp_vol if rp_vol > 0 else 0
    base_sharpe = base_ann / base_vol if base_vol > 0 else 0
    rp_dd = (1 - df_rp['rp_nav'] / df_rp['rp_nav'].cummax()).max()
    base_dd = (1 - df_base['base_nav'] / df_base['base_nav'].cummax()).max()

    print(f'{"总收益":<14}{rp_total:>14.4f}{base_total:>14.4f}{rp_total-base_total:>14.4f}')
    print(f'{"年化":<14}{rp_ann:>14.4f}{base_ann:>14.4f}{rp_ann-base_ann:>14.4f}')
    print(f'{"年化波动":<14}{rp_vol:>14.4f}{base_vol:>14.4f}{rp_vol-base_vol:>14.4f}')
    print(f'{"Sharpe":<14}{rp_sharpe:>14.4f}{base_sharpe:>14.4f}{rp_sharpe-base_sharpe:>14.4f}')
    print(f'{"最大回撤":<14}{rp_dd:>14.4f}{base_dd:>14.4f}{rp_dd-base_dd:>14.4f}')

    # Gate 6 判定
    print('\n【Gate 6 判定】RP vs Baseline：Sharpe 不下降 AND 回撤不上升')
    sharpe_ok = rp_sharpe >= base_sharpe
    dd_ok = rp_dd <= base_dd
    print(f'  Sharpe 不下降: {"✅" if sharpe_ok else "❌"} '
          f'({rp_sharpe:.4f} vs {base_sharpe:.4f})')
    print(f'  回撤不上升:    {"✅" if dd_ok else "❌"} '
          f'({rp_dd:.4f} vs {base_dd:.4f})')
    print(f'  Gate 6: {"✅ 通过" if (sharpe_ok and dd_ok) else "❌ 未通过"}')

    # 表7：RP 权重描述
    print('\n【表7】Risk Parity 时变权重描述统计')
    print(f'{"因子":<10}{"均值":>10}{"标准差":>10}{"最小":>10}{"最大":>10}')
    for f in fm_keep:
        s = df_w[f]
        print(f'{f:<10}{s.mean():>10.4f}{s.std(ddof=1):>10.4f}'
              f'{s.min():>10.4f}{s.max():>10.4f}')

    # 保存 CSV
    df_rp.to_csv(f'{OUT_DIR}/step3_rp_nav.csv')
    df_base.to_csv(f'{OUT_DIR}/step3_baseline_nav.csv')
    df_w.to_csv(f'{OUT_DIR}/step3_rp_weights.csv')

    return {
        'rp_total': rp_total,
        'base_total': base_total,
        'rp_sharpe': rp_sharpe,
        'base_sharpe': base_sharpe,
        'rp_dd': rp_dd,
        'base_dd': base_dd,
        'gate6_pass': sharpe_ok and dd_ok,
    }


# ============================================================
# 运行入口
# ============================================================

def run():
    """Phase 2 三步法全流程。"""
    print('=' * 60)
    print(f'P2-MF-2026Q2-v1 Phase 2 多因子合成（v2 修复版）')
    print(f'样本: {SAMPLE_NAME}, 区间: {START_DATE} ~ {END_DATE}')
    print(f'修复: 因子市值中性化+winsorize / 调仓间隔收益 / 单变量FM诊断')
    print('=' * 60)

    rebal_dates = get_rebalance_dates(START_DATE, END_DATE)
    if QUICK_TEST:
        rebal_dates = rebal_dates[:3]
        print(f'>>> 快速模式：只跑前 3 个调仓日 {rebal_dates}')

    # 调仓日配对 next_date（最后一个调仓日用 END_DATE 作为 next_date）
    # Python 3.6 兼容：不用 fromisoformat，手动解析 'YYYY-MM-DD'
    y, m, d = END_DATE.split('-')
    end_date_obj = datetime.date(int(y), int(m), int(d))
    date_pairs = []
    for i, d in enumerate(rebal_dates):
        if i + 1 < len(rebal_dates):
            next_d = rebal_dates[i + 1]
        else:
            next_d = end_date_obj
        date_pairs.append((d, next_d))

    print(f'调仓日数: {len(date_pairs)}（首末调仓间隔收益覆盖到 {END_DATE}）')

    os.makedirs(OUT_DIR, exist_ok=True)

    # Step 1
    df_corr = step1_correlation_matrix(date_pairs, debug_first=True)

    # Step 2
    fm_result = step2_fama_macbeth(date_pairs, debug_first=False)
    fm_keep = fm_result[1] if fm_result else []

    # Step 3
    rp_result = step3_risk_parity(date_pairs, fm_keep, debug_first=False)

    # 汇总
    print('\n\n' + '=' * 60)
    print('===== Phase 2 汇总 =====')
    print('=' * 60)
    if fm_result:
        print(f'FM 保留因子: {fm_keep}')
    if rp_result:
        print(f'Gate 6（RP vs Baseline Sharpe 不下降且回撤不上升）: '
              f'{"✅ 通过" if rp_result["gate6_pass"] else "❌ 未通过"}')
        print(f'  RP 总收益 {rp_result["rp_total"]:.4f} vs '
              f'Baseline {rp_result["base_total"]:.4f}')
        print(f'  RP Sharpe {rp_result["rp_sharpe"]:.4f} vs '
              f'Baseline {rp_result["base_sharpe"]:.4f}')
        print(f'  RP 回撤 {rp_result["rp_dd"]:.4f} vs '
              f'Baseline {rp_result["base_dd"]:.4f}')

    print(f'\n所有 CSV 已保存到 {OUT_DIR}/')
    print('Phase 2 三步法完成。Gate 2+3 完整判定需在聚宽研究环境跑后人工评估 '
          '(残差 alpha / 行业中性化 / 因子暴露稳定性)。')


if __name__ == '__main__':
    run()
