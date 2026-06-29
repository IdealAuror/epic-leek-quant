"""
P3-ST-2026Q2-v1 Phase 3 压力测试+仓位管理（standalone 版）
============================================================

4 模块并行：
  Module 1: Purged Walk-Forward CV（防泄漏交叉验证）
  Module 2: 制度断点分段（4 个 A 股断点）
  Module 3: DSR 多重比较修正
  Module 4: 目标波动率仓位管理（新增，基于 F5 信号）

Baseline: P1-F2F5-EP-LowVol-2026Q2-v1（F2+F5 等权 z-score，收益 372.71%，Sharpe 0.37，回撤 47.62%）

Gate 4 通过标准（预注册锁定）：
  M1: 样本外 Sharpe ≥ 0.2 AND 样本外/内 Sharpe ≥ 0.5
  M2: 任一断点后年化不转负 AND Sharpe 下降 < 50% AND Alpha 方向不反转
  M3: DSR ≥ 0.95
  M4: V2 回撤 < 35% AND 年化下降 < 3pp AND Sharpe 不下降

在聚宽【研究环境】中直接粘贴运行。

【符号约定】
  F2_ep:  市值中性化+winsorize 后的 EP（越高=越便宜=好）
  F5_vol: 市值中性化+winsorize 后的 -vol_60d（越高=越低波动=好）
"""

import datetime
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

# ============================================================
# 聚宽对象解析
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
SAMPLE_NAME = 'AllA'

# 因子参数
VOL_LOOKBACK = 60
VOL_MIN_OBS_RATIO = 0.5

# Walk-Forward CV
WF_TRAIN_MONTHS = 60       # 5 年训练窗
WF_TEST_MONTHS = 12        # 1 年验证窗
WF_PURGE_DAYS = 63         # 约 3 个月 purging
WF_EMBARGO_DAYS = 21       # 约 1 个月 embargoing
WF_STEP_MONTHS = 12        # 滚动步长

# 目标波动率仓位管理
TARGET_VOL = 0.15          # 目标年化波动率 15%
TARGET_VOL_CONSERVATIVE = 0.12
MAX_LEVERAGE = 1.0         # 不加杠杆
F5_OVERLAY_THRESHOLD = 1.0 # F5 信号 z-score 阈值
F5_OVERLAY_PENALTY = 0.5   # 高波动时额外减仓 50%

# 多重检验
N_TESTS = 6  # F1-F5 + F2+F5 组合

# 制度断点
BREAKPOINTS = [
    {'date': '2017-06-01', 'name': 'MSCI 纳入 A 股'},
    {'date': '2019-06-01', 'name': '注册制科创板开板'},
    {'date': '2024-02-01', 'name': '量化 DMA 踩踏'},
    {'date': '2024-04-01', 'name': '新国九条'},
]

# 输出
OUT_DIR = 'results/P3-ST-2026Q2-v1'

# QUICK_TEST=True 只跑前 3 个调仓日验证数据链
QUICK_TEST = False

# 金融股剔除
_FINANCE_NAMES = {'银行I', '非银金融I'}


# ============================================================
# 内联 data_layer（与 P2 脚本一致）
# ============================================================

def _get_stock_pool(index_id, date_str, min_listed_days=180):
    if index_id is None:
        sec_df = get_all_securities(['stock'], date=date_str)
        stocks = list(sec_df.index)
    else:
        stocks = get_index_stocks(index_id, date=date_str)
    if stocks:
        st_df = get_extras('is_st', stocks, end_date=date_str, count=1)
        if st_df is not None and not st_df.empty:
            st_today = st_df.iloc[-1]
            stocks = [s for s in stocks
                      if s in st_today.index and not st_today[s]]
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
    if stocks:
        stocks = _exclude_finance_stocks(stocks, date_str)
    return stocks


def _exclude_finance_stocks(stocks, date_str):
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
    for c in candidates:
        if c in df.columns:
            return df[c]
    return None


def _fetch_fundamentals_pit(date_str, stocks):
    pieces = []
    try:
        q_val = query(valuation.code, valuation.market_cap)
        df_val = get_fundamentals(q_val, date=date_str)
        if df_val is not None and not df_val.empty:
            if 'code' in df_val.columns:
                df_val = df_val.set_index('code')
            pieces.append(df_val)
    except Exception:
        pass
    try:
        df_bal = get_fundamentals(query(balance), date=date_str)
        if df_bal is not None and not df_bal.empty:
            if 'code' in df_bal.columns:
                df_bal = df_bal.set_index('code')
            pieces.append(df_bal)
    except Exception:
        pass
    try:
        df_inc = get_fundamentals(query(income), date=date_str)
        if df_inc is not None and not df_inc.empty:
            if 'code' in df_inc.columns:
                df_inc = df_inc.set_index('code')
            pieces.append(df_inc)
    except Exception:
        pass
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


def _calc_realized_volatility(date_str, stocks, lookback_days=VOL_LOOKBACK):
    if not stocks:
        return {}
    try:
        df_px = get_price(stocks, end_date=date_str, count=lookback_days + 1,
                          fields=['close'], skip_paused=False, panel=False, fq='post')
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
        wide = df_px.pivot_table(index=df_px.index,
                                 columns='code', values='close')
    except Exception:
        return {}
    rets = wide.pct_change().iloc[1:]
    min_obs = int(lookback_days * VOL_MIN_OBS_RATIO)
    valid_count = rets.notna().sum()
    valid_codes = valid_count[valid_count >= min_obs].index
    if len(valid_codes) == 0:
        return {}
    vols = rets[valid_codes].std(ddof=1)
    return dict(vols)


# ============================================================
# 工具函数
# ============================================================

def get_rebalance_dates(start, end):
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


def neutralize_ols(factor_values, regressor):
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
    s = pd.Series(s, dtype=float)
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return s * 0
    return (s - mu) / sd


def winsorize_cross_section(s, lower=0.01, upper=0.99):
    s = pd.Series(s, dtype=float)
    if s.notna().sum() < 10:
        return s
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lower=lo, upper=hi)


# ============================================================
# 价格缓存
# ============================================================
_PRICE_CACHE = {}


def _load_period_prices(start_str, end_str, codes):
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


def forward_period_return(codes, date_str, next_date_str):
    px = _load_period_prices(date_str, next_date_str, codes)
    if px is None or px.empty:
        return None
    try:
        px.index = pd.to_datetime(px.index)
    except Exception:
        return None
    d_ts = pd.Timestamp(date_str)
    nd_ts = pd.Timestamp(next_date_str)
    valid_start = px.index[px.index >= d_ts]
    if len(valid_start) == 0:
        return None
    close_start = px.loc[valid_start[0]]
    valid_end = px.index[px.index <= nd_ts]
    if len(valid_end) == 0:
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
    return ce / cs - 1


def get_industry_map(stocks, date_str):
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
# 横截面构建（F2+F5 双因子，与 Phase 1 Baseline 一致）
# ============================================================

def build_cross_section(date, next_date=None, debug=False):
    """构建横截面，计算 F2_ep + F5_vol（市值中性化+winsorize 后）。

    与 Phase 1 F2+F5 Baseline 口径一致。
    """
    date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
    stocks = _get_stock_pool(None, date_str)
    if not stocks:
        return None

    df = _fetch_fundamentals_pit(date_str, stocks)
    if df is None or df.empty:
        return None

    mcap = _get_col(df, 'market_cap')
    np_col = _get_col(df, 'net_profit', 'np_parent_company_owners',
                      'net_profit_is_parent_company')
    if mcap is None or np_col is None:
        return None
    df['market_cap'] = mcap.astype(float)
    df['net_profit'] = np_col.astype(float)
    df['F2_ep_raw'] = df['net_profit'] / (df['market_cap'] * 1e8)

    mask = df['net_profit'].fillna(0) > 0
    tl = _get_col(df, 'total_liability', 'total_liabilities')
    ta = _get_col(df, 'total_assets')
    if tl is not None and ta is not None:
        df['debt_to_assets'] = tl.astype(float) / ta.astype(float).replace(0, np.nan)
        mask &= df['debt_to_assets'].fillna(0.5) <= 1.0

    vol_map = _calc_realized_volatility(date_str, list(df.index), VOL_LOOKBACK)
    df['vol_60d'] = df.index.map(lambda c: vol_map.get(c, np.nan))
    df['F5_vol_raw'] = -df['vol_60d']

    df = df.dropna(subset=['F2_ep_raw', 'F5_vol_raw', 'market_cap']).copy()
    if len(df) < 30:
        return None

    # 市值中性化 + winsorize
    log_mcap = np.log(df['market_cap'].astype(float).replace(0, np.nan))
    for raw_col, neut_col in [('F2_ep_raw', 'F2_ep'), ('F5_vol_raw', 'F5_vol')]:
        raw = df[raw_col].astype(float)
        neut = neutralize_ols(raw.values, log_mcap.values)
        neut = winsorize_cross_section(pd.Series(neut, index=df.index))
        df[neut_col] = neut

    if next_date is None:
        return df
    next_date_str = next_date.strftime('%Y-%m-%d') if hasattr(next_date, 'strftime') else str(next_date)
    codes = list(df.index)
    fwd = forward_period_return(codes, date_str, next_date_str)
    if fwd is None:
        return None
    df['fwd_return'] = df.index.map(fwd)
    df = df.dropna(subset=['fwd_return'])
    if len(df) < 30:
        return None
    return df


def build_portfolio(df, f2_weight=0.5, f5_weight=0.5, n_hold=50):
    """构建 F2+F5 加权组合，返回 (持仓 codes, 持仓 weights)。"""
    df = df.copy()
    df['F2_z'] = zscore_cross_section(df['F2_ep'])
    df['F5_z'] = zscore_cross_section(df['F5_vol'])
    df['score'] = f2_weight * df['F2_z'] + f5_weight * df['F5_z']
    n = min(n_hold, len(df) // 5)
    n = max(n, 10)
    df_sel = df.nlargest(n, 'score')
    weights = zscore_cross_section(df_sel['score']).clip(lower=0)
    if weights.sum() > 0:
        weights = weights / weights.sum()
    return list(df_sel.index), weights


# ============================================================
# Module 1: Purged Walk-Forward CV
# ============================================================

def module_1_walk_forward(date_pairs, debug_first=True):
    """Purged Walk-Forward CV：验证 F2+F5 Baseline 是否过拟合。"""
    print('\n' + '=' * 60)
    print('===== Module 1: Purged Walk-Forward CV =====')
    print('=' * 60)

    # 参数网格（预注册）
    param_grid = [
        {'n_hold': 30, 'f5_weight': 0.3},
        {'n_hold': 50, 'f5_weight': 0.5},  # Baseline
        {'n_hold': 100, 'f5_weight': 0.7},
    ]

    wf_records = []
    rebal_dates = [d for d, _ in date_pairs]

    # 滚动窗口
    i = 0
    while i + WF_TRAIN_MONTHS // 4 < len(rebal_dates):  # 训练窗约 5 年（15 个季度调仓日）
        train_end_idx = i + WF_TRAIN_MONTHS // 4
        test_start_idx = train_end_idx + 1  # +1 季度 purging
        test_end_idx = min(test_start_idx + WF_TEST_MONTHS // 4, len(rebal_dates) - 1)

        if test_end_idx <= test_start_idx:
            break

        train_dates = rebal_dates[i:train_end_idx + 1]
        test_dates = rebal_dates[test_start_idx:test_end_idx + 1]

        if len(train_dates) < 8 or len(test_dates) < 2:
            i += WF_STEP_MONTHS // 4
            continue

        # 训练窗：每个参数组合计算样本内 Sharpe
        in_sample_sharpes = {}
        for params in param_grid:
            rets = []
            for j, d in enumerate(train_dates):
                if j + 1 >= len(train_dates):
                    continue
                next_d = train_dates[j + 1] if j + 1 < len(train_dates) else date_pairs[train_dates.index(d)][1]
                df = build_cross_section(d, next_d, debug=False)
                if df is None or len(df) < 30:
                    continue
                codes, weights = build_portfolio(
                    df, f2_weight=1-params['f5_weight'],
                    f5_weight=params['f5_weight'], n_hold=params['n_hold'])
                if not codes:
                    continue
                df_sel = df[df.index.isin(codes)]
                ret = float((weights * df_sel['fwd_return']).sum())
                rets.append(ret)
            if len(rets) >= 3:
                sharpe_in = np.mean(rets) / (np.std(rets, ddof=1) + 1e-10) * np.sqrt(3)
                in_sample_sharpes[str(params)] = sharpe_in

        if not in_sample_sharpes:
            i += WF_STEP_MONTHS // 4
            continue

        # 选样本内最优参数
        best_params_str = max(in_sample_sharpes, key=in_sample_sharpes.get)
        best_params = eval(best_params_str)
        sharpe_in = in_sample_sharpes[best_params_str]

        # 验证窗：用最优参数计算样本外 Sharpe
        rets_out = []
        for j, d in enumerate(test_dates):
            idx = rebal_dates.index(d)
            next_d = date_pairs[idx][1]
            df = build_cross_section(d, next_d, debug=False)
            if df is None or len(df) < 30:
                continue
            codes, weights = build_portfolio(
                df, f2_weight=1-best_params['f5_weight'],
                f5_weight=best_params['f5_weight'], n_hold=best_params['n_hold'])
            if not codes:
                continue
            df_sel = df[df.index.isin(codes)]
            ret = float((weights * df_sel['fwd_return']).sum())
            rets_out.append(ret)

        if len(rets_out) >= 2:
            sharpe_out = np.mean(rets_out) / (np.std(rets_out, ddof=1) + 1e-10) * np.sqrt(3)
            wf_records.append({
                'train_start': str(train_dates[0]),
                'train_end': str(train_dates[-1]),
                'test_start': str(test_dates[0]),
                'test_end': str(test_dates[-1]),
                'best_params': best_params_str,
                'sharpe_in': sharpe_in,
                'sharpe_out': sharpe_out,
                'n_test': len(rets_out),
            })
            print(f'  [M1] {train_dates[0]}~{train_dates[-1]} → '
                  f'{test_dates[0]}~{test_dates[-1]}: '
                  f'in={sharpe_in:.3f} out={sharpe_out:.3f} '
                  f'params={best_params_str}')

        i += WF_STEP_MONTHS // 4

    if not wf_records:
        print('  无有效数据')
        return None

    df_wf = pd.DataFrame(wf_records)

    # 表1：样本内 vs 样本外
    print('\n【表1】Walk-Forward CV 样本内 vs 样本外 Sharpe')
    print(f'{"训练期":<28}{"验证期":<28}{"样本内":>10}{"样本外":>10}{"比率":>8}')
    for _, r in df_wf.iterrows():
        ratio = r['sharpe_out'] / r['sharpe_in'] if r['sharpe_in'] != 0 else 0
        print(f'{r["train_start"]+"~"+r["train_end"]:<28}'
              f'{r["test_start"]+"~"+r["test_end"]:<28}'
              f'{r["sharpe_in"]:>10.3f}{r["sharpe_out"]:>10.3f}{ratio:>8.2f}')

    # 汇总统计
    mean_in = df_wf['sharpe_in'].mean()
    mean_out = df_wf['sharpe_out'].mean()
    ratio = mean_out / mean_in if mean_in != 0 else 0
    print(f'\n  样本内 Sharpe 均值: {mean_in:.3f}')
    print(f'  样本外 Sharpe 均值: {mean_out:.3f}')
    print(f'  样本外/内 比率: {ratio:.2f}')

    # Gate 4 模块 1 判定
    print('\n【Gate 4 模块 1 判定】')
    m1_pass_sharpe = mean_out >= 0.2
    m1_pass_ratio = ratio >= 0.5
    # 参数稳定性：最优参数非边际极端值
    param_set = set(df_wf['best_params'])
    extreme = "{'n_hold': 30, 'f5_weight': 0.3}" in param_set and len(param_set) == 1
    m1_pass_params = not extreme
    print(f'  样本外 Sharpe ≥ 0.2: {"✅" if m1_pass_sharpe else "❌"} ({mean_out:.3f})')
    print(f'  样本外/内 ≥ 0.5: {"✅" if m1_pass_ratio else "❌"} ({ratio:.2f})')
    print(f'  参数非边际极端值: {"✅" if m1_pass_params else "❌"} (参数集: {param_set})')
    m1_pass = m1_pass_sharpe and m1_pass_ratio and m1_pass_params
    print(f'  模块 1: {"✅ 通过" if m1_pass else "❌ 未通过"}')

    df_wf.to_csv(f'{OUT_DIR}/module1_walk_forward.csv', index=False)
    return {'df': df_wf, 'pass': m1_pass, 'mean_out_sharpe': mean_out, 'ratio': ratio}


# ============================================================
# Module 2: 制度断点分段
# ============================================================

def module_2_breakpoints(date_pairs, debug_first=True):
    """按 A 股制度断点分段，验证跨断点稳健性。"""
    print('\n' + '=' * 60)
    print('===== Module 2: 制度断点分段 =====')
    print('=' * 60)

    results = []
    for bp in BREAKPOINTS:
        bp_date = pd.Timestamp(bp['date'])
        # 断点前后各 2 年
        pre_start = bp_date - pd.Timedelta(days=730)
        post_end = bp_date + pd.Timedelta(days=730)

        pre_rets = []
        post_rets = []
        for date, next_date in date_pairs:
            d_ts = pd.Timestamp(date)
            if d_ts < pre_start or d_ts >= post_end:
                continue
            df = build_cross_section(date, next_date, debug=False)
            if df is None or len(df) < 30:
                continue
            codes, weights = build_portfolio(df, f2_weight=0.5, f5_weight=0.5, n_hold=50)
            if not codes:
                continue
            df_sel = df[df.index.isin(codes)]
            ret = float((weights * df_sel['fwd_return']).sum())
            if d_ts < bp_date:
                pre_rets.append(ret)
            else:
                post_rets.append(ret)

        pre_sharpe = (np.mean(pre_rets) / (np.std(pre_rets, ddof=1) + 1e-10) * np.sqrt(3)
                      if len(pre_rets) >= 2 else 0)
        post_sharpe = (np.mean(post_rets) / (np.std(post_rets, ddof=1) + 1e-10) * np.sqrt(3)
                       if len(post_rets) >= 2 else 0)
        pre_ann = (1 + np.mean(pre_rets)) ** 3 - 1 if pre_rets else 0
        post_ann = (1 + np.mean(post_rets)) ** 3 - 1 if post_rets else 0

        # 回撤
        def max_dd(rets):
            if not rets:
                return 0
            nav = np.cumprod([1 + r for r in rets])
            return max(1 - n / max(nav[:i+1]) for i, n in enumerate(nav))

        pre_dd = max_dd(pre_rets)
        post_dd = max_dd(post_rets)

        results.append({
            'breakpoint': bp['name'],
            'date': bp['date'],
            'pre_n': len(pre_rets),
            'post_n': len(post_rets),
            'pre_ann': pre_ann,
            'post_ann': post_ann,
            'pre_sharpe': pre_sharpe,
            'post_sharpe': post_sharpe,
            'pre_dd': pre_dd,
            'post_dd': post_dd,
        })
        print(f'  [M2] {bp["name"]} ({bp["date"]}): '
              f'前 ann={pre_ann:.4f} Sharpe={pre_sharpe:.3f} dd={pre_dd:.3f} | '
              f'后 ann={post_ann:.4f} Sharpe={post_sharpe:.3f} dd={post_dd:.3f}')

    if not results:
        print('  无有效数据')
        return None

    df_bp = pd.DataFrame(results)

    # 表3：断点前后表现
    print('\n【表3】制度断点前后表现对照')
    print(f'{"断点":<22}{"前段年化":>10}{"后段年化":>10}{"前Sharpe":>10}{"后Sharpe":>10}{"前回撤":>10}{"后回撤":>10}')
    for _, r in df_bp.iterrows():
        print(f'{r["breakpoint"]:<22}{r["pre_ann"]:>10.4f}{r["post_ann"]:>10.4f}'
              f'{r["pre_sharpe"]:>10.3f}{r["post_sharpe"]:>10.3f}'
              f'{r["pre_dd"]:>10.3f}{r["post_dd"]:>10.3f}')

    # Gate 4 模块 2 判定
    print('\n【Gate 4 模块 2 判定】')
    m2_pass_ann = all(r['post_ann'] > 0 for r in results)
    m2_pass_sharpe = all(
        r['post_sharpe'] >= r['pre_sharpe'] * 0.5 if r['pre_sharpe'] > 0
        else r['post_sharpe'] > 0
        for r in results)
    m2_pass_alpha = all(
        np.sign(r['pre_ann']) == np.sign(r['post_ann']) or r['post_ann'] > 0
        for r in results)
    print(f'  断点后年化不转负: {"✅" if m2_pass_ann else "❌"}')
    print(f'  断点后 Sharpe 下降 < 50%: {"✅" if m2_pass_sharpe else "❌"}')
    print(f'  Alpha 方向不反转: {"✅" if m2_pass_alpha else "❌"}')
    m2_pass = m2_pass_ann and m2_pass_sharpe and m2_pass_alpha
    print(f'  模块 2: {"✅ 通过" if m2_pass else "❌ 未通过"}')

    df_bp.to_csv(f'{OUT_DIR}/module2_breakpoints.csv', index=False)
    return {'df': df_bp, 'pass': m2_pass}


# ============================================================
# Module 3: DSR 多重比较修正
# ============================================================

def module_3_dsr(date_pairs, observed_sharpe=None, debug_first=True):
    """Deflated Sharpe Ratio 多重比较修正。"""
    print('\n' + '=' * 60)
    print('===== Module 3: DSR 多重比较修正 =====')
    print('=' * 60)

    # 若未提供 observed_sharpe，用 Baseline 跑一遍计算
    if observed_sharpe is None:
        rets = []
        for date, next_date in date_pairs:
            df = build_cross_section(date, next_date, debug=False)
            if df is None or len(df) < 30:
                continue
            codes, weights = build_portfolio(df, f2_weight=0.5, f5_weight=0.5, n_hold=50)
            if not codes:
                continue
            df_sel = df[df.index.isin(codes)]
            ret = float((weights * df_sel['fwd_return']).sum())
            rets.append(ret)
        if len(rets) < 2:
            print('  无有效数据')
            return None
        sr_observed = np.mean(rets) / (np.std(rets, ddof=1) + 1e-10) * np.sqrt(3)
    else:
        sr_observed = observed_sharpe
        rets = [0]  # 占位

    # DSR 计算（López de Prado 2014）
    # SR_observed: 实测 Sharpe（年化）
    # N: 试验次数 = 6（F1-F5 + F2+F5 组合）
    # T: 样本长度（月数）
    # skew/kurt of returns
    T = len(date_pairs) * 4  # 估算月数（每调仓日约 4 个月）
    N = N_TESTS
    rets_arr = np.array(rets)
    skew = stats.skew(rets_arr) if len(rets_arr) >= 3 else 0
    kurt = stats.kurtosis(rets_arr, fisher=False) if len(rets_arr) >= 3 else 3

    # 期望最大 Sharpe（多重检验下）
    # E[max(SR)] ≈ sqrt(2*log(N)) * sigma_SR（极端值理论近似）
    # 更精确：用正态分布次序统计量
    # SR 的标准误：SE(SR) = sqrt((1 - skew*SR + (kurt-1)/4 * SR^2) / T)
    # 用年化 SR，T 用月数
    sr_monthly = sr_observed / np.sqrt(12)  # 转月度
    se_sr = np.sqrt((1 - skew * sr_monthly + (kurt - 1) / 4 * sr_monthly ** 2) / T)

    # 期望最大 Sharpe（按正态分布次序统计量近似）
    # E[max(SR)] = sigma_SR * E[max of N standard normals]
    # E[max of N standard normals] ≈ sqrt(2*ln(N)) for large N
    if N > 1:
        e_max_sr = np.sqrt(2 * np.log(N)) * se_sr
    else:
        e_max_sr = 0

    # DSR = P(SR_true > 0 | SR_observed, multiple testing)
    # = Φ((SR_observed - E[max(SR)]) / SE(SR))
    dsr = stats.norm.cdf((sr_monthly - e_max_sr) / se_sr) if se_sr > 0 else 0.5

    # 表4：DSR 计算明细
    print('\n【表4】DSR 计算明细')
    print(f'  实测 Sharpe（年化）: {sr_observed:.4f}')
    print(f'  实测 Sharpe（月度）: {sr_monthly:.4f}')
    print(f'  试验次数 N: {N}')
    print(f'  样本长度 T（月）: {T}')
    print(f'  收益偏度: {skew:.4f}')
    print(f'  收益峰度: {kurt:.4f}')
    print(f'  Sharpe 标准误: {se_sr:.4f}')
    print(f'  期望最大 Sharpe（多重检验）: {e_max_sr:.4f}')
    print(f'  DSR: {dsr:.4f}')

    # Gate 4 模块 3 判定
    print('\n【Gate 4 模块 3 判定】')
    m3_pass = dsr >= 0.95
    print(f'  DSR ≥ 0.95: {"✅" if m3_pass else "❌"} ({dsr:.4f})')
    print(f'  模块 3: {"✅ 通过" if m3_pass else "❌ 未通过"}')

    # 保存
    df_dsr = pd.DataFrame([{
        'sr_observed': sr_observed,
        'sr_monthly': sr_monthly,
        'N': N, 'T': T, 'skew': skew, 'kurt': kurt,
        'se_sr': se_sr, 'e_max_sr': e_max_sr, 'dsr': dsr,
    }])
    df_dsr.to_csv(f'{OUT_DIR}/module3_dsr.csv', index=False)
    return {'df': df_dsr, 'pass': m3_pass, 'dsr': dsr, 'sr_observed': sr_observed}


# ============================================================
# Module 4: 目标波动率仓位管理
# ============================================================

def module_4_target_volatility(date_pairs, debug_first=True):
    """目标波动率仓位管理：用 F5 信号动态调仓。"""
    print('\n' + '=' * 60)
    print('===== Module 4: 目标波动率仓位管理 =====')
    print('=' * 60)

    # 先跑 Baseline（无仓位管理）
    print('  [M4] 跑 Baseline（无仓位管理）...')
    base_rets = []
    base_market_vols = []  # 每期全市场波动信号
    for date, next_date in date_pairs:
        df = build_cross_section(date, next_date, debug=False)
        if df is None or len(df) < 30:
            continue
        codes, weights = build_portfolio(df, f2_weight=0.5, f5_weight=0.5, n_hold=50)
        if not codes:
            continue
        df_sel = df[df.index.isin(codes)]
        ret = float((weights * df_sel['fwd_return']).sum())
        base_rets.append({'date': date, 'ret': ret})

        # 全市场波动信号（横截面 mean(F5_vol) 的代理）
        # F5_vol = -vol_60d，越低=市场越波动
        market_vol_signal = float(df['F5_vol'].mean())
        base_market_vols.append({'date': date, 'vol_signal': market_vol_signal})

    if not base_rets:
        print('  无有效数据')
        return None

    df_base = pd.DataFrame(base_rets).set_index('date')
    df_vol = pd.DataFrame(base_market_vols).set_index('date')
    # 波动信号 z-score
    df_vol['vol_z'] = zscore_cross_section(df_vol['vol_signal'])

    # Baseline 指标
    base_total = (1 + df_base['ret']).prod() - 1
    if len(df_base) > 1:
        first_date = pd.to_datetime(df_base.index[0])
        last_date = pd.to_datetime(df_base.index[-1])
        years = max((last_date - first_date).days / 365.25, 0.1)
    else:
        years = 1.0
    base_ann = (1 + base_total) ** (1.0 / years) - 1
    base_vol = df_base['ret'].std() * np.sqrt(3)
    base_sharpe = base_ann / base_vol if base_vol > 0 else 0
    base_dd = (1 - (1 + df_base['ret']).cumprod() / (1 + df_base['ret']).cumprod().cummax()).max()

    print(f'  Baseline: total={base_total:.4f} ann={base_ann:.4f} '
          f'vol={base_vol:.4f} sharpe={base_sharpe:.4f} dd={base_dd:.4f}')

    # 三个变体
    variants = [
        {'id': 'V1', 'name': '纯目标波动率', 'target_vol': TARGET_VOL, 'f5_overlay': False},
        {'id': 'V2', 'name': 'F5 叠加目标波动率', 'target_vol': TARGET_VOL, 'f5_overlay': True},
        {'id': 'V3', 'name': '保守版（12%）', 'target_vol': TARGET_VOL_CONSERVATIVE, 'f5_overlay': True},
    ]

    variant_results = []
    for v in variants:
        print(f'\n  [M4] 跑 {v["id"]} ({v["name"]})...')
        v_rets = []
        v_weights = []
        for idx, row in df_base.iterrows():
            date = idx
            ret_base = row['ret']
            # 估算实现波动率（用过去 3 期收益标准差年化）
            past_rets = df_base.loc[:date, 'ret'].iloc[-6:]  # 过去 6 期
            if len(past_rets) >= 2:
                realized_vol = past_rets.std() * np.sqrt(3)
            else:
                realized_vol = base_vol
            realized_vol = max(realized_vol, 0.05)  # 下限 5%

            # 目标仓位
            target_weight = min(v['target_vol'] / realized_vol, MAX_LEVERAGE)

            # F5 信号叠加
            if v['f5_overlay'] and date in df_vol.index:
                vol_z = df_vol.loc[date, 'vol_z']
                if vol_z < -F5_OVERLAY_THRESHOLD:
                    # 市场高波动，额外减仓
                    target_weight *= F5_OVERLAY_PENALTY
                # vol_z > 1 时不额外加仓（避免追高）

            target_weight = max(min(target_weight, 1.0), 0.1)  # [0.1, 1.0]
            v_rets.append({'date': date, 'ret': ret_base * target_weight})
            v_weights.append({'date': date, 'weight': target_weight})

        df_v = pd.DataFrame(v_rets).set_index('date')
        df_w = pd.DataFrame(v_weights).set_index('date')

        v_total = (1 + df_v['ret']).prod() - 1
        v_ann = (1 + v_total) ** (1.0 / years) - 1
        v_vol = df_v['ret'].std() * np.sqrt(3)
        v_sharpe = v_ann / v_vol if v_vol > 0 else 0
        v_dd = (1 - (1 + df_v['ret']).cumprod() / (1 + df_v['ret']).cumprod().cummax()).max()

        variant_results.append({
            'id': v['id'], 'name': v['name'],
            'total': v_total, 'ann': v_ann, 'vol': v_vol,
            'sharpe': v_sharpe, 'dd': v_dd,
            'mean_weight': float(df_w['weight'].mean()),
            'min_weight': float(df_w['weight'].min()),
            'max_weight': float(df_w['weight'].max()),
            'df_rets': df_v, 'df_weights': df_w,
        })
        print(f'  {v["id"]}: total={v_total:.4f} ann={v_ann:.4f} '
              f'vol={v_vol:.4f} sharpe={v_sharpe:.4f} dd={v_dd:.4f} '
              f'weight mean={df_w["weight"].mean():.3f}')

    # 表5：对照
    print('\n【表5】目标波动率 V1/V2/V3 vs Baseline 收益风险对照')
    print(f'{"指标":<14}{"Baseline":>14}{"V1 纯TV":>14}{"V2 F5+TV":>14}{"V3 保守":>14}')
    print(f'{"总收益":<14}{base_total:>14.4f}'
          f'{variant_results[0]["total"]:>14.4f}'
          f'{variant_results[1]["total"]:>14.4f}'
          f'{variant_results[2]["total"]:>14.4f}')
    print(f'{"年化":<14}{base_ann:>14.4f}'
          f'{variant_results[0]["ann"]:>14.4f}'
          f'{variant_results[1]["ann"]:>14.4f}'
          f'{variant_results[2]["ann"]:>14.4f}')
    print(f'{"年化波动":<14}{base_vol:>14.4f}'
          f'{variant_results[0]["vol"]:>14.4f}'
          f'{variant_results[1]["vol"]:>14.4f}'
          f'{variant_results[2]["vol"]:>14.4f}')
    print(f'{"Sharpe":<14}{base_sharpe:>14.4f}'
          f'{variant_results[0]["sharpe"]:>14.4f}'
          f'{variant_results[1]["sharpe"]:>14.4f}'
          f'{variant_results[2]["sharpe"]:>14.4f}')
    print(f'{"最大回撤":<14}{base_dd:>14.4f}'
          f'{variant_results[0]["dd"]:>14.4f}'
          f'{variant_results[1]["dd"]:>14.4f}'
          f'{variant_results[2]["dd"]:>14.4f}')

    # 表6：仓位统计
    print('\n【表6】目标波动率仓位时序描述统计')
    print(f'{"变体":<10}{"均值":>10}{"标准差":>10}{"最小":>10}{"最大":>10}')
    for v in variant_results:
        print(f'{v["id"]:<10}{v["mean_weight"]:>10.3f}'
              f'{v["df_weights"]["weight"].std():>10.3f}'
              f'{v["min_weight"]:>10.3f}{v["max_weight"]:>10.3f}')

    # Gate 4 模块 4 判定（以 V2 为准）
    print('\n【Gate 4 模块 4 判定】（以 V2 F5+TV 为准）')
    v2 = variant_results[1]
    m4_pass_dd = v2['dd'] < 0.35
    m4_pass_ann = base_ann - v2['ann'] < 0.03
    m4_pass_sharpe = v2['sharpe'] >= base_sharpe
    print(f'  V2 回撤 < 35%: {"✅" if m4_pass_dd else "❌"} ({v2["dd"]:.4f})')
    print(f'  V2 年化下降 < 3pp: {"✅" if m4_pass_ann else "❌"} '
          f'({base_ann - v2["ann"]:.4f})')
    print(f'  V2 Sharpe 不下降: {"✅" if m4_pass_sharpe else "❌"} '
          f'({v2["sharpe"]:.4f} vs {base_sharpe:.4f})')
    m4_pass = m4_pass_dd and m4_pass_ann and m4_pass_sharpe
    print(f'  模块 4: {"✅ 通过" if m4_pass else "❌ 未通过"}')

    # 保存
    df_base.to_csv(f'{OUT_DIR}/module4_baseline_rets.csv')
    for v in variant_results:
        v['df_rets'].to_csv(f'{OUT_DIR}/module4_{v["id"]}_rets.csv')
        v['df_weights'].to_csv(f'{OUT_DIR}/module4_{v["id"]}_weights.csv')

    return {
        'base': {'total': base_total, 'ann': base_ann, 'sharpe': base_sharpe, 'dd': base_dd},
        'variants': variant_results,
        'pass': m4_pass,
        'v2_dd': v2['dd'],
        'v2_ann_drop': base_ann - v2['ann'],
    }


# ============================================================
# 运行入口
# ============================================================

def run():
    """Phase 3 四模块全流程。"""
    print('=' * 60)
    print(f'P3-ST-2026Q2-v1 Phase 3 压力测试+仓位管理')
    print(f'样本: {SAMPLE_NAME}, 区间: {START_DATE} ~ {END_DATE}')
    print(f'4 模块: Walk-Forward + 断点 + DSR + 目标波动率')
    print('=' * 60)

    rebal_dates = get_rebalance_dates(START_DATE, END_DATE)
    if QUICK_TEST:
        rebal_dates = rebal_dates[:3]
        print(f'>>> 快速模式：只跑前 3 个调仓日 {rebal_dates}')

    # 调仓日配对 next_date
    y, m, d = END_DATE.split('-')
    end_date_obj = datetime.date(int(y), int(m), int(d))
    date_pairs = []
    for i, d in enumerate(rebal_dates):
        if i + 1 < len(rebal_dates):
            next_d = rebal_dates[i + 1]
        else:
            next_d = end_date_obj
        date_pairs.append((d, next_d))

    print(f'调仓日数: {len(date_pairs)}')

    os.makedirs(OUT_DIR, exist_ok=True)

    # Module 1: Walk-Forward CV
    m1 = module_1_walk_forward(date_pairs, debug_first=True)

    # Module 2: 制度断点
    m2 = module_2_breakpoints(date_pairs, debug_first=False)

    # Module 3: DSR
    m3 = module_3_dsr(date_pairs, debug_first=False)

    # Module 4: 目标波动率仓位管理
    m4 = module_4_target_volatility(date_pairs, debug_first=False)

    # 汇总
    print('\n\n' + '=' * 60)
    print('===== Phase 3 汇总 =====')
    print('=' * 60)
    print('\n【Gate 4 总判定】')
    modules_pass = 0
    if m1 and m1['pass']:
        modules_pass += 1
        print(f'  M1 Walk-Forward: ✅ 通过 (样本外 Sharpe={m1["mean_out_sharpe"]:.3f})')
    else:
        print(f'  M1 Walk-Forward: ❌ 未通过')

    if m2 and m2['pass']:
        modules_pass += 1
        print(f'  M2 制度断点: ✅ 通过')
    else:
        print(f'  M2 制度断点: ❌ 未通过')

    if m3 and m3['pass']:
        modules_pass += 1
        print(f'  M3 DSR: ✅ 通过 (DSR={m3["dsr"]:.4f})')
    else:
        print(f'  M3 DSR: ❌ 未通过 (DSR={m3["dsr"]:.4f})' if m3 else f'  M3 DSR: ❌ 未通过')

    if m4 and m4['pass']:
        modules_pass += 1
        print(f'  M4 目标波动率: ✅ 通过 (V2 回撤={m4["v2_dd"]:.4f})')
    else:
        v2_dd = m4['v2_dd'] if m4 else 0
        v2_drop = m4['v2_ann_drop'] if m4 else 0
        print(f'  M4 目标波动率: ❌ 未通过 (V2 回撤={v2_dd:.4f}, 年化下降={v2_drop:.4f})')

    print(f'\n  通过模块数: {modules_pass}/4')

    if modules_pass == 4:
        print('\n  >>> Phase 3 通过，进 Phase 4 实盘校准')
    elif modules_pass == 3:
        print('\n  >>> Phase 3 部分通过（3/4），策略稳健但需关注未通过模块')
        print('  >>> 进 Phase 4 但限制初始规模')
    else:
        print('\n  >>> Phase 3 失败（≤2/4），策略不稳健，停止迭代')

    print(f'\n所有 CSV 已保存到 {OUT_DIR}/')


if __name__ == '__main__':
    run()
