"""
P4-PL-2026Q2-v1 Phase 4 实盘校准（standalone 版）
============================================================

5 模块：
  Module 1: 成本后净收益（毛 vs 净）
  Module 2: 容量估算（5000 万资金冲击成本）
  Module 3: 因子拥挤度监测（F2/F5 Z-score 均值时序）
  Module 4: 换手-alpha 闭环（边际信息收益 vs 边际交易成本）
  Module 5: 仓位管理改进（最大回撤约束 + 动态止损，替代目标波动率）

Baseline: P1-F2F5-EP-LowVol-2026Q2-v1（日频口径，收益 372.71%，Sharpe 0.37，回撤 47.62%）
目标资金: 5000 万（限制初始规模）
收益口径: 日频（与 Phase 1 一致，暴露真实回撤）

Gate 5 通过标准（预注册锁定）：
  M1: 成本后年化净收益 > 5% AND Sharpe > 0.3 AND 成本占比 < 30%
  M2: 5000万冲击成本 < 净收益20% AND 持仓市值/日均成交额>20%的股票数 < 5
  M3: F2/F5 拥挤度 < 历史 90 分位
  M4: 边际信息收益 ≥ 边际交易成本
  M5: V2 回撤 < 30% AND 年化下降 < 5pp AND Sharpe 不下降

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

# 成本模型
COMMISSION = 0.0003           # 万三佣金（双边）
STAMP_DUTY = 0.001            # 印花税（卖出单边）
SLIPPAGE = 0.001              # 千一滑点（双边）
BASE_ROUND_TRIP = 0.0026      # 基础双边成本 = 0.0003*2 + 0.001 + 0.001
IMPACT_COEFF = 0.1            # 线性冲击成本系数

# 资金规模
TARGET_CAPITAL = 50000000     # 5000 万
POSITION_LIMIT = 0.10         # 个股 10% 上限
LIQUIDITY_THRESHOLD = 5000000 # 日均成交额 ≥ 500 万

# 仓位管理（最大回撤约束 + 动态止损）
DD_THRESHOLD_1 = 0.10         # 回撤 > 10% → 仓位 70%
DD_THRESHOLD_2 = 0.15         # 回撤 > 15% → 仓位 50%
DD_THRESHOLD_3 = 0.20         # 回撤 > 20% → 仓位 30%
DD_STOPLOSS = 0.25            # 回撤 > 25% → 止损，仓位 20%
DD_RECOVERY = 0.05            # 从低点反弹 5% → 恢复加仓
STOPLOSS_COOLDOWN = 63        # 止损后 3 个月（63 交易日）观察期
F5_OVERLAY_THRESHOLD = -0.5   # F5 信号 z-score 阈值（修复 Phase 3 bug，从 -1 调到 -0.5）
F5_OVERLAY_PENALTY = 0.8      # 高波动时额外减仓 20%

# 输出
OUT_DIR = 'results/P4-PL-2026Q2-v1'

# QUICK_TEST=True 只跑前 3 个调仓日验证数据链
QUICK_TEST = False

# 金融股剔除
_FINANCE_NAMES = {'银行I', '非银金融I'}


# ============================================================
# 内联 data_layer（与 P3 脚本一致）
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
        q_val = query(valuation.code, valuation.market_cap, valuation.circulating_market_cap)
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


def _calc_avg_volume(date_str, stocks, lookback_days=20):
    """计算日均成交额（用于流动性过滤和冲击成本估算）。"""
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
        wide = df_px.pivot_table(index=df_px.index,
                                 columns='code', values='money')
    except Exception:
        return {}
    avg_money = wide.mean()
    return dict(avg_money)


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


# ============================================================
# 横截面构建（F2+F5 双因子，与 Phase 1/3 一致）
# ============================================================

def build_cross_section(date, next_date=None, debug=False, liquidity_filter=True):
    """构建横截面，计算 F2_ep + F5_vol（市值中性化+winsorize 后）。"""
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

    # 流动性过滤（仅 Phase 4 启用）
    if liquidity_filter:
        avg_vol_map = _calc_avg_volume(date_str, list(df.index))
        df['avg_money'] = df.index.map(lambda c: avg_vol_map.get(c, 0))
        mask &= df['avg_money'].fillna(0) >= LIQUIDITY_THRESHOLD

    df = df.dropna(subset=['F2_ep_raw', 'F5_vol_raw', 'market_cap']).copy()
    df = df[mask].copy()
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
# 日频净值计算（与 Phase 1 一致）
# ============================================================

def compute_daily_nav(date_pairs, target_capital=TARGET_CAPITAL,
                      apply_cost=True, apply_impact=True):
    """计算日频净值，含成本和冲击成本。

    返回:
      daily_nav: pd.Series, 日频净值
      rebal_records: list, 每次调仓记录（含换手率、成本、持仓详情）
    """
    print('  [日频回测] 开始计算日频净值...')
    all_trade_days = get_trade_days(start_date=START_DATE, end_date=END_DATE)
    daily_nav = pd.Series(index=pd.to_datetime(all_trade_days), dtype=float)
    daily_nav.iloc[0] = 1.0

    rebal_records = []
    prev_holdings = {}  # code -> weight

    for idx, (date, next_date) in enumerate(date_pairs):
        date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
        next_date_str = next_date.strftime('%Y-%m-%d') if hasattr(next_date, 'strftime') else str(next_date)

        df = build_cross_section(date, next_date, debug=False, liquidity_filter=True)
        if df is None or len(df) < 30:
            print(f'  [日频回测] {date_str}: 横截面构建失败，跳过')
            continue

        codes, weights = build_portfolio(df, f2_weight=0.5, f5_weight=0.5, n_hold=50)
        if not codes:
            continue

        new_holdings = dict(zip(codes, weights.values))

        # 换手率
        turnover = 0.0
        all_codes = set(prev_holdings.keys()) | set(new_holdings.keys())
        for c in all_codes:
            old_w = prev_holdings.get(c, 0)
            new_w = new_holdings.get(c, 0)
            turnover += abs(new_w - old_w)

        # 成本
        base_cost = turnover * BASE_ROUND_TRIP / 2 if apply_cost else 0

        # 冲击成本（线性模型）
        impact_cost = 0.0
        if apply_impact and 'avg_money' in df.columns:
            df_sel = df[df.index.isin(codes)]
            for code in codes:
                if code not in df_sel.index:
                    continue
                w = new_holdings.get(code, 0)
                avg_money = df_sel.loc[code, 'avg_money']
                if avg_money > 0:
                    position_value = target_capital * w
                    impact = position_value / avg_money * IMPACT_COEFF * 0.01
                    impact_cost += impact * w

        # 调仓日净值扣成本（用 current_nav 累积，避免 NaN 传播）
        if idx == 0:
            current_nav = 1.0
        current_nav *= (1 - base_cost - impact_cost)
        daily_nav.loc[pd.Timestamp(date)] = current_nav

        # 持仓期间日频收益
        period_days = daily_nav.index[
            (daily_nav.index >= pd.Timestamp(date)) &
            (daily_nav.index <= pd.Timestamp(next_date))
        ]
        if len(period_days) == 0:
            continue

        # 加载持仓股日频价格
        px = _load_period_prices(date_str, next_date_str, codes)
        if px is None or px.empty:
            continue
        px.index = pd.to_datetime(px.index)

        # 计算日频收益（用 current_nav 累积，显式赋值）
        for i in range(1, len(period_days)):
            prev_day = period_days[i - 1]
            curr_day = period_days[i]
            if prev_day not in px.index or curr_day not in px.index:
                continue
            daily_ret = 0.0
            for code in codes:
                if code not in px.columns:
                    continue
                w = new_holdings.get(code, 0)
                if w == 0:
                    continue
                p_prev = px.loc[prev_day, code]
                p_curr = px.loc[curr_day, code]
                if p_prev > 0 and not np.isnan(p_prev) and not np.isnan(p_curr):
                    daily_ret += w * (p_curr / p_prev - 1)
            current_nav *= (1 + daily_ret)
            daily_nav.loc[curr_day] = current_nav

        rebal_records.append({
            'date': date_str,
            'n_hold': len(codes),
            'turnover': turnover,
            'base_cost': base_cost,
            'impact_cost': impact_cost,
            'total_cost': base_cost + impact_cost,
            'holdings': new_holdings,
        })
        prev_holdings = new_holdings
        print(f'  [日频回测] {date_str}: n={len(codes)} turnover={turnover:.3f} '
              f'cost={base_cost+impact_cost:.5f}')

    daily_nav = daily_nav.dropna()
    return daily_nav, rebal_records


# ============================================================
# Module 1: 成本后净收益
# ============================================================

def module_1_net_return(date_pairs):
    """成本后净收益分析（毛 vs 净）。"""
    print('\n' + '=' * 60)
    print('===== Module 1: 成本后净收益 =====')
    print('=' * 60)

    # 毛收益（无成本）
    print('\n  [M1] 计算毛收益（无成本）...')
    nav_gross, records_gross = compute_daily_nav(
        date_pairs, apply_cost=False, apply_impact=False)

    # 净收益（含成本+冲击）
    print('\n  [M1] 计算净收益（含成本+冲击）...')
    nav_net, records_net = compute_daily_nav(
        date_pairs, apply_cost=True, apply_impact=True)

    if nav_gross.empty or nav_net.empty:
        print('  无有效数据')
        return None

    # 计算指标
    def calc_metrics(nav):
        total_ret = nav.iloc[-1] / nav.iloc[0] - 1
        years = (nav.index[-1] - nav.index[0]).days / 365.25
        ann_ret = (1 + total_ret) ** (1 / years) - 1
        daily_ret = nav.pct_change().dropna()
        ann_vol = daily_ret.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        dd = 1 - nav / nav.cummax()
        max_dd = dd.max()
        return {'total': total_ret, 'ann': ann_ret, 'vol': ann_vol,
                'sharpe': sharpe, 'dd': max_dd}

    m_gross = calc_metrics(nav_gross)
    m_net = calc_metrics(nav_net)

    # 成本占比
    total_cost = sum(r['total_cost'] for r in records_net)
    cost_ratio = total_cost / m_gross['total'] if m_gross['total'] > 0 else 0

    # 表1
    print('\n【表1】毛收益 vs 净收益对照')
    print(f'{"指标":<14}{"毛收益":>14}{"净收益":>14}{"差异":>14}')
    print(f'{"总收益":<14}{m_gross["total"]:>14.4f}{m_net["total"]:>14.4f}'
          f'{m_net["total"]-m_gross["total"]:>14.4f}')
    print(f'{"年化":<14}{m_gross["ann"]:>14.4f}{m_net["ann"]:>14.4f}'
          f'{m_net["ann"]-m_gross["ann"]:>14.4f}')
    print(f'{"年化波动":<14}{m_gross["vol"]:>14.4f}{m_net["vol"]:>14.4f}'
          f'{m_net["vol"]-m_gross["vol"]:>14.4f}')
    print(f'{"Sharpe":<14}{m_gross["sharpe"]:>14.4f}{m_net["sharpe"]:>14.4f}'
          f'{m_net["sharpe"]-m_gross["sharpe"]:>14.4f}')
    print(f'{"最大回撤":<14}{m_gross["dd"]:>14.4f}{m_net["dd"]:>14.4f}'
          f'{m_net["dd"]-m_gross["dd"]:>14.4f}')
    print(f'\n  累计成本: {total_cost:.4f}')
    print(f'  成本占比: {cost_ratio:.2%}')

    # 表2：年度换手率与成本占比
    print('\n【表2】年度换手率与成本占比')
    df_recs = pd.DataFrame([{k: v for k, v in r.items() if k != 'holdings'}
                            for r in records_net])
    df_recs['date'] = pd.to_datetime(df_recs['date'])
    df_recs['year'] = df_recs['date'].dt.year
    yearly = df_recs.groupby('year').agg({
        'turnover': 'sum',
        'base_cost': 'sum',
        'impact_cost': 'sum',
        'total_cost': 'sum',
    })
    print(f'{"年份":<8}{"换手率":>10}{"基础成本":>12}{"冲击成本":>12}{"总成本":>12}')
    for year, row in yearly.iterrows():
        print(f'{year:<8}{row["turnover"]:>10.3f}{row["base_cost"]:>12.5f}'
              f'{row["impact_cost"]:>12.5f}{row["total_cost"]:>12.5f}')

    # Gate 5 模块 1 判定
    print('\n【Gate 5 模块 1 判定】')
    m1_pass_ret = m_net['ann'] > 0.05
    m1_pass_sharpe = m_net['sharpe'] > 0.3
    m1_pass_cost = cost_ratio < 0.30
    print(f'  成本后年化净收益 > 5%: {"✅" if m1_pass_ret else "❌"} ({m_net["ann"]:.4f})')
    print(f'  成本后 Sharpe > 0.3: {"✅" if m1_pass_sharpe else "❌"} ({m_net["sharpe"]:.4f})')
    print(f'  成本占比 < 30%: {"✅" if m1_pass_cost else "❌"} ({cost_ratio:.2%})')
    m1_pass = m1_pass_ret and m1_pass_sharpe and m1_pass_cost
    print(f'  模块 1: {"✅ 通过" if m1_pass else "❌ 未通过"}')

    # 保存
    nav_gross.to_csv(f'{OUT_DIR}/module1_nav_gross.csv')
    nav_net.to_csv(f'{OUT_DIR}/module1_nav_net.csv')
    df_recs.to_csv(f'{OUT_DIR}/module1_rebal_records.csv', index=False)

    return {
        'gross': m_gross, 'net': m_net, 'cost_ratio': cost_ratio,
        'records': records_net, 'nav_net': nav_net, 'pass': m1_pass,
    }


# ============================================================
# Module 2: 容量估算
# ============================================================

def module_2_capacity(date_pairs, m1_result):
    """5000 万资金下的冲击成本估算。"""
    print('\n' + '=' * 60)
    print('===== Module 2: 容量估算 =====')
    print('=' * 60)

    if m1_result is None:
        print('  依赖 Module 1 数据，跳过')
        return None

    records = m1_result['records']

    # 持仓市值/日均成交额分档统计
    impact_ratios = []
    high_impact_count = 0
    for rec in records:
        df = build_cross_section(rec['date'], liquidity_filter=True)
        if df is None:
            continue
        holdings = rec['holdings']
        for code, w in holdings.items():
            if code not in df.index or 'avg_money' not in df.columns:
                continue
            avg_money = df.loc[code, 'avg_money']
            if avg_money > 0:
                position_value = TARGET_CAPITAL * w
                ratio = position_value / avg_money
                impact_ratios.append({
                    'date': rec['date'],
                    'code': code,
                    'weight': w,
                    'position_value': position_value,
                    'avg_money': avg_money,
                    'ratio': ratio,
                })
                if ratio > 0.20:
                    high_impact_count += 1

    if not impact_ratios:
        print('  无有效数据')
        return None

    df_impact = pd.DataFrame(impact_ratios)

    # 分档统计
    bins = [0, 0.05, 0.20, 1.0]
    labels = ['<5%', '5-20%', '>20%']
    df_impact['bucket'] = pd.cut(df_impact['ratio'], bins=bins, labels=labels)
    bucket_counts = df_impact['bucket'].value_counts()

    # 表3
    print('\n【表3】持仓市值/日均成交额分档统计')
    print(f'{"分档":<10}{"次数":>10}{"占比":>10}')
    for label in labels:
        count = bucket_counts.get(label, 0)
        pct = count / len(df_impact)
        print(f'{label:<10}{count:>10}{pct:>10.2%}')

    # 冲击成本时序
    impact_by_date = df_impact.groupby('date')['ratio'].agg(['mean', 'max'])
    avg_impact_ratio = impact_by_date['mean'].mean()
    max_impact_ratio = impact_by_date['max'].max()

    # 表4
    print('\n【表4】5000 万资金下冲击成本时序（年度）')
    print(f'{"年份":<8}{"均值":>10}{"最大":>10}{">20%次数":>12}')
    df_impact['year'] = pd.to_datetime(df_impact['date']).dt.year
    # 兼容旧版 pandas（聚宽 Python 3.6 不支持 named aggregation）
    _g = df_impact.groupby('year')['ratio']
    yearly_impact = pd.DataFrame({
        'mean_ratio': _g.mean(),
        'max_ratio': _g.max(),
        'high_count': _g.apply(lambda x: (x > 0.20).sum()),
    })
    for year, row in yearly_impact.iterrows():
        print(f'{year:<8}{row["mean_ratio"]:>10.4f}{row["max_ratio"]:>10.4f}'
              f'{int(row["high_count"]):>12}')

    # 容量上限估算（冲击成本 <20% 对应的最大资金）
    # ratio = capital * w / avg_money < 0.20
    # capital < 0.20 * avg_money / w
    # 保守估计：用最受限的持仓（ratio 最大）反推
    max_ratio = df_impact['ratio'].max()
    if max_ratio > 0:
        capacity_max = int(0.20 * df_impact['avg_money'].quantile(0.5) /
                           df_impact['weight'].quantile(0.95))
    else:
        capacity_max = 0

    print(f'\n  平均冲击成本比例: {avg_impact_ratio:.4f}')
    print(f'  最大冲击成本比例: {max_impact_ratio:.4f}')
    print(f'  >20% 高冲击次数: {high_impact_count}')
    print(f'  容量上限估算（冲击成本<20%）: {capacity_max/1e8:.2f} 亿')

    # Gate 5 模块 2 判定
    print('\n【Gate 5 模块 2 判定】')
    total_impact_cost = sum(r['impact_cost'] for r in records)
    net_return = m1_result['net']['total']
    impact_ratio_of_return = total_impact_cost / net_return if net_return > 0 else 1
    m2_pass_impact = impact_ratio_of_return < 0.20
    m2_pass_high = high_impact_count < 5
    print(f'  5000万冲击成本 < 净收益20%: {"✅" if m2_pass_impact else "❌"} '
          f'({impact_ratio_of_return:.2%})')
    print(f'  >20% 高冲击股票数 < 5: {"✅" if m2_pass_high else "❌"} ({high_impact_count})')
    m2_pass = m2_pass_impact and m2_pass_high
    print(f'  模块 2: {"✅ 通过" if m2_pass else "❌ 未通过"}')

    df_impact.to_csv(f'{OUT_DIR}/module2_impact_detail.csv', index=False)

    return {
        'avg_impact_ratio': avg_impact_ratio,
        'max_impact_ratio': max_impact_ratio,
        'high_impact_count': high_impact_count,
        'capacity_max': capacity_max,
        'pass': m2_pass,
    }


# ============================================================
# Module 3: 因子拥挤度监测
# ============================================================

def module_3_crowding(date_pairs):
    """F2/F5 因子拥挤度监测。"""
    print('\n' + '=' * 60)
    print('===== Module 3: 因子拥挤度监测 =====')
    print('=' * 60)

    crowding_records = []
    for date, next_date in date_pairs:
        df = build_cross_section(date, debug=False, liquidity_filter=False)
        if df is None or len(df) < 30:
            continue

        f2_mean = float(df['F2_ep'].mean())
        f5_mean = float(df['F5_vol'].mean())
        f2_z_mean = float(zscore_cross_section(df['F2_ep']).mean())
        f5_z_mean = float(zscore_cross_section(df['F5_vol']).mean())

        # 纳入 vs 剔除股对照
        df_temp = df.copy()
        df_temp['F2_z'] = zscore_cross_section(df_temp['F2_ep'])
        df_temp['F5_z'] = zscore_cross_section(df_temp['F5_vol'])
        df_temp['score'] = 0.5 * df_temp['F2_z'] + 0.5 * df_temp['F5_z']
        top_50 = df_temp.nlargest(50, 'score')
        bottom_50 = df_temp.nsmallest(50, 'score')

        crowding_records.append({
            'date': date,
            'f2_mean': f2_mean,
            'f5_mean': f5_mean,
            'f2_z_mean': f2_z_mean,
            'f5_z_mean': f5_z_mean,
            'top50_f2_mean': float(top_50['F2_ep'].mean()),
            'bottom50_f2_mean': float(bottom_50['F2_ep'].mean()),
            'top50_f5_mean': float(top_50['F5_vol'].mean()),
            'bottom50_f5_mean': float(bottom_50['F5_vol'].mean()),
            'asymmetry_f2': float(top_50['F2_ep'].mean() - bottom_50['F2_ep'].mean()),
            'asymmetry_f5': float(top_50['F5_vol'].mean() - bottom_50['F5_vol'].mean()),
        })

    if not crowding_records:
        print('  无有效数据')
        return None

    df_crowd = pd.DataFrame(crowding_records).set_index('date')

    # 历史分位
    current_f2 = df_crowd['f2_z_mean'].iloc[-1]
    current_f5 = df_crowd['f5_z_mean'].iloc[-1]
    f2_percentile = (df_crowd['f2_z_mean'] < current_f2).sum() / len(df_crowd)
    f5_percentile = (df_crowd['f5_z_mean'] < current_f5).sum() / len(df_crowd)

    # 表5
    print('\n【表5】F2/F5 因子拥挤度时序（年度）')
    print(f'{"年份":<8}{"F2均值":>10}{"F5均值":>10}{"F2_z":>10}{"F5_z":>10}'
          f'{"F2不对称":>12}{"F5不对称":>12}')
    df_crowd['year'] = pd.to_datetime(df_crowd.index).year
    yearly_crowd = df_crowd.groupby('year').mean()
    for year, row in yearly_crowd.iterrows():
        print(f'{year:<8}{row["f2_mean"]:>10.4f}{row["f5_mean"]:>10.4f}'
              f'{row["f2_z_mean"]:>10.4f}{row["f5_z_mean"]:>10.4f}'
              f'{row["asymmetry_f2"]:>12.4f}{row["asymmetry_f5"]:>12.4f}')

    print(f'\n  当前 F2 拥挤度: {current_f2:.4f} (历史分位 {f2_percentile:.2%})')
    print(f'  当前 F5 拥挤度: {current_f5:.4f} (历史分位 {f5_percentile:.2%})')

    # Gate 5 模块 3 判定
    print('\n【Gate 5 模块 3 判定】')
    m3_pass_f2 = f2_percentile < 0.90
    m3_pass_f5 = f5_percentile < 0.90
    print(f'  F2 拥挤度 < 90分位: {"✅" if m3_pass_f2 else "❌"} ({f2_percentile:.2%})')
    print(f'  F5 拥挤度 < 90分位: {"✅" if m3_pass_f5 else "❌"} ({f5_percentile:.2%})')
    m3_pass = m3_pass_f2 and m3_pass_f5
    print(f'  模块 3: {"✅ 通过" if m3_pass else "❌ 未通过"}')

    df_crowd.to_csv(f'{OUT_DIR}/module3_crowding.csv')

    return {
        'current_f2': current_f2, 'current_f5': current_f5,
        'f2_percentile': f2_percentile, 'f5_percentile': f5_percentile,
        'pass': m3_pass,
    }


# ============================================================
# Module 4: 换手-alpha 闭环
# ============================================================

def module_4_turnover_alpha(date_pairs, m1_result):
    """换手-alpha 闭环：边际信息收益 vs 边际交易成本。"""
    print('\n' + '=' * 60)
    print('===== Module 4: 换手-alpha 闭环 =====')
    print('=' * 60)

    if m1_result is None:
        print('  依赖 Module 1 数据，跳过')
        return None

    records = m1_result['records']

    # 计算每次调仓的信息收益（新组合 vs 旧组合的下期收益差）
    info_records = []
    for i in range(1, len(records)):
        prev_rec = records[i - 1]
        curr_rec = records[i]

        # 当前调仓日的下期收益（用 forward_period_return）
        date = curr_rec['date']
        next_date = date_pairs[i][1] if i < len(date_pairs) else None
        if next_date is None:
            continue

        df = build_cross_section(date, next_date, debug=False, liquidity_filter=True)
        if df is None:
            continue

        # 新组合收益
        new_codes = list(curr_rec['holdings'].keys())
        new_weights = curr_rec['holdings']
        df_new = df[df.index.isin(new_codes)]
        if df_new.empty:
            continue
        new_ret = float(sum(new_weights.get(c, 0) * df_new.loc[c, 'fwd_return']
                            for c in new_codes if c in df_new.index))

        # 旧组合收益（如果保持不变）
        old_codes = list(prev_rec['holdings'].keys())
        old_weights = prev_rec['holdings']
        df_old = df[df.index.isin(old_codes)]
        if df_old.empty:
            continue
        old_ret = float(sum(old_weights.get(c, 0) * df_old.loc[c, 'fwd_return']
                            for c in old_codes if c in df_old.index))

        # 信息收益 = 新组合收益 - 旧组合收益
        info_return = new_ret - old_ret

        # 边际信息收益 = 信息收益 / 换手率
        turnover = curr_rec['turnover']
        marginal_info_return = info_return / turnover if turnover > 0 else 0

        # 边际交易成本 = 总成本 / 换手率
        total_cost = curr_rec['total_cost']
        marginal_cost = total_cost / turnover if turnover > 0 else 0

        info_records.append({
            'date': date,
            'turnover': turnover,
            'new_ret': new_ret,
            'old_ret': old_ret,
            'info_return': info_return,
            'total_cost': total_cost,
            'marginal_info_return': marginal_info_return,
            'marginal_cost': marginal_cost,
            'net_benefit': marginal_info_return - marginal_cost,
        })

    if not info_records:
        print('  无有效数据')
        return None

    df_info = pd.DataFrame(info_records)

    # 表6
    print('\n【表6】换手-alpha 闭环')
    print(f'{"日期":<12}{"换手率":>10}{"信息收益":>12}{"边际信息收益":>14}'
          f'{"边际成本":>12}{"净收益":>12}')
    for _, r in df_info.iterrows():
        print(f'{r["date"]:<12}{r["turnover"]:>10.3f}{r["info_return"]:>12.5f}'
              f'{r["marginal_info_return"]:>14.5f}{r["marginal_cost"]:>12.5f}'
              f'{r["net_benefit"]:>12.5f}')

    avg_marginal_info = df_info['marginal_info_return'].mean()
    avg_marginal_cost = df_info['marginal_cost'].mean()
    positive_ratio = (df_info['net_benefit'] > 0).mean()

    print(f'\n  平均边际信息收益: {avg_marginal_info:.5f}')
    print(f'  平均边际交易成本: {avg_marginal_cost:.5f}')
    print(f'  净收益为正比例: {positive_ratio:.2%}')

    # Gate 5 模块 4 判定
    print('\n【Gate 5 模块 4 判定】')
    m4_pass = avg_marginal_info >= avg_marginal_cost
    print(f'  边际信息收益 ≥ 边际交易成本: {"✅" if m4_pass else "❌"} '
          f'({avg_marginal_info:.5f} vs {avg_marginal_cost:.5f})')
    print(f'  模块 4: {"✅ 通过" if m4_pass else "❌ 未通过"}')

    df_info.to_csv(f'{OUT_DIR}/module4_turnover_alpha.csv', index=False)

    return {
        'avg_marginal_info': avg_marginal_info,
        'avg_marginal_cost': avg_marginal_cost,
        'positive_ratio': positive_ratio,
        'pass': m4_pass,
    }


# ============================================================
# Module 5: 仓位管理改进（最大回撤约束 + 动态止损）
# ============================================================

def module_5_position_management(date_pairs, m1_result):
    """仓位管理改进：最大回撤约束 + 动态止损 + F5 信号叠加。"""
    print('\n' + '=' * 60)
    print('===== Module 5: 仓位管理改进 =====')
    print('=' * 60)

    if m1_result is None:
        print('  依赖 Module 1 数据，跳过')
        return None

    nav_baseline = m1_result['nav_net'].copy()
    if nav_baseline.empty:
        print('  无 Baseline 净值数据')
        return None

    # Baseline 指标（净收益口径）
    def calc_metrics(nav):
        total_ret = nav.iloc[-1] / nav.iloc[0] - 1
        years = (nav.index[-1] - nav.index[0]).days / 365.25
        ann_ret = (1 + total_ret) ** (1 / years) - 1
        daily_ret = nav.pct_change().dropna()
        ann_vol = daily_ret.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        dd = 1 - nav / nav.cummax()
        max_dd = dd.max()
        return {'total': total_ret, 'ann': ann_ret, 'vol': ann_vol,
                'sharpe': sharpe, 'dd': max_dd}

    base_metrics = calc_metrics(nav_baseline)
    print(f'\n  Baseline (净收益口径): ann={base_metrics["ann"]:.4f} '
          f'sharpe={base_metrics["sharpe"]:.4f} dd={base_metrics["dd"]:.4f}')

    # 计算全市场 F5 信号时序（用于 V2 的 F5 叠加）
    print('  [M5] 计算全市场 F5 信号时序...')
    f5_signals = {}
    for date, next_date in date_pairs:
        df = build_cross_section(date, debug=False, liquidity_filter=False)
        if df is None:
            continue
        f5_signals[date] = float(df['F5_vol'].mean())

    f5_series = pd.Series(f5_signals)
    f5_z_series = zscore_cross_section(f5_series)

    # 三个变体
    variants = [
        {'id': 'V1', 'name': '纯最大回撤约束', 'f5_overlay': False, 'conservative': False},
        {'id': 'V2', 'name': '最大回撤约束 + F5 叠加', 'f5_overlay': True, 'conservative': False},
        {'id': 'V3', 'name': '保守版（阈值 8%/12%/18%）', 'f5_overlay': True, 'conservative': True},
    ]

    variant_results = []
    for v in variants:
        print(f'\n  [M5] 跑 {v["id"]} ({v["name"]})...')

        # 阈值
        if v['conservative']:
            t1, t2, t3, stoploss = 0.08, 0.12, 0.18, 0.25
        else:
            t1, t2, t3, stoploss = DD_THRESHOLD_1, DD_THRESHOLD_2, DD_THRESHOLD_3, DD_STOPLOSS

        nav_v = nav_baseline.copy()
        weights = pd.Series(1.0, index=nav_v.index)

        # 状态变量
        peak = nav_v.iloc[0]
        low = nav_v.iloc[0]
        in_stoploss = False
        stoploss_cooldown = 0

        for i in range(1, len(nav_v)):
            curr_nav = nav_v.iloc[i]
            prev_nav = nav_v.iloc[i - 1]

            # 更新 peak 和 low
            if curr_nav > peak:
                peak = curr_nav
                low = curr_nav
            if curr_nav < low:
                low = curr_nav

            # 当前回撤
            dd = 1 - curr_nav / peak

            # 回撤修复判断
            recovery = (curr_nav - low) / low if low > 0 else 0

            # 止损冷却
            if in_stoploss:
                stoploss_cooldown -= 1
                if stoploss_cooldown <= 0 and recovery >= DD_RECOVERY:
                    in_stoploss = False

            # 仓位决策
            if in_stoploss:
                target_w = 0.20
            elif dd > stoploss:
                target_w = 0.20
                in_stoploss = True
                stoploss_cooldown = STOPLOSS_COOLDOWN
            elif dd > t3:
                target_w = 0.30
            elif dd > t2:
                target_w = 0.50
            elif dd > t1:
                target_w = 0.70
            elif recovery >= DD_RECOVERY and not in_stoploss:
                target_w = 1.00
            else:
                target_w = weights.iloc[i - 1]

            # F5 信号叠加
            if v['f5_overlay']:
                # 找最近的调仓日
                nearest_date = None
                for d in f5_z_series.index:
                    if pd.Timestamp(d) <= nav_v.index[i]:
                        nearest_date = d
                    else:
                        break
                if nearest_date is not None:
                    vol_z = f5_z_series.get(nearest_date, 0)
                    if vol_z < F5_OVERLAY_THRESHOLD:
                        target_w *= F5_OVERLAY_PENALTY

            target_w = max(min(target_w, 1.0), 0.10)
            weights.iloc[i] = target_w

            # 应用仓位：当日收益 = Baseline 收益 × 仓位
            daily_ret = nav_v.iloc[i] / nav_v.iloc[i - 1] - 1
            adjusted_ret = daily_ret * target_w
            nav_v.iloc[i] = nav_v.iloc[i - 1] * (1 + adjusted_ret)

        m_v = calc_metrics(nav_v)
        variant_results.append({
            'id': v['id'], 'name': v['name'],
            'metrics': m_v, 'nav': nav_v, 'weights': weights,
        })
        print(f'  {v["id"]}: ann={m_v["ann"]:.4f} sharpe={m_v["sharpe"]:.4f} '
              f'dd={m_v["dd"]:.4f} weight_mean={weights.mean():.3f}')

    # 表7
    print('\n【表7】仓位管理 V1/V2/V3 vs Baseline 收益风险对照（日频口径）')
    print(f'{"指标":<14}{"Baseline":>14}{"V1 纯DD":>14}{"V2 DD+F5":>14}{"V3 保守":>14}')
    print(f'{"总收益":<14}{base_metrics["total"]:>14.4f}'
          f'{variant_results[0]["metrics"]["total"]:>14.4f}'
          f'{variant_results[1]["metrics"]["total"]:>14.4f}'
          f'{variant_results[2]["metrics"]["total"]:>14.4f}')
    print(f'{"年化":<14}{base_metrics["ann"]:>14.4f}'
          f'{variant_results[0]["metrics"]["ann"]:>14.4f}'
          f'{variant_results[1]["metrics"]["ann"]:>14.4f}'
          f'{variant_results[2]["metrics"]["ann"]:>14.4f}')
    print(f'{"年化波动":<14}{base_metrics["vol"]:>14.4f}'
          f'{variant_results[0]["metrics"]["vol"]:>14.4f}'
          f'{variant_results[1]["metrics"]["vol"]:>14.4f}'
          f'{variant_results[2]["metrics"]["vol"]:>14.4f}')
    print(f'{"Sharpe":<14}{base_metrics["sharpe"]:>14.4f}'
          f'{variant_results[0]["metrics"]["sharpe"]:>14.4f}'
          f'{variant_results[1]["metrics"]["sharpe"]:>14.4f}'
          f'{variant_results[2]["metrics"]["sharpe"]:>14.4f}')
    print(f'{"最大回撤":<14}{base_metrics["dd"]:>14.4f}'
          f'{variant_results[0]["metrics"]["dd"]:>14.4f}'
          f'{variant_results[1]["metrics"]["dd"]:>14.4f}'
          f'{variant_results[2]["metrics"]["dd"]:>14.4f}')

    # 表8：仓位统计
    print('\n【表8】仓位时序描述统计')
    print(f'{"变体":<10}{"均值":>10}{"标准差":>10}{"最小":>10}{"最大":>10}')
    for v in variant_results:
        w = v['weights']
        print(f'{v["id"]:<10}{w.mean():>10.3f}{w.std():>10.3f}'
              f'{w.min():>10.3f}{w.max():>10.3f}')

    # Gate 5 模块 5 判定（以 V2 为准）
    print('\n【Gate 5 模块 5 判定】（以 V2 DD+F5 为准）')
    v2 = variant_results[1]['metrics']
    m5_pass_dd = v2['dd'] < 0.30
    m5_pass_ann = base_metrics['ann'] - v2['ann'] < 0.05
    m5_pass_sharpe = v2['sharpe'] >= base_metrics['sharpe']
    print(f'  V2 回撤 < 30%: {"✅" if m5_pass_dd else "❌"} ({v2["dd"]:.4f})')
    print(f'  V2 年化下降 < 5pp: {"✅" if m5_pass_ann else "❌"} '
          f'({base_metrics["ann"] - v2["ann"]:.4f})')
    print(f'  V2 Sharpe 不下降: {"✅" if m5_pass_sharpe else "❌"} '
          f'({v2["sharpe"]:.4f} vs {base_metrics["sharpe"]:.4f})')
    m5_pass = m5_pass_dd and m5_pass_ann and m5_pass_sharpe
    print(f'  模块 5: {"✅ 通过" if m5_pass else "❌ 未通过"}')

    # 保存
    nav_baseline.to_csv(f'{OUT_DIR}/module5_baseline_nav.csv')
    for v in variant_results:
        v['nav'].to_csv(f'{OUT_DIR}/module5_{v["id"]}_nav.csv')
        v['weights'].to_csv(f'{OUT_DIR}/module5_{v["id"]}_weights.csv')

    return {
        'base': base_metrics,
        'variants': [{'id': v['id'], 'metrics': v['metrics']} for v in variant_results],
        'pass': m5_pass,
        'v2_dd': v2['dd'],
        'v2_ann_drop': base_metrics['ann'] - v2['ann'],
    }


# ============================================================
# 运行入口
# ============================================================

def run():
    """Phase 4 五模块全流程。"""
    print('=' * 60)
    print(f'P4-PL-2026Q2-v1 Phase 4 实盘校准')
    print(f'样本: {SAMPLE_NAME}, 区间: {START_DATE} ~ {END_DATE}')
    print(f'目标资金: {TARGET_CAPITAL/1e4:.0f} 万')
    print(f'5 模块: 成本净收益 + 容量 + 拥挤度 + 换手-alpha + 仓位管理')
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

    # Module 1: 成本后净收益
    m1 = module_1_net_return(date_pairs)

    # Module 2: 容量估算
    m2 = module_2_capacity(date_pairs, m1)

    # Module 3: 因子拥挤度
    m3 = module_3_crowding(date_pairs)

    # Module 4: 换手-alpha 闭环
    m4 = module_4_turnover_alpha(date_pairs, m1)

    # Module 5: 仓位管理改进
    m5 = module_5_position_management(date_pairs, m1)

    # 汇总
    print('\n\n' + '=' * 60)
    print('===== Phase 4 汇总 =====')
    print('=' * 60)
    print('\n【Gate 5 总判定】')
    modules_pass = 0
    if m1 and m1['pass']:
        modules_pass += 1
        print(f'  M1 成本净收益: ✅ 通过 (年化={m1["net"]["ann"]:.4f})')
    else:
        print(f'  M1 成本净收益: ❌ 未通过')

    if m2 and m2['pass']:
        modules_pass += 1
        print(f'  M2 容量: ✅ 通过 (高冲击={m2["high_impact_count"]})')
    else:
        print(f'  M2 容量: ❌ 未通过')

    if m3 and m3['pass']:
        modules_pass += 1
        print(f'  M3 拥挤度: ✅ 通过 (F2={m3["f2_percentile"]:.2%}, F5={m3["f5_percentile"]:.2%})')
    else:
        print(f'  M3 拥挤度: ❌ 未通过')

    if m4 and m4['pass']:
        modules_pass += 1
        print(f'  M4 换手-alpha: ✅ 通过 (净收益为正比例={m4["positive_ratio"]:.2%})')
    else:
        print(f'  M4 换手-alpha: ❌ 未通过')

    if m5 and m5['pass']:
        modules_pass += 1
        print(f'  M5 仓位管理: ✅ 通过 (V2 回撤={m5["v2_dd"]:.4f})')
    else:
        v2_dd = m5['v2_dd'] if m5 else 0
        v2_drop = m5['v2_ann_drop'] if m5 else 0
        print(f'  M5 仓位管理: ❌ 未通过 (V2 回撤={v2_dd:.4f}, 年化下降={v2_drop:.4f})')

    print(f'\n  通过模块数: {modules_pass}/5')

    if modules_pass == 5:
        print('\n  >>> Phase 4 通过，策略实盘就绪')
    elif modules_pass == 4:
        print('\n  >>> Phase 4 部分通过（4/5），策略可实盘但需人工风控')
    else:
        print('\n  >>> Phase 4 失败（≤3/5），策略未达实盘标准')

    print(f'\n所有 CSV 已保存到 {OUT_DIR}/')


if __name__ == '__main__':
    run()
