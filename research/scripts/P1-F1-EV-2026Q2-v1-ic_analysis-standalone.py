"""
P1-F1-EV-2026Q2-v1 IC 分析脚本（standalone 版）
=================================================

在聚宽【研究环境】中直接粘贴运行，无需上传 data_layer.py / factor_lib.py。
所有依赖逻辑已内联。

产出 spec 要求的表1-7 + CSV 序列，用于 Gate 1 七条判定。

【符号约定】（重要，避免 IC 方向混淆）
- 原始 ev = market_cap + 有息负债 - cash，越低（越负）= 净现金越多 = "好"信号
- 定义 signal = -ev（越高 = 净现金越多 = 越好），使 IC>0 表示因子有效（Gate 1 规则1）
- Q1-Q5 按 ev 升序分位：Q1 = 最低 ev（净现金最多）= 多头组（Gate 1 规则2 "Q1多头组"）
- 同时报告二值 factor_ev_negative 的 IC（spec 定义因子），作为辅助校验

【spec 歧义处理记录】
- factor_ev_negative 为二值因子，无法做 Q1-Q5 五分组 → 改用连续 ev 排序分组，
  Q1=最低ev(净现金最多)=多头组；二值 IC 单独报告。
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
# 聚宽研究环境会把 valuation/balance/income/cash_flow/query/get_fundamentals 等
# 注入到全局命名空间。但不同 kernel 配置下注入位置可能不同（globals 或 builtins）。
# 这里用多层 fallback 解析，避免 globals()[name] 字典查找绕过 builtins 的问题。

from jqdata import *  # noqa: F403,F401

# 显式导入兜底（若 jqdata __all__ 未导出，依赖 kernel 预加载/builtins）
try:
    from jqdata import (  # noqa: F401
        valuation, balance, income, cash_flow, query,
        get_fundamentals, get_price, get_all_securities,
        get_index_stocks, get_extras, get_security_info,
        get_trade_days, get_industry, finance,
    )
except Exception:
    pass  # 依赖 kernel 预加载或 builtins 注入，后续直接按名字引用


warnings.filterwarnings('ignore')


# ============================================================
# 参数
# ============================================================
START_DATE = '2014-01-01'
END_DATE = '2026-06-30'
INDEX_IDS = {
    'CSI300': '000300.XSHG',
    'CSI500': '000905.XSHG',
    # 'AllA': None,  # 全A股票多、资源消耗大，单独跑
}

# 快速验证开关：True=只跑2014年3个调仓日+只CSI300，验证数据链；
# 验证通过后改 False 跑全量
QUICK_TEST = False
FACTOR_BINARY = 'factor_ev_negative'
SHELL_THRESHOLD_PRE = 20e8      # 注册制前 20 亿
SHELL_THRESHOLD_POST = 30e8     # 注册制后 30 亿
BREAKPOINT = datetime.date(2019, 6, 1)
OUT_DIR = 'results/P1-F1-EV-2026Q2-v1'


# ============================================================
# 内联 data_layer 必要函数（standalone，无需 import data_layer）
# ============================================================

def _get_stock_pool(index_id, date_str, min_listed_days=180):
    """构建股票池：成分股（或全A）→ 剔ST → 剔次新股。"""
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
    return result


def _fetch_fundamentals_pit(date_str, stocks):
    """PIT 财务数据查询（动态字段探测，彻底兼容聚宽各版本字段名差异）。

    聚宽 balance/income/cash_flow 表的字段名在不同 API 版本间存在单复数/
    命名风格差异（如 short_term_loans vs short_term_loan），属性访问会抛
    AttributeError。本函数改用 query(Table) 全字段查询 + DataFrame 列名访问，
    完全规避属性访问。

    返回以 code 为 index 的 DataFrame，列名为聚宽原始列名。
    """
    # 分别查四张表（query(Entity) 选该表全部列），再按 code merge
    pieces = []
    # valuation（code/market_cap 是核心，用属性访问确保必中）
    try:
        q_val = query(valuation.code, valuation.market_cap)
        df_val = get_fundamentals(q_val, date=date_str)
        if df_val is not None and not df_val.empty:
            if 'code' in df_val.columns:
                df_val = df_val.set_index('code')
            pieces.append(df_val)
    except Exception:
        pass

    # balance 全字段（避免属性访问）
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

    # 按股票池过滤并 merge
    df = pieces[0]
    for p in pieces[1:]:
        # 避免重复列
        overlap = [c for c in p.columns if c in df.columns]
        if overlap:
            p = p.drop(columns=overlap)
        df = df.join(p, how='outer')

    if stocks:
        df = df[df.index.isin(list(stocks))]
    if df is None or df.empty:
        return None
    return df


def _fetch_actual_controller(stocks, date_str):
    """查询实际控制人 {code: actual_controller}。

    finance.STK_COMPANY_INFO 的属性访问在某些环境可能失败，try/except 保护，
    失败返回空 dict（不影响 EV<0 因子，仅影响国企因子辅助校验）。
    """
    if not stocks:
        return {}
    try:
        q = query(
            finance.STK_COMPANY_INFO.actual_controller,
            finance.STK_COMPANY_INFO.code,
        ).filter(finance.STK_COMPANY_INFO.code.in_(list(stocks)))
        df = get_fundamentals(q, date=date_str)
        if df is None or df.empty:
            return {}
        return dict(zip(df['code'], df['actual_controller']))
    except Exception:
        return {}


# ============================================================
# 内联 factor_lib 必要函数（standalone）
# ============================================================

def _get_col(df, *candidates):
    """从 df 中按候选列名找第一个存在的列，返回 Series；找不到返回 None。"""
    for c in candidates:
        if c in df.columns:
            return df[c]
    return None


def _calculate_all_factors(df):
    """因子计算（内联版，聚宽真实字段名 + 单位统一，2026-06 诊断确认）。

    诊断确认的聚宽真实字段名（与文档/猜测的差异）：
    - balance: cash_equivalents, total_assets, total_liability,
              total_current_assets, total_current_liability,
              shortterm_loan（无下划线！）, longterm_loan（无下划线！）,
              accounts_payable, salaries_payable, taxs_payable（拼写！）
    - income: net_profit
    - cash_flow: net_operate_cash_flow

    【关键：单位统一】
    - valuation.market_cap 单位是【亿元】（诊断：茅台 24608.9 亿）
    - balance/cash_flow 字段单位是【元】（诊断：茅台 cash_equivalents 507 亿=5.07e10 元）
    - EV 计算必须统一为元：market_cap × 1e8

    有息负债口径：shortterm_loan + longterm_loan（聚宽真实字段，比估算更精确）
    """
    # market_cap（valuation 查询，单位亿元）
    mcap = _get_col(df, 'market_cap')
    if mcap is None:
        return None  # 无市值数据，无法计算因子
    df['market_cap'] = mcap.astype(float)

    # 有息负债：聚宽真实字段 shortterm_loan + longterm_loan（诊断确认，无下划线）
    stl = _get_col(df, 'shortterm_loan', 'short_term_loan', 'short_term_loans')
    ltl = _get_col(df, 'longterm_loan', 'long_term_loan', 'long_term_loans')
    if stl is not None and ltl is not None:
        interest_bearing_debt = stl.fillna(0) + ltl.fillna(0)
    elif stl is not None:
        interest_bearing_debt = stl.fillna(0)
    elif ltl is not None:
        interest_bearing_debt = ltl.fillna(0)
    else:
        # 回退：用 total_liability 整体代理（高估有息负债，保守）
        tl = _get_col(df, 'total_liability', 'total_liabilities')
        interest_bearing_debt = tl.fillna(0) if tl is not None else 0
    df['interest_bearing_debt'] = interest_bearing_debt

    # cash_equivalents（货币资金，单位元）
    cash = _get_col(df, 'cash_equivalents', 'monetary_funds')
    if cash is None:
        return None  # 无现金数据，无法算 EV
    df['cash_equivalents'] = cash.astype(float)

    # EV = market_cap（亿元→元）+ 有息负债（元）- cash（元）
    # 统一为元，避免单位不一致导致因子失真
    df['ev'] = (df['market_cap'] * 1e8) + df['interest_bearing_debt'] - df['cash_equivalents']
    df['factor_ev_negative'] = (df['ev'] < 0).astype(int)
    df['cash_available_ext'] = df['cash_equivalents']
    df['ev_ext'] = df['ev']

    # current_ratio = 流动资产 / 流动负债（同单位，无碍）
    tca = _get_col(df, 'total_current_assets')
    tcl = _get_col(df, 'total_current_liability', 'total_current_liabilities')
    if tca is not None and tcl is not None:
        df['current_ratio'] = tca.astype(float) / tcl.astype(float).replace(0, np.nan)
    else:
        df['current_ratio'] = np.nan

    # earnings_yield = net_profit（元）/ market_cap（亿元→元）
    np_col = _get_col(df, 'net_profit', 'np_parent_company_owners',
                      'net_profit_is_parent_company')
    if np_col is not None:
        df['net_profit'] = np_col.astype(float)
        df['earnings_yield'] = df['net_profit'] / (df['market_cap'] * 1e8)
    else:
        df['net_profit'] = np.nan
        df['earnings_yield'] = np.nan

    # debt_to_assets = total_liability / total_assets（同单位，无碍）
    tl = _get_col(df, 'total_liability', 'total_liabilities')
    ta = _get_col(df, 'total_assets')
    if tl is not None and ta is not None:
        df['total_liability'] = tl.astype(float)
        df['total_assets'] = ta.astype(float)
        df['debt_to_assets'] = df['total_liability'] / df['total_assets'].replace(0, np.nan)
    else:
        df['debt_to_assets'] = np.nan

    return df


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
                    # 聚宽 get_trade_days 可能返回 date/datetime，统一转 Timestamp
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
# 按月批量取指定股票池收盘价，缓存
_PRICE_CACHE = {}  # {(year_month, frozenset_codes_hash): DataFrame}


def _load_month_prices(year_month, codes):
    """加载某年某月指定 codes 的收盘价，缓存。

    返回 DataFrame（index=交易日, columns=code, values=close）。
    用 get_price(codes_list, ...) 按股票池批量取，不用 None 全市场
    （全市场 get_price(None) 在研究环境可能返回空或超时）。
    """
    cache_key = (year_month, hash(frozenset(codes)))
    if cache_key in _PRICE_CACHE:
        return _PRICE_CACHE[cache_key]

    y, m = int(year_month[:4]), int(year_month[5:7])
    start = datetime.date(y, m, 1)
    end = (start.replace(day=28) + datetime.timedelta(days=7))
    codes_list = list(codes)
    close = pd.DataFrame()
    try:
        # 按股票池批量取（panel=False 长表）
        df = get_price(codes_list,
                       start_date=start.strftime('%Y-%m-%d'),
                       end_date=end.strftime('%Y-%m-%d'),
                       fields=['close'], skip_paused=False,
                       panel=False, fq='post')
        if df is not None and not df.empty:
            # 聚宽 panel=False 长表：日期在 'time' 列（不是 index）
            # index 是默认整数行号，直接 to_datetime 会当 Unix 时间戳
            if 'time' in df.columns:
                df = df.set_index('time')
            elif 'date' in df.columns:
                df = df.set_index('date')
            # 现在 index 应是日期
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
    ym0 = f'{d.year:04d}-{d.month:02d}'
    if d.month == 12:
        ym1 = f'{d.year + 1:04d}-01'
    else:
        ym1 = f'{d.year:04d}-{d.month + 1:02d}'

    px0 = _load_month_prices(ym0, codes)
    px1 = _load_month_prices(ym1, codes)
    if debug:
        print(f'    [debug] {date_str}: px0={px0.shape if px0 is not None else None}, '
              f'px1={px1.shape if px1 is not None else None}')

    if px0 is None or px0.empty or px1 is None or px1.empty:
        return None

    # 确保 index 是 datetime（防止整数行号）
    try:
        px0.index = pd.to_datetime(px0.index)
        px1.index = pd.to_datetime(px1.index)
    except Exception:
        return None

    if debug:
        print(f'    [debug] {date_str}: px0.index 范围 {px0.index.min()} ~ {px0.index.max()}, '
              f'px1.index 范围 {px1.index.min()} ~ {px1.index.max()}')

    # 取 date_str 当天（或之后首个交易日）的收盘价
    d_ts = pd.Timestamp(date_str)
    valid0 = px0.index[px0.index >= d_ts]
    if debug:
        print(f'    [debug] {date_str}: d_ts={d_ts}, valid0 数量={len(valid0)}')
    if len(valid0) == 0:
        # 回退：取 px0 最后一行（月末价作为起始价的近似）
        valid0_idx = px0.index[-1:]
        if debug:
            print(f'    [debug] {date_str}: 回退用 px0 最后一行 {valid0_idx}')
    else:
        valid0_idx = valid0[:1]
    close_start = px0.loc[valid0_idx[0]]
    close_end = px1.iloc[-1]
    if debug:
        print(f'    [debug] {date_str}: close_start 非空={close_start.notna().sum()}/'
              f'{len(close_start)}, close_end 非空={close_end.notna().sum()}/{len(close_end)}')
    if len(valid0) == 0:
        return None
    close_start = px0.loc[valid0[0]]
    close_end = px1.iloc[-1]

    # 对齐 codes
    codes_set = set(codes)
    cs = close_start[close_start.index.isin(codes_set)]
    ce = close_end[close_end.index.isin(codes_set)]
    if cs.empty or ce.empty:
        return None

    common = cs.index.intersection(ce.index)
    cs = cs.loc[common]
    ce = ce.loc[common]
    # 排除 0 价（停牌等）
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

def build_cross_section(date, index_id, apply_shell_filter=False, debug=False):
    """构建单个调仓日的横截面 DataFrame。None 表示数据不足。"""
    date_str = date.strftime('%Y-%m-%d')
    stocks = _get_stock_pool(index_id, date_str)
    if not stocks:
        if debug:
            print(f'    [debug] {date_str}: 股票池为空')
        return None
    if debug:
        print(f'    [debug] {date_str}: 股票池 {len(stocks)} 只')

    df = _fetch_fundamentals_pit(date_str, stocks)
    if df is None or df.empty:
        if debug:
            print(f'    [debug] {date_str}: fundamentals 查询为空')
        return None
    if debug:
        print(f'    [debug] {date_str}: fundamentals {df.shape}, 列数={len(df.columns)}')

    ctrl_map = _fetch_actual_controller(list(df.index), date_str)
    if ctrl_map:
        df['actual_controller'] = df.index.map(ctrl_map)
    df = _calculate_all_factors(df)
    if df is None or df.empty:
        if debug:
            print(f'    [debug] {date_str}: _calculate_all_factors 返回空')
        return None
    if debug:
        print(f'    [debug] {date_str}: 因子计算后 {len(df)} 只, '
              f'ev非空={df["ev"].notna().sum() if "ev" in df.columns else 0}')

    # 基础过滤——对缺失字段宽容（缺失不全部过滤掉）
    # net_profit 缺失时视为通过（不因 income 表查询失败而误杀）
    if 'net_profit' in df.columns and df['net_profit'].notna().any():
        mask = df['net_profit'].fillna(0) > 0
    else:
        mask = pd.Series(True, index=df.index)
    # debt_to_assets 缺失时视为通过
    if 'debt_to_assets' in df.columns and df['debt_to_assets'].notna().any():
        mask &= df['debt_to_assets'].fillna(0.5) <= 1.0
    # current_ratio 缺失时视为通过
    if 'current_ratio' in df.columns and df['current_ratio'].notna().any():
        mask &= df['current_ratio'].fillna(2.0) > 1.5
    # 必须 ev 可计算
    if 'ev' not in df.columns or df['ev'].isna().all():
        if debug:
            print(f'    [debug] {date_str}: ev 列不存在或全空')
        return None
    mask &= df['ev'].notna()
    df = df[mask].copy()
    if df.empty:
        if debug:
            print(f'    [debug] {date_str}: 过滤后为空 '
                  f'(net_profit>0={mask.sum() if "net_profit" in df.columns else "N/A"})')
        return None
    if debug:
        print(f'    [debug] {date_str}: 过滤后 {len(df)} 只')

    # V2 壳价值剔除
    # market_cap 单位是亿元，阈值也用亿元（20/30 亿，不是 20e8/30e8 元）
    if apply_shell_filter:
        threshold_yi = (20.0 if date < BREAKPOINT else 30.0)  # 亿元
        df = df[df['market_cap'] >= threshold_yi]
        if df.empty:
            return None

    # signal = -ev（高=净现金多=好）
    df['signal'] = -df['ev']

    # 行业
    ind_map = get_industry_map(list(df.index), date_str)
    df['industry'] = df.index.map(lambda c: ind_map.get(c, '未知'))

    # 前向收益
    codes = list(df.index)
    fwd = forward_month_return(codes, date_str, debug=debug)
    if fwd is None:
        if debug:
            print(f'    [debug] {date_str}: forward_month_return 返回 None')
        return None
    df['fwd_return'] = df.index.map(fwd)
    df = df.dropna(subset=['fwd_return', 'signal', 'ev'])
    if len(df) < 10:
        if debug:
            print(f'    [debug] {date_str}: dropna 后不足 10 只 ({len(df)})')
        return None
    if debug:
        print(f'    [debug] {date_str}: 最终横截面 {len(df)} 只')
    return df


# ============================================================
# 主分析
# ============================================================

def analyze_sample(sample_name, index_id, variant='V1'):
    """对单样本计算完整 IC 表。"""
    shell_filter = (variant == 'V2')
    tag = f'{sample_name}-{variant}'
    print(f'\n{"=" * 60}\n===== {tag} =====\n{"=" * 60}')

    rebal_dates = get_rebalance_dates(START_DATE, END_DATE)
    if QUICK_TEST:  # 快速模式：只取前 3 个调仓日（2014 年）
        rebal_dates = rebal_dates[:3]
        print(f'  [快速模式] 只跑前 3 个调仓日: {rebal_dates}')
    records = []
    ic_list, ic_binary_list, dates_list = [], [], []
    q_rets = {q: [] for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']}

    for i, date in enumerate(rebal_dates):
        df = build_cross_section(date, index_id,
                                 apply_shell_filter=shell_filter,
                                 debug=(i == 0))  # 首日调试
        if df is None:
            continue
        if i == 0:  # 首日调试：确认数据流
            print(f'  [调试] {date}: 横截面 {len(df)} 只股票')
            print(f'         ev 范围: {df["ev"].min():.2e} ~ {df["ev"].max():.2e}')
            print(f'         EV<0 数量: {int((df["ev"]<0).sum())}')
            print(f'         fwd_return 非空: {df["fwd_return"].notna().sum()}')

        ic = stats.spearmanr(df['signal'], df['fwd_return'])[0]
        if np.isnan(ic):
            continue
        ic_list.append(ic)
        ic_binary = stats.spearmanr(df[FACTOR_BINARY], df['fwd_return'])[0]
        ic_binary_list.append(0.0 if np.isnan(ic_binary) else ic_binary)
        dates_list.append(date)

        # Q1-Q5（按 ev 升序：Q1=最低ev=净现金最多=多头）
        try:
            df['group'] = pd.qcut(df['ev'].rank(method='first'), 5,
                                  labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        except Exception:
            df['group'] = pd.qcut(df['ev'], 5,
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
            'rank_ic_binary': (0.0 if np.isnan(ic_binary) else ic_binary),
            'ic_mcap_neutral': (np.nan if np.isnan(ic_mcap) else ic_mcap),
            'ic_ind_neutral': (np.nan if np.isnan(ic_ind) else ic_ind),
            'ic_small_cap': ic_by_tier['小'],
            'ic_mid_cap': ic_by_tier['中'],
            'ic_large_cap': ic_by_tier['大'],
            'median_mcap': float(df['market_cap'].median()),
            'n_ev_neg': int(df[FACTOR_BINARY].sum()),
        })

        if (i + 1) % 6 == 0:
            print(f'  [{tag}] 进度 {i+1}/{len(rebal_dates)} 日处理完成')

    if not ic_list:
        print(f'[{tag}] 无有效数据')
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
    icb_mean, _, icb_t = newey_west_t(pd.Series(ic_binary_list), lags=4)
    icb_std = float(pd.Series(ic_binary_list).std(ddof=1))
    icb_ir = icb_mean / icb_std if icb_std > 0 else 0.0

    print('\n【表1】月度 Rank IC 描述统计')
    print(f'  连续signal(-ev): mean={ic_mean:.6f} std={ic_std:.6f} '
          f'NW-t={ic_t:.4f} IC_IR={ic_ir:.4f} 胜率={win_rate:.2%} '
          f'正/负={pos_pct:.2%}/{neg_pct:.2%}')
    print(f'  二值factor_ev_negative: mean={icb_mean:.6f} std={icb_std:.6f} '
          f'NW-t={icb_t:.4f} IC_IR={icb_ir:.4f}')

    # 表2
    print('\n【表2】Q1-Q5 分组（Q1=最低ev=净现金最多=多头组）')
    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        r = q_rets[q]
        mean_m = float(np.nanmean(r))
        std_m = float(np.nanstd(r))
        ann_ret = (1 + mean_m) ** 12 - 1
        ann_vol = std_m * np.sqrt(12)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
        print(f'  {q}: 月均={mean_m:.6f} 年化={ann_ret:.4f} '
              f'年化波动={ann_vol:.4f} 夏普={sharpe:.4f}')

    # 表3
    print('\n【表3】分市值档 Rank IC（横截面三分位：小/中/大）')
    for tier, col in [('小', 'ic_small_cap'), ('中', 'ic_mid_cap'),
                      ('大', 'ic_large_cap')]:
        s = rec_df[col].dropna()
        if len(s) > 1:
            m, _, t = newey_west_t(s, lags=4)
            print(f'  {tier}盘: mean={m:.6f} NW-t={t:.4f} N={len(s)}')
        else:
            print(f'  {tier}盘: 样本不足')

    # 表4
    print('\n【表4】分行业 Rank IC（行业中性化后残差 IC）')
    s = rec_df['ic_ind_neutral'].dropna()
    if len(s) > 1:
        m, _, t = newey_west_t(s, lags=4)
        print(f'  行业中性化 IC: mean={m:.6f} NW-t={t:.4f} '
              f'IC_IR={m / float(s.std(ddof=1)):.4f} N={len(s)}')
    else:
        print('  样本不足')

    # 表5
    print('\n【表5】市值中性化后 Rank IC')
    s = rec_df['ic_mcap_neutral'].dropna()
    if len(s) > 1:
        m, _, t = newey_west_t(s, lags=4)
        print(f'  市值中性化 IC: mean={m:.6f} NW-t={t:.4f} '
              f'IC_IR={m / float(s.std(ddof=1)):.4f} N={len(s)}')
    else:
        print('  样本不足')

    # 表6
    print('\n【表6】2019.06 断点前后分段 IC')
    # 统一 index 为 Timestamp，避免 date vs Timestamp 比较报错
    ic_series.index = pd.to_datetime(ic_series.index)
    bp_ts = pd.Timestamp(BREAKPOINT)
    pre_ic = ic_series[ic_series.index < bp_ts]
    post_ic = ic_series[ic_series.index >= bp_ts]
    pre_mean = post_mean = np.nan
    if len(pre_ic) > 1:
        pm, _, pt = newey_west_t(pre_ic, lags=4)
        pre_mean = pm
        print(f'  前段({pre_ic.index[0].date()}~{pre_ic.index[-1].date()}): '
              f'mean={pm:.6f} NW-t={pt:.4f} N={len(pre_ic)}')
    if len(post_ic) > 1:
        qm, _, qt = newey_west_t(post_ic, lags=4)
        post_mean = qm
        print(f'  后段({post_ic.index[0].date()}~{post_ic.index[-1].date()}): '
              f'mean={qm:.6f} NW-t={qt:.4f} N={len(post_ic)}')

    # CSV
    os.makedirs(OUT_DIR, exist_ok=True)
    out_ic = pd.DataFrame({
        'date': dates_list,
        'rank_ic': ic_list,
        'rank_ic_binary': ic_binary_list,
        'ic_mcap_neutral': rec_df['ic_mcap_neutral'].values,
        'ic_ind_neutral': rec_df['ic_ind_neutral'].values,
        'n': rec_df['n'].values,
        'n_ev_neg': rec_df['n_ev_neg'].values,
    })
    out_ic.to_csv(f'{OUT_DIR}/{sample_name}_{variant}_ic_monthly.csv',
                  index=False)
    out_q = pd.DataFrame({'date': dates_list})
    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        out_q[q] = q_rets[q]
    out_q.to_csv(f'{OUT_DIR}/{sample_name}_{variant}_quantile.csv',
                 index=False)
    print(f'\n  CSV 已保存到 {OUT_DIR}/{sample_name}_{variant}_*.csv')

    return {
        'tag': tag,
        'ic_ir': ic_ir,
        'ic_mean': ic_mean,
        'ic_t': ic_t,
        'ic_ir_binary': icb_ir,
        'q1_monthly_mean': float(np.nanmean(q_rets['Q1'])),
        'q5_monthly_mean': float(np.nanmean(q_rets['Q5'])),
        'pre_ic_mean': pre_mean,
        'post_ic_mean': post_mean,
    }


# ============================================================
# 运行入口
# ============================================================

def run():
    """三样本 × V1/V2 双变体全量跑 IC 分析。全量上报，禁止择优。"""
    if QUICK_TEST:
        print('>>> 快速验证模式：只跑 2014 年 3 个调仓日 + 只 CSI300')
    samples = {'CSI300': '000300.XSHG'} if QUICK_TEST else INDEX_IDS
    summary = []
    for name, idx in samples.items():
        r1 = analyze_sample(name, idx, variant='V1')
        if r1:
            summary.append(r1)
        if not QUICK_TEST:  # 快速模式只跑 V1
            r2 = analyze_sample(name, idx, variant='V2')
            if r2:
                summary.append(r2)

    print('\n\n' + '=' * 60)
    print('===== 全样本汇总（V1 vs V2 对比 / 表7）=====')
    print('=' * 60)
    print(f'{"tag":<16}{"IC_IR":>10}{"IC_mean":>12}{"NW-t":>10}'
          f'{"Q1月均":>10}{"Q5月均":>10}{"前段IC":>10}{"后段IC":>10}')
    for r in summary:
        print(f'{r["tag"]:<16}{r["ic_ir"]:>10.4f}{r["ic_mean"]:>12.6f}'
              f'{r["ic_t"]:>10.4f}{r["q1_monthly_mean"]:>10.6f}'
              f'{r["q5_monthly_mean"]:>10.6f}'
              f'{r["pre_ic_mean"]:>10.6f}{r["post_ic_mean"]:>10.6f}')

    if summary:
        pd.DataFrame(summary).to_csv(f'{OUT_DIR}/_summary.csv', index=False)
        print(f'\n汇总已保存到 {OUT_DIR}/_summary.csv')


if __name__ == '__main__':
    run()
