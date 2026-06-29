"""
P1-F3-Dividend-2026Q2-v1 IC 分析脚本（Phase 1.3 standalone 版）
================================================================
聚宽【研究环境】直接粘贴运行，无外部依赖。

Phase 1.3 因子：股息率（F3），目标下行保护，与 EP 组合降回撤。

因子定义:
  主信号: dividend_yield = 过去12个月累计每股现金分红 / 换仓日收盘价
          （越高=分红越多=好，IC>0 有效）
  过滤测试: OCF >= net_profit（现金流质量过滤，验证假设 H12）

数据源:
  分红明细: finance.STK_XR_XD 除权除息表（bonus_ratio_rmb 每10股派息，a_xr_date 除权日）
  财务数据: get_fundamentals(query(cash_flow, income)) PIT 查询
  收盘价: get_price 不复权（股息率=实际分红/实际股价）

注: 聚宽因子库 jqfactor 无股息率因子（260个因子中不含），finance.STK_DIVIDEND
不存在，改用 finance.STK_XR_XD 除权除息表手算（探测确认 2026-06）。

Size 共线风险:
  股息率 = 分红/市值，P 在分母，与 EP 同属 Type-B 信号。
  市值中性化后 IC NW-t>=2 是决定性门槛（与 F1/F2/F4/F5 一致）。

Gate 1 七条（同 F2/F5）:
  1. IC 方向正确(>0)
  2. Q5多头 > Q1
  3. 全样本 NW-t>=2
  4. 分市值档稳定
  5. 行业中性化后 t>=2
  6. 2019.06 断点后增强
  7. 多空组合夏普 > 0.3
  决定性门槛：市值中性化后 IC NW-t>=2

QUICK_TEST=True 先验证数据链（3个调仓日），确认后改 False 全量跑。
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
QUICK_TEST = False  # 全量跑 Gate 1 七条
BREAKPOINT = datetime.date(2019, 6, 1)
OUT_DIR = 'results/P1-F3-Dividend-2026Q2-v1'

# 金融股剔除（银行/非银金融天然高股息，不剔会主导信号）
_EXCLUDE_FINANCE = True


# ============================================================
# 内联 data_layer 必要函数（standalone，无需 import data_layer）
# ============================================================

def _get_col(df, *candidates):
    """从 df 中按候选列名找第一个存在的列，返回 Series；找不到返回 None。"""
    for c in candidates:
        if c in df.columns:
            return df[c]
    return None


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

    # 金融股剔除（银行/非银金融天然高股息，不剔会主导信号）
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
    """PIT 市值查询。valuation.market_cap 单位是【亿元】。"""
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


def _fetch_financials(date_str, stocks):
    """PIT 查询 OCF 和 net_profit（现金流质量过滤用）。

    已确认聚宽字段名（2026-06 诊断）：
      cash_flow.net_operate_cash_flow（不是 operating_cash_flow！）
      income.net_profit
    """
    try:
        q = query(
            cash_flow.code,
            cash_flow.net_operate_cash_flow,
            income.code,
            income.net_profit,
        ).filter(cash_flow.code.in_(list(stocks)))
        df = get_fundamentals(q, date=date_str)
        if df is None or df.empty:
            return None
        # 合并 cash_flow 和 income（get_fundamentals 可能分表返回）
        if 'code' in df.columns:
            df = df.set_index('code')
        # 去重列名（code 出现两次）
        df = df[~df.index.duplicated(keep='first')]
        return df
    except Exception:
        # 降级：分表查
        try:
            q_cf = query(
                cash_flow.code, cash_flow.net_operate_cash_flow
            ).filter(cash_flow.code.in_(list(stocks)))
            df_cf = get_fundamentals(q_cf, date=date_str)
            q_inc = query(
                income.code, income.net_profit
            ).filter(income.code.in_(list(stocks)))
            df_inc = get_fundamentals(q_inc, date=date_str)
            if df_cf is None or df_inc is None:
                return None
            df_cf = df_cf.set_index('code')
            df_inc = df_inc.set_index('code')
            df = df_cf.join(df_inc, how='outer')
            return df if not df.empty else None
        except Exception:
            return None


# ============================================================
# 股息率数据获取（F3 核心，用聚宽因子库 get_factor_values）
# ============================================================

# 分红查询窗口（天），过去 400 天的除权除息记录
DIV_LOOKBACK_DAYS = 400
# 分红查询分批大小（finance.run_query 有行数限制）
DIV_BATCH_SIZE = 300


def _fetch_dividend_data(date_str, stocks, debug=False):
    """用 finance.STK_XR_XD 查询过去 DIV_LOOKBACK_DAYS 天已实施的现金分红。

    返回 {code: 累计每股税前现金分红}。

    STK_XR_XD 表关键字段（探测确认 2026-06）:
      code: 股票代码（带 .XSHE/.XSHG 后缀）
      bonus_ratio_rmb: 每10股派息金额（人民币，含税）→ 除以10得每股
      a_xr_date: A 股除权除息日
      plan_progress: 方案进度（"实施方案"为已实施）

    处理逻辑：
    1. 分批查询（DIV_BATCH_SIZE 只/批，绕过 run_query 行数限制）
    2. 日期过滤：a_xr_date 在 [start, end] 内
    3. 只取已实施的（plan_progress == '实施方案'）
    4. bonus_ratio_rmb > 0
    5. 按 code 累加每股现金分红（bonus_ratio_rmb / 10）
    """
    if not stocks:
        return {}

    d = pd.Timestamp(date_str)
    start = (d - pd.Timedelta(days=DIV_LOOKBACK_DAYS)).strftime('%Y-%m-%d')
    end = date_str

    all_recs = []
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
                finance.STK_XR_XD.a_xr_date >= start,
                finance.STK_XR_XD.a_xr_date <= end,
            )
            df_batch = finance.run_query(q)
            if df_batch is not None and not df_batch.empty:
                all_recs.append(df_batch)
                if debug and i == 0:
                    print('    [debug] 首批查询成功, %d 条, 列=%s' % (
                        len(df_batch), list(df_batch.columns)))
        except Exception as e:
            if debug and i == 0:
                print('    [debug] 查询失败: %s' % str(e)[:100])

    if not all_recs:
        if debug:
            print('    [debug] %s: 分红查询全量为空' % date_str)
        return {}

    df = pd.concat(all_recs, ignore_index=True)
    if debug:
        print('    [debug] %s: 分红记录合并 %d 条' % (date_str, len(df)))

    # 只取已实施的
    if 'plan_progress' in df.columns:
        df = df[df['plan_progress'].astype(str) == '实施方案']
        if debug:
            print('    [debug] %s: 实施过滤后 %d 条' % (date_str, len(df)))

    # bonus_ratio_rmb > 0（每10股派息金额）
    if 'bonus_ratio_rmb' in df.columns:
        df['bonus_ratio_rmb'] = pd.to_numeric(df['bonus_ratio_rmb'], errors='coerce')
        df = df[df['bonus_ratio_rmb'] > 0]
        if debug:
            print('    [debug] %s: 派息>0过滤后 %d 条' % (date_str, len(df)))
    else:
        if debug:
            print('    [debug] %s: 无 bonus_ratio_rmb 列' % date_str)
        return {}

    if df.empty:
        return {}

    # 按 code 累加每股分红（bonus_ratio_rmb 是每10股，除以10得每股）
    div_per_share = df.groupby('code')['bonus_ratio_rmb'].sum() / 10.0

    stocks_set = set(stocks)
    result = {}
    for code, val in div_per_share.items():
        if code in stocks_set:
            result[code] = float(val)

    if debug:
        print('    [debug] %s: 分红匹配 %d/%d 只' % (
            date_str, len(result), len(stocks)))
    return result


def _get_close_prices(date_str, stocks):
    """获取换仓日收盘价（不复权，用于股息率计算）。"""
    if not stocks:
        return {}
    try:
        df = get_price(list(stocks), end_date=date_str, count=1,
                       fields=['close'], skip_paused=False,
                       panel=False, fq=None)
        if df is None or df.empty:
            return {}
        if 'time' in df.columns:
            df = df.set_index('time')
        elif 'date' in df.columns:
            df = df.set_index('date')
        if 'code' in df.columns:
            close = df.pivot_table(index=df.index,
                                   columns='code', values='close')
        else:
            close = df
        if close is None or close.empty:
            return {}
        last_row = close.iloc[-1]
        return {code: float(last_row[code]) for code in close.columns
                if not np.isnan(last_row[code]) and last_row[code] > 0}
    except Exception:
        return {}


# ============================================================
# 内联 factor_lib（股息率因子计算）
# ============================================================

def _calculate_all_factors(df, div_map=None, close_map=None, fin_df=None):
    """F3 股息率因子计算。

    主信号: dividend_yield = 累计每股现金分红 / 收盘价
    对照:   OCF>=net_profit 过滤后的 IC

    输入:
      df: 含 market_cap 列的 DataFrame（index=code）
      div_map: {code: 累计每股税前现金分红}（来自 STK_XR_XD）
      close_map: {code: 换仓日收盘价（不复权）}
      fin_df: 含 net_operate_cash_flow, net_profit 列的 DataFrame
    """
    mcap = _get_col(df, 'market_cap')
    if mcap is None:
        return None
    df['market_cap'] = mcap.astype(float)

    # 股息率 = 每股分红 / 收盘价
    if div_map and close_map:
        df['div_per_share'] = df.index.map(lambda c: div_map.get(c, 0.0))
        df['close_price'] = df.index.map(lambda c: close_map.get(c, np.nan))
        df['dividend_yield'] = df['div_per_share'] / df['close_price'].replace(0, np.nan)
        df['dividend_yield'] = df['dividend_yield'].where(
            df['div_per_share'] > 0, np.nan)
    else:
        df['div_per_share'] = 0.0
        df['close_price'] = np.nan
        df['dividend_yield'] = np.nan

    # OCF 过滤指标
    if fin_df is not None:
        df['ocf'] = df.index.map(
            lambda c: fin_df.loc[c, 'net_operate_cash_flow']
            if c in fin_df.index else np.nan)
        df['net_profit'] = df.index.map(
            lambda c: fin_df.loc[c, 'net_profit']
            if c in fin_df.index else np.nan)
        df['ocf_ratio'] = df['ocf'] / df['net_profit'].replace(0, np.nan)
    else:
        df['ocf'] = np.nan
        df['net_profit'] = np.nan
        df['ocf_ratio'] = np.nan

    # 主信号 = dividend_yield（越高=分红越多=好）
    df['signal'] = df['dividend_yield']

    # OCF 过滤标记（用于对照分析）
    df['ocf_pass'] = (df['ocf_ratio'] >= 1.0).fillna(False)

    return df


# ============================================================
# 工具函数（复用 F2-EP/F5-LowVol）
# ============================================================

def get_rebalance_dates(start, end):
    """季度调仓日：每年 5/9/11 月首个交易日。"""
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
    """Newey-West 调整 t 统计量。"""
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
    """获取 {code: 行业名}。"""
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
# 批量价格缓存（用于 forward_month_return）
# ============================================================
_PRICE_CACHE = {}


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
    """计算 codes 在 date_str 之后约一个月的累计收益。"""
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

    # 1. 市值
    df = _fetch_market_cap(date_str, stocks)
    if df is None or df.empty:
        if debug:
            print('    [debug] %s: market_cap 查询为空' % date_str)
        return None
    if debug:
        print('    [debug] %s: market_cap %d 只' % (date_str, len(df)))

    # 2. 分红数据（finance.STK_XR_XD 除权除息表）
    div_map = _fetch_dividend_data(date_str, list(df.index), debug=debug)
    if debug:
        print('    [debug] %s: 分红数据 %d 只' % (date_str, len(div_map)))

    # 3. 收盘价（不复权，算股息率用）
    close_map = _get_close_prices(date_str, list(df.index))
    if debug:
        print('    [debug] %s: 收盘价 %d 只' % (date_str, len(close_map)))

    # 4. 财务数据（OCF + net_profit，用于过滤对照）
    fin_df = _fetch_financials(date_str, list(df.index))
    if debug:
        n_fin = len(fin_df) if fin_df is not None else 0
        print('    [debug] %s: 财务数据 %d 只' % (date_str, n_fin))

    # 5. 因子计算
    df = _calculate_all_factors(df, div_map=div_map, close_map=close_map,
                                fin_df=fin_df)
    if df is None or df.empty:
        if debug:
            print('    [debug] %s: _calculate_all_factors 返回空' % date_str)
        return None
    if debug:
        n_div = df['dividend_yield'].notna().sum()
        n_ocf = df['ocf_pass'].sum() if 'ocf_pass' in df.columns else 0
        print('    [debug] %s: 因子计算后 %d 只, 有分红=%d, OCF通过=%d' % (
            date_str, len(df), n_div, n_ocf))

    # 6. 过滤：必须有 dividend_yield（有分红）
    mask = df['dividend_yield'].notna() & (df['dividend_yield'] > 0)
    df = df[mask].copy()
    if df.empty:
        if debug:
            print('    [debug] %s: 过滤后为空（无分红股票）' % date_str)
        return None
    if debug:
        print('    [debug] %s: 过滤后 %d 只' % (date_str, len(df)))

    # 7. 行业
    ind_map = get_industry_map(list(df.index), date_str)
    df['industry'] = df.index.map(lambda c: ind_map.get(c, '未知'))

    # 8. 前向收益
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
    ic_list, dates_list = [], []
    ic_filtered_list = []  # OCF 过滤后 IC
    q_rets = {q: [] for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']}

    for i, date in enumerate(rebal_dates):
        df = build_cross_section(date, index_id, debug=(i == 0))
        if df is None:
            continue
        if i == 0:
            print('  [调试] %s: 横截面 %d 只股票 (主信号: dividend_yield)' % (
                date, len(df)))
            v = df['dividend_yield'].dropna()
            if len(v) > 0:
                print('         dividend_yield: %.6f ~ %.6f (中位 %.6f, 非空 %d/%d)' % (
                    v.min(), v.max(), v.median(), len(v), len(df)))
            if 'ocf_ratio' in df.columns:
                or_ = df['ocf_ratio'].dropna()
                if len(or_) > 0:
                    print('         ocf_ratio: %.4f ~ %.4f (中位 %.4f, OCF通过 %d/%d)' % (
                        or_.min(), or_.max(), or_.median(),
                        int(df['ocf_pass'].sum()), len(df)))
            print('         fwd_return 非空: %d' % df['fwd_return'].notna().sum())

        # 主信号 IC（全样本 dividend_yield）
        ic = stats.spearmanr(df['signal'], df['fwd_return'])[0]
        if np.isnan(ic):
            continue
        ic_list.append(ic)
        dates_list.append(date)

        # OCF 过滤后 IC（对照）
        df_filtered = df[df['ocf_pass']] if 'ocf_pass' in df.columns else df
        if len(df_filtered) >= 10:
            ic_f = stats.spearmanr(df_filtered['signal'],
                                   df_filtered['fwd_return'])[0]
            ic_filtered_list.append(0.0 if np.isnan(ic_f) else ic_f)
        else:
            ic_filtered_list.append(np.nan)

        # Q1-Q5：按 signal(dividend_yield) 升序（Q5=最高股息率=多头）
        try:
            df['group'] = pd.qcut(df['signal'].rank(method='first'), 5,
                                  labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        except Exception:
            df['group'] = pd.qcut(df['signal'], 5,
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
            'n_filtered': len(df_filtered) if 'ocf_pass' in df.columns else len(df),
            'rank_ic': ic,
            'ic_mcap_neutral': (np.nan if np.isnan(ic_mcap) else ic_mcap),
            'ic_ind_neutral': (np.nan if np.isnan(ic_ind) else ic_ind),
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
    print('  主信号-dividend_yield: mean=%.6f std=%.6f NW-t=%.4f IC_IR=%.4f 胜率=%.2f%% 正/负=%.2f%%/%.2f%%' % (
        ic_mean, ic_std, ic_t, ic_ir, win_rate * 100, pos_pct * 100, neg_pct * 100))
    # OCF 过滤后对照
    filt_valid = [x for x in ic_filtered_list if not np.isnan(x)]
    if len(filt_valid) > 1:
        f_mean, _, f_t = newey_west_t(pd.Series(filt_valid), lags=4)
        f_std = float(np.std(filt_valid, ddof=1))
        f_ir = f_mean / f_std if f_std > 0 else 0
        print('  对照-OCF过滤后:        mean=%.6f std=%.6f NW-t=%.4f IC_IR=%.4f N=%d' % (
            f_mean, f_std, f_t, f_ir, len(filt_valid)))

    # 表2
    print('\n【表2】Q1-Q5 分组（Q5=最高股息率=多头组）')
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
        print('  dividend_yield 行业中性化: mean=%.6f NW-t=%.4f IC_IR=%.4f N=%d' % (
            m, t, m / float(s.std(ddof=1)), len(s)))
    else:
        print('  样本不足')

    # 表5（决定性门槛）
    print('\n【表5】市值中性化后 Rank IC（决定性门槛 t>=2）')
    s = rec_df['ic_mcap_neutral'].dropna()
    if len(s) > 1:
        m, _, t = newey_west_t(s, lags=4)
        flag = 'PASS' if t >= 2.0 else 'FAIL'
        print('  dividend_yield 市值中性化: mean=%.6f NW-t=%.4f IC_IR=%.4f N=%d [%s]' % (
            m, t, m / float(s.std(ddof=1)), len(s), flag))
    else:
        print('  样本不足')

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
    q_means = [float(np.nanmean(q_rets[q])) for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']]
    q5_gt_q1 = q_means[4] > q_means[0]
    print('  2. Q5多头>Q1:      Q1=%.4f Q5=%.4f %s' % (
        q_means[0], q_means[4], 'OK' if q5_gt_q1 else 'FAIL(倒挂)'))
    print('  3. 全样本NW-t>=2:  t=%.4f %s' % (
        ic_t, 'OK' if ic_t >= 2.0 else 'FAIL'))
    tier_ok = True
    for tier, col in [('小', 'ic_small_cap'), ('中', 'ic_mid_cap'),
                      ('大', 'ic_large_cap')]:
        s_t = rec_df[col].dropna()
        if len(s_t) > 1:
            _, _, t_t = newey_west_t(s_t, lags=4)
            if t_t < 1.0:
                tier_ok = False
    print('  4. 分市值档稳定:   %s' % ('OK' if tier_ok else 'FAIL'))
    s_ind = rec_df['ic_ind_neutral'].dropna()
    ind_t = 0.0
    if len(s_ind) > 1:
        _, _, ind_t = newey_west_t(s_ind, lags=4)
    print('  5. 行业中性t>=2:   t=%.4f %s' % (
        ind_t, 'OK' if ind_t >= 2.0 else 'FAIL'))
    post_ok = False
    if not np.isnan(post_mean):
        s_post = post_ic
        if len(s_post) > 1:
            _, _, post_t = newey_west_t(s_post, lags=4)
            post_ok = post_t >= 1.5
    print('  6. 断点后增强:     %s' % ('OK' if post_ok else 'FAIL'))
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
        'rank_ic_ocf_filtered': ic_filtered_list,
        'ic_mcap_neutral': rec_df['ic_mcap_neutral'].values,
        'ic_ind_neutral': rec_df['ic_ind_neutral'].values,
        'n': rec_df['n'].values,
        'n_filtered': rec_df['n_filtered'].values,
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
        print('>>> 确认数据链正确后，将 QUICK_TEST 改为 False 全量跑')
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
