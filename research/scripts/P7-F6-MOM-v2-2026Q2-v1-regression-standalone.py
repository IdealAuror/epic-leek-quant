"""
P7-F6-MOM-v2-2026Q2-v1 加权回归动量研究脚本（standalone 版）
============================================================

Phase 7 探索脚本：验证加权对数回归动量是否优于两点法（61-21 momentum）。

灵感来源：七星高照 v3.0 ETF 动量轮动策略的方法论
- 对数加权线性回归（近期权重更高）替代两点法
- R² 趋势稳定性作为辅助过滤/加权

对比版本：
  1. f2f5f6_40d         : 两点法 61-21（当前生产版本，baseline）
  2. f2f5f6_wreg        : 加权回归斜率年化（40d信号窗口）
  3. f2f5f6_wreg_r2     : 加权回归动量 × R²（趋势稳定性加权）
  4. f2f5f6_40d_r2filt  : 两点法 + R²<0.4 过滤（剔除假动量）
  5. f2f5f6_wreg24d     : 加权回归（24d短窗口，七星高照原码参数）
  6. f2f5f6_wreg_r2sq   : 加权回归动量 × R²²（更严格惩罚低R²）

防过拟合红线：
  - 训练集 2014-2020，验证集 2021-2026
  - 三样本（AllA/CSI300/CSI500）同时改善才接受
  - 改善幅度 > 2% Sharpe 才切换

在聚宽【研究环境】中直接粘贴运行。

背景：
- P4-PL-2026Q2-v2 baseline 策略 7 段分段回测中 6 段跑赢基准
- 唯一失效段是 2019-2020 核心资产牛市（Alpha -0.11，跑输 46pp）
- 改进方案：加入 F6-MOM（12-1 momentum，Carhart 1997）动量因子

本脚本一次运行即可输出五版本对比：
  1. baseline (F2+F5) 全样本 + 7 段指标
  2. F2F5F6-12-1 (三因子, 12-1 momentum, 信号231天) 全样本 + 7 段指标
  3. F2F5F6-6-1  (三因子, 6-1 momentum,  信号105天) 全样本 + 7 段指标
  4. F2F5F6-40d  (三因子, 61-21 momentum, 信号40天)  全样本 + 7 段指标
  5. F2F5F6-consistency (三因子, 动量一致性过滤) 全样本 + 7 段指标
  6. 五版本差异对比表（重点看段5/6/7）

复用 P4-PL-2026Q2-v2-standalone.py 的回测逻辑（日频净值计算），
增加 F6 动量因子支持和分段统计能力。

变更记录：
- v1: 12-1 momentum 修复段5 但恶化段6/7（窗口信号滞后）
- v2: 新增 6-1 momentum 版本，窗口减半降低信号滞后性
- v3: 新增 40d momentum 版本（61-21），短窗口连续看趋势
      实测全样本Sharpe 0.82最高，段5/6修复，但段7恶化-5.23pp
- v4: 新增动量一致性过滤版本（12-1与6-1同向时启用F6，反向时中性化）
      对症"震荡市动量反复"问题，验证是否能同时修复段5/6/7
- v5: 新增反向F6版本（f2f5_neg_f6_40d），基于IC分析MOM_40d NW-t=-1.63负IC
      假设反向使用F6（选低动量股）更优。实测disprove：正向40d收益694% vs 反向274%
      段5超额正向-2.69pp vs 反向-27.01pp。IC分析局限性：单因子截面排序无法
      衡量组合分散化价值（与F5教训一致）。正向40d确认为最终版本
- v6: 新增跨样本验证模式（CROSS_SAMPLE_MODE），支持CSI300/CSI500验证
      F6仅在AllA验证，需跨样本确认大盘股有效性（规避F1壳价值污染教训）

最终结论：正向40d版本（61-21 momentum）全样本Sharpe 0.82最高，段5修复-2.69pp
达成，策略环境回撤34.93%达标。Phase 5通过，项目收尾。
详见 research/decisions/P5-F6-MOM-2026Q2-v1-decision.md

注：研究环境的回测引擎是手写的（撮合简化），与策略回测环境有小差异。
    本脚本用于快速验证因子改进方向，最终结论仍需策略回测环境确认。

在聚宽【研究环境】中直接粘贴运行。
"""

import datetime
import os
import warnings

import numpy as np
import pandas as pd

# ============================================================
# 聚宽对象解析
# ============================================================
from jqdata import *  # noqa: F403,F401

try:
    from jqdata import (  # noqa: F401
        valuation, balance, income, query,
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

# --- 样本配置 ---
# 切换样本时同时修改 SAMPLE_NAME 和 BENCHMARK_CODE：
#   AllA    -> SAMPLE_NAME='AllA',    BENCHMARK_CODE='000985.XSHG' (中证全指)
#   CSI300  -> SAMPLE_NAME='CSI300',  BENCHMARK_CODE='000300.XSHG' (沪深300)
#   CSI500  -> SAMPLE_NAME='CSI500',  BENCHMARK_CODE='000905.XSHG' (中证500)
SAMPLE_NAME = 'AllA'
BENCHMARK_CODE = '000985.XSHG'  # 中证全指

# 因子参数
VOL_LOOKBACK = 40              # P6 扫描最优：40d (776%/S=0.67) > 60d (732%/S=0.64)
VOL_MIN_OBS_RATIO = 0.5

# F6 动量因子参数（12-1 momentum, Carhart 1997）
MOM_LOOKBACK_LONG = 252      # 12 个月
MOM_SKIP_RECENT = 21         # 剔除最近 1 个月
MOM_MIN_OBS_RATIO = 0.8

# 6-1 momentum 对照版本（窗口减半，降低信号滞后性）
MOM_LOOKBACK_LONG_ALT = 126  # 6 个月

# 40d momentum 版本（61-21，信号长度40天，短窗口连续看趋势）
# 实测全样本Sharpe 0.82最高，段5/6修复，但段7恶化
MOM_LOOKBACK_LONG_40D = 61   # 61 天 = 21(剔除) + 40(信号)

# 成本模型
COMMISSION = 0.0003
STAMP_DUTY = 0.001
SLIPPAGE = 0.001
BASE_ROUND_TRIP = 0.0026     # 双边基础成本
IMPACT_COEFF = 0.1

# 资金规模
TARGET_CAPITAL = 50000000    # 5000 万
LIQUIDITY_THRESHOLD = 10000000  # 1000 万

# 持仓
N_HOLD = 50

# 分段定义（与分段回测报告完全一致）
SEGMENTS = [
    ('段1:大牛市',     '2014-01-01', '2015-06-30'),
    ('段2:股灾',       '2015-07-01', '2016-01-31'),
    ('段3:白马慢牛',   '2016-02-01', '2017-12-31'),
    ('段4:熊市',       '2018-01-01', '2018-12-31'),
    ('段5:核心资产牛市', '2019-01-01', '2020-12-31'),
    ('段6:震荡下跌',   '2021-01-01', '2022-10-31'),
    ('段7:结构性行情', '2022-11-01', '2026-06-30'),
]

OUT_DIR = 'results/P7-F6-MOM-v2-2026Q2-v1'

# P7 加权回归动量参数
R2_FILTER_THRESHOLD = 0.4     # R² 过滤阈值（参考七星高照 v3.0）
WREG_WEIGHT_END = 2.0         # 加权回归近期权重上限（linspace(1, WREG_WEIGHT_END, n)）

# 24 天短窗口版本（七星高照原码参数 LOOKBACK_DAYS=24）
# 信号窗口 = 24 + 21(skip) = 45 天，取 t-22 到 t-45 的收盘价
MOM_LOOKBACK_LONG_24D = 45    # 45 天 = 21(剔除) + 24(信号)

# 训练/验证集分割（防过拟合红线）
TRAIN_END = '2020-12-31'      # 训练集截止日
VALID_START = '2021-01-01'    # 验证集起始日

# QUICK_TEST=True 只跑前 6 个调仓日验证数据链
QUICK_TEST = False

# 跨样本验证模式：True 时一次性跑 AllA + CSI300 + CSI500 三个样本
# 每个样本跑 baseline + 40d+ 两版本对比，输出对比汇总表（不用 PRECOMPUTED）
# 用于确认 F6 在 CSI300/CSI500 大盘股是否有效（规避 F1 壳价值污染教训）
# 使用方法：设 CROSS_SAMPLE_MODE=True 即可，无需手动切换 SAMPLE_NAME
# 验证标准：40d+ Sharpe >= 0.3 且不显著跑输 baseline（大盘股动量弱预期）
CROSS_SAMPLE_MODE = True

# 跨样本验证的样本列表（无需修改，run_cross_sample 自动遍历）
CROSS_SAMPLES = [
    ('AllA', '000985.XSHG'),    # 中证全指（全A股，对照基准）
    ('CSI300', '000300.XSHG'),  # 沪深300（大盘股）
    ('CSI500', '000905.XSHG'),  # 中证500（中盘股）
]


def _get_sample_index_id():
    """Return index_id for _get_stock_pool based on SAMPLE_NAME.

    AllA   -> None (use get_all_securities)
    CSI300 -> '000300.XSHG'
    CSI500 -> '000905.XSHG'
    """
    if SAMPLE_NAME == 'AllA':
        return None
    elif SAMPLE_NAME == 'CSI300':
        return '000300.XSHG'
    elif SAMPLE_NAME == 'CSI500':
        return '000905.XSHG'
    else:
        return None

# ============================================================
# 已跑过的四个版本数据（2026-06-29 运行结果，避免重复跑浪费时间）
# 格式: (全样本收益%, Sharpe, 回撤%, Alpha, 换手均值,
#        [全样本超额pp, 段1超额pp, 段2超额pp, 段3超额pp, 段4超额pp, 段5超额pp, 段6超额pp, 段7超额pp])
# ============================================================
PRECOMPUTED = {
    'baseline': (515.18, 0.78, 39.72, 0.102, 1.608,
                 [394.51, 79.46, 3.99, 26.58, 7.42, -22.40, 40.22, 15.10]),
    'f2f5f6_12-1': (507.46, 0.70, 42.33, 0.100, 1.674,
                    [386.78, 96.47, 2.18, 56.18, 4.71, -0.33, 10.86, 3.45]),
    'f2f5f6_6-1': (334.25, 0.61, 43.87, 0.071, 1.738,
                   [213.58, 87.16, 0.81, 48.13, -3.27, -4.78, 13.52, -13.17]),
    'f2f5f6_40d': (694.64, 0.82, 43.56, 0.123, 1.750,
                   [573.97, 98.84, 4.47, 43.25, 5.99, -2.69, 54.17, -5.23]),
    'f2f5f6_consistency': (395.17, 0.64, 47.58, 0.084, 1.705,
                           [274.49, 94.03, -3.98, 68.85, 5.38, -1.31, 4.61, -10.85]),
}
# 只跑 f2f5_neg_f6_40d 一个新版本（反向F6），前五个版本用 PRECOMPUTED 数据
ONLY_NEG_F6 = True

# 金融股剔除
_FINANCE_NAMES = {'银行I', '非银金融I'}


# ============================================================
# 内联 data_layer（复用 P4 脚本）
# ============================================================

def _get_stock_pool(index_id, date_str, min_listed_days=365):
    """构建股票池：成分股（或全A） -> 剔ST -> 剔次新股 -> 剔金融股。

    min_listed_days=365：动量因子需要 252+21=273 天历史价格。
    """
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
    """PIT 财务数据查询，返回以 code 为 index 的 DataFrame。"""
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
    """计算实现波动率：过去 lookback_days 交易日日收益率标准差。"""
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


def _calc_momentum(date_str, stocks,
                   lookback_long=MOM_LOOKBACK_LONG,
                   skip_recent=MOM_SKIP_RECENT):
    """计算 12-1 动量因子（F6-MOM）。

    MOM_12_1 = price[t-skip_recent] / price[t-lookback_long] - 1
    剔除最近 1 个月避免短期反转污染（Carhart 1997）。
    """
    if not stocks:
        return {}
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


def _calc_momentum_weighted_regression(date_str, stocks,
                                        lookback_long=MOM_LOOKBACK_LONG_40D,
                                        skip_recent=MOM_SKIP_RECENT):
    """加权对数回归动量（P7 新增，参考七星高照 v3.0 方法论）。

    对信号窗口内的 log(price) 做加权线性回归：
    - 近期权重更高（linspace(1, 2, n)），捕捉趋势加速度
    - 斜率年化作为动量信号：mom = exp(slope * 250) - 1
    - R² 衡量趋势稳定性（1=完美线性趋势，0=无趋势）

    信号窗口 = lookback_long - skip_recent = 61 - 21 = 40 天
    取 t-22 到 t-61 的收盘价（跳过最近21天避免短期反转污染）

    R² 计算说明（与七星高照原码一致）：
    - ss_res 用加权残差平方和
    - ss_tot 用未加权均值 np.mean(y)（非加权均值）
    这在统计上略不严谨（加权回归应配加权均值），但为复现原策略保持一致

    返回 (mom_map, r2_map)。
    """
    if not stocks:
        return {}, {}
    signal_len = lookback_long - skip_recent  # 40
    total_count = lookback_long + 5
    try:
        df = get_price(stocks, end_date=date_str, count=total_count,
                       fields=['close'], skip_paused=False,
                       panel=False, fq='post')
        if df is None or df.empty:
            return {}, {}
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
        return {}, {}

    if close is None or close.empty:
        return {}, {}
    if len(close) < lookback_long + 1:
        return {}, {}

    # 信号窗口：跳过最近 skip_recent 天，取前 signal_len 天
    signal_close = close.iloc[-(lookback_long + 1):-(skip_recent)]
    if len(signal_close) < signal_len:
        # 边界情况：取所有可用数据
        signal_close = close.iloc[-(lookback_long + 1):]

    actual_len = len(signal_close)
    if actual_len < int(signal_len * MOM_MIN_OBS_RATIO):
        return {}, {}

    valid_counts = signal_close.count()
    min_obs = int(signal_len * MOM_MIN_OBS_RATIO)

    x = np.arange(actual_len, dtype=float)
    weights = np.linspace(1.0, WREG_WEIGHT_END, actual_len)

    mom_map = {}
    r2_map = {}
    for code in stocks:
        if code not in valid_counts.index:
            continue
        cnt = valid_counts.get(code, 0)
        if cnt < min_obs:
            continue
        # 取价格序列，前向填充缺失值
        prices = signal_close[code].ffill().bfill().values
        if len(prices) != actual_len:
            continue
        prices = prices.astype(float)
        if np.any(prices <= 0) or np.any(np.isnan(prices)):
            continue
        y = np.log(prices)
        try:
            # 加权线性回归
            coeffs = np.polyfit(x, y, 1, w=weights)
            slope = coeffs[0]
            intercept = coeffs[1]
            y_pred = slope * x + intercept
            # 加权 R²（与七星高照原码一致：ss_tot 用未加权均值）
            y_mean = np.mean(y)
            ss_res = np.sum(weights * (y - y_pred) ** 2)
            ss_tot = np.sum(weights * (y - y_mean) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            # 年化收益作为动量信号
            mom = float(np.exp(slope * 250) - 1.0)
            if not np.isnan(mom) and np.isfinite(mom):
                mom_map[code] = mom
                r2_map[code] = float(r2)
        except Exception:
            continue
    return mom_map, r2_map


def _calc_avg_volume(date_str, stocks, lookback_days=20):
    """计算日均成交额（流动性过滤用）。"""
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
    return dict(wide.mean())


# ============================================================
# 工具函数
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


def neutralize_ols(factor_values, regressor):
    """OLS 残差中性化。"""
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


# ============================================================
# 横截面构建（支持 baseline 和 f2f5f6 两种模式）
# ============================================================

def build_cross_section(date, mode='baseline', debug=False):
    """构建横截面。

    mode='baseline':           F2-EP + F5-LV 双因子
    mode='f2f5f6':             F2-EP + F5-LV + F6-MOM(12-1, 信号231天) 三因子
    mode='f2f5f6_6_1':         F2-EP + F5-LV + F6-MOM(6-1,  信号105天) 三因子
    mode='f2f5f6_40d':         F2-EP + F5-LV + F6-MOM(61-21, 信号40天)  三因子
    mode='f2f5f6_consistency': F2-EP + F5-LV + F6(动量一致性过滤) 三因子
        - 同时计算 12-1 和 6-1 动量
        - 同向时 F6 = MOM_12_1（正常发挥动量作用）
        - 反向时 F6 = 0（中性化，相当于回到双因子排序）
        - 趋势市（段5）大部分同向→F6正常；震荡市（段6/7）多反向→F6被中性化

    返回 DataFrame，含 F2_ep / F5_vol (and F6_mom if mode is f2f5f6*).
    """
    date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
    stocks = _get_stock_pool(_get_sample_index_id(), date_str)
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

    # F5 波动率
    vol_map = _calc_realized_volatility(date_str, list(df.index), VOL_LOOKBACK)
    df['vol_60d'] = df.index.map(lambda c: vol_map.get(c, np.nan))
    df['F5_vol_raw'] = -df['vol_60d']

    # F6 动量
    use_f6 = mode in ('f2f5f6', 'f2f5f6_6_1', 'f2f5f6_40d', 'f2f5f6_consistency',
                      'f2f5_neg_f6_40d', 'f2f5f6_wreg', 'f2f5f6_wreg_r2',
                      'f2f5f6_40d_r2filt', 'f2f5f6_wreg24d', 'f2f5f6_wreg_r2sq')
    if use_f6:
        if mode == 'f2f5f6':
            # 12-1 momentum（默认长窗口 252，信号长度231天）
            mom_map = _calc_momentum(date_str, list(df.index),
                                     lookback_long=MOM_LOOKBACK_LONG,
                                     skip_recent=MOM_SKIP_RECENT)
            df['F6_mom_raw'] = df.index.map(lambda c: mom_map.get(c, np.nan))
            mask &= df['F6_mom_raw'].notna() & np.isfinite(df['F6_mom_raw'])
        elif mode == 'f2f5f6_6_1':
            # 6-1 momentum（短窗口 126，信号长度105天）
            mom_map = _calc_momentum(date_str, list(df.index),
                                     lookback_long=MOM_LOOKBACK_LONG_ALT,
                                     skip_recent=MOM_SKIP_RECENT)
            df['F6_mom_raw'] = df.index.map(lambda c: mom_map.get(c, np.nan))
            mask &= df['F6_mom_raw'].notna() & np.isfinite(df['F6_mom_raw'])
        elif mode in ('f2f5f6_40d', 'f2f5_neg_f6_40d'):
            # 40d momentum（61-21，信号长度40天）
            # 反向版本F6计算与正向相同，区别在build_portfolio里用负号
            mom_map = _calc_momentum(date_str, list(df.index),
                                     lookback_long=MOM_LOOKBACK_LONG_40D,
                                     skip_recent=MOM_SKIP_RECENT)
            df['F6_mom_raw'] = df.index.map(lambda c: mom_map.get(c, np.nan))
            mask &= df['F6_mom_raw'].notna() & np.isfinite(df['F6_mom_raw'])
        elif mode in ('f2f5f6_wreg', 'f2f5f6_wreg_r2', 'f2f5f6_wreg24d',
                      'f2f5f6_wreg_r2sq'):
            # P7: 加权对数回归动量（参考七星高照 v3.0 方法论）
            wreg_lookback = MOM_LOOKBACK_LONG_24D if mode == 'f2f5f6_wreg24d' \
                            else MOM_LOOKBACK_LONG_40D
            wreg_mom, wreg_r2 = _calc_momentum_weighted_regression(
                date_str, list(df.index),
                lookback_long=wreg_lookback,
                skip_recent=MOM_SKIP_RECENT)
            if mode in ('f2f5f6_wreg', 'f2f5f6_wreg24d'):
                df['F6_mom_raw'] = df.index.map(lambda c: wreg_mom.get(c, np.nan))
            elif mode == 'f2f5f6_wreg_r2':
                # 动量 × R²（趋势稳定性加权）
                df['F6_mom_raw'] = df.index.map(
                    lambda c: wreg_mom.get(c, np.nan) * wreg_r2.get(c, 0)
                    if c in wreg_mom else np.nan)
            else:
                # f2f5f6_wreg_r2sq: 动量 × R²²（更严格惩罚低R²，七星高照注释变体）
                df['F6_mom_raw'] = df.index.map(
                    lambda c: wreg_mom.get(c, np.nan) * wreg_r2.get(c, 0) ** 2
                    if c in wreg_mom else np.nan)
            mask &= df['F6_mom_raw'].notna() & np.isfinite(df['F6_mom_raw'])
        elif mode == 'f2f5f6_40d_r2filt':
            # P7: 两点法动量 + R² 过滤（剔除趋势不稳定的假动量）
            mom_map = _calc_momentum(date_str, list(df.index),
                                     lookback_long=MOM_LOOKBACK_LONG_40D,
                                     skip_recent=MOM_SKIP_RECENT)
            _, r2_map = _calc_momentum_weighted_regression(
                date_str, list(df.index),
                lookback_long=MOM_LOOKBACK_LONG_40D,
                skip_recent=MOM_SKIP_RECENT)
            df['F6_mom_raw'] = df.index.map(lambda c: mom_map.get(c, np.nan))
            r2_series = pd.Series(df.index.map(lambda c: r2_map.get(c, 0)),
                                  index=df.index)
            mask &= r2_series >= R2_FILTER_THRESHOLD
            mask &= df['F6_mom_raw'].notna() & np.isfinite(df['F6_mom_raw'])
        else:
            # 动量一致性过滤：同时计算 12-1 和 6-1
            mom_12_map = _calc_momentum(date_str, list(df.index),
                                        lookback_long=MOM_LOOKBACK_LONG,
                                        skip_recent=MOM_SKIP_RECENT)
            mom_6_map = _calc_momentum(date_str, list(df.index),
                                       lookback_long=MOM_LOOKBACK_LONG_ALT,
                                       skip_recent=MOM_SKIP_RECENT)
            df['mom_12_1'] = df.index.map(lambda c: mom_12_map.get(c, np.nan))
            df['mom_6_1'] = df.index.map(lambda c: mom_6_map.get(c, np.nan))
            # 两个动量都需要有效值
            mask &= df['mom_12_1'].notna() & df['mom_6_1'].notna()
            # 一致性判断：同向时用 12-1，反向时置 0（中性化）
            concord = (np.sign(df['mom_12_1']) == np.sign(df['mom_6_1']))
            df['F6_mom_raw'] = np.where(concord, df['mom_12_1'], 0.0)

    # 流动性过滤
    avg_vol_map = _calc_avg_volume(date_str, list(df.index))
    df['avg_money'] = df.index.map(lambda c: avg_vol_map.get(c, 0))
    mask &= df['avg_money'].fillna(0) >= LIQUIDITY_THRESHOLD

    drop_cols = ['F2_ep_raw', 'F5_vol_raw', 'market_cap']
    if use_f6:
        drop_cols.append('F6_mom_raw')
    df = df.dropna(subset=drop_cols).copy()
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
    if use_f6:
        raw = df['F6_mom_raw'].astype(float)
        neut = neutralize_ols(raw.values, log_mcap.values)
        neut = winsorize_cross_section(pd.Series(neut, index=df.index))
        df['F6_mom'] = neut

    if debug:
        print('    [debug] %s %s: 横截面 %d 只' % (date_str, mode, len(df)))
    return df


def build_portfolio(df, mode='baseline', n_hold=N_HOLD):
    """构建组合，返回 (持仓 codes, 持仓 weights)。

    mode='baseline':           F2+F5 双因子等权
    mode='f2f5f6':             F2+F5+F6(12-1) 三因子等权（正向F6）
    mode='f2f5f6_6_1':         F2+F5+F6(6-1)  三因子等权（正向F6）
    mode='f2f5f6_40d':         F2+F5+F6(40d)  三因子等权（正向F6）
    mode='f2f5f6_consistency': F2+F5+F6(一致性过滤) 三因子等权
    mode='f2f5_neg_f6_40d':    F2+F5-F6(40d)  三因子等权（反向F6，选低动量股）
        - IC分析显示A股动量整体负IC（反转效应），反向使用F6
        - score = (F2_z + F5_z - F6_z) / 3，选低动量+低估值+低波动
    """
    df = df.copy()
    df['F2_z'] = zscore_cross_section(df['F2_ep'])
    df['F5_z'] = zscore_cross_section(df['F5_vol'])
    if mode == 'f2f5_neg_f6_40d':
        # 反向F6：选低动量股（反转策略）
        df['F6_z'] = zscore_cross_section(df['F6_mom'])
        df['score'] = (1.0 / 3.0) * df['F2_z'] + (1.0 / 3.0) * df['F5_z'] - (1.0 / 3.0) * df['F6_z']
    elif mode in ('f2f5f6', 'f2f5f6_6_1', 'f2f5f6_40d', 'f2f5f6_consistency',
                  'f2f5f6_wreg', 'f2f5f6_wreg_r2', 'f2f5f6_40d_r2filt',
                  'f2f5f6_wreg24d', 'f2f5f6_wreg_r2sq'):
        df['F6_z'] = zscore_cross_section(df['F6_mom'])
        # 三因子等权（各 1/3）
        df['score'] = (1.0 / 3.0) * df['F2_z'] + (1.0 / 3.0) * df['F5_z'] + (1.0 / 3.0) * df['F6_z']
    else:
        # baseline 双因子等权
        df['score'] = 0.5 * df['F2_z'] + 0.5 * df['F5_z']

    n = min(n_hold, len(df) // 5)
    n = max(n, 10)
    df_sel = df.nlargest(n, 'score')
    weights = zscore_cross_section(df_sel['score']).clip(lower=0)
    if weights.sum() > 0:
        weights = weights / weights.sum()
    return list(df_sel.index), weights


# ============================================================
# 日频净值计算
# ============================================================

def compute_daily_nav(date_pairs, mode='baseline', target_capital=TARGET_CAPITAL,
                      apply_cost=True, apply_impact=True):
    """计算日频净值，含成本和冲击成本。

    返回:
      daily_nav: pd.Series, 日频净值
      rebal_records: list, 每次调仓记录
    """
    print('  [%s] 开始计算日频净值...' % mode)
    all_trade_days = get_trade_days(start_date=START_DATE, end_date=END_DATE)
    daily_nav = pd.Series(index=pd.to_datetime(all_trade_days), dtype=float)
    daily_nav.iloc[0] = 1.0

    rebal_records = []
    prev_holdings = {}

    for idx, (date, next_date) in enumerate(date_pairs):
        date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
        next_date_str = next_date.strftime('%Y-%m-%d') if hasattr(next_date, 'strftime') else str(next_date)

        df = build_cross_section(date, mode=mode, debug=(idx == 0))
        if df is None or len(df) < 30:
            print('  [%s] %s: 横截面构建失败，跳过' % (mode, date_str))
            continue

        codes, weights = build_portfolio(df, mode=mode, n_hold=N_HOLD)
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

        # 冲击成本
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

        px = _load_period_prices(date_str, next_date_str, codes)
        if px is None or px.empty:
            continue
        px.index = pd.to_datetime(px.index)

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
        })
        prev_holdings = new_holdings
        if (idx + 1) % 6 == 0:
            print('  [%s] 进度 %d/%d, %s: n=%d turnover=%.3f' % (
                mode, idx + 1, len(date_pairs), date_str, len(codes), turnover))

    daily_nav = daily_nav.dropna()
    print('  [%s] 日频净值计算完成，共 %d 天' % (mode, len(daily_nav)))
    return daily_nav, rebal_records


# ============================================================
# 基准日净值
# ============================================================

def compute_benchmark_nav():
    """计算基准（中证全指）日频净值。"""
    print('  [基准] 加载 %s 日频价格...' % BENCHMARK_CODE)
    try:
        df = get_price(BENCHMARK_CODE, start_date=START_DATE, end_date=END_DATE,
                       fields=['close'], skip_paused=False, fq='post')
        if df is None or df.empty:
            return None
        df.index = pd.to_datetime(df.index)
        nav = df['close'] / df['close'].iloc[0]
        return nav
    except Exception as e:
        print('  [基准] 加载失败: %s' % e)
        return None


# ============================================================
# 分段统计
# ============================================================

def compute_segment_stats(daily_nav, benchmark_nav, segment_name,
                          start_str, end_str):
    """计算单段的收益/Sharpe/回撤/Alpha 等指标。"""
    start_ts = pd.Timestamp(start_str)
    end_ts = pd.Timestamp(end_str)

    # 切分（用 <= end_ts，包含 end_date 当天）
    nav_seg = daily_nav[(daily_nav.index >= start_ts) & (daily_nav.index <= end_ts)]
    bench_seg = benchmark_nav[(benchmark_nav.index >= start_ts) & (benchmark_nav.index <= end_ts)]

    if len(nav_seg) < 2 or len(bench_seg) < 2:
        return {
            'segment': segment_name,
            'period': '%s ~ %s' % (start_str[:10], end_str[:10]),
            'strategy_return': np.nan,
            'benchmark_return': np.nan,
            'excess': np.nan,
            'alpha': np.nan,
            'sharpe': np.nan,
            'max_drawdown': np.nan,
            'n_days': len(nav_seg),
        }

    # 收益率
    strategy_return = float(nav_seg.iloc[-1] / nav_seg.iloc[0] - 1)
    benchmark_return = float(bench_seg.iloc[-1] / bench_seg.iloc[0] - 1)
    excess = strategy_return - benchmark_return

    # 日收益率
    strat_daily_ret = nav_seg.pct_change().dropna()
    bench_daily_ret = bench_seg.pct_change().dropna()

    # Sharpe（年化，无风险利率=0）
    if strat_daily_ret.std() > 0:
        sharpe = float(strat_daily_ret.mean() / strat_daily_ret.std() * np.sqrt(252))
    else:
        sharpe = 0.0

    # 最大回撤
    cummax = nav_seg.cummax()
    drawdown = (nav_seg - cummax) / cummax
    max_dd = float(-drawdown.min()) if len(drawdown) > 0 else 0.0

    # Alpha/Beta（对基准回归）
    common_idx = strat_daily_ret.index.intersection(bench_daily_ret.index)
    if len(common_idx) > 30:
        s = strat_daily_ret.loc[common_idx]
        b = bench_daily_ret.loc[common_idx]
        # 简单 OLS: s = alpha + beta * b
        x_mat = np.column_stack([np.ones(len(b)), b.values])
        try:
            beta_vec = np.linalg.lstsq(x_mat, s.values, rcond=None)[0]
            alpha_daily = float(beta_vec[0])
            alpha = alpha_daily * 252  # 年化
        except Exception:
            alpha = np.nan
    else:
        alpha = np.nan

    return {
        'segment': segment_name,
        'period': '%s ~ %s' % (start_str[:10], end_str[:10]),
        'strategy_return': strategy_return,
        'benchmark_return': benchmark_return,
        'excess': excess,
        'alpha': alpha,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'n_days': len(nav_seg),
    }


def compute_full_stats(daily_nav, benchmark_nav):
    """计算全样本指标。"""
    return compute_segment_stats(daily_nav, benchmark_nav,
                                 '全样本', START_DATE, END_DATE)


# ============================================================
# 主分析
# ============================================================

def run():
    """主入口：只跑 f2f5_neg_f6_40d（反向F6），与前五个版本（PRECOMPUTED）对比。

    IC分析显示F6-MOM整体负IC（A股反转效应），反向使用F6可能更有效。
    反向F6：score = (F2_z + F5_z - F6_z) / 3，选低动量+低估值+低波动。
    """
    os.makedirs(OUT_DIR, exist_ok=True)

    print('=' * 70)
    print('Phase 5 F6-MOM 分段回测分析（六版本对比，只跑反向F6）')
    print('=' * 70)
    print('样本: %s' % SAMPLE_NAME)
    print('基准: %s' % BENCHMARK_CODE)
    print('时段: %s ~ %s' % (START_DATE, END_DATE))
    print('资金: %d 万' % (TARGET_CAPITAL / 1e4))
    print('分段数: %d' % len(SEGMENTS))
    print('版本:')
    print('  - baseline           : F2+F5 双因子 [PRECOMPUTED]')
    print('  - f2f5f6_12-1        : F2+F5+F6(12-1正向) [PRECOMPUTED]')
    print('  - f2f5f6_6-1         : F2+F5+F6(6-1正向)  [PRECOMPUTED]')
    print('  - f2f5f6_40d         : F2+F5+F6(40d正向)  [PRECOMPUTED]')
    print('  - f2f5f6_consistency : F2+F5+F6(一致性)   [PRECOMPUTED]')
    print('  - f2f5_neg_f6_40d    : F2+F5-F6(40d反向)  [实时计算]')
    print('  IC分析依据: MOM_40d 市值中性化 NW-t=-1.63（负IC），')
    print('              反向使用F6（选低动量股）可能提升Alpha')
    print('=' * 70)

    # 1. 调仓日
    rebal_dates = get_rebalance_dates(START_DATE, END_DATE)
    if QUICK_TEST:
        rebal_dates = rebal_dates[:6]
        print('>>> [快速模式] 只跑前 6 个调仓日: %s' % rebal_dates)
    if len(rebal_dates) < 2:
        print('调仓日不足，退出')
        return

    # 调仓日配对
    date_pairs = []
    for i in range(len(rebal_dates) - 1):
        date_pairs.append((rebal_dates[i], rebal_dates[i + 1]))
    last_date = rebal_dates[-1]
    date_pairs.append((last_date, pd.Timestamp(END_DATE).date()))

    # 2. 只跑 反向F6
    print('\n>>> 运行 f2f5_neg_f6_40d (F2+F5-F6, 反向40d动量，选低动量股) ...')
    neg_nav, neg_records = compute_daily_nav(
        date_pairs, mode='f2f5_neg_f6_40d', target_capital=TARGET_CAPITAL)

    # 3. 基准
    print('\n>>> 加载基准 ...')
    benchmark_nav = compute_benchmark_nav()
    if benchmark_nav is None:
        print('基准加载失败，无法计算 Alpha')
        return

    # 4. 反向F6 分段统计
    print('\n' + '=' * 70)
    print('分段统计结果')
    print('=' * 70)

    neg_stats = [compute_full_stats(neg_nav, benchmark_nav)]
    for seg_name, start, end in SEGMENTS:
        neg_stats.append(compute_segment_stats(
            neg_nav, benchmark_nav, seg_name, start, end))

    # 5. 输出 反向F6 分段指标表
    print('\n【表1】f2f5_neg_f6_40d (F2+F5-F6, 反向40d动量) 分段指标')
    print('%-22s %-22s %10s %10s %10s %8s %8s %8s %6s' % (
        '段', '时段', '策略收益', '基准收益', '超额', 'Alpha', 'Sharpe', '回撤', '天数'))
    for r in neg_stats:
        print('%-22s %-22s %9.2f%% %9.2f%% %9.2fpp %7.3f %7.2f %7.2f%% %5d' % (
            r['segment'], r['period'],
            r['strategy_return'] * 100, r['benchmark_return'] * 100,
            r['excess'] * 100, r['alpha'] if not np.isnan(r['alpha']) else 0,
            r['sharpe'], r['max_drawdown'] * 100, r['n_days']))

    # 6. 六版本超额收益对比（前五个用 PRECOMPUTED，反向F6用实时数据）
    print('\n【表2】六版本超额收益对比（重点看段5/6/7）')
    print('%-22s %9s %9s %9s %9s %11s %11s' % (
        '段', 'baseline', '12-1+', '6-1+', '40d+', 'consist', '40d-'))
    seg_labels = ['全样本'] + [s[0] for s in SEGMENTS]
    for i, label in enumerate(seg_labels):
        b_excess = PRECOMPUTED['baseline'][5][i]
        f12_excess = PRECOMPUTED['f2f5f6_12-1'][5][i]
        f6_excess = PRECOMPUTED['f2f5f6_6-1'][5][i]
        f40_excess = PRECOMPUTED['f2f5f6_40d'][5][i]
        c_excess = PRECOMPUTED['f2f5f6_consistency'][5][i]
        n_excess = neg_stats[i]['excess'] * 100
        print('%-22s %8.2fpp %8.2fpp %8.2fpp %8.2fpp %10.2fpp %10.2fpp' % (
            label, b_excess, f12_excess, f6_excess, f40_excess, c_excess, n_excess))

    # 7. 六版本全样本核心指标
    print('\n【表3】六版本全样本核心指标对比')
    print('%-22s %10s %10s %10s %10s %10s' % (
        '版本', '总收益', 'Sharpe', '回撤', 'Alpha', '换手均值'))
    for ver_name in ['baseline', 'f2f5f6_12-1', 'f2f5f6_6-1',
                     'f2f5f6_40d', 'f2f5f6_consistency']:
        ret, sharpe, dd, alpha, turn, _ = PRECOMPUTED[ver_name]
        print('%-22s %9.2f%% %9.2f %9.2f%% %9.3f %9.3f' % (
            ver_name, ret, sharpe, dd, alpha, turn))
    full_n = neg_stats[0]
    print('%-22s %9.2f%% %9.2f %9.2f%% %9.3f %9.3f' % (
        'f2f5_neg_f6_40d', full_n['strategy_return'] * 100, full_n['sharpe'],
        full_n['max_drawdown'] * 100,
        full_n['alpha'] if not np.isnan(full_n['alpha']) else 0,
        np.mean([r['turnover'] for r in neg_records])))

    # 8. 关键判定（对比正向40d和反向40d）
    print('\n' + '=' * 70)
    print('关键判定（正向40d vs 反向40d，IC分析的实战验证）')
    print('=' * 70)

    # 段5 修复
    seg5_b_excess = PRECOMPUTED['baseline'][5][5]
    seg5_40_excess = PRECOMPUTED['f2f5f6_40d'][5][5]
    seg5_n = neg_stats[5]
    print('\n[1] 段5（核心资产牛市）检查:')
    print('    baseline:         超额 %.2fpp' % seg5_b_excess)
    print('    f2f5f6_40d(正向): 超额 %.2fpp' % seg5_40_excess)
    print('    f2f5_neg_f6(反向): 超额 %.2fpp' % (seg5_n['excess'] * 100))
    if seg5_40_excess >= -10:
        print('    ✅ 正向40d 修复段5 (超额 >= -10pp)')
    else:
        print('    ❌ 正向40d 未修复段5')
    if seg5_n['excess'] * 100 >= -10:
        print('    ✅ 反向40d 段5 超额 >= -10pp')
    else:
        print('    ⚠️ 反向40d 段5 超额 < -10pp（反向F6在趋势市可能恶化）')

    # 段6/7
    print('\n[2] 段6/7 检查（正向40d vs 反向40d）:')
    for seg_idx, seg_label in [(6, '段6'), (7, '段7')]:
        b_excess = PRECOMPUTED['baseline'][5][seg_idx]
        f40_excess = PRECOMPUTED['f2f5f6_40d'][5][seg_idx]
        n_stat = neg_stats[seg_idx]
        print('    %s:' % seg_label)
        print('      baseline:         超额 %.2fpp' % b_excess)
        print('      f2f5f6_40d(正向): 超额 %.2fpp (相对baseline %.2fpp)' % (
            f40_excess, f40_excess - b_excess))
        print('      f2f5_neg(反向):   超额 %.2fpp (相对baseline %.2fpp)' % (
            n_stat['excess'] * 100, n_stat['excess'] * 100 - b_excess))

    # 全样本指标
    print('\n[3] 全样本指标对比（正向40d vs 反向40d）:')
    f40_ret, f40_sharpe, f40_dd, _, _, _ = PRECOMPUTED['f2f5f6_40d']
    print('    正向40d: 收益 %.2f%%, Sharpe %.2f, 回撤 %.2f%%' % (
        f40_ret, f40_sharpe, f40_dd))
    print('    反向40d: 收益 %.2f%%, Sharpe %.2f, 回撤 %.2f%%' % (
        full_n['strategy_return'] * 100, full_n['sharpe'],
        full_n['max_drawdown'] * 100))
    if f40_sharpe >= 0.5:
        print('    ✅ 正向40d Sharpe >= 0.5')
    else:
        print('    ❌ 正向40d Sharpe < 0.5')
    if full_n['sharpe'] >= 0.5:
        print('    ✅ 反向40d Sharpe >= 0.5')
    else:
        print('    ❌ 反向40d Sharpe < 0.5')
    if f40_dd <= 40:
        print('    ✅ 正向40d 回撤 <= 40%%')
    else:
        print('    ❌ 正向40d 回撤 > 40%%')
    if full_n['max_drawdown'] <= 0.40:
        print('    ✅ 反向40d 回撤 <= 40%%')
    else:
        print('    ❌ 反向40d 回撤 > 40%%')

    # 9. CSV 落盘
    df_n = pd.DataFrame(neg_stats)
    df_n['version'] = 'f2f5_neg_f6_40d'
    df_n.to_csv('%s/segmented_neg_f6.csv' % OUT_DIR, index=False)
    print('\nCSV 已保存到 %s/segmented_neg_f6.csv' % OUT_DIR)

    # 日净值落盘
    nav_df = pd.DataFrame({
        'f2f5_neg_f6_40d': neg_nav,
        'benchmark': benchmark_nav,
    })
    nav_df.to_csv('%s/daily_nav_neg_f6.csv' % OUT_DIR)
    print('日净值已保存到 %s/daily_nav_neg_f6.csv' % OUT_DIR)

    print('\n' + '=' * 70)
    print('分析完成')
    print('=' * 70)


def run_cross_sample():
    """Cross-sample verification: run baseline + 40d+ for AllA/CSI300/CSI500 at once.

    Purpose: F6 was only validated on AllA. F1 lesson (shell value pollution)
    warns that factors effective on AllA may fail on large-cap stocks.
    This mode runs two versions (baseline + 40d+) on three samples in one pass,
    outputs a side-by-side comparison table + a summary gate check.

    Usage:
      1. Set CROSS_SAMPLE_MODE = True
      2. Run in JoinQuant research environment (no need to switch SAMPLE_NAME)

    Pass criteria (per sample):
      - 40d+ Sharpe >= 0.3 (large-cap momentum expected to be weaker)
      - 40d+ does not significantly underperform baseline (excess >= -10pp)
      - Segment 5 (core asset bull market) excess >= -10pp (F6 should fix seg5)
    """
    global SAMPLE_NAME, BENCHMARK_CODE
    os.makedirs(OUT_DIR, exist_ok=True)

    print('=' * 70)
    print('Phase 5 F6-MOM Cross-Sample Verification (AllA + CSI300 + CSI500)')
    print('=' * 70)
    print('Period: %s ~ %s' % (START_DATE, END_DATE))
    print('Capital: %d wan' % (TARGET_CAPITAL / 1e4))
    print('Versions: baseline (F2+F5) vs 40d+ (F2+F5+F6, 61-21 momentum)')
    print('Samples: %s' % ', '.join([s[0] for s in CROSS_SAMPLES]))
    print('=' * 70)

    # 1. Rebalance dates (shared across samples)
    rebal_dates = get_rebalance_dates(START_DATE, END_DATE)
    if QUICK_TEST:
        rebal_dates = rebal_dates[:6]
        print('>>> [QUICK_TEST] Only running first 6 rebalance dates: %s' % rebal_dates)
    if len(rebal_dates) < 2:
        print('Insufficient rebalance dates, exiting')
        return

    date_pairs = []
    for i in range(len(rebal_dates) - 1):
        date_pairs.append((rebal_dates[i], rebal_dates[i + 1]))
    last_date = rebal_dates[-1]
    date_pairs.append((last_date, pd.Timestamp(END_DATE).date()))

    # 2. Run each sample: baseline + 40d+
    all_results = {}  # {sample_name: {'base_stats':..., 'f40_stats':..., 'base_turn':, 'f40_turn':, 'benchmark_nav':}}
    for sample_name, benchmark_code in CROSS_SAMPLES:
        print('\n' + '#' * 70)
        print('# Sample: %s (benchmark: %s)' % (sample_name, benchmark_code))
        print('#' * 70)

        # Switch global sample
        SAMPLE_NAME = sample_name
        BENCHMARK_CODE = benchmark_code

        # Run baseline
        print('\n>>> [%s] Running baseline (F2+F5) ...' % sample_name)
        base_nav, base_records = compute_daily_nav(
            date_pairs, mode='baseline', target_capital=TARGET_CAPITAL)

        # Run 40d+
        print('\n>>> [%s] Running f2f5f6_40d (F2+F5+F6, 40d momentum) ...' % sample_name)
        f40_nav, f40_records = compute_daily_nav(
            date_pairs, mode='f2f5f6_40d', target_capital=TARGET_CAPITAL)

        # Benchmark
        print('\n>>> [%s] Loading benchmark ...' % sample_name)
        benchmark_nav = compute_benchmark_nav()
        if benchmark_nav is None:
            print('Benchmark load failed for %s, skipping' % sample_name)
            continue

        # Segment stats
        base_stats = [compute_full_stats(base_nav, benchmark_nav)]
        f40_stats = [compute_full_stats(f40_nav, benchmark_nav)]
        for seg_name, start, end in SEGMENTS:
            base_stats.append(compute_segment_stats(
                base_nav, benchmark_nav, seg_name, start, end))
            f40_stats.append(compute_segment_stats(
                f40_nav, benchmark_nav, seg_name, start, end))

        all_results[sample_name] = {
            'base_stats': base_stats,
            'f40_stats': f40_stats,
            'base_turn': np.mean([r['turnover'] for r in base_records]),
            'f40_turn': np.mean([r['turnover'] for r in f40_records]),
            'benchmark_nav': benchmark_nav,
        }

        # Save per-sample CSV
        suffix = sample_name.lower()
        df_out = pd.DataFrame(f40_stats)
        df_out['version'] = 'f2f5f6_40d'
        df_out['sample'] = sample_name
        df_out.to_csv('%s/cross_sample_40d_%s.csv' % (OUT_DIR, suffix), index=False)
        nav_df = pd.DataFrame({
            'baseline_%s' % suffix: base_nav,
            'f2f5f6_40d_%s' % suffix: f40_nav,
            'benchmark': benchmark_nav,
        })
        nav_df.to_csv('%s/daily_nav_cross_%s.csv' % (OUT_DIR, suffix))
        print('>>> [%s] CSV saved (cross_sample_40d_%s.csv + daily_nav_cross_%s.csv)'
              % (sample_name, suffix, suffix))

    if not all_results:
        print('\nNo sample completed successfully, exiting')
        return

    # 3. Comparison Table 1: Full-sample core metrics across samples
    print('\n' + '=' * 70)
    print('[Table 1] Full-Sample Core Metrics Across Samples')
    print('=' * 70)
    print('%-10s %-10s %10s %10s %10s %10s %10s' % (
        'Sample', 'Version', 'Return', 'Sharpe', 'DD', 'Alpha', 'Turnover'))
    for sample_name in [s[0] for s in CROSS_SAMPLES]:
        if sample_name not in all_results:
            continue
        r = all_results[sample_name]
        bs = r['base_stats'][0]
        fs = r['f40_stats'][0]
        print('%-10s %-10s %9.2f%% %9.2f %9.2f%% %9.3f %9.3f' % (
            sample_name, 'baseline',
            bs['strategy_return'] * 100, bs['sharpe'],
            bs['max_drawdown'] * 100,
            bs['alpha'] if not np.isnan(bs['alpha']) else 0,
            r['base_turn']))
        print('%-10s %-10s %9.2f%% %9.2f %9.2f%% %9.3f %9.3f' % (
            sample_name, '40d+',
            fs['strategy_return'] * 100, fs['sharpe'],
            fs['max_drawdown'] * 100,
            fs['alpha'] if not np.isnan(fs['alpha']) else 0,
            r['f40_turn']))

    # 4. Comparison Table 2: Excess (40d+ vs baseline) across samples and segments
    print('\n' + '=' * 70)
    print('[Table 2] 40d+ Excess vs Baseline by Segment (pp)')
    print('=' * 70)
    header = '%-22s' % 'Segment'
    for sample_name in [s[0] for s in CROSS_SAMPLES]:
        if sample_name in all_results:
            header += ' %10s' % sample_name
    print(header)
    seg_labels = ['全样本'] + [s[0] for s in SEGMENTS]
    for seg_idx in range(len(SEGMENTS) + 1):
        line = '%-22s' % seg_labels[seg_idx]
        for sample_name in [s[0] for s in CROSS_SAMPLES]:
            if sample_name not in all_results:
                continue
            r = all_results[sample_name]
            diff = (r['f40_stats'][seg_idx]['excess']
                    - r['base_stats'][seg_idx]['excess']) * 100
            line += ' %9.2fpp' % diff
        print(line)

    # 5. Summary Gate Check across samples
    print('\n' + '=' * 70)
    print('[Table 3] Cross-Sample Gate Check Summary')
    print('=' * 70)
    print('%-10s %-22s %-22s %-22s %-10s' % (
        'Sample', '[1] 40d+ Sharpe>=0.3', '[2] Excess>=-10pp',
        '[3] Seg5 Excess>=-10pp', 'Verdict'))
    overall_pass = True
    for sample_name in [s[0] for s in CROSS_SAMPLES]:
        if sample_name not in all_results:
            continue
        r = all_results[sample_name]
        f40_sharpe = r['f40_stats'][0]['sharpe']
        f40_excess_vs_base = (r['f40_stats'][0]['excess']
                              - r['base_stats'][0]['excess']) * 100
        seg5_f40_excess = r['f40_stats'][5]['excess'] * 100

        c1 = f40_sharpe >= 0.3
        c2 = f40_excess_vs_base >= -10
        c3 = seg5_f40_excess >= -10
        # For AllA, seg5 must PASS (it's the original validation sample)
        # For CSI300/CSI500, seg5 is informational (large-cap may not fix seg5)
        sample_pass = c1 and c2
        if sample_name == 'AllA':
            sample_pass = sample_pass and c3
        if not sample_pass:
            overall_pass = False

        print('%-10s %-22s %-22s %-22s %-10s' % (
            sample_name,
            '%.2f %s' % (f40_sharpe, 'PASS' if c1 else 'FAIL'),
            '%.2fpp %s' % (f40_excess_vs_base, 'PASS' if c2 else 'FAIL'),
            '%.2fpp %s' % (seg5_f40_excess, 'PASS' if c3 else 'WARN'),
            'PASS' if sample_pass else 'FAIL'))

    print('\n' + '=' * 70)
    if overall_pass:
        print('OVERALL: PASS - F6 effectiveness confirmed across all samples')
        print('Phase 5 cross-sample verification complete, project can close')
    else:
        print('OVERALL: FAIL - F6 may be ineffective on some samples')
        print('Review per-sample results above before closing Phase 5')
    print('=' * 70)

    # 6. Save summary CSV
    summary_rows = []
    for sample_name in [s[0] for s in CROSS_SAMPLES]:
        if sample_name not in all_results:
            continue
        r = all_results[sample_name]
        for seg_idx, seg_label in enumerate(seg_labels):
            summary_rows.append({
                'sample': sample_name,
                'segment': seg_label,
                'baseline_return': r['base_stats'][seg_idx]['strategy_return'],
                'f40_return': r['f40_stats'][seg_idx]['strategy_return'],
                'baseline_excess': r['base_stats'][seg_idx]['excess'],
                'f40_excess': r['f40_stats'][seg_idx]['excess'],
                'diff_excess_pp': (r['f40_stats'][seg_idx]['excess']
                                   - r['base_stats'][seg_idx]['excess']) * 100,
                'f40_sharpe': r['f40_stats'][seg_idx]['sharpe'],
                'f40_max_drawdown': r['f40_stats'][seg_idx]['max_drawdown'],
                'f40_alpha': r['f40_stats'][seg_idx]['alpha'],
            })
    pd.DataFrame(summary_rows).to_csv(
        '%s/cross_sample_summary.csv' % OUT_DIR, index=False)
    print('\nSummary CSV saved to %s/cross_sample_summary.csv' % OUT_DIR)

    print('\n' + '=' * 70)
    print('Cross-sample verification complete')
    print('=' * 70)


def run_p7():
    """P7: 加权回归动量 vs 两点法对比，跨样本 + 训练/验证集分割。

    对每个样本（AllA/CSI300/CSI500）跑 4 个版本：
      1. f2f5f6_40d       : 两点法 61-21（当前生产版本，baseline）
      2. f2f5f6_wreg      : 加权回归斜率年化
      3. f2f5f6_wreg_r2   : 加权回归动量 × R²
      4. f2f5f6_40d_r2filt: 两点法 + R²<0.4 过滤

    防过拟合检查：
      - 训练集 2014-2020 vs 验证集 2021-2026
      - 三样本同时改善才接受
      - 改善幅度 > 2% Sharpe 才切换
    """
    global SAMPLE_NAME, BENCHMARK_CODE
    os.makedirs(OUT_DIR, exist_ok=True)

    P7_VERSIONS = [
        ('f2f5f6_40d',        '两点法61-21 (baseline)'),
        ('f2f5f6_wreg',       '加权回归动量(40d)'),
        ('f2f5f6_wreg_r2',    '加权回归xR2(40d)'),
        ('f2f5f6_40d_r2filt', '两点法+R2过滤'),
        ('f2f5f6_wreg24d',    '加权回归动量(24d)'),
        ('f2f5f6_wreg_r2sq',  '加权回归xR2^2(40d)'),
    ]
    baseline_mode = 'f2f5f6_40d'

    print('=' * 70)
    print('P7: 加权回归动量研究 (跨样本 + 训练/验证分割)')
    print('=' * 70)
    print('Period: %s ~ %s' % (START_DATE, END_DATE))
    print('Train: %s ~ %s | Valid: %s ~ %s' % (
        START_DATE, TRAIN_END, VALID_START, END_DATE))
    print('Capital: %d wan' % (TARGET_CAPITAL / 1e4))
    print('Versions:')
    for mode, desc in P7_VERSIONS:
        print('  - %-20s : %s' % (mode, desc))
    print('Samples: %s' % ', '.join([s[0] for s in CROSS_SAMPLES]))
    print('=' * 70)

    # 调仓日
    rebal_dates = get_rebalance_dates(START_DATE, END_DATE)
    if QUICK_TEST:
        rebal_dates = rebal_dates[:6]
        print('>>> [QUICK_TEST] Only first 6 rebalance dates')
    if len(rebal_dates) < 2:
        print('Insufficient rebalance dates, exiting')
        return

    date_pairs = []
    for i in range(len(rebal_dates) - 1):
        date_pairs.append((rebal_dates[i], rebal_dates[i + 1]))
    last_date = rebal_dates[-1]
    date_pairs.append((last_date, pd.Timestamp(END_DATE).date()))

    # 运行每个样本的每个版本
    all_results = {}
    for sample_name, benchmark_code in CROSS_SAMPLES:
        print('\n' + '#' * 70)
        print('# Sample: %s (benchmark: %s)' % (sample_name, benchmark_code))
        print('#' * 70)

        SAMPLE_NAME = sample_name
        BENCHMARK_CODE = benchmark_code

        sample_data = {}
        benchmark_nav = None
        for mode, desc in P7_VERSIONS:
            print('\n>>> [%s] Running %s (%s) ...' % (sample_name, mode, desc))
            nav, records = compute_daily_nav(
                date_pairs, mode=mode, target_capital=TARGET_CAPITAL)

            if benchmark_nav is None:
                print('>>> [%s] Loading benchmark ...' % sample_name)
                benchmark_nav = compute_benchmark_nav()
                if benchmark_nav is None:
                    print('Benchmark load failed, skipping sample')
                    break

            # 全样本 + 训练集 + 验证集 统计
            full_stats = compute_full_stats(nav, benchmark_nav)
            train_stats = compute_segment_stats(
                nav, benchmark_nav, 'train', START_DATE, TRAIN_END)
            valid_stats = compute_segment_stats(
                nav, benchmark_nav, 'valid', VALID_START, END_DATE)

            sample_data[mode] = {
                'full': full_stats,
                'train': train_stats,
                'valid': valid_stats,
                'turnover': np.mean([r['turnover'] for r in records]) if records else 0,
                'nav': nav,
            }

        if benchmark_nav is not None and len(sample_data) == len(P7_VERSIONS):
            all_results[sample_name] = sample_data
            # 保存净值
            nav_cols = {'benchmark': benchmark_nav}
            for mode, _ in P7_VERSIONS:
                nav_cols[mode] = sample_data[mode]['nav']
            pd.DataFrame(nav_cols).to_csv(
                '%s/daily_nav_p7_%s.csv' % (OUT_DIR, sample_name.lower()))
            print('>>> [%s] NAV saved' % sample_name)

    if not all_results:
        print('\nNo sample completed, exiting')
        return

    # === Table 1: 全样本核心指标 ===
    print('\n' + '=' * 70)
    print('[Table 1] Full-Sample Core Metrics')
    print('=' * 70)
    print('%-10s %-22s %8s %8s %8s %8s %8s' % (
        'Sample', 'Version', 'Return', 'Sharpe', 'DD', 'Alpha', 'Turn'))
    for sample_name in [s[0] for s in CROSS_SAMPLES]:
        if sample_name not in all_results:
            continue
        sd = all_results[sample_name]
        for mode, desc in P7_VERSIONS:
            if mode not in sd:
                continue
            s = sd[mode]['full']
            print('%-10s %-22s %7.1f%% %7.2f %7.1f%% %7.3f %7.3f' % (
                sample_name, mode,
                s['strategy_return'] * 100, s['sharpe'],
                s['max_drawdown'] * 100,
                s['alpha'] if not np.isnan(s['alpha']) else 0,
                sd[mode]['turnover']))

    # === Table 2: 训练集 vs 验证集 Sharpe ===
    print('\n' + '=' * 70)
    print('[Table 2] Train (2014-2020) vs Valid (2021-2026) Sharpe')
    print('=' * 70)
    print('%-10s %-22s %10s %10s %10s' % (
        'Sample', 'Version', 'Train', 'Valid', 'Delta'))
    for sample_name in [s[0] for s in CROSS_SAMPLES]:
        if sample_name not in all_results:
            continue
        sd = all_results[sample_name]
        for mode, desc in P7_VERSIONS:
            if mode not in sd:
                continue
            t_sharpe = sd[mode]['train']['sharpe']
            v_sharpe = sd[mode]['valid']['sharpe']
            print('%-10s %-22s %9.2f %9.2f %+9.2f' % (
                sample_name, mode, t_sharpe, v_sharpe, v_sharpe - t_sharpe))

    # === Table 3: 相对 baseline 的 Sharpe 改善 ===
    print('\n' + '=' * 70)
    print('[Table 3] Sharpe Delta vs Baseline (%s)' % baseline_mode)
    print('=' * 70)
    print('%-10s %-22s %10s %10s %10s' % (
        'Sample', 'Version', 'Full', 'Train', 'Valid'))
    for sample_name in [s[0] for s in CROSS_SAMPLES]:
        if sample_name not in all_results:
            continue
        sd = all_results[sample_name]
        if baseline_mode not in sd:
            continue
        b_full = sd[baseline_mode]['full']['sharpe']
        b_train = sd[baseline_mode]['train']['sharpe']
        b_valid = sd[baseline_mode]['valid']['sharpe']
        for mode, desc in P7_VERSIONS:
            if mode not in sd:
                continue
            d_full = sd[mode]['full']['sharpe'] - b_full
            d_train = sd[mode]['train']['sharpe'] - b_train
            d_valid = sd[mode]['valid']['sharpe'] - b_valid
            marker = ''
            if mode != baseline_mode:
                if d_train > 0 and d_valid < 0:
                    marker = '  ⚠️overfit'
                elif d_valid > 0.02:
                    marker = '  ✅'
            print('%-10s %-22s %+9.2f %+9.2f %+9.2f%s' % (
                sample_name, mode, d_full, d_train, d_valid, marker))

    # === Gate Check ===
    print('\n' + '=' * 70)
    print('[Gate Check] 过拟合红线：验证集三样本同时改善 > 0.02 Sharpe')
    print('=' * 70)
    for mode, desc in P7_VERSIONS:
        if mode == baseline_mode:
            continue
        print('\n%-22s (%s):' % (mode, desc))
        all_pass = True
        for sample_name in [s[0] for s in CROSS_SAMPLES]:
            if sample_name not in all_results or mode not in all_results[sample_name]:
                print('  %-10s: N/A' % sample_name)
                all_pass = False
                continue
            sd = all_results[sample_name]
            if baseline_mode not in sd:
                all_pass = False
                continue
            d_valid = sd[mode]['valid']['sharpe'] - sd[baseline_mode]['valid']['sharpe']
            status = '✅' if d_valid > 0.02 else '❌'
            print('  %-10s: valid Sharpe delta %+6.2f %s' % (
                sample_name, d_valid, status))
            if d_valid <= 0.02:
                all_pass = False
        print('  => %s' % ('PASS — 可考虑切换' if all_pass else 'FAIL — 不切换'))

    # === 保存汇总 CSV ===
    summary_rows = []
    for sample_name in [s[0] for s in CROSS_SAMPLES]:
        if sample_name not in all_results:
            continue
        sd = all_results[sample_name]
        for mode, desc in P7_VERSIONS:
            if mode not in sd:
                continue
            s = sd[mode]
            b_valid_sharpe = sd[baseline_mode]['valid']['sharpe'] if baseline_mode in sd else np.nan
            summary_rows.append({
                'sample': sample_name,
                'version': mode,
                'desc': desc,
                'full_return': s['full']['strategy_return'],
                'full_sharpe': s['full']['sharpe'],
                'full_dd': s['full']['max_drawdown'],
                'full_alpha': s['full']['alpha'],
                'train_sharpe': s['train']['sharpe'],
                'valid_sharpe': s['valid']['sharpe'],
                'valid_delta_vs_baseline': s['valid']['sharpe'] - b_valid_sharpe,
                'turnover': s['turnover'],
            })
    pd.DataFrame(summary_rows).to_csv(
        '%s/p7_summary.csv' % OUT_DIR, index=False)
    print('\nSummary CSV saved to %s/p7_summary.csv' % OUT_DIR)

    print('\n' + '=' * 70)
    print('P7 analysis complete')
    print('=' * 70)


if __name__ == '__main__':
    if CROSS_SAMPLE_MODE:
        run_p7()
    else:
        run()
