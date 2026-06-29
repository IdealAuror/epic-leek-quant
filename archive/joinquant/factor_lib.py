"""factor_lib.py — 五大因子计算引擎。

本模块实现 epic-leek-quant 项目的五大价值因子计算逻辑。所有函数设计为
接收已从聚宽查询好的 DataFrame（列名为 `table.column` 格式），在 DataFrame
上新增因子列，不涉及聚宽 API 调用。

因子列表（按 PROJECT-PLAN.md §1.2 口径）：
    F1 EV<0     — 企业价值为负（净现金>总市值+有息负债），二值
    F2 EP       — 盈利收益率 = 净利润/市值，连续
    F3 股息率    — 股利收益率，连续
    F4 国企背景  — 实际控制人是否国资，二值
    F5 财务质量  — 低杠杆+正经营现金流，二值

修复记录（按 plan-review.md B1/B2）：
    - B1: debt_to_assets 先算比率再取中位数
    - B2: 二值因子直接等权求和，不做 Z-score

运行环境：
    - 聚宽研究/策略环境：依赖 jqdata 注入，但本模块不直接调用 jqdata
    - 本地审核环境：可独立 import（仅 numpy/pandas 依赖）
"""

import datetime
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

# 国企实际控制人关键词列表
STATE_OWNERS: list[str] = [
    '国务院国有资产监督管理委员会',
    '地方国有资产监督管理委员会',
    '地方政府',
    '中央国家机关',
    '中央汇金投资有限责任公司',
    '中国证券金融股份有限公司',
]


def calculate_all_factors(df: pd.DataFrame) -> pd.DataFrame:
    """计算五大因子，在传入的 DataFrame 上新增因子列。

    输入 DataFrame 的列名需为聚宽标准格式（`table.column`），例如：
        `valuation.market_cap`、`balance.total_liability`、`income.net_profit` 等。

    Parameters
    ----------
    df : pd.DataFrame
        已从聚宽查询的横截面数据，每行一只股票。

    Returns
    -------
    pd.DataFrame
        原 DataFrame 加上以下新增列：
        - factor_ev_negative : int (0/1)
        - earnings_yield : float
        - factor_low_pe : int (0/1)
        - div_yield_continuous : float
        - factor_state_owned : int (0/1)
        - debt_to_assets : float
        - factor_low_leverage : int (0/1)
        - factor_positive_cf : int (0/1)
        - interest_bearing_debt : float
        - ev / ev_ext / cash_available_ext / current_ratio : float
    """
    # F1: EV 为负
    interest_bearing_debt = (
        df['total_liability'] - df['accounts_payable']
        - df['advances_from_customers'] - df['wages_payable']
        - df['taxes_payable'] - df['other_current_liabilities']
    )
    df['interest_bearing_debt'] = interest_bearing_debt
    df['ev'] = df['market_cap'] + interest_bearing_debt - df['cash_equivalents']
    df['factor_ev_negative'] = (df['ev'] < 0).astype(int)

    # F1 扩展变体
    df['cash_available_ext'] = (
        df['monetary_funds'] + df['financial_assets_held_for_trading']
    )
    df['ev_ext'] = (
        df['market_cap'] + interest_bearing_debt - df['cash_available_ext']
    )
    df['current_ratio'] = (
        df['total_current_assets'] / df['total_current_liabilities']
    )

    # F2: 盈利收益率 + 低 PE 阈值
    df['earnings_yield'] = df['net_profit'] / df['market_cap']
    df['factor_low_pe'] = (df['pe_ttm'] < 10).astype(int)

    # F3: 股息率
    df['div_yield_continuous'] = df['div_yield']

    # F4: 国企背景
    df['factor_state_owned'] = (
        df['actual_controller'].isin(STATE_OWNERS).astype(int)
    )

    # F5: 财务质量（修复 B1：先算比率再取中位数）
    df['debt_to_assets'] = df['total_liability'] / df['total_assets']
    median_debt = df['debt_to_assets'].median()
    df['factor_low_leverage'] = (df['debt_to_assets'] < median_debt).astype(int)
    df['factor_positive_cf'] = (df['operating_cash_flow'] > 0).astype(int)

    return df


def composite_score(df: pd.DataFrame) -> pd.DataFrame:
    """多因子合成打分。

    二值因子（0/1）等权求和；连续因子做 1%/99% winsorize 后
    Z-score 标准化再加到总分（对应 PROJECT-PLAN.md §1.2 C12/B2）。

    Parameters
    ----------
    df : pd.DataFrame
        需包含 calculate_all_factors 输出的所有因子列。

    Returns
    -------
    pd.DataFrame
        原 DataFrame 加上 `composite_score` 列（float，越高越好）。
    """
    binary_cols: list[str] = [
        'factor_ev_negative', 'factor_low_pe',
        'factor_state_owned', 'factor_low_leverage',
        'factor_positive_cf',
    ]
    continuous_cols: list[str] = ['earnings_yield', 'div_yield_continuous']

    score: pd.Series = df[binary_cols].sum(axis=1).astype(float)

    for col in continuous_cols:
        s = df[col].dropna()
        if len(s) < 2:
            continue
        lo, hi = s.quantile(0.01), s.quantile(0.99)
        clipped = s.clip(lo, hi)
        std = clipped.std()
        mean = clipped.mean()
        if std > 0:
            score.loc[s.index] += (clipped - mean) / std

    df['composite_score'] = score
    return df
