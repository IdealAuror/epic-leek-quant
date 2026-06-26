import datetime
import numpy as np
import pandas as pd

STATE_OWNERS = [
    '国务院国有资产监督管理委员会',
    '地方国有资产监督管理委员会',
    '地方政府',
    '中央国家机关',
    '中央汇金投资有限责任公司',
    '中国证券金融股份有限公司',
]

def calculate_all_factors(df):
    interest_bearing_debt = (
        df['total_liability'] - df['accounts_payable']
        - df['advances_from_customers'] - df['wages_payable']
        - df['taxes_payable'] - df['other_current_liabilities']
    )
    df['interest_bearing_debt'] = interest_bearing_debt
    df['ev'] = df['market_cap'] + interest_bearing_debt - df['cash_equivalents']
    df['factor_ev_negative'] = (df['ev'] < 0).astype(int)
    df['cash_available_ext'] = df['monetary_funds'] + df['financial_assets_held_for_trading']
    df['ev_ext'] = df['market_cap'] + interest_bearing_debt - df['cash_available_ext']
    df['current_ratio'] = df['total_current_assets'] / df['total_current_liabilities']
    df['earnings_yield'] = df['net_profit'] / df['market_cap']
    df['factor_low_pe'] = (df['pe_ttm'] < 10).astype(int)
    df['div_yield_continuous'] = df['div_yield']
    df['factor_state_owned'] = df['actual_controller'].isin(STATE_OWNERS).astype(int)
    df['debt_to_assets'] = df['total_liability'] / df['total_assets']
    median_debt = df['debt_to_assets'].median()
    df['factor_low_leverage'] = (df['debt_to_assets'] < median_debt).astype(int)
    df['factor_positive_cf'] = (df['operating_cash_flow'] > 0).astype(int)
    return df

def composite_score(df):
    binary_cols = [
        'factor_ev_negative', 'factor_low_pe',
        'factor_state_owned', 'factor_low_leverage',
        'factor_positive_cf',
    ]
    continuous_cols = ['earnings_yield', 'div_yield_continuous']
    score = df[binary_cols].sum(axis=1).astype(float)
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
