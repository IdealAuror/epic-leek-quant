"""
P6-VolWindow Scan — AllA only, [40,60,90] days
预取全量价格 + 股票池，预计 2-3 分钟
"""

import datetime
import numpy as np
import pandas as pd

# === config ===
VOLS = [40, 60, 90]
N = 50
ML = 61; MS = 21
START = '2014-01-01'; END = '2026-06-26'
TRAIN_END = '2020-12-31'; TEST_START = '2021-01-01'
FIN = ['银行', '证券', '保险', '多元金融']
MX = max(VOLS + [ML]) + 10
PSTART = '2013-01-01'

print("="*60)
print("P6 VolWindow — AllA [40,60,90]")
print("="*60)

# ======== 1. 调仓日 + 全量代码 ========
print("\n[1/3] Loading universe...")
mons = pd.date_range(START, END, freq='MS')
rd = [d for d in mons if d.month in [5,9,11] and d >= pd.Timestamp('2014-01-01')]

all_stocks = get_all_securities(['stock']).index.tolist()
# pre-filter: remove stocks listed after 2013 (keep only those older than 1yr by 2014)
old = [s for s in all_stocks if (datetime.date(2014,1,1) - get_security_info(s).start_date).days >= 252]
print(f"  Stocks (pre-2013): {len(old)}")

# build date->codes once
by_date = {}
all_codes = set()
for i, dt in enumerate(rd):
    d = dt.date()
    codes = old[:]

    # ST
    try:
        st = get_extras('is_st', codes, start_date=d, end_date=d, df=True)
        if st is not None and len(st) > 0:
            sr = st.iloc[-1]
            codes = [c for c in codes if not sr.get(c, False)]
    except:
        pass

    # financial
    try:
        fin_codes = set()
        for f in FIN:
            fin_codes.update(get_industry_stocks(f, date=d))
        codes = [c for c in codes if c not in fin_codes]
    except:
        pass

    if len(codes) > 600:
        codes = codes[:600]
    by_date[dt] = codes
    all_codes.update(codes)

print(f"  Rebalance dates: {len(rd)}, unique codes: {len(all_codes)}")

# ======== 2. 预取全量价格 ========
print(f"\n[2/3] Loading price ({len(all_codes)} stocks)...")
sc = sorted(all_codes)
pr = get_price(sc, start_date=PSTART, end_date=END, frequency='daily',
               fields=['close'], skip_paused=True, fq='pre', panel=False)
pp = pr.pivot(index='time', columns='code', values='close')
print(f"  Price: {pp.shape}")

# ======== 3. 回测 ========
print(f"\n[3/3] Running backtests...")

def run(vol_w):
    pv = 1.0; log = []
    rl = [d for d in rd if d in by_date]
    for i, dt in enumerate(rl):
        av = [c for c in by_date[dt] if c in pp.columns]
        if len(av) < N*2: continue
        pw = pp[av].loc[:dt].dropna(axis=1, how='all')
        if pw.empty or len(pw) < MX: continue

        # EP
        try:
            q = query(valuation.code, valuation.pe_ratio).filter(valuation.code.in_(av))
            fd = get_fundamentals(q, date=dt.date())
            if fd is None or len(fd) == 0: continue
            fd['ep'] = 1.0 / fd['pe_ratio'].replace(0, np.nan)
            fd = fd.dropna(subset=['ep'])
            em = dict(zip(fd['code'], fd['ep']))
        except: continue

        # vol
        ret = pw.pct_change().dropna(how='all')
        vo = ret.tail(vol_w).std()
        # mom
        if len(pw) < ML: continue
        mo = pw.iloc[-MS] / pw.iloc[-ML] - 1

        cm = list(set(em.keys()) & set(vo.dropna().index) & set(mo.dropna().index))
        if len(cm) < N: continue

        sc_df = pd.DataFrame({
            'c': cm, 'ep': [em[c] for c in cm],
            'vo': [vo[c] for c in cm], 'mo': [mo[c] for c in cm]
        })
        for col in ['ep','vo','mo']:
            mu, sg = sc_df[col].mean(), sc_df[col].std()
            sc_df[f'z_{col}'] = (sc_df[col]-mu)/sg if sg>0 else 0
        sc_df['s'] = (sc_df['z_ep'] - sc_df['z_vo'] + sc_df['z_mo']) / 3
        sel = sc_df.nlargest(N, 's')['c'].tolist()

        ni = min(i+1, len(rl)-1)
        nx = rl[ni]
        su = pp[sel].loc[dt:nx]
        if su.empty or len(su) < 2: continue
        rt = su.iloc[-1] / su.iloc[0] - 1
        rt = rt.dropna()
        if len(rt) == 0: continue
        pv *= (1 + rt.mean())
        log.append({'date': nx, 'val': pv})

    if len(log) < 4: return None
    df = pd.DataFrame(log).set_index('date')
    df['r'] = df['val'].pct_change()
    rr = df['r'].dropna()
    if len(rr) < 3: return None

    ar = (1+rr.mean())**12 - 1
    sh = (rr.mean()/rr.std())*np.sqrt(12) if rr.std()>0 else 0
    cu = (1+rr).cumprod()
    dd = (cu/cu.cummax()-1).min()

    tr = df.index <= pd.Timestamp(TRAIN_END)
    te = df.index >= pd.Timestamp(TEST_START)
    def ms(s): 
        s = s.dropna()
        if len(s)<2: return np.nan
        return (s.mean()/s.std())*np.sqrt(12) if s.std()>0 else 0

    return {
        'w': vol_w, 'bl': vol_w==60,
        'ar': ar, 'sh': sh, 'dd': dd,
        'ts': ms(rr[tr]), 'vs': ms(rr[te]),
        'n': len(df)
    }

results = []
for w in VOLS:
    r = run(w)
    if r:
        results.append(r)
        bl = " ★ BASELINE" if r['bl'] else ""
        print(f"  {w:3d}d | ret={r['ar']*100:5.1f}% S={r['sh']:.3f} DD={r['dd']*100:5.1f}% testS={r['vs']:.3f}{bl}")

# ==== verdict ====
print(f"\n{'='*60}")
if results:
    best = max(results, key=lambda x: x['vs'] if not np.isnan(x['vs']) else -999)
    bl = [r for r in results if r['bl']]
    if bl and best['vs'] > bl[0]['vs']:
        print(f"  → {best['w']}d beats 60d on test Sharpe ({best['vs']:.3f} vs {bl[0]['vs']:.3f})")
    else:
        print(f"  → 60d remains optimal (test Sharpe={bl[0]['vs']:.3f})")
