"""
Does crypto breadth say anything about forward BTC / ETH returns?

Prints bucket tables, event studies and a period split. Writes cache/breadth.csv
for the charts.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import breadth as B

START = '2016-01-01'            # top-100 universe reliably full with 30w history
HORIZONS = [4, 13, 26, 52]      # weeks ≈ 1, 3, 6, 12 months
pd.set_option('display.width', 220)
pd.set_option('display.max_columns', 40)


def fwd_table(df, col, bins, asset='btc', labels=None):
    x = df.dropna(subset=[col])
    cat = pd.cut(x[col], bins, labels=labels, include_lowest=True)
    rows = []
    for k, g in x.groupby(cat, observed=True):
        row = {'bucket': k, 'weeks': len(g),
               'episodes': int((cat == k).astype(int).diff().fillna(0).eq(1).sum() + (cat.iloc[0] == k))}
        for h in HORIZONS:
            f = g[f'f_{asset}_{h}'].dropna()
            row[f'{h}w med'] = np.expm1(f.median()) if len(f) else np.nan
            row[f'{h}w %up'] = (f > 0).mean() if len(f) else np.nan
        rows.append(row)
    base = {'bucket': 'ALL', 'weeks': len(x), 'episodes': np.nan}
    for h in HORIZONS:
        f = df.loc[x.index, f'f_{asset}_{h}'].dropna()
        base[f'{h}w med'] = np.expm1(f.median())
        base[f'{h}w %up'] = (f > 0).mean()
    rows.append(base)
    return pd.DataFrame(rows).set_index('bucket')


def events(df, mask, cooldown=26):
    """First week of each signal, then ignore repeats for `cooldown` weeks."""
    out, last = [], None
    for d in df.index[mask.reindex(df.index, fill_value=False)]:
        if last is None or (d - last).days > cooldown * 7:
            out.append(d)
            last = d
    return out


def event_table(df, dates, assets=('btc', 'eth')):
    rows = []
    for d in dates:
        r = {'date': d.date()}
        for a in assets:
            for h in [13, 26, 52]:
                v = df[f'f_{a}_{h}'].get(d)
                r[f'{a} {h}w'] = np.expm1(v) if pd.notna(v) else np.nan
        rows.append(r)
    t = pd.DataFrame(rows).set_index('date')
    if len(t):
        t.loc['median'] = t.median()
        t.loc['% up'] = (t.drop('median') > 0).sum() / t.drop('median').notna().sum()
    return t


def fmt(t):
    return t.map(lambda v: f'{v:+.0%}' if isinstance(v, (float, np.floating)) and pd.notna(v) and abs(v) < 50 else v)


if __name__ == '__main__':
    p = B.build_panel()
    b = B.breadth(p)
    b.to_csv(os.path.join(B.HERE, 'cache', 'breadth.csv'))
    df = b.loc[START:].copy()
    df['btc_ma'] = b['btc'].rolling(B.MA_WEEKS).mean().reindex(df.index)
    df['btc_trend'] = np.where(df['btc'] > df['btc_ma'], 'BTC above 30w', 'BTC below 30w')
    df['ma_chg8'] = df['pct_above_ma'].diff(8)
    # forward returns from the full series, so period / regime subsets aren't cut short
    for a in ['btc', 'eth']:
        for h in HORIZONS:
            df[f'f_{a}_{h}'] = B.forward(b[a], h).reindex(df.index)

    print(f"weeks {df.index[0].date()} → {df.index[-1].date()}  n={len(df)}  "
          f"median eligible coins: MA {df.n_ma.median():.0f}, YoY {df.n_yoy.median():.0f}")

    pct_bins = [0, .2, .4, .6, .8, 1]
    pct_lbl = ['0–20%', '20–40%', '40–60%', '60–80%', '80–100%']
    for col in ['pct_above_ma', 'pct_up_yoy']:
        for a in ['btc', 'eth']:
            print(f'\n=== {col} → forward {a.upper()} (median return, % of weeks up) ===')
            print(fmt(fwd_table(df, col, pct_bins, a, pct_lbl)))
    print('\n=== net new 52w highs − lows → forward BTC ===')
    print(fmt(fwd_table(df, 'net_hi_lo', [-1, -.3, -.1, .1, .3, 1], 'btc',
                        ['≤−30', '−30…−10', '±10', '+10…+30', '≥+30'])))

    # Does breadth add anything beyond BTC's own trend?
    print('\n=== Breadth within BTC trend regimes → forward BTC 26w ===')
    for reg, g in df.groupby('btc_trend'):
        print(reg)
        print(fmt(fwd_table(g, 'pct_above_ma', [0, .3, .6, 1], 'btc',
                            ['weak <30%', 'mid', 'strong >60%'])[['weeks', 'episodes', '13w med', '26w med', '26w %up', '52w med']]))

    # Period split
    for lo, hi in [('2016', '2020'), ('2021', '2026')]:
        g = df.loc[lo:hi]
        print(f'\n=== {lo}–{hi}: pct_above_ma → forward BTC ===')
        print(fmt(fwd_table(g, 'pct_above_ma', pct_bins, 'btc', pct_lbl)[['weeks', 'episodes', '13w med', '26w med', '26w %up', '52w med', '52w %up']]))

    # Rank correlations
    print('\n=== Spearman: breadth vs forward log return ===')
    rows = []
    for col in ['pct_above_ma', 'pct_up_yoy', 'net_hi_lo', 'btc_dom']:
        for a in ['btc', 'eth']:
            r = {'measure': col, 'asset': a}
            for h in HORIZONS:
                r[f'{h}w'] = df[col].corr(df[f'f_{a}_{h}'], method='spearman')
            rows.append(r)
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    # Events
    wash = events(df, df['pct_above_ma'] <= 0.10)
    print('\n=== WASHOUT: ≤10% of top-100 above their 30w MA (first week, 26w cooldown) ===')
    print(fmt(event_table(df, wash)))
    thrust_mask = (df['pct_above_ma'] >= 0.60) & (df['pct_above_ma'].rolling(7).min() <= 0.20)
    thr = events(df, thrust_mask)
    print('\n=== THRUST: from ≤20% to ≥60% above 30w MA within 6 weeks ===')
    print(fmt(event_table(df, thr)))
    near_hi = df['btc'] >= 0.9 * df['btc'].rolling(52).max()
    div_mask = near_hi & (df['pct_above_ma'] < 0.5) & (df['pct_above_ma'].diff(8) < -0.15)
    div = events(df, div_mask)
    print('\n=== DIVERGENCE: BTC within 10% of 52w high, <50% of top-100 above MA and falling 15pts+ over 8w ===')
    print(fmt(event_table(df, div)))
    print('\n=== latest ===')
    print(df[['pct_above_ma', 'pct_up_yoy', 'net_hi_lo', 'btc_dom', 'stable_share', 'btc_trend']].tail(6))
