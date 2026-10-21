"""Series, event tables and current readings for the preferreds research page → cache/page_data.json"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import events as E
import control as K

px, p = E.px, E.p
b = px['BTC-USD']
tnx = p['^TNX'].reindex(px.index).ffill()
CPN = {'STRK': 8, 'STRF': 10, 'STRD': 10}
dv = E.dv


def strc_rate():
    """Annualised STRC rate from its dividends (monthly to Jun 2026, then twice a month)."""
    d = dv['STRC'][dv['STRC'] > 0]
    per_year = pd.Series(12.0, index=d.index)
    per_year[d.index >= '2026-06-30'] = 24.0
    return (d * per_year).reindex(px.index).ffill()


rate = strc_rate()
spreads = pd.DataFrame({t: CPN[t] / px[t] * 100 - tnx for t in CPN})
spreads['STRC'] = rate / px['STRC'] * 100 - tnx
r = lambda v, n=2: None if pd.isna(v) else round(float(v), n)
S = px.loc['2025-02-03':]

out = {'series': {
    'date': [d.strftime('%Y-%m-%d') for d in S.index],
    **{t: [r(v) for v in S[t]] for t in ('STRC', 'STRK', 'STRF', 'STRD', 'BMNP', 'SATA')},
    'btc': [r(v, 0) for v in S['BTC-USD']],
    **{f'sp_{t}': [r(v) for v in spreads[t].reindex(S.index)] for t in ('STRK', 'STRF', 'STRD', 'STRC')},
    'tnx': [r(v) for v in tnx.reindex(S.index)],
    'strc_rate': [r(v) for v in rate.reindex(S.index)],
}}


def ev_rows(dates):
    rows = []
    for d in dates:
        rows.append(dict(date=d.strftime('%Y-%m-%d'), strc=r(px.at[d, 'STRC']), btc=r(b[d], 0),
                         btc_20=r(E.fwd(b, d, 20), 4), btc_60=r(E.fwd(b, d, 60), 4),
                         btc_worst=r(E.worst_ahead(b, d), 4),
                         strc_tr_60=r(E.fwd(E.TR['STRC'], d, 60), 4),
                         strd_tr_60=r(E.fwd(E.TR['STRD'], d, 60), 4)))
    return rows


out['events'] = {
    'strc_below_97': ev_rows(E.first_crosses(px['STRC'], 97, 99)),
    'strc_back_98': ev_rows(E.first_recover(px['STRC'], 98, 95)),
    'strd_break': ev_rows(K.cross_down(px['STRD'])),
}


def summ(dates, h=60):
    v = [x for x in (E.fwd(b, d, h) for d in dates) if pd.notna(x)]
    w = [E.worst_ahead(b, d) for d in dates]
    return dict(n=len(v), med=r(np.median(v), 4), lower=int(sum(x < 0 for x in v)), worst=r(np.median(w), 4))


ma = b.rolling(50).mean()
out['breaks'] = [
    dict(name='All trading days', **summ(list(px.index[:-60]))),
    dict(name='BTC closes below its own 50-day average', **summ(K.cross_down(b))),
    dict(name='MSTR closes below its 50-day average', **summ(K.cross_down(px['MSTR']))),
    dict(name='STRF closes below its 50-day average', **summ(K.cross_down(px['STRF']))),
    dict(name='STRK closes below its 50-day average', **summ(K.cross_down(px['STRK']))),
    dict(name='STRD closes below its 50-day average', **summ(K.cross_down(px['STRD']))),
    dict(name='STRD breaks while BTC is still above its 50-day', **summ([x for x in K.cross_down(px['STRD']) if b[x] > ma[x]])),
]

# Order of events in the three selloffs
dd = px / px.rolling(60, min_periods=1).max() - 1
order = []
for name, a, z in [('Oct–Nov 2025', '2025-10-25', '2025-12-10'), ('Jan–Feb 2026', '2026-01-20', '2026-03-10'),
                   ('May–Jun 2026', '2026-05-10', '2026-07-20')]:
    row = {'episode': name, 'cols': {}}
    for t in ('BTC-USD', 'ETH-USD', 'MSTR', 'STRK', 'STRD', 'STRF', 'STRC'):
        y, x = dd[t].loc[a:z], px[t].loc[a:z]
        f3 = y.index[y <= -0.03]
        row['cols'][t] = dict(first3=f3[0].strftime('%Y-%m-%d') if len(f3) else None,
                              low=x.idxmin().strftime('%Y-%m-%d'), dd=r(y.min(), 4))
    order.append(row)
out['order'] = order

last = px.index[-1]
out['now'] = dict(
    date=last.strftime('%Y-%m-%d'),
    strc=r(px.at[last, 'STRC']), strc_rate=r(rate.iloc[-1]),
    strd=r(px.at[last, 'STRD']), strd_vs_50=r(px['STRD'].iloc[-1] / px['STRD'].rolling(50).mean().iloc[-1] - 1, 4),
    strk_vs_50=r(px['STRK'].iloc[-1] / px['STRK'].rolling(50).mean().iloc[-1] - 1, 4),
    btc_vs_50=r(b.iloc[-1] / ma.iloc[-1] - 1, 4),
    strd_spread=r(spreads['STRD'].iloc[-1]), strd_spread_chg20=r(spreads['STRD'].diff(20).iloc[-1]),
    strd_spread_peak=r(spreads['STRD'].max()), tnx=r(tnx.iloc[-1]), tnx_chg20=r(tnx.diff(20).iloc[-1]),
    bmnp=r(px['BMNP'].dropna().iloc[-1]), bmnp_low=r(px['BMNP'].min()),
    sata=r(px['SATA'].dropna().iloc[-1]),
    strd_break_date=K.cross_down(px['STRD'])[-1].strftime('%Y-%m-%d'),
)
json.dump(out, open(os.path.join(E.HERE, 'cache', 'page_data.json'), 'w'), separators=(',', ':'))
print(json.dumps(out['now'], indent=1))
print(json.dumps(out['breaks'], indent=0))
