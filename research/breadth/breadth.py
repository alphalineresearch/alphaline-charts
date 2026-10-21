"""
Crypto market breadth from CoinMarketCap weekly top-500 snapshots.

build_panel()   → weekly price / mcap panels (coin id × week), pegged assets removed
breadth(panel)  → weekly breadth measures over that week's top-N universe
Run directly to print a summary.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SNAP = os.path.join(HERE, 'cache', 'snapshots.csv.gz')

TOP_N = 100
MA_WEEKS = 30          # ≈ 200 trading days, the weekly analogue of the 200-day line


def load():
    df = pd.read_csv(SNAP, parse_dates=['date'])
    df = df[(df.price > 0) & (df.mcap > 0)]
    return df


RWA_TAGS = {'tokenized-stock', 'tokenized-etfs', 'tokenized-gold', 'tokenized-commodities',
            'tokenized-treasury-bonds'}
PEG_REFS = {1: 'BTC', 1027: 'ETH', 5426: 'SOL', 1839: 'BNB'}
NOT_RWA = {1975, 28299}        # Chainlink, TokenFi: tagged tokenized-stock, but ordinary tokens


def pegged_ids(df):
    """Coins that are really another asset. CMC's stablecoin / staking tags are too broad
    (LDO, RPL, ANKR are tagged liquid-staking; DGD stablecoin), so pegs are found from
    price behaviour: the price tracks $1, BTC, ETH, SOL or BNB week to week. Tokenized
    stocks and commodities are excluded by tag."""
    tags = df.groupby('id')['tags'].apply(lambda s: set('|'.join(s.dropna()).split('|')))
    rwa = set(tags[tags.apply(lambda t: bool(RWA_TAGS & t))].index) - NOT_RWA
    px = df.pivot_table(index='date', columns='id', values='price')
    refs = {'USD': pd.Series(1.0, index=px.index)}
    refs.update({name: px[i] for i, name in PEG_REFS.items() if i in px})
    n_obs = px.notna().sum()
    pegged = set()
    for name, r in refs.items():
        lr = np.log(px.div(r, axis=0))
        wobble = lr.diff().abs().median()              # typical weekly move vs the reference
        level = lr.median().abs()                      # ratio within ~2× of the reference
        pegged |= set(wobble[(wobble < 0.01) & (level < np.log(2)) & (n_obs >= 3)].index)
    pegged -= set(PEG_REFS)
    return rwa | pegged


def build_panel():
    df = load()
    drop = pegged_ids(df)
    clean = df[~df.id.isin(drop)].copy()
    clean['rank_clean'] = clean.groupby('date')['mcap'].rank(ascending=False, method='first')
    px = clean.pivot_table(index='date', columns='id', values='price')
    mc = clean.pivot_table(index='date', columns='id', values='mcap')
    rk = clean.pivot_table(index='date', columns='id', values='rank_clean')
    allmc = df.pivot_table(index='date', columns='id', values='mcap')   # incl. stables, for TOTAL
    names = df.drop_duplicates('id', keep='last').set_index('id')['symbol']
    return dict(px=px, mc=mc, rk=rk, allmc=allmc, names=names, dropped=drop,
                stable_ids=set(df.loc[df.tags.fillna('').str.contains('stablecoin'), 'id']))


def breadth(p, top_n=TOP_N, ma_weeks=MA_WEEKS):
    px, mc, rk = p['px'], p['mc'], p['rk']
    uni = rk <= top_n                                  # this week's universe (no look-ahead)
    hist_ok = px.rolling(ma_weeks, min_periods=int(ma_weeks * 0.85)).count() >= int(ma_weeks * 0.85)
    ma = px.rolling(ma_weeks, min_periods=int(ma_weeks * 0.85)).mean()
    px52 = px.shift(52)
    hi52 = px.rolling(52, min_periods=40).max()
    lo52 = px.rolling(52, min_periods=40).min()
    ret1 = px / px.shift(1) - 1

    def share(cond, valid):
        v = uni & valid
        return (cond & v).sum(axis=1) / v.sum(axis=1).replace(0, np.nan), v.sum(axis=1)

    out = pd.DataFrame(index=px.index)
    out['pct_above_ma'], out['n_ma'] = share(px > ma, hist_ok)
    out['pct_up_yoy'], out['n_yoy'] = share(px > px52, px52.notna())
    h, n_hl = share(px >= hi52, hi52.notna())
    l, _ = share(px <= lo52, lo52.notna())
    out['pct_new_hi'], out['pct_new_lo'], out['n_hl'] = h, l, n_hl
    out['net_hi_lo'] = h - l
    adv = ((ret1 > 0) & uni).sum(axis=1)
    dec = ((ret1 < 0) & uni).sum(axis=1)
    out['ad_line'] = (adv - dec).cumsum()
    out['n_universe'] = uni.sum(axis=1)

    # Equal-weight alts (top-N ex BTC/ETH) vs cap-weighted, from weekly returns
    alts = uni.copy()
    for c in (1, 1027):
        if c in alts:
            alts[c] = False
    r = ret1.where(alts.shift(1, fill_value=False))   # held from last week's universe
    r = r.clip(-0.9, 3.0)                               # guard against bad prints
    # equal weight as the average log return (the "typical alt"); an arithmetic
    # mean is dominated by a few micro-cap pumps
    out['alt_ew_ret'] = np.expm1(np.log1p(r).mean(axis=1))
    w = mc.shift(1).where(alts.shift(1, fill_value=False))
    out['alt_cw_ret'] = (r * w).sum(axis=1) / w.where(r.notna()).sum(axis=1)
    out['alt_ew_index'] = (1 + out['alt_ew_ret'].fillna(0)).cumprod()
    # a holdable equal-weight basket, rebalanced weekly (arithmetic mean)
    out['alt_ewp_ret'] = r.mean(axis=1)
    out['alt_ewp_index'] = (1 + out['alt_ewp_ret'].fillna(0)).cumprod()
    out['alt_cw_index'] = (1 + out['alt_cw_ret'].fillna(0)).cumprod()

    # TradingView-style totals from snapshot market caps (top 500 only)
    allmc = p['allmc'].reindex(px.index)
    total = allmc.sum(axis=1)
    btc = allmc[1]
    eth = allmc[1027] if 1027 in allmc else pd.Series(np.nan, index=allmc.index)
    out['total'] = total
    out['total2'] = total - btc
    out['total3'] = total - btc - eth.fillna(0)
    top10 = allmc.apply(lambda s: s.nlargest(10).sum(), axis=1)
    out['others'] = total - top10
    out['btc_dom'] = btc / total
    stab = [c for c in p['stable_ids'] if c in allmc]
    out['stable_share'] = allmc[stab].sum(axis=1) / total

    out['btc'] = px.get(1)
    out['eth'] = px[1027] if 1027 in px else np.nan
    return out


def forward(series, weeks):
    return np.log(series.shift(-weeks) / series)


if __name__ == '__main__':
    p = build_panel()
    print(f"panel: {p['px'].shape[1]} coins × {p['px'].shape[0]} weeks; dropped {len(p['dropped'])} pegged")
    print('dropped (sample):', sorted(p['names'].reindex(list(p['dropped'])).dropna().tolist())[:60])
    b = breadth(p)
    pd.set_option('display.width', 200)
    print(b[['n_universe', 'n_ma', 'pct_above_ma', 'pct_up_yoy', 'net_hi_lo', 'btc_dom', 'stable_share']]
          .resample('YE').last().round(2))
