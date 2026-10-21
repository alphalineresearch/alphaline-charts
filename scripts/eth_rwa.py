"""
eth_rwa.py — real-world assets on Ethereum, rebuilt after DefiLlama's Sep 16, 2026 change.

On Sep 16, 2026 DefiLlama stopped counting tokenized funds (BUIDL, gold tokens, Centrifuge,
Securitize funds…) as TVL and rewrote the history, so its Ethereum "RWA" fell from ~$14B to
~$2–3B on every date. This module adds those products back from the chain itself:

  RWA on Ethereum = DefiLlama RWA protocols (as now reported, minus any we count directly)
                  + tokenized funds and gold read from their Ethereum contracts

Token supply: totalSupply() on each Ethereum contract, read from a public archive node —
weekly back to launch, then daily as the job runs (data/eth_rwa/supply.csv, appended).
Token price:
  gold_oz / gold_g — gold futures close (Yahoo GC=F), per troy ounce or per gram
  usd              — $1 (BUIDL-style funds that pay out in new tokens)
  nav              — CoinGecko daily price for the last year; before that, the current NAV
                     discounted at the yield the last year implies (these funds accrue
                     smoothly, so the error is small)
The token list (data/eth_rwa/tokens.csv) is reviewed by hand; add a token there and the
next run backfills it.

    from eth_rwa import fetch_rwa_tvl          # drop-in for the old DefiLlama-only version
"""

import os
import time

import numpy as np
import pandas as pd
import requests

HERE      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(HERE, '..', 'data', 'eth_rwa')
TOKENS    = os.path.join(DATA_DIR, 'tokens.csv')
SUPPLY    = os.path.join(DATA_DIR, 'supply.csv')
CACHE     = os.path.join(DATA_DIR, 'rwa_daily.csv')
NAV_FILE  = os.path.join(DATA_DIR, 'nav_prices.csv')    # observed NAVs: history beyond CoinGecko's year, and the fallback
RPCS      = ['https://eth.drpc.org', 'https://ethereum-rpc.publicnode.com', 'https://eth.llamarpc.com']
TROY_OZ_G = 31.1034768
ETH_TVL_THRESHOLD = 1_000_000
# DefiLlama RWA protocols whose tokens are counted directly below (avoid double counting)
COUNTED_DIRECTLY = {'Anemoy Capital'}

MERGE_BLOCK, MERGE_TS = 15537394, 1663224162      # first proof-of-stake block, 2022-09-15
BATCH = 3                                          # drpc free plan: ≤ 3 calls per batch
SEL = {'decimals': '0x313ce567', 'totalSupply': '0x18160ddd', 'symbol': '0x95d89b41'}


# ════════════════════════════════════════════
# CHAIN
# ════════════════════════════════════════════
def _rpc(payload):
    last = None
    for rpc in RPCS:
        for attempt in range(3):
            try:
                r = requests.post(rpc, json=payload, timeout=30)
                j = r.json()
                return j
            except Exception as e:
                last = e
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f'all RPCs failed: {last}')


def _latest_block():
    j = _rpc({'jsonrpc': '2.0', 'id': 1, 'method': 'eth_getBlockByNumber', 'params': ['latest', False]})['result']
    return int(j['number'], 16), int(j['timestamp'], 16)


def _block_at(ts, latest):
    """Approximate block number at a unix time (12 s slots after the merge, ~13.2 s before).
    Weekly / daily sampling only needs it to within a few minutes."""
    n, t = latest
    if ts >= MERGE_TS:
        return min(n, MERGE_BLOCK + int((ts - MERGE_TS) / 12.05))
    return max(1, MERGE_BLOCK - int((MERGE_TS - ts) / 13.2))


def _batch_calls(calls, block):
    """[(address, selector)] at one block → list of int (None when the contract
    doesn't exist yet or the call reverts)."""
    payload = [{'jsonrpc': '2.0', 'id': i, 'method': 'eth_call',
                'params': [{'to': a, 'data': s}, hex(block)]} for i, (a, s) in enumerate(calls)]
    out = [None] * len(calls)
    for k in range(0, len(payload), BATCH):         # free public nodes cap batch size
        res = _rpc(payload[k:k + BATCH])
        if isinstance(res, dict):
            res = [res]
        for r in res:
            if 'error' in r and 'not allowed' in str(r['error']):
                raise RuntimeError(r['error'])
            v = r.get('result')
            if v and v != '0x':
                out[r['id']] = int(v, 16)
    return out


def _decode_str(h):
    b = bytes.fromhex(h[2:])
    if len(b) >= 96:
        n = int.from_bytes(b[32:64], 'big')
        return b[64:64 + n].decode(errors='replace')
    return b.rstrip(b'\0').decode(errors='replace')


def token_meta(tokens):
    """Decimals and symbol straight from each contract (the CSV is not trusted for these)."""
    meta = {}
    for _, t in tokens.iterrows():
        dec = _rpc({'jsonrpc': '2.0', 'id': 1, 'method': 'eth_call',
                    'params': [{'to': t.address, 'data': SEL['decimals']}, 'latest']})['result']
        sym = _rpc({'jsonrpc': '2.0', 'id': 1, 'method': 'eth_call',
                    'params': [{'to': t.address, 'data': SEL['symbol']}, 'latest']})['result']
        meta[t.symbol] = dict(decimals=int(dec, 16), onchain_symbol=_decode_str(sym))
        if meta[t.symbol]['onchain_symbol'].lower() != t.symbol.lower():
            print(f'  ! {t.symbol}: contract symbol is {meta[t.symbol]["onchain_symbol"]!r}')
    return meta


def update_supply(start='2019-09-01'):
    """Weekly supply back to `start` for any token not yet stored, then one reading per
    day since the last stored date. Appends to data/eth_rwa/supply.csv."""
    os.makedirs(DATA_DIR, exist_ok=True)
    tokens = pd.read_csv(TOKENS)
    meta = token_meta(tokens)
    have = pd.read_csv(SUPPLY, parse_dates=['date']) if os.path.exists(SUPPLY) else \
        pd.DataFrame(columns=['date', 'symbol', 'supply'])
    latest = _latest_block()
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()

    rows = []
    stored = set(have['symbol'].unique())
    new = [s for s in tokens.symbol if s not in stored]
    if new:                                         # backfill new tokens weekly
        print(f'  backfilling {", ".join(new)} weekly from {start}')
        sub = tokens[tokens.symbol.isin(new)]
        for d in pd.date_range(start, today, freq='W-SUN'):
            blk = _block_at(int(d.timestamp()) + 43200, latest)
            vals = _batch_calls([(a, SEL['totalSupply']) for a in sub.address], blk)
            for sym, v in zip(sub.symbol, vals):
                if v:
                    rows.append((d, sym, v / 10 ** meta[sym]['decimals']))
            time.sleep(0.2)

    last = have['date'].max() if len(have) else today - pd.Timedelta(days=1)
    for d in pd.date_range(last + pd.Timedelta(days=1), today, freq='D'):
        blk = _block_at(int(d.timestamp()) + 43200, latest) if d < today else latest[0]
        vals = _batch_calls([(a, SEL['totalSupply']) for a in tokens.address], blk)
        for sym, v in zip(tokens.symbol, vals):
            if v:
                rows.append((d, sym, v / 10 ** meta[sym]['decimals']))
    if rows:
        add = pd.DataFrame(rows, columns=['date', 'symbol', 'supply'])
        out = pd.concat([have, add]).drop_duplicates(['date', 'symbol'], keep='last')
        out.sort_values(['date', 'symbol']).to_csv(SUPPLY, index=False, float_format='%.6f')
        print(f'  supply: +{len(add)} readings → {SUPPLY}')
    return pd.read_csv(SUPPLY, parse_dates=['date'])


# ════════════════════════════════════════════
# PRICES
# ════════════════════════════════════════════
def _gold(index):
    import yfinance as yf
    g = yf.Ticker('GC=F').history(period='max', auto_adjust=False)['Close']
    g.index = g.index.tz_localize(None).normalize()
    return g.reindex(index.union(g.index)).ffill().reindex(index)


def _coingecko_daily(cid):
    for attempt in range(5):
        r = requests.get(f'https://api.coingecko.com/api/v3/coins/{cid}/market_chart',
                         params={'vs_currency': 'usd', 'days': 365, 'interval': 'daily'}, timeout=30)
        if r.status_code == 429:
            time.sleep(20 * (attempt + 1))
            continue
        r.raise_for_status()
        p = pd.Series({pd.Timestamp(t, unit='ms').normalize(): v for t, v in r.json()['prices']})
        return p[~p.index.duplicated(keep='last')].sort_index()
    raise RuntimeError(f'CoinGecko rate limit for {cid}')


def _nav(cid, index, saved=None):
    """Daily NAV from observed prices: CoinGecko's last year plus every price saved on
    earlier runs (so the observed history grows past CoinGecko's one-year limit). Dates
    before the first observation are discounted at the yield the latest year implies
    (clipped to 0–15% a year). Returns (nav on index, observed prices)."""
    p = _coingecko_daily(cid)
    p = p[p > 0]
    if saved is not None and len(saved.dropna()):
        p = p.combine_first(saved.dropna())          # fresh CoinGecko prices win
    yr = p[p.index >= p.index[-1] - pd.Timedelta(days=365)]
    years = (yr.index[-1] - yr.index[0]).days / 365.25
    r = np.log(yr.iloc[-1] / yr.iloc[0]) / years if years > 0.25 else 0.0
    r = float(np.clip(r, 0, 0.15))
    nav = p.reindex(index.union(p.index)).interpolate(limit_area='inside').reindex(index)
    early = index < p.index[0]
    nav[early] = p.iloc[0] * np.exp(-r * (p.index[0] - index[early]).days / 365.25)
    return nav.ffill(), p


def _saved_navs():
    return pd.read_csv(NAV_FILE, index_col=0, parse_dates=True) if os.path.exists(NAV_FILE) else pd.DataFrame()


def token_values(supply, index):
    """USD value of each token on Ethereum, daily on `index`. NAV tokens fall back to the
    saved NAVs (data/eth_rwa/nav_prices.csv) if CoinGecko is unavailable."""
    tokens = pd.read_csv(TOKENS).set_index('symbol')
    saved, fresh = _saved_navs(), {}
    sup = supply.pivot_table(index='date', columns='symbol', values='supply')
    sup = sup.reindex(index.union(sup.index)).interpolate(limit_area='inside').reindex(index)
    gold = _gold(index) if (tokens.pricing.str.startswith('gold')).any() else None
    vals = {}
    for sym, t in tokens.iterrows():
        if sym not in sup:
            continue
        if t.pricing == 'gold_oz':
            px = gold
        elif t.pricing == 'gold_g':
            px = gold / TROY_OZ_G
        elif t.pricing == 'usd':
            px = pd.Series(1.0, index=index)
        else:
            try:
                px, fresh[sym] = _nav(t.coingecko_id, index, saved.get(sym))
                time.sleep(4)                       # CoinGecko free tier
            except Exception as e:
                if sym not in saved:
                    print(f'  ! {sym}: no NAV ({e}); left out today')
                    continue
                print(f'  ! {sym}: CoinGecko failed ({e}); using saved NAVs')
                px = saved[sym].reindex(index.union(saved.index)).ffill().bfill().reindex(index)
        vals[sym] = (sup[sym] * px).fillna(0)
    if fresh:                                       # observed prices only, never estimates
        obs = pd.DataFrame(fresh)
        nav = obs.combine_first(saved) if len(saved) else obs
        nav.sort_index().to_csv(NAV_FILE, float_format='%.6f')
    return pd.DataFrame(vals, index=index)


# ════════════════════════════════════════════
# DEFILLAMA (what it still reports)
# ════════════════════════════════════════════
def defillama_rwa(get):
    """Sum of DefiLlama RWA-category protocols on Ethereum, minus COUNTED_DIRECTLY."""
    protocols = get('https://api.llama.fi/protocols').json()
    picks = [(p['name'], p['slug']) for p in protocols
             if ('rwa' in (p.get('category') or '').lower() or 'real world' in (p.get('category') or '').lower())
             and (p.get('chainTvls') or {}).get('Ethereum', 0) > ETH_TVL_THRESHOLD
             and p['name'] not in COUNTED_DIRECTLY]
    series = {}
    for name, slug in picks:
        try:
            data = get(f'https://api.llama.fi/protocol/{slug}').json()
            hist = data.get('chainTvls', {}).get('Ethereum', {}).get('tvl', [])
            if hist:
                series[name] = pd.Series({pd.Timestamp(p['date'], unit='s'): p['totalLiquidityUSD'] for p in hist})
            time.sleep(0.25)
        except Exception as e:
            print(f'  ! {name}: {e}')
    if not series:
        return pd.Series(dtype=float)
    df = pd.DataFrame(series)
    df.index = pd.to_datetime(df.index).normalize()
    return df.resample('D').last().ffill().sum(axis=1).iloc[:-1]


# ════════════════════════════════════════════
# PUBLIC
# ════════════════════════════════════════════
def fetch_rwa_tvl(get=None, update=True, detail=False):
    """Daily RWA on Ethereum in USD as a one-column frame 'rwa_tvl_usd' (the shape the
    ETH chart scripts expect). `get` is the caller's HTTP helper (with retries).
    The first call of the day caches the result (data/eth_rwa/rwa_daily.csv) and later
    calls that day reuse it, so the seven ETH charts don't each hit CoinGecko."""
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    if not detail and os.path.exists(CACHE):
        cached = pd.read_csv(CACHE, index_col=0, parse_dates=True)
        if len(cached) and cached.index[-1].normalize() == today:     # computed today (file dates lie after a checkout)
            print(f'RWA on Ethereum: cached today | latest ${cached["rwa_tvl_usd"].iloc[-1]/1e9:.2f}B')
            return cached
    get = get or (lambda url: requests.get(url, timeout=90))
    print('Fetching RWA on Ethereum (DefiLlama protocols + tokenized funds and gold on-chain)...')
    dl = defillama_rwa(get)
    try:
        supply = update_supply() if update else pd.read_csv(SUPPLY, parse_dates=['date'])
    except Exception as e:                          # node down: use what is stored
        print(f'  ! supply update failed ({e}); using stored supply')
        supply = pd.read_csv(SUPPLY, parse_dates=['date'])
    end = max(dl.index.max() if len(dl) else supply['date'].max(), supply['date'].max())
    index = pd.date_range(min(dl.index.min() if len(dl) else supply['date'].min(), supply['date'].min()), end, freq='D')
    tok = token_values(supply, index)
    total = dl.reindex(index).ffill().fillna(0) + tok.sum(axis=1)
    print(f'  {len(index)} days | latest: ${total.iloc[-1]/1e9:.2f}B '
          f'(DefiLlama protocols ${dl.iloc[-1]/1e9 if len(dl) else 0:.2f}B, tokens ${tok.iloc[-1].sum()/1e9:.2f}B)')
    out = total.to_frame(name='rwa_tvl_usd')
    out.to_csv(CACHE, float_format='%.0f')
    if detail:
        return out, dl, tok
    return out
