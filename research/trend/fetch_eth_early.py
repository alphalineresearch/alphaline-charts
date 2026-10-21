"""
One-off: ETH/USD daily closes from Kraken's public trade history, Aug 7 2015 (first day of
trading) to May 17 2016 (the day before Coinbase's ETH-USD history starts).
Writes data/eth_early_daily.csv (date, close = last trade of the UTC day, trades, volume).

    python research/trend/fetch_eth_early.py
"""
import os
import sys
import time

import pandas as pd
import requests

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT = os.path.join(ROOT, 'data', 'eth_early_daily.csv')
END = pd.Timestamp('2016-05-18')

rows, since = [], '0'
while True:
    for attempt in range(5):
        r = requests.get('https://api.kraken.com/0/public/Trades',
                         params=dict(pair='ETHUSD', since=since), timeout=30).json()
        if not r.get('error'):
            break
        time.sleep(3 * (attempt + 1))          # rate limit
    res = r['result']
    since = res.pop('last')
    trades = next(iter(res.values()))
    rows += [(float(t[2]), float(t[0]), float(t[1])) for t in trades]
    last = pd.Timestamp(rows[-1][0], unit='s')
    print(f'\r  {len(rows):,} trades → {last:%Y-%m-%d}', end='', flush=True)
    if last >= END or not trades:
        break
    time.sleep(1.1)

df = pd.DataFrame(rows, columns=['ts', 'price', 'vol'])
df['date'] = pd.to_datetime(df['ts'], unit='s').dt.normalize()
df = df[df['date'] < END]
daily = df.groupby('date').agg(close=('price', 'last'), trades=('price', 'size'), volume=('vol', 'sum'))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
daily.to_csv(OUT)
print(f'\n  {len(daily)} days {daily.index[0]:%Y-%m-%d} → {daily.index[-1]:%Y-%m-%d} → {OUT}')
