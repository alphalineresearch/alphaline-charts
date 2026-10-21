"""
Download CoinMarketCap's weekly historical top-500 snapshots (Sundays, 2013-04-28 →)
into research/breadth/cache/snapshots.csv.gz. Resumable: weeks already cached are skipped.

Source: CoinMarketCap website data API (unofficial, no key). Snapshots include coins
that later died, so breadth built from them is free of survivorship bias.
"""
import gzip, json, os, sys, time
import pandas as pd
import requests

HERE  = os.path.dirname(os.path.abspath(__file__))
RAW   = os.path.join(HERE, 'cache', 'raw')
OUT   = os.path.join(HERE, 'cache', 'snapshots.csv.gz')
URL   = ('https://api.coinmarketcap.com/data-api/v3/cryptocurrency/listings/historical'
         '?date={}&limit=500&start=1&convertId=2781')
os.makedirs(RAW, exist_ok=True)


def fetch(day):
    path = os.path.join(RAW, f'{day}.json.gz')
    if os.path.exists(path):
        return
    for attempt in range(6):
        try:
            r = requests.get(URL.format(day), timeout=30,
                             headers={'User-Agent': 'Mozilla/5.0 (alphaline research)'})
            j = r.json()
            data = j.get('data')
            if r.ok and isinstance(data, list):
                with gzip.open(path, 'wt', encoding='utf-8') as f:
                    json.dump(data, f)
                print(f'{day}: {len(data)} coins', flush=True)
                return
            msg = str(j.get('status', {}).get('error_message', r.status_code))
        except Exception as e:
            msg = str(e)
        wait = 10 * 2 ** attempt
        print(f'{day}: retry in {wait}s ({msg[:80]})', flush=True)
        time.sleep(wait)
    print(f'{day}: GAVE UP', flush=True)


def build():
    rows = []
    for fn in sorted(os.listdir(RAW)):
        day = fn[:10]
        with gzip.open(os.path.join(RAW, fn), 'rt', encoding='utf-8') as f:
            for c in json.load(f):
                q = (c.get('quotes') or [{}])[0]
                rows.append(dict(date=day, id=c['id'], symbol=c.get('symbol'), slug=c.get('slug'),
                                 name=c.get('name'), rank=c.get('cmcRank'),
                                 tags='|'.join(c.get('tags') or []),
                                 price=q.get('price'), mcap=q.get('marketCap'),
                                 volume=q.get('volume24h'), supply=c.get('circulatingSupply')))
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False, compression='gzip')
    print(f'built {OUT}: {len(df):,} rows, {df.date.nunique()} weeks')


if __name__ == '__main__':
    end = pd.Timestamp.utcnow().tz_localize(None).normalize()
    days = pd.date_range('2013-04-28', end, freq='W-SUN')
    if 'reverse' in sys.argv:
        days = days[::-1]
    for d in days:
        fetch(d.strftime('%Y-%m-%d'))
        time.sleep(1.2)
    build()
