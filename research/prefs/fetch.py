"""Daily prices + dividends for crypto-treasury preferreds and context series → cache/"""
import os, json
import pandas as pd, yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
PREFS = ['STRC', 'STRK', 'STRF', 'STRD', 'BMNP', 'SATA']
OTHER = ['MSTR', 'BMNR', 'ASST', 'BTC-USD', 'ETH-USD', '^TNX', '^IRX']

if __name__ == '__main__':
    px, divs, info = {}, {}, {}
    for t in PREFS + OTHER:
        tk = yf.Ticker(t)
        h = tk.history(period='max', auto_adjust=False)
        h.index = h.index.tz_localize(None).normalize()
        px[t] = h['Close']
        if t in PREFS:
            d = h['Dividends'][h['Dividends'] > 0]
            divs[t] = d
            try:
                i = tk.get_info()
                info[t] = {k: i.get(k) for k in ('longName', 'shortName', 'quoteType', 'longBusinessSummary', 'dividendRate', 'dividendYield', 'exDividendDate')}
            except Exception as e:
                info[t] = {'err': str(e)}
        print(f'{t:8} {len(h):5} rows {h.index[0].date()} → {h.index[-1].date()} last {h["Close"].iloc[-1]:.2f}')
    pd.DataFrame(px).to_csv(os.path.join(HERE, 'cache', 'prices.csv'))
    pd.DataFrame(divs).to_csv(os.path.join(HERE, 'cache', 'dividends.csv'))
    json.dump(info, open(os.path.join(HERE, 'cache', 'info.json'), 'w'), indent=1, default=str)
    for t in PREFS:
        d = divs[t]
        print(f'\n{t}: {len(d)} dividends; last 8:', [(x.date().isoformat(), round(v, 4)) for x, v in d.tail(8).items()])
        print('  ', {k: (str(v)[:300] if v else v) for k, v in info[t].items()})
