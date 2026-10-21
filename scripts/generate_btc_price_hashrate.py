"""
generate_btc_price_hashrate.py
Alphaline Research — BTC Price & Hash Rate

Fetches BTC price and hash rate history from blockchain.info.
No API keys required.
Writes docs/btc_price_hashrate.html to docs/.

Usage:
    python scripts/generate_btc_price_hashrate.py
"""

import os
import sys
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from alphaline_theme import apply_theme, write_chart, date_long

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

# ════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════
OUTPUT_PATH = os.path.join('docs', 'btc_price_hashrate.html')

# ════════════════════════════════════════════
# BRAND COLORS
# ════════════════════════════════════════════
GOLD      = '#D4A843'
GOLD_LIT  = '#ECC96A'
GREEN_LIT = '#2ABF7A'


def hex_to_rgba(h, a=0.35):
    h = h.lstrip('#')
    return f'rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})'


# ════════════════════════════════════════════
# DATA FETCHER
# ════════════════════════════════════════════
def fetch_blockchain_chart(chart_name):
    url = (f'https://api.blockchain.info/charts/{chart_name}'
           f'?timespan=all&format=json&sampled=false')
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    rows = [
        {'date': pd.Timestamp(int(pt['x']), unit='s').normalize(),
         'value': float(pt['y'])}
        for pt in resp.json().get('values', [])
    ]
    df = pd.DataFrame(rows).set_index('date').sort_index()
    return df[~df.index.duplicated(keep='last')]


def build_dataframe():
    print('Fetching BTC price history (blockchain.info)...')
    price_df = fetch_blockchain_chart('market-price')
    price_df.columns = ['price_usd']
    price_df = price_df[price_df['price_usd'] > 0].dropna()
    print(f'  {len(price_df)} rows | {price_df.index[0].date()} → {price_df.index[-1].date()} | latest ${price_df["price_usd"].iloc[-1]:,.0f}')

    print('Fetching hash rate history (blockchain.info)...')
    hr_df = fetch_blockchain_chart('hash-rate')
    hr_df.columns = ['hashrate_raw']
    hr_df['hashrate_eh'] = hr_df['hashrate_raw'] / 1e6
    hr_df = hr_df[hr_df['hashrate_eh'] > 0].dropna()
    hr_df = hr_df[hr_df.index >= price_df.index[0]]
    print(f'  {len(hr_df)} rows | {hr_df.index[0].date()} → {hr_df.index[-1].date()} | latest {hr_df["hashrate_eh"].iloc[-1]:.1f} EH/s')

    price_rolling_ath = price_df['price_usd'].cummax()
    hr_rolling_ath    = hr_df['hashrate_eh'].cummax()

    return price_df, hr_df, price_rolling_ath, hr_rolling_ath


# ════════════════════════════════════════════
# CHART — BTC PRICE & HASH RATE
# ════════════════════════════════════════════
def plot_btc_price_hashrate(price_df, hr_df, price_rolling_ath, hr_rolling_ath):
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.52, 0.48],
        vertical_spacing=0.04,
    )

    # ── Panel 1: BTC Price ────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=price_df.index, y=price_df['price_usd'],
        mode='lines', name='BTC Price',
        line=dict(color=GOLD, width=1.4),
        hovertemplate='%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>',
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=price_rolling_ath.index, y=price_rolling_ath,
        mode='lines', name='Price ATH',
        line=dict(color=GOLD_LIT, width=1.0, dash='dot'),
        hovertemplate='%{x|%Y-%m-%d}<br>ATH: $%{y:,.0f}<extra></extra>',
    ), row=1, col=1)

    # ── Panel 2: Hash Rate ────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=hr_df.index, y=hr_df['hashrate_eh'],
        mode='lines', name='Hash Rate',
        line=dict(color=GREEN_LIT, width=1.4),
        hovertemplate='%{x|%Y-%m-%d}<br>%{y:.1f} EH/s<extra></extra>',
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=hr_rolling_ath.index, y=hr_rolling_ath,
        mode='lines', name='Hash Rate ATH',
        line=dict(color='#58a6ff', width=1.0, dash='dot'),
        hovertemplate='%{x|%Y-%m-%d}<br>ATH: %{y:.1f} EH/s<extra></extra>',
    ), row=2, col=1)

    apply_theme(fig)
    fig.update_yaxes(type='log', title_text='Price (USD, log)', row=1, col=1)
    fig.update_yaxes(type='log', title_text='Hash Rate (EH/s, log)', row=2, col=1)
    fig.update_xaxes(rangeslider_visible=False)
    return fig


# ════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════
if __name__ == '__main__':
    os.makedirs('docs', exist_ok=True)

    print('=== Building BTC Price & Hash Rate chart ===')
    price_df, hr_df, price_rolling_ath, hr_rolling_ath = build_dataframe()
    fig = plot_btc_price_hashrate(price_df, hr_df, price_rolling_ath, hr_rolling_ath)

    _px, _hr = price_df['price_usd'].iloc[-1], hr_df['hashrate_eh'].iloc[-1]
    _hr_below = (1 - _hr / hr_rolling_ath.iloc[-1]) * 100
    write_chart(fig, OUTPUT_PATH,
                title='BTC Price & Hash Rate',
                subtitle=(f'<b>BTC ${_px:,.0f}</b> · hash rate {_hr:,.0f} EH/s · '
                          + ('at its high' if _hr_below < 0.5 else f'{_hr_below:.0f}% below its high')
                          + f' · {date_long(hr_df.index[-1])}'),
                note='Dotted lines = running all-time highs',
                source='blockchain.info', default='5Y')
    print(f'Saved: {OUTPUT_PATH}')
