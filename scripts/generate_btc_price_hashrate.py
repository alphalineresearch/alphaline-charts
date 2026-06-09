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
NAVY      = '#0A1628'
NAVY_MID  = '#102240'
GOLD      = '#D4A843'
GOLD_LIT  = '#ECC96A'
WHITE     = '#F8FAFB'
MIST      = '#7A8F9F'
STEEL     = '#374D61'
GREEN_LIT = '#2ABF7A'

CHART_HEIGHT = 720


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

    # ── Range buttons ─────────────────────────────────────────────────────────
    end_dt    = price_df.index[-1]
    jan1      = pd.Timestamp(end_dt.year, 1, 1)
    all_start = price_df.index[0]

    def _log_range(series_list, pad=0.08):
        combined = pd.concat([s.dropna() for s in series_list])
        combined = combined[combined > 0]
        lo = np.log10(combined.min())
        hi = np.log10(combined.max())
        span = max(hi - lo, 0.01)
        return [lo - pad * span, hi + pad * span]

    def _rng(start, end=end_dt):
        s, e = start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')
        p_win  = price_df.loc[s:e, 'price_usd']
        ath_p  = price_rolling_ath.loc[s:e]
        hr_win = hr_df.loc[s:e, 'hashrate_eh']
        ath_hr = hr_rolling_ath.loc[s:e]
        return [{
            'xaxis.range':       [s, e],
            'xaxis2.range':      [s, e],
            'yaxis.range':       _log_range([p_win, ath_p]),
            'yaxis2.range':      _log_range([hr_win, ath_hr]),
            'yaxis.autorange':   False,
            'yaxis2.autorange':  False,
        }]

    fig.update_layout(
        updatemenus=[dict(
            type='buttons',
            direction='right',
            x=0.5, xanchor='center',
            y=-0.13, yanchor='top',
            pad=dict(r=4, t=4, b=4),
            showactive=True,
            bgcolor=MIST,
            bordercolor=STEEL,
            font=dict(family='Courier New, monospace', color=NAVY, size=10),
            buttons=[
                dict(label='1M',  method='relayout', args=_rng(end_dt - pd.DateOffset(months=1))),
                dict(label='3M',  method='relayout', args=_rng(end_dt - pd.DateOffset(months=3))),
                dict(label='6M',  method='relayout', args=_rng(end_dt - pd.DateOffset(months=6))),
                dict(label='YTD', method='relayout', args=_rng(jan1)),
                dict(label='1Y',  method='relayout', args=_rng(end_dt - pd.DateOffset(years=1))),
                dict(label='3Y',  method='relayout', args=_rng(end_dt - pd.DateOffset(years=3))),
                dict(label='5Y',  method='relayout', args=_rng(end_dt - pd.DateOffset(years=5))),
                dict(label='ALL', method='relayout', args=_rng(all_start)),
            ],
        )]
    )

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=NAVY, plot_bgcolor=NAVY_MID,
        height=CHART_HEIGHT, autosize=True,
        title=dict(
            text=(
                f'<span style="font-family:Georgia,serif; font-size:15px; color:{WHITE};">BTC Price & Hash Rate</span>'
                f'<br><span style="font-family:Courier New,monospace; font-size:8px; color:{GOLD};">ALPHALINE RESEARCH</span>'
            ),
            x=0.02, xanchor='left', y=0.985, yanchor='top',
        ),
        font=dict(family='Courier New, monospace', color=MIST, size=10),
        margin=dict(l=65, r=90, t=65, b=160),
        showlegend=True,
        legend=dict(
            orientation='h', yanchor='top', y=-0.08, xanchor='center', x=0.5,
            bgcolor='rgba(10,22,40,0.8)', bordercolor=STEEL, font=dict(size=9),
        ),
        hoverlabel=dict(bgcolor=NAVY_MID, bordercolor=GOLD,
                        font=dict(family='Courier New, monospace', size=11, color=WHITE)),
        annotations=[
            dict(text='Source: blockchain.info',
                 xref='paper', yref='paper', x=1.0, y=-0.21,
                 xanchor='right', yanchor='top',
                 font=dict(family='Courier New, monospace', size=10, color=MIST),
                 showarrow=False),
            dict(text='<b>ALPHALINE RESEARCH</b>',
                 xref='paper', yref='paper', x=0.0, y=-0.21,
                 xanchor='left', yanchor='top',
                 font=dict(family='Courier New, monospace', size=10, color=GOLD),
                 showarrow=False),
        ],
    )

    fig.update_yaxes(type='log', gridcolor='rgba(212,168,67,0.06)', gridwidth=0.5,
                     zeroline=False, showspikes=True, spikecolor=MIST,
                     spikethickness=1, spikedash='dot',
                     title_text='Price (USD, log)', row=1, col=1)
    fig.update_yaxes(type='log', gridcolor='rgba(212,168,67,0.06)', gridwidth=0.5,
                     zeroline=False, showspikes=True, spikecolor=MIST,
                     spikethickness=1, spikedash='dot',
                     title_text='Hash Rate (EH/s, log)', row=2, col=1)
    fig.update_xaxes(gridcolor='rgba(212,168,67,0.06)', gridwidth=0.5, zeroline=False,
                     showspikes=True, spikecolor=MIST, spikethickness=1, spikedash='dot',
                     rangeslider_visible=False)

    return fig


# ════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════
if __name__ == '__main__':
    os.makedirs('docs', exist_ok=True)

    print('=== Building BTC Price & Hash Rate chart ===')
    price_df, hr_df, price_rolling_ath, hr_rolling_ath = build_dataframe()
    fig = plot_btc_price_hashrate(price_df, hr_df, price_rolling_ath, hr_rolling_ath)

    fig.write_html(OUTPUT_PATH, include_plotlyjs='cdn', config={'responsive': True})
    print(f'Saved: {OUTPUT_PATH}')
