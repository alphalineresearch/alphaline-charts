"""
generate_btc_production_cost_charts.py
Alphaline Research — BTC Production Cost + Risk Score Dashboard

Fetches BTC price from Yahoo Finance, MSTR price from Yahoo Finance,
hash rate + mining difficulty from blockchain.info, current hash rate from
mempool.space, and BTC spot from CoinGecko. No API keys required.

Writes four files to docs/:
  docs/btc_price_vs_production_cost.html  — Chart 5
  docs/btc_risk_score_history.html        — Chart 7
  docs/btc_risk_score_recency.html        — Chart 7b
  docs/btc_signal_mstr.html               — Chart 7c

Usage:
    python scripts/generate_btc_production_cost_charts.py
"""

import os
import sys
import bisect
import warnings
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import linregress
from datetime import datetime, timedelta
import yfinance as yf

warnings.filterwarnings('ignore')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

# ════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════
electricity_cost_kwh       = 0.05
fleet_efficiency_jth       = 22
PUE                        = 1.10
efficiency_improvement_pct = 0.05
MIN_EFFICIENCY_JTH         = 3.0
EFFICIENCY_IMPROVEMENT_AFTER_2036 = 0.02
_EFF_CUTOFF_YRS            = 10

diminishing_scenarios = {
    'Bear': {'base_cagr': 0.15, 'decay_factor': 0.40},
    'Base': {'base_cagr': 0.15, 'decay_factor': 0.55},
    'Bull': {'base_cagr': 0.25, 'decay_factor': 0.70},
}
SCENARIO_ORDER = ['Bear', 'Base', 'Bull']

BLOCKS_PER_HALVING = 210_000
GENESIS_SUBSIDY    = 50.0
GENESIS_DATE       = datetime(2009, 1, 3)
BTC_MAX_SUPPLY     = 21_000_000

HASH_NORM_WINDOW   = 730
SIGNAL_THRESHOLD   = 40

_HIST_EFF = {
    '2013-01-01': 1000.0, '2014-01-01': 500.0,  '2016-01-01': 200.0,
    '2018-01-01': 100.0,  '2019-01-01': 75.0,   '2020-01-01': 60.0,
    '2021-01-01': 40.0,   '2022-06-01': 32.0,   '2023-01-01': 25.0,
    '2024-01-01': 22.0,
}
_HIST_HALVINGS = [
    ('2012-11-28', 25.0), ('2016-07-09', 12.5),
    ('2020-05-11', 6.25), ('2024-04-20', 3.125),
]

FALLBACK_BLOCK_HEIGHT = 945_000
FALLBACK_HASH_RATE_EH = 1010.0
FALLBACK_BTC_PRICE    = 85_000

OUTPUT_COST     = os.path.join('docs', 'btc_price_vs_production_cost.html')
OUTPUT_HISTORY  = os.path.join('docs', 'btc_risk_score_history.html')
OUTPUT_RECENCY  = os.path.join('docs', 'btc_risk_score_recency.html')
OUTPUT_MSTR     = os.path.join('docs', 'btc_signal_mstr.html')

# ════════════════════════════════════════════
# BRAND COLORS
# ════════════════════════════════════════════
NAVY      = '#0A1628'
NAVY_MID  = '#102240'
NAVY_LIT  = '#1A3A6E'
GOLD      = '#D4A843'
GOLD_LIT  = '#ECC96A'
GOLD_DIM  = '#8A6B25'
WHITE     = '#F8FAFB'
MIST      = '#7A8F9F'
STEEL     = '#374D61'
GREEN     = '#1A8A5A'
RED       = '#8A1A1A'
GREEN_LIT = '#2ABF7A'
RED_LIT   = '#D64444'
PURPLE    = '#9B72CF'

SCENARIO_COLORS = {'Bear': RED_LIT, 'Base': GOLD, 'Bull': GREEN_LIT}
CHART_HEIGHT    = 750

RECENCY_BANDS = [
    (  0,  30,   GREEN_LIT, '0–30d (fresh)'),
    ( 30,  90,   GOLD,      '30–90d'),
    ( 90,  180,  GOLD_DIM,  '90–180d'),
    (180,  365,  RED_LIT,   '180–365d'),
    (365,  9999, MIST,      '>1yr / never'),
]


def hex_to_rgba(h, a=0.35):
    h = h.lstrip('#')
    return f'rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})'


# ════════════════════════════════════════════
# CHART TEMPLATE
# ════════════════════════════════════════════
def alphaline_layout(fig, title, height=CHART_HEIGHT, subtitle='',
                     source='alphalineresearch.com  |  blockchain.info · Mempool.space · CoinGecko · Yahoo Finance'):
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=NAVY, plot_bgcolor=NAVY_MID,
        height=height, autosize=True,
        title=dict(
            text=(
                f'<span style="font-family:Georgia,serif; font-size:15px; color:{WHITE};">{title}</span>'
                f'<br><span style="font-family:Courier New,monospace; font-size:8px; color:{GOLD};">ALPHALINE RESEARCH</span>'
                + (f'<br><span style="font-family:Courier New,monospace; font-size:9px; color:{MIST};">{subtitle}</span>' if subtitle else '')
            ),
            x=0.02, xanchor='left', y=0.985, yanchor='top'
        ),
        font=dict(family='Courier New, monospace', color=MIST, size=10),
        margin=dict(l=55, r=80, t=80, b=130),
        xaxis=dict(gridcolor='rgba(212,168,67,0.06)', gridwidth=0.5, zeroline=False,
                   showspikes=True, spikecolor=MIST, spikethickness=1, spikedash='dot'),
        yaxis=dict(gridcolor='rgba(212,168,67,0.06)', gridwidth=0.5, zeroline=False),
        hoverlabel=dict(bgcolor=NAVY_MID, bordercolor=GOLD,
                        font=dict(family='Courier New, monospace', size=11, color=WHITE)),
        legend=dict(
            orientation='h', yanchor='top', y=-0.09,
            xanchor='center', x=0.5,
            bgcolor='rgba(10,22,40,0.8)', bordercolor=STEEL,
            font=dict(size=9),
        ),
        annotations=[
            dict(text=f'Source: {source}',
                 xref='paper', yref='paper', x=1.0, y=-0.13,
                 xanchor='right', yanchor='top',
                 font=dict(family='Courier New, monospace', size=10, color=MIST),
                 showarrow=False),
            dict(text='<b>ALPHALINE RESEARCH</b>',
                 xref='paper', yref='paper', x=0.0, y=-0.13,
                 xanchor='left', yanchor='top',
                 font=dict(family='Courier New, monospace', size=10, color=GOLD),
                 showarrow=False),
        ],
    )
    return fig


# ════════════════════════════════════════════
# PROJECTION + COST HELPERS
# ════════════════════════════════════════════
def projected_efficiency(years):
    if years <= _EFF_CUTOFF_YRS:
        eff = fleet_efficiency_jth * ((1 - efficiency_improvement_pct) ** years)
    else:
        eff_at_cutoff = fleet_efficiency_jth * ((1 - efficiency_improvement_pct) ** _EFF_CUTOFF_YRS)
        eff = eff_at_cutoff * ((1 - EFFICIENCY_IMPROVEMENT_AFTER_2036) ** (years - _EFF_CUTOFF_YRS))
    return max(eff, MIN_EFFICIENCY_JTH)


def production_cost_per_btc(hash_rate_eh, eff_jth, elec_kwh, subsidy_btc, pue=PUE):
    hashrate_th    = hash_rate_eh * 1e6
    daily_kwh      = hashrate_th * eff_jth * pue * 86400 / 3.6e6
    daily_cost_usd = daily_kwh * elec_kwh
    daily_btc      = 144 * subsidy_btc
    return daily_cost_usd / daily_btc if daily_btc else float('nan')


def project_hr_cycle_decay(years_out, base_cagr, decay_factor, base_hr):
    full_cycles = int(years_out / 4)
    partial_yrs = years_out - full_cycles * 4
    growth = 1.0
    for c in range(full_cycles):
        growth *= (1 + base_cagr * (decay_factor ** c)) ** 4
    if partial_yrs > 0:
        growth *= (1 + base_cagr * (decay_factor ** full_cycles)) ** partial_yrs
    return base_hr * growth


def subsidy_at_t(y, halvings_yrs, current_sub):
    sub = current_sub
    for yth in halvings_yrs:
        if y >= yth:
            sub /= 2
        else:
            break
    return sub


def _hist_eff(date):
    e = 1000.0
    for ts, v in sorted(_HIST_EFF.items()):
        if date >= pd.Timestamp(ts):
            e = v
    return e


def _hist_sub(date):
    r = 50.0
    for ts, v in _HIST_HALVINGS:
        if date >= pd.Timestamp(ts):
            r = v
    return r


# ════════════════════════════════════════════
# DATA FETCHERS
# ════════════════════════════════════════════
def fetch_json(url, timeout=15):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


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


def fetch_btc_price():
    print('Fetching BTC price (yfinance)...')
    raw = yf.download('BTC-USD', period='max', interval='1d', progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    raw.index.name = 'date'
    px = raw['close'].dropna().rename('btc_price')
    today = pd.Timestamp.utcnow().normalize().tz_localize(None)
    if len(px) and px.index[-1] >= today:
        px = px.iloc[:-1]
    print(f'  {len(px)} rows | latest: ${px.iloc[-1]:,.0f}')
    return px


def fetch_mstr_price():
    print('Fetching MSTR price (yfinance)...')
    raw = yf.download('MSTR', start='2020-01-01', auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    px = raw['close'].dropna().sort_index()
    print(f'  {len(px)} rows | latest: ${px.iloc[-1]:,.2f}')
    return px


def fetch_current_state():
    print('Fetching current block height (mempool.space)...')
    try:
        tip_height = int(requests.get('https://mempool.space/api/blocks/tip/height', timeout=10).text.strip())
        print(f'  Block height: {tip_height:,}')
    except Exception as e:
        tip_height = FALLBACK_BLOCK_HEIGHT
        print(f'  Block height (fallback): {tip_height:,}  [{e}]')

    print('Fetching hash rate (mempool.space)...')
    try:
        hr_data    = fetch_json('https://mempool.space/api/v1/mining/hashrate/1m')
        live_hr_eh = hr_data['currentHashrate'] / 1e18
        print(f'  Hash rate: {live_hr_eh:.1f} EH/s')
    except Exception as e:
        live_hr_eh = FALLBACK_HASH_RATE_EH
        print(f'  Hash rate (fallback): {live_hr_eh:.1f} EH/s  [{e}]')

    print('Fetching BTC spot (CoinGecko)...')
    try:
        btc_spot = fetch_json(
            'https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd'
        )['bitcoin']['usd']
        print(f'  BTC spot: ${btc_spot:,.0f}')
    except Exception as e:
        btc_spot = FALLBACK_BTC_PRICE
        print(f'  BTC spot (fallback): ${btc_spot:,.0f}  [{e}]')

    current_halving_era = tip_height // BLOCKS_PER_HALVING
    current_subsidy     = GENESIS_SUBSIDY / (2 ** current_halving_era)
    print(f'  Current era: {current_halving_era}  |  subsidy: {current_subsidy} BTC/block')
    return tip_height, live_hr_eh, btc_spot, current_halving_era, current_subsidy


def fetch_hash_rate_history():
    print('Fetching hash rate history (blockchain.info)...')
    raw = fetch_blockchain_chart('hash-rate')
    raw.columns = ['hashrate_eh']
    raw['hashrate_eh'] = raw['hashrate_eh'] / 1e6
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    historical_hr = raw[raw['hashrate_eh'] > 0].dropna()
    print(f'  {len(historical_hr)} rows | {historical_hr.index[0].date()} → {historical_hr.index[-1].date()}')
    return historical_hr


def build_halving_schedule(tip_height, current_halving_era, current_subsidy, n=3):
    recs = []
    for i in range(1, n + 1):
        era        = current_halving_era + i
        hblock     = era * BLOCKS_PER_HALVING
        secs_away  = (hblock - tip_height) * 600
        est_date   = datetime.utcnow() + timedelta(seconds=secs_away)
        years_away = secs_away / (365.25 * 24 * 3600)
        recs.append({
            'era':       era,
            'est_date':  est_date,
            'years_away': years_away,
            'subsidy':   GENESIS_SUBSIDY / (2 ** era),
            'label':     f"H#{era} ({est_date.strftime('%Y')})",
        })
    return recs


def build_historical_cost(historical_hr):
    hc = historical_hr.loc['2013-01-01':].copy()
    hc['eff_jth']     = hc.index.map(_hist_eff)
    hc['subsidy']     = hc.index.map(_hist_sub)
    hc['hashrate_th'] = hc['hashrate_eh'] * 1e6
    hc['prod_cost']   = (
        hc['hashrate_th'] * hc['eff_jth'] * PUE * 86400 / 3.6e6
        * electricity_cost_kwh / (144 * hc['subsidy'])
    )
    hist_cost_30d = hc['prod_cost'].rolling(30).mean().dropna()
    print(f'Historical cost: {len(hist_cost_30d)} rows | latest ${hist_cost_30d.iloc[-1]:,.0f}')
    return hist_cost_30d


def build_signal_df(btc_px, historical_hr, hist_cost_30d):
    hr_series  = historical_hr['hashrate_eh']
    hr_90d_chg = hr_series.pct_change(90) * 100
    hr_daily   = hr_90d_chg.reindex(btc_px.index, method='ffill')
    cost_daily = hist_cost_30d.reindex(btc_px.index, method='ffill')
    prem_daily = (btc_px / cost_daily - 1) * 100

    sig = pd.DataFrame({
        'btc_price':   btc_px,
        'cost':        cost_daily,
        'premium_pct': prem_daily,
        'hr_90d_pct':  hr_daily,
    }).dropna()
    sig = sig[sig.index >= pd.Timestamp('2016-01-01')]

    _ALL_HALVINGS_DT = sorted(
        [pd.Timestamp(d) for d, _ in _HIST_HALVINGS] +
        [pd.Timestamp(datetime.utcnow() + timedelta(seconds=(
            ((sig.index[-1].year - 2024) // 4 + 5) * BLOCKS_PER_HALVING * 600
        )))]
    )

    def days_to_next_halving_at(date):
        future = [h for h in _ALL_HALVINGS_DT if h > pd.Timestamp(date)]
        return (future[0] - pd.Timestamp(date)).days if future else 9999

    sig['days_to_halving'] = [days_to_next_halving_at(d) for d in sig.index]
    sig['hr_90d_pctrank']  = (
        sig['hr_90d_pct'].rolling(HASH_NORM_WINDOW, min_periods=180).rank(pct=True)
    )

    def score_cost_proximity(p):
        return float(np.interp(p, [-20, 0, 15, 35, 70, 120], [40, 35, 25, 14, 4, 0]))

    def score_hashrate_momentum(r):
        return float(r * 30)

    def score_halving_urgency(d):
        return float(np.interp(min(float(d), 730.0), [0, 30, 90, 180, 365, 730], [30, 28, 22, 15, 7, 0]))

    sig['s_cost']    = sig['premium_pct'].apply(score_cost_proximity)
    sig['s_hash']    = sig['hr_90d_pctrank'].apply(score_hashrate_momentum)
    sig['s_halving'] = sig['days_to_halving'].apply(score_halving_urgency)
    sig['score']     = sig['s_cost'] + sig['s_hash'] + sig['s_halving']

    print(f'Signal df: {len(sig)} rows | {sig.index[0].date()} → {sig.index[-1].date()}')
    return sig


def compute_current_score(btc_spot, current_cost, historical_hr, halving_records):
    hr_series   = historical_hr['hashrate_eh']
    hr_90d_chg  = hr_series.pct_change(90) * 100
    _hr_now     = hr_series.iloc[-1]
    _hr_90d_ago = hr_series.iloc[-91] if len(hr_series) > 91 else hr_series.iloc[0]
    hr_90d_now  = (_hr_now / _hr_90d_ago - 1) * 100
    prem_now    = (btc_spot / current_cost - 1) * 100

    _HALVINGS_DT = [pd.Timestamp(d) for d, _ in _HIST_HALVINGS] + [
        pd.Timestamp(halving_records[0]['est_date'].strftime('%Y-%m-%d'))
    ]
    now_ts    = pd.Timestamp.utcnow().tz_localize(None)
    future_h  = [h for h in _HALVINGS_DT if h > now_ts]
    next_h_days = (future_h[0] - now_ts).days if future_h else 9999

    _recent_chg = hr_90d_chg.dropna().iloc[-HASH_NORM_WINDOW:]
    pctrank_now = float((_recent_chg < hr_90d_now).sum() / len(_recent_chg))

    s_cost    = float(np.interp(prem_now, [-20, 0, 15, 35, 70, 120], [40, 35, 25, 14, 4, 0]))
    s_hash    = float(pctrank_now * 30)
    s_halving = float(np.interp(min(float(next_h_days), 730.0), [0, 30, 90, 180, 365, 730], [30, 28, 22, 15, 7, 0]))
    score_now = s_cost + s_hash + s_halving

    zone = 'FAVOURABLE' if score_now >= 66 else 'NEUTRAL' if score_now >= 33 else 'RISK OFF'
    print(f'BTC Risk Score: {score_now:.0f}/100  [{zone}]')
    return score_now


def _make_recency_segments(index, sig, threshold=SIGNAL_THRESHOLD):
    above_dates = sorted(sig.index[sig['score'] >= threshold])

    def days_since(date):
        idx = bisect.bisect_right(above_dates, date) - 1
        return (date - above_dates[idx]).days if idx >= 0 else 99999

    def bucket(ds):
        for lo, hi, color, label in RECENCY_BANDS:
            if ds < hi:
                return (color, label)
        return (MIST, '>1yr / never')

    bkts = [bucket(days_since(d)) for d in index]
    segs, prev, s0 = [], None, 0
    for i, bkt in enumerate(bkts):
        if bkt != prev:
            if prev is not None:
                segs.append((s0, i, prev))
            s0, prev = i, bkt
    if prev is not None:
        segs.append((s0, len(index), prev))
    return segs, above_dates


# ════════════════════════════════════════════
# CHART 5 — BTC Price vs Production Cost History + Base Projection → 2032
# ════════════════════════════════════════════
def plot_btc_price_vs_production_cost(btc_px, hist_cost_30d, halving_records,
                                       live_hr_eh, historical_hr, current_subsidy):
    sp_base    = diminishing_scenarios['Base']
    halvings_yrs_3 = [h['years_away'] for h in halving_records]
    halving_dates_3  = [h['est_date'] for h in halving_records]
    halving_labels_3 = [h['label']    for h in halving_records]

    proj_start = datetime.utcnow()
    proj_end   = halving_dates_3[1] - timedelta(days=60)
    proj_anchor_hr = (
        historical_hr['hashrate_eh'].rolling(30).mean().iloc[-1]
        if historical_hr is not None else live_hr_eh
    )

    n_proj     = 1500
    proj_yrs   = np.linspace(0, (proj_end - proj_start).days / 365.25, n_proj)
    proj_dates = [proj_start + timedelta(days=y * 365.25) for y in proj_yrs]
    proj_cost  = [
        production_cost_per_btc(
            project_hr_cycle_decay(y, sp_base['base_cagr'], sp_base['decay_factor'], proj_anchor_hr),
            projected_efficiency(y), electricity_cost_kwh,
            subsidy_at_t(y, halvings_yrs_3, current_subsidy)
        ) for y in proj_yrs
    ]
    proj_cost_upper = [c * 1.5 for c in proj_cost]

    fig = go.Figure()

    # BTC price
    fig.add_trace(go.Scatter(
        x=btc_px.index, y=btc_px.values,
        mode='lines', name='BTC Price',
        line=dict(color=GOLD, width=2.0),
        hovertemplate='%{x|%Y-%m-%d}<br>BTC: $%{y:,.0f}<extra></extra>'
    ))

    # Historical cost band
    cost_lo = hist_cost_30d
    cost_hi = hist_cost_30d * 1.5
    fig.add_trace(go.Scatter(
        x=cost_lo.index, y=cost_lo.values,
        mode='lines', name='Production Cost (hist)',
        line=dict(color=GREEN_LIT, width=1.8),
        hovertemplate='%{x|%Y-%m-%d}<br>Cost: $%{y:,.0f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=list(cost_lo.index) + list(cost_lo.index[::-1]),
        y=list(cost_hi.values) + list(cost_lo.values[::-1]),
        fill='toself', fillcolor=hex_to_rgba(GREEN_LIT, 0.12),
        line=dict(width=0), name='Cost band (to 1.5×)', showlegend=True, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=cost_hi.index, y=cost_hi.values,
        mode='lines', showlegend=False,
        line=dict(color=GREEN_LIT, width=0.8, dash='dot'), opacity=0.40,
        hovertemplate='%{x|%Y-%m-%d}<br>1.5× Cost: $%{y:,.0f}<extra></extra>'
    ))

    # Projected cost band
    fig.add_trace(go.Scatter(
        x=proj_dates, y=proj_cost,
        mode='lines',
        name=f'Proj. Cost — Base (C0={sp_base["base_cagr"]*100:.0f}%)',
        line=dict(color=GREEN_LIT, width=2.0, dash='dash'),
        hovertemplate='%{x|%Y-%m-%d}<br>Proj. cost: $%{y:,.0f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=proj_dates + proj_dates[::-1],
        y=proj_cost_upper + proj_cost[::-1],
        fill='toself', fillcolor=hex_to_rgba(GREEN_LIT, 0.08),
        line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=proj_dates, y=proj_cost_upper,
        mode='lines', showlegend=False,
        line=dict(color=GREEN_LIT, width=0.7, dash='dot'), opacity=0.35,
        hovertemplate='%{x|%Y-%m-%d}<br>1.5× Proj.: $%{y:,.0f}<extra></extra>'
    ))

    # Halving vlines — past
    _HV_COLOR = hex_to_rgba(PURPLE, 0.55)
    for date_str, sub_lbl in _HIST_HALVINGS:
        fig.add_vline(
            x=pd.Timestamp(date_str).timestamp() * 1000,
            line_color=_HV_COLOR, line_width=1.4, line_dash='dash',
            annotation_text=str(sub_lbl), annotation_position='top left',
            annotation_font=dict(family='Courier New, monospace', size=8, color=PURPLE)
        )
    # 2028 halving only (chart ends before 2032)
    fig.add_vline(
        x=halving_dates_3[0].timestamp() * 1000,
        line_color=_HV_COLOR, line_width=1.4, line_dash='dash',
        annotation_text=halving_labels_3[0], annotation_position='top left',
        annotation_font=dict(family='Courier New, monospace', size=8, color=PURPLE)
    )

    alphaline_layout(fig,
        f'BTC Price vs Miner Production Cost  |  History + Base Scenario → 2032',
        height=CHART_HEIGHT)
    fig.update_layout(
        yaxis_type='log', yaxis_title='$/BTC (log)',
        xaxis_title='Date',
        xaxis=dict(range=[btc_px.index[0].isoformat(), proj_end.isoformat()])
    )
    return fig


# ════════════════════════════════════════════
# CHART 7 — Risk Score History + Signal Recency Background
# ════════════════════════════════════════════
def plot_risk_score_history(sig, score_now):
    today_dt    = pd.Timestamp.utcnow().tz_localize(None)
    segs, above_dates = _make_recency_segments(sig.index, sig)

    def days_since(date):
        idx = bisect.bisect_right(above_dates, date) - 1
        return (date - above_dates[idx]).days if idx >= 0 else 99999

    def bucket(ds):
        for lo, hi, color, label in RECENCY_BANDS:
            if ds < hi:
                return (color, label)
        return (MIST, '>1yr / never')

    bkts = [(color, label) for color, label in [bucket(days_since(d)) for d in sig.index]]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.42, 0.58], vertical_spacing=0.04
    )

    # Recency vrects
    _segments = []
    _prev = None; _seg_start = None
    for _dt, _bkt in zip(sig.index, bkts):
        if _bkt != _prev:
            if _prev is not None:
                _segments.append((_seg_start, _dt, _prev))
            _seg_start = _dt; _prev = _bkt
    if _prev is not None:
        _segments.append((_seg_start, sig.index[-1] + pd.Timedelta(days=1), _prev))

    for _x0, _x1, (_col, _lbl) in _segments:
        fig.add_vrect(x0=_x0, x1=_x1,
                      fillcolor=hex_to_rgba(_col, 0.20), line_width=0,
                      row=1, col=1)

    # Score threshold lines
    for threshold in [33, 66]:
        fig.add_hline(y=threshold, line_color=hex_to_rgba(MIST, 0.25),
                      line_width=0.5, line_dash='dot', row=1, col=1)

    # Stacked component fills
    fig.add_trace(go.Scatter(
        x=sig.index, y=sig['s_cost'], name='Cost Proximity (0–40)',
        fill='tozeroy', fillcolor=hex_to_rgba(GREEN_LIT, 0.30),
        line=dict(color='rgba(0,0,0,0)', width=0), stackgroup='score',
        hovertemplate='%{x|%Y-%m-%d}<br>Cost Proximity: %{y:.0f}<extra></extra>'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=sig.index, y=sig['s_hash'], name='Hashrate Momentum (0–30)',
        fill='tonexty', fillcolor=hex_to_rgba(GOLD, 0.30),
        line=dict(color='rgba(0,0,0,0)', width=0), stackgroup='score',
        hovertemplate='%{x|%Y-%m-%d}<br>Hashrate Momentum: %{y:.0f}<extra></extra>'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=sig.index, y=sig['s_halving'], name='Halving Urgency (0–30)',
        fill='tonexty', fillcolor=hex_to_rgba(PURPLE, 0.30),
        line=dict(color='rgba(0,0,0,0)', width=0), stackgroup='score',
        hovertemplate='%{x|%Y-%m-%d}<br>Halving Urgency: %{y:.0f}<extra></extra>'
    ), row=1, col=1)

    # Smoothed composite line
    score_smooth = sig['score'].rolling(14, center=True, min_periods=7).mean()
    fig.add_trace(go.Scatter(
        x=score_smooth.index, y=score_smooth, name='Composite Score (14d avg)',
        mode='lines', line=dict(color=hex_to_rgba(WHITE, 0.50), width=1.2),
        hovertemplate='%{x|%Y-%m-%d}<br>Score: %{y:.0f}/100<extra></extra>'
    ), row=1, col=1)

    # Today annotation
    fig.add_vline(x=today_dt.timestamp() * 1000,
                  line_color=hex_to_rgba(WHITE, 0.25), line_width=0.8, line_dash='dot',
                  row=1, col=1)
    fig.add_annotation(
        x=today_dt, y=score_now + 5, xref='x', yref='y',
        text=f'  {score_now:.0f}', showarrow=False,
        font=dict(family='Courier New, monospace', size=9, color=GOLD_LIT), xanchor='left'
    )

    # Recency legend dummy traces
    _days_now  = days_since(today_dt)
    _bkt_now   = bucket(_days_now)[1]
    for lo, hi, color, label in RECENCY_BANDS:
        marker = ' ◄ now' if label == _bkt_now else ''
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(symbol='square', size=10, color=color),
            name=f'█ {label}{marker}',
            showlegend=True, legendgroup='recency'
        ), row=1, col=1)

    # Halving vlines both panels
    signal_start = sig.index[0]
    for date_str, _ in _HIST_HALVINGS:
        if pd.Timestamp(date_str) >= signal_start:
            for r in [1, 2]:
                fig.add_vline(
                    x=pd.Timestamp(date_str).timestamp() * 1000,
                    line_color=hex_to_rgba(PURPLE, 0.40), line_width=0.7, line_dash='dash',
                    row=r, col=1
                )

    # Panel 2: BTC price
    fig.add_trace(go.Scatter(
        x=sig.index, y=sig['btc_price'], name='BTC Price',
        mode='lines', line=dict(color=GOLD, width=1.5),
        hovertemplate='%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>'
    ), row=2, col=1)

    # Production cost band
    _cost_lo = sig['cost'].dropna()
    _cost_hi = _cost_lo * 1.5
    fig.add_trace(go.Scatter(
        x=_cost_lo.index, y=_cost_lo.values,
        mode='lines', name='Production Cost Floor',
        line=dict(color=MIST, width=1.6),
        hovertemplate='%{x|%Y-%m-%d}<br>Cost: $%{y:,.0f}<extra></extra>'
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=list(_cost_lo.index) + list(_cost_lo.index[::-1]),
        y=list(_cost_hi.values) + list(_cost_lo.values[::-1]),
        fill='toself', fillcolor=hex_to_rgba(MIST, 0.10),
        line=dict(width=0), name='Cost band (1.5×)',
        showlegend=True, hoverinfo='skip'
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=_cost_hi.index, y=_cost_hi.values,
        mode='lines', showlegend=False,
        line=dict(color=MIST, width=0.7, dash='dot'), opacity=0.35,
        hovertemplate='%{x|%Y-%m-%d}<br>1.5× Cost: $%{y:,.0f}<extra></extra>'
    ), row=2, col=1)

    fig.add_vline(x=today_dt.timestamp() * 1000,
                  line_color=hex_to_rgba(WHITE, 0.25), line_width=0.8, line_dash='dot',
                  row=2, col=1)

    # Zone labels
    for y_mid, label, color in [(16, 'RISK OFF', RED_LIT), (49, 'NEUTRAL', GOLD), (83, 'FAVOURABLE', GREEN_LIT)]:
        fig.add_annotation(
            xref='paper', x=0.995, yref='y', y=y_mid, text=label, showarrow=False,
            font=dict(family='Courier New, monospace', size=8, color=color),
            xanchor='right', opacity=0.55
        )

    fig.update_yaxes(title_text='Risk Score  (0–100)', range=[0, 100],
                     title_font=dict(size=11), zeroline=False, row=1, col=1)
    fig.update_yaxes(title_text='$/BTC  (log)', type='log',
                     title_font=dict(size=11), zeroline=False, autorange=True, row=2, col=1)
    _x_future = today_dt + pd.Timedelta(days=120)
    fig.update_xaxes(showticklabels=False, range=[sig.index[0], _x_future], row=1, col=1)
    fig.update_xaxes(title_text='Date', title_font=dict(size=11),
                     range=[sig.index[0], _x_future], row=2, col=1)

    alphaline_layout(fig,
        'BTC Risk Score History  |  Background = Days Since Signal (score ≥ 40)',
        height=CHART_HEIGHT)
    fig.update_layout(margin=dict(b=160), legend=dict(font=dict(size=8)))
    return fig


# ════════════════════════════════════════════
# CHART 7b — Risk Score Recency (area colored by signal recency)
# ════════════════════════════════════════════
def plot_risk_score_recency(sig):
    segs, _ = _make_recency_segments(sig.index, sig)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.42, 0.58], vertical_spacing=0.04
    )

    # Panel 1: filled score area, one segment per color
    _shown = set()
    for _s, _e, (_col, _lbl) in segs:
        _e_ext = min(_e + 1, len(sig))
        _sl = sig.iloc[_s:_e_ext]
        _show = _lbl not in _shown
        if _show:
            _shown.add(_lbl)
        fig.add_trace(go.Scatter(
            x=_sl.index, y=_sl['score'],
            fill='tozeroy', fillcolor=hex_to_rgba(_col, 0.55),
            line=dict(color=hex_to_rgba(_col, 0.70), width=0.8),
            name=_lbl, legendgroup=_lbl, showlegend=_show,
            hovertemplate='%{x|%Y-%m-%d}<br>Score: %{y:.0f}  |  ' + _lbl + '<extra></extra>'
        ), row=1, col=1)

    score_smooth = sig['score'].rolling(14, center=True, min_periods=7).mean()
    fig.add_trace(go.Scatter(
        x=score_smooth.index, y=score_smooth, name='Score (14d avg)',
        mode='lines', line=dict(color=hex_to_rgba(WHITE, 0.50), width=1.2), showlegend=True,
        hovertemplate='%{x|%Y-%m-%d}<br>Score: %{y:.0f}/100<extra></extra>'
    ), row=1, col=1)

    fig.add_hline(y=SIGNAL_THRESHOLD, line_color=hex_to_rgba(WHITE, 0.35),
                  line_width=0.8, line_dash='dot', row=1, col=1)
    fig.add_annotation(
        xref='paper', x=0.995, yref='y', y=SIGNAL_THRESHOLD + 2,
        text=f'Signal threshold ({SIGNAL_THRESHOLD})', showarrow=False,
        font=dict(family='Courier New, monospace', size=8, color=hex_to_rgba(WHITE, 0.50)),
        xanchor='right'
    )

    # Halving vlines
    signal_start = sig.index[0]
    for date_str, _ in _HIST_HALVINGS:
        if pd.Timestamp(date_str) >= signal_start:
            for r in [1, 2]:
                fig.add_vline(
                    x=pd.Timestamp(date_str).timestamp() * 1000,
                    line_color=hex_to_rgba(PURPLE, 0.40), line_width=0.7, line_dash='dash',
                    row=r, col=1
                )

    # Panel 2: BTC price colored by recency
    for _s, _e, (_col, _lbl) in segs:
        _e_ext = min(_e + 1, len(sig))
        _sl = sig.iloc[_s:_e_ext]
        fig.add_trace(go.Scatter(
            x=_sl.index, y=_sl['btc_price'],
            mode='lines', line=dict(color=_col, width=1.5),
            name=_lbl, legendgroup=_lbl, showlegend=False,
            hovertemplate='%{x|%Y-%m-%d}<br>$%{y:,.0f}  |  ' + _lbl + '<extra></extra>'
        ), row=2, col=1)

    # Production cost band
    _c_lo = sig['cost'].dropna()
    _c_hi = _c_lo * 1.5
    fig.add_trace(go.Scatter(
        x=_c_lo.index, y=_c_lo.values,
        mode='lines', name='Production Cost Floor',
        line=dict(color=MIST, width=1.6),
        hovertemplate='%{x|%Y-%m-%d}<br>Cost: $%{y:,.0f}<extra></extra>'
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=list(_c_lo.index) + list(_c_lo.index[::-1]),
        y=list(_c_hi.values) + list(_c_lo.values[::-1]),
        fill='toself', fillcolor=hex_to_rgba(MIST, 0.10),
        line=dict(width=0), name='Cost band (1.5×)',
        showlegend=True, hoverinfo='skip'
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=_c_hi.index, y=_c_hi.values,
        mode='lines', showlegend=False,
        line=dict(color=MIST, width=0.7, dash='dot'), opacity=0.35,
        hovertemplate='%{x|%Y-%m-%d}<br>1.5× Cost: $%{y:,.0f}<extra></extra>'
    ), row=2, col=1)

    fig.update_yaxes(title_text='Risk Score  (0–100)', range=[0, 100],
                     title_font=dict(size=11), zeroline=False, row=1, col=1)
    fig.update_yaxes(title_text='$/BTC  (log)', type='log',
                     title_font=dict(size=11), zeroline=False, row=2, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(title_text='Date', title_font=dict(size=11), row=2, col=1)

    alphaline_layout(fig,
        'BTC Risk Score — Area & Price Color = Days Since Signal Fired (score ≥ 40)  |  2016+',
        height=CHART_HEIGHT)
    fig.update_layout(margin=dict(b=160), legend=dict(font=dict(size=8)))
    return fig


# ════════════════════════════════════════════
# CHART 7c — MSTR Price vs BTC Signal Recency
# ════════════════════════════════════════════
def plot_mstr_signal_recency(sig, mstr_px):
    _MSTR_START = mstr_px.index[0]
    sig_m = sig[sig.index >= _MSTR_START]

    segs_sig,  _ = _make_recency_segments(sig_m.index, sig)
    segs_mstr, _ = _make_recency_segments(mstr_px.index, sig)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.35, 0.65], vertical_spacing=0.04
    )

    # Panel 1: BTC score area (recency-colored)
    _shown = set()
    for _s, _e, (_col, _lbl) in segs_sig:
        _e_ext = min(_e + 1, len(sig_m))
        _sl = sig_m.iloc[_s:_e_ext]
        _show = _lbl not in _shown
        if _show:
            _shown.add(_lbl)
        fig.add_trace(go.Scatter(
            x=_sl.index, y=_sl['score'],
            fill='tozeroy', fillcolor=hex_to_rgba(_col, 0.55),
            line=dict(color=hex_to_rgba(_col, 0.70), width=0.8),
            name=_lbl, legendgroup=_lbl, showlegend=_show,
            hovertemplate='%{x|%Y-%m-%d}<br>BTC Score: %{y:.0f}  |  ' + _lbl + '<extra></extra>'
        ), row=1, col=1)

    score_smooth_m = sig_m['score'].rolling(14, center=True, min_periods=7).mean()
    fig.add_trace(go.Scatter(
        x=score_smooth_m.index, y=score_smooth_m, name='BTC Score (14d avg)',
        mode='lines', line=dict(color=WHITE, width=1.5), showlegend=True,
        hovertemplate='%{x|%Y-%m-%d}<br>Score: %{y:.0f}/100<extra></extra>'
    ), row=1, col=1)
    fig.add_hline(y=40, line_color=hex_to_rgba(WHITE, 0.35),
                  line_width=0.8, line_dash='dot', row=1, col=1)

    # Panel 2: MSTR price colored by BTC signal recency
    for _s, _e, (_col, _lbl) in segs_mstr:
        _e_ext = min(_e + 1, len(mstr_px))
        _sl = mstr_px.iloc[_s:_e_ext]
        fig.add_trace(go.Scatter(
            x=_sl.index, y=_sl.values,
            mode='lines', line=dict(color=_col, width=1.5),
            name=_lbl, legendgroup=_lbl, showlegend=False,
            hovertemplate='%{x|%Y-%m-%d}<br>MSTR $%{y:,.2f}  |  ' + _lbl + '<extra></extra>'
        ), row=2, col=1)

    # Halving vlines
    for date_str, _ in _HIST_HALVINGS:
        if pd.Timestamp(date_str) >= _MSTR_START:
            for r in [1, 2]:
                fig.add_vline(
                    x=pd.Timestamp(date_str).timestamp() * 1000,
                    line_color=hex_to_rgba(PURPLE, 0.40), line_width=0.7, line_dash='dash',
                    row=r, col=1
                )

    fig.update_yaxes(title_text='BTC Score  (0–100)', range=[0, 100],
                     title_font=dict(size=11), zeroline=False, row=1, col=1)
    fig.update_yaxes(title_text='MSTR $/share  (log)', type='log',
                     title_font=dict(size=11), zeroline=False, row=2, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(title_text='Date', title_font=dict(size=11), row=2, col=1)

    alphaline_layout(fig,
        'MSTR Price vs BTC Signal Recency  |  BTC score ≥ 40 buy windows mapped to MSTR',
        height=CHART_HEIGHT)
    return fig


# ════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════
if __name__ == '__main__':
    os.makedirs('docs', exist_ok=True)
    print('=== Building BTC Production Cost + Risk Score Charts ===\n')

    # Fetch all data
    tip_height, live_hr_eh, btc_spot, current_halving_era, current_subsidy = fetch_current_state()
    historical_hr  = fetch_hash_rate_history()
    btc_px         = fetch_btc_price()
    mstr_px        = fetch_mstr_price()
    halving_records = build_halving_schedule(tip_height, current_halving_era, current_subsidy, n=3)

    # Derived data
    current_cost  = production_cost_per_btc(live_hr_eh, fleet_efficiency_jth,
                                             electricity_cost_kwh, current_subsidy)
    hist_cost_30d = build_historical_cost(historical_hr)
    sig           = build_signal_df(btc_px, historical_hr, hist_cost_30d)
    score_now     = compute_current_score(btc_spot, current_cost, historical_hr, halving_records)

    # Chart 5
    print('\n--- Chart 5: BTC Price vs Production Cost ---')
    fig5 = plot_btc_price_vs_production_cost(
        btc_px, hist_cost_30d, halving_records, live_hr_eh, historical_hr, current_subsidy
    )
    fig5.write_html(OUTPUT_COST, include_plotlyjs='cdn', config={'responsive': True})
    print(f'Saved: {OUTPUT_COST}')

    # Chart 7
    print('\n--- Chart 7: Risk Score History ---')
    fig7 = plot_risk_score_history(sig, score_now)
    fig7.write_html(OUTPUT_HISTORY, include_plotlyjs='cdn', config={'responsive': True})
    print(f'Saved: {OUTPUT_HISTORY}')

    # Chart 7b
    print('\n--- Chart 7b: Risk Score Recency ---')
    fig7b = plot_risk_score_recency(sig)
    fig7b.write_html(OUTPUT_RECENCY, include_plotlyjs='cdn', config={'responsive': True})
    print(f'Saved: {OUTPUT_RECENCY}')

    # Chart 7c
    print('\n--- Chart 7c: MSTR Signal Recency ---')
    fig7c = plot_mstr_signal_recency(sig, mstr_px)
    fig7c.write_html(OUTPUT_MSTR, include_plotlyjs='cdn', config={'responsive': True})
    print(f'Saved: {OUTPUT_MSTR}')

    print('\n=== All 4 charts complete ===')
