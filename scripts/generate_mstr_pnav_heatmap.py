"""
generate_mstr_pnav_heatmap.py
Alphaline Research — MSTR mNAV Z-Score Heatmap + BTC Production Cost

mNAV (net) = MSTR market cap ÷ (BTC held × BTC price + USD cash − debt − preferred)

  Strategy has ~$20B of convertible debt and preferred stock that rank ahead of
  common shareholders, so comparing market cap to the gross BTC value overstates
  what common equity owns. This chart uses the net figure.

Data sources (no API keys required):
  - Yahoo Finance: MSTR price, BTC price (via yfinance)
  - blockchain.com: mining difficulty → BTC production cost
  - Strategy (strategy.com ledger) + SEC filings: BTC holdings, basic shares
    (Class A + Class B), debt, preferred and USD reserve — hardcoded tables,
    update weekly from Strategy's Monday 8-K.

Writes docs/mstr_pnav_heatmap.html.

Usage:
    python generate_mstr_pnav_heatmap.py
"""

import os
import warnings
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import yfinance as yf

warnings.filterwarnings('ignore')

# ════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════
OUTPUT_PATH   = os.path.join('docs', 'mstr_pnav_heatmap.html')
START         = '2020-08-01'
ZSCORE_START  = '2022-01-01'
SPLIT_DATE    = pd.Timestamp('2024-08-01')
SPLIT_FACTOR  = 10
BULL_THRESH   = 2.0
BEAR_THRESH   = 1.2
SMOOTH_SPAN   = 14
ZSCORE_CAP    = 6.0    # cap outliers when calibrating mean / std
COST_BAND_MULT = 1.5

EFFICIENCY_SCHEDULE = {
    '2013-01-01': 1000, '2014-01-01': 500,  '2016-01-01': 200,
    '2018-01-01': 100,  '2019-01-01': 75,   '2020-01-01': 60,
    '2021-01-01': 40,   '2022-06-01': 32,   '2023-01-01': 25,
    '2024-01-01': 22,
}
ELECTRICITY_COST = 0.05   # $/kWh
PUE              = 1.10
HALVING_SCHEDULE = [
    ('2009-01-03', 50.0), ('2012-11-28', 25.0),
    ('2016-07-09', 12.5), ('2020-05-11', 6.25),
    ('2024-04-20', 3.125),
]

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
CHART_WIDTH  = 1100
CHART_HEIGHT = 870

def hex_to_rgba(h, a=0.35):
    h = h.lstrip('#')
    return f'rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})'

# ════════════════════════════════════════════
# ALPHALINE LAYOUT
# ════════════════════════════════════════════
def alphaline_layout(fig, title, height=CHART_HEIGHT, subtitle='',
                     source='alphalineresearch.com  |  Yahoo Finance · Strategy (MSTR) · blockchain.com'):
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
        margin=dict(l=40, r=20, t=80, b=115),
        xaxis=dict(gridcolor='rgba(212,168,67,0.06)', gridwidth=0.5, zeroline=False,
                   showspikes=True, spikecolor=MIST, spikethickness=1, spikedash='dot'),
        yaxis=dict(gridcolor='rgba(212,168,67,0.06)', gridwidth=0.5, zeroline=False),
        hoverlabel=dict(bgcolor=NAVY_MID, bordercolor=GOLD,
                        font=dict(family='Courier New, monospace', size=11, color=WHITE)),
        legend=dict(
            bgcolor='rgba(10,22,40,0.0)', bordercolor='rgba(0,0,0,0)', borderwidth=0,
            font=dict(size=9, color=MIST),
            orientation='h', x=0.5, xanchor='center',
            y=-0.09, yanchor='top',
            tracegroupgap=0,
        ),
        annotations=[
            dict(text=f'Source: {source}',
                 xref='paper', yref='paper', x=0.99, y=-0.04,
                 xanchor='right', yanchor='top',
                 font=dict(family='Courier New, monospace', size=8, color=STEEL),
                 showarrow=False),
        ],
    )
    return fig

# ════════════════════════════════════════════
# BTC HOLDINGS LOOKUP TABLE
# Source: Strategy (MSTR) SEC filings / press releases
# Verify & update from: saylortracker.com  or  sec.gov
# ════════════════════════════════════════════
BTC_HOLDINGS_RAW = [
    ('2020-08-11',   21_454),
    ('2020-09-14',   38_250),
    ('2020-12-04',   40_824),
    ('2020-12-21',   70_470),
    ('2021-01-22',   70_784),
    ('2021-02-02',   71_079),
    ('2021-02-24',   90_531),
    ('2021-03-12',   91_326),
    ('2021-05-18',   92_079),
    ('2021-06-21',  105_085),
    ('2021-08-24',  108_992),
    ('2021-09-13',  114_042),
    ('2021-11-01',  121_044),
    ('2021-12-09',  122_478),
    ('2021-12-30',  124_391),
    ('2022-01-31',  125_051),
    ('2022-03-29',  129_218),
    ('2022-06-28',  129_699),
    ('2022-12-27',  132_500),
    ('2023-03-27',  138_955),
    ('2023-06-27',  152_333),
    ('2023-09-24',  158_245),
    ('2023-12-26',  189_150),
    ('2024-03-18',  214_278),
    ('2024-06-20',  226_500),
    ('2024-09-13',  252_220),
    ('2024-11-11',  279_420),
    ('2024-11-18',  331_200),
    ('2024-11-25',  386_700),
    ('2024-12-02',  402_100),
    ('2024-12-09',  423_650),
    ('2024-12-22',  446_400),
    ('2025-01-27',  471_107),
    ('2025-02-24',  499_096),
    ('2025-03-31',  506_137),
    ('2025-04-07',  528_185),
    # ── strategy.com/ledger (weekly 8-Ks) ──
    ('2025-04-14',  531_644), ('2025-04-21',  538_200), ('2025-04-28',  553_555),
    ('2025-05-05',  555_450), ('2025-05-12',  568_840), ('2025-05-19',  576_230),
    ('2025-05-26',  580_250), ('2025-06-02',  580_955), ('2025-06-09',  582_000),
    ('2025-06-16',  592_100), ('2025-06-23',  592_345), ('2025-06-30',  597_325),
    ('2025-07-14',  601_550), ('2025-07-21',  607_770), ('2025-07-29',  628_791),
    ('2025-08-11',  628_946), ('2025-08-18',  629_376), ('2025-08-25',  632_457),
    ('2025-09-02',  636_505), ('2025-09-08',  638_460), ('2025-09-15',  638_985),
    ('2025-09-22',  639_835), ('2025-09-29',  640_031), ('2025-10-13',  640_250),
    ('2025-10-20',  640_418), ('2025-10-27',  640_808), ('2025-11-03',  641_205),
    ('2025-11-10',  641_692), ('2025-11-17',  649_870), ('2025-12-01',  650_000),
    ('2025-12-08',  660_624), ('2025-12-15',  671_268), ('2025-12-29',  672_497),
    ('2025-12-31',  672_500), ('2026-01-05',  673_783), ('2026-01-12',  687_410),
    ('2026-01-20',  709_715), ('2026-01-26',  712_647), ('2026-02-02',  713_502),
    ('2026-02-09',  714_644), ('2026-02-17',  717_131), ('2026-02-23',  717_722),
    ('2026-03-02',  720_737), ('2026-03-09',  738_731), ('2026-03-16',  761_068),
    ('2026-03-23',  762_099), ('2026-04-06',  766_970), ('2026-04-13',  780_897),
    ('2026-04-20',  815_061), ('2026-04-27',  818_334), ('2026-05-11',  818_869),
    ('2026-05-18',  843_738), ('2026-06-01',  843_706), ('2026-06-08',  845_256),
    ('2026-06-15',  846_842), ('2026-06-22',  847_363), ('2026-06-30',  846_000),
    ('2026-07-06',  843_775), ('2026-08-03',  842_138), ('2026-08-10',  840_447),
    ('2026-08-31',  845_050), ('2026-09-21',  846_000), ('2026-09-28',  847_666),
    ('2026-10-05',  848_000), ('2026-10-12',  848_250), ('2026-10-19',  848_517),
]

# ════════════════════════════════════════════
# BASIC SHARES OUTSTANDING (Class A + Class B, split-adjusted)
# Source: strategy.com share history + weekly ledger. Yahoo's share series
# misses the 19.64M Class B shares and lags ATM issuance, so we use
# Strategy's own counts and interpolate between them.
# ════════════════════════════════════════════
SHARES_RAW = [
    ('2020-08-01',  96_600_000),
    ('2020-12-31',  95_870_000), ('2021-12-31', 112_855_000),
    ('2022-12-31', 115_488_000), ('2023-12-31', 168_681_000),
    ('2024-12-31', 245_778_000), ('2025-06-30', 261_318_000),
    ('2025-07-21', 283_553_000), ('2025-08-25', 284_641_000),
    ('2025-09-29', 287_093_000), ('2025-11-03', 287_354_000),
    ('2025-12-08', 300_800_000), ('2025-12-31', 312_062_000),
    ('2026-01-20', 330_174_000), ('2026-02-23', 334_098_000),
    ('2026-03-23', 345_594_000), ('2026-04-20', 348_992_000),
    ('2026-05-18', 351_659_000), ('2026-06-30', 371_603_000),
    ('2026-08-03', 387_614_000), ('2026-08-31', 420_483_000),
    ('2026-09-28', 421_977_908), ('2026-10-19', 422_353_257),
]

# ════════════════════════════════════════════
# CAPITAL STRUCTURE — step changes ($)
#   debt  = convertible / secured notes + loans (principal)
#   pref  = preferred stock liquidation preference (STRK/STRF/STRD/STRC/STRE)
#   cash  = USD reserve + cash (only counted from the USD-reserve era)
# Exact figures from 10-Q/10-K and strategy.com where marked ✓;
# others are approximate quarter-end estimates — VERIFY against filings.
# ════════════════════════════════════════════
CAPITAL_RAW = [
    # date,          debt,           pref,            cash
    ('2020-08-01',            0,               0,               0),
    ('2020-12-11',  650_000_000,               0,               0),   # 2025 converts
    ('2021-02-19', 1_700_000_000,              0,               0),   # + 2027 converts
    ('2021-06-14', 2_200_000_000,              0,               0),   # + secured notes
    ('2022-03-29', 2_405_000_000,              0,               0),   # + Silvergate loan
    ('2023-03-24', 2_200_000_000,              0,               0),   # loan repaid
    ('2024-03-09', 3_000_000_000,              0,               0),   # + 2030A ✓
    ('2024-03-19', 3_603_750_000,              0,               0),   # + 2031 ✓
    ('2024-06-18', 3_903_750_000,              0,               0),   # + 2032 ✓, secured notes redeemed
    ('2024-09-20', 4_263_750_000,              0,               0),   # + 2028 ✓, 2025s retired
    ('2024-11-22', 7_263_750_000,              0,               0),   # + 2029 ($3.0B)
    ('2025-01-31', 7_263_750_000,    730_000_000,               0),   # STRK IPO
    ('2025-02-22', 8_213_750_000,    730_000_000,               0),   # + 2030B ✓, 2027s converted
    ('2025-03-31', 8_213_750_000,  1_900_000_000,               0),   # + STRF (≈)
    ('2025-06-30', 8_213_750_000,  3_400_000_000,               0),   # + STRD (≈)
    ('2025-07-29', 8_213_750_000,  6_200_000_000,               0),   # + STRC IPO (≈)
    ('2025-09-30', 8_213_750_000,  6_900_000_000,               0),   # (≈)
    ('2025-12-01', 8_213_750_000,  7_700_000_000,   1_440_000_000),   # USD reserve created
    ('2025-12-31', 8_213_750_000,  8_032_324_000,   2_301_470_000),   # ✓ 10-K
    ('2026-03-31', 8_213_750_000, 11_700_000_000,   2_210_000_000),   # pref ≈, cash ✓
    ('2026-05-31', 6_713_750_000, 13_500_000_000,   2_300_000_000),   # $1.5B 2029s repurchased ✓
    ('2026-06-30', 6_713_750_000, 15_462_056_000,   2_447_982_000),   # ✓ 10-Q
    ('2026-07-26', 6_713_750_000, 15_200_000_000,   3_750_000_000),   # reserve ✓
    ('2026-10-19', 6_714_000_000, 13_556_668_500,   5_280_000_000),   # ✓ strategy.com
]

def _build_btc_holdings():
    _btc = pd.DataFrame(BTC_HOLDINGS_RAW, columns=['date', 'btc_held'])
    _btc['date'] = pd.to_datetime(_btc['date'])
    _btc = _btc.set_index('date').sort_index()
    _daily_idx = pd.date_range(_btc.index[0], datetime.utcnow().date(), freq='D')
    return _btc.reindex(_daily_idx).ffill().rename_axis('date')

def _build_shares():
    _s = pd.DataFrame(SHARES_RAW, columns=['date', 'shares'])
    _s['date'] = pd.to_datetime(_s['date'])
    _s = _s.set_index('date').sort_index()['shares'].astype(float)
    _idx = pd.date_range(_s.index[0], datetime.utcnow().date(), freq='D')
    # interpolate between disclosures, then hold the latest count flat
    return _s.reindex(_idx).interpolate(method='time').ffill().rename_axis('date')

def _build_capital():
    _c = pd.DataFrame(CAPITAL_RAW, columns=['date', 'debt', 'pref', 'cash'])
    _c['date'] = pd.to_datetime(_c['date'])
    _c = _c.set_index('date').sort_index().astype(float)
    _idx = pd.date_range(_c.index[0], datetime.utcnow().date(), freq='D')
    _c = _c.reindex(_idx)
    # Debt moves in discrete issues/redemptions → step.
    # Preferred (ATM sales, buybacks) and cash move continuously → interpolate.
    _c['debt'] = _c['debt'].ffill()
    _c[['pref', 'cash']] = _c[['pref', 'cash']].interpolate(method='time').ffill()
    return _c.rename_axis('date')

# ════════════════════════════════════════════
# DATA FETCHERS
# ════════════════════════════════════════════
def fetch_price_data():
    """Fetch MSTR and BTC daily closes via yfinance."""
    print('Fetching MSTR price...')
    mstr_raw = yf.download('MSTR', start=START, interval='1d', progress=False)
    if isinstance(mstr_raw.columns, pd.MultiIndex):
        mstr_raw.columns = [c[0].lower() for c in mstr_raw.columns]
    else:
        mstr_raw.columns = [c.lower() for c in mstr_raw.columns]
    mstr_raw.index = pd.to_datetime(mstr_raw.index).tz_localize(None)
    mstr_px = mstr_raw['close'].dropna().rename('mstr_price')
    print(f'  MSTR: {len(mstr_px)} rows | latest ${mstr_px.iloc[-1]:,.2f}')

    print('Fetching BTC price...')
    btc_raw = yf.download('BTC-USD', start=START, interval='1d', progress=False)
    if isinstance(btc_raw.columns, pd.MultiIndex):
        btc_raw.columns = [c[0].lower() for c in btc_raw.columns]
    else:
        btc_raw.columns = [c.lower() for c in btc_raw.columns]
    btc_raw.index = pd.to_datetime(btc_raw.index).tz_localize(None)
    btc_px = btc_raw['close'].dropna().rename('btc_price')
    today = pd.Timestamp.utcnow().normalize().tz_localize(None)
    if len(btc_px) and btc_px.index[-1] >= today:
        btc_px = btc_px.iloc[:-1]
    print(f'  BTC:  {len(btc_px)} rows | latest ${btc_px.iloc[-1]:,.0f}')

    return mstr_px, btc_px


def fetch_production_cost():
    """Fetch BTC mining difficulty from blockchain.com and compute production cost."""
    print('Fetching mining difficulty from blockchain.com...')
    try:
        url = ('https://api.blockchain.info/charts/difficulty'
               '?timespan=all&format=json&sampled=false')
        resp = requests.get(url, timeout=45)
        resp.raise_for_status()
        rows = [{'date': pd.Timestamp(int(pt['x']), unit='s').normalize(),
                 'value': float(pt['y'])} for pt in resp.json().get('values', [])]
        diff_raw = pd.DataFrame(rows).set_index('date').sort_index()
        diff_raw = diff_raw[~diff_raw.index.duplicated(keep='last')]
        diff_raw.index = pd.to_datetime(diff_raw.index).tz_localize(None)
        diff_raw.columns = ['avg_difficulty']
        diff_raw['hashrate_th'] = diff_raw['avg_difficulty'] * (2**32) / 600 / 1e12

        def _block_reward(date):
            r = 50.0
            for h_date, h_r in HALVING_SCHEDULE:
                if date >= pd.Timestamp(h_date):
                    r = h_r
            return r

        def _efficiency(date):
            e = 1000.0
            for ts, val in sorted(EFFICIENCY_SCHEDULE.items()):
                if date >= pd.Timestamp(ts):
                    e = val
            return e

        diff_raw['block_reward']    = diff_raw.index.map(_block_reward)
        diff_raw['btc_mined_daily'] = diff_raw['block_reward'] * 144
        diff_raw['efficiency_jth']  = diff_raw.index.map(_efficiency)
        diff_raw['production_cost'] = (
            diff_raw['hashrate_th'] * diff_raw['efficiency_jth'] *
            PUE * 86400 / 3.6e6 * ELECTRICITY_COST / diff_raw['btc_mined_daily']
        )
        cost_series = diff_raw['production_cost'].rolling(30).mean().dropna()
        cost_series.name = 'cost_30d'
        print(f'  Production cost: {len(cost_series)} rows | latest ${cost_series.iloc[-1]:,.0f}')
        return cost_series, True
    except Exception as e:
        print(f'  Production cost fetch failed: {e} — proceeding without')
        return pd.Series(dtype=float, name='cost_30d'), False


# ════════════════════════════════════════════
# BUILD DATAFRAME
# ════════════════════════════════════════════
def build_dataframe():
    btc_holdings = _build_btc_holdings()
    shares_daily = _build_shares()
    capital      = _build_capital()
    mstr_px, btc_px = fetch_price_data()
    cost_series, has_cost = fetch_production_cost()

    df = pd.concat([mstr_px, btc_px], axis=1).dropna()
    df = df.join(btc_holdings['btc_held'], how='left')
    df = df.join(shares_daily.rename('shares'), how='left')
    df = df.join(capital, how='left')
    for c in ['btc_held', 'shares', 'debt', 'pref', 'cash']:
        df[c] = df[c].ffill()
    df = df.dropna(subset=['btc_held', 'shares'])
    df[['debt', 'pref', 'cash']] = df[['debt', 'pref', 'cash']].fillna(0)

    df['btc_nav']     = df['btc_held'] * df['btc_price']
    df['net_assets']  = df['btc_nav'] + df['cash'] - df['debt'] - df['pref']
    df['mstr_mktcap'] = df['mstr_price'] * df['shares']
    df['pnav_gross']  = df['mstr_mktcap'] / df['btc_nav']
    # Net mNAV is only meaningful when common equity has a real claim on the BTC.
    # In late 2022 debt was close to the whole BTC stack, so the ratio explodes;
    # leave those days blank rather than plot a meaningless spike.
    MIN_NET_SHARE = 0.25
    df['pnav'] = (df['mstr_mktcap'] / df['net_assets']).where(df['net_assets'] > MIN_NET_SHARE * df['btc_nav'])
    df = df.dropna(subset=['pnav'])
    df['pnav_smooth'] = df['pnav'].ewm(span=SMOOTH_SPAN, adjust=False).mean()

    # Z-score (calibrated to post-2022 era only)
    cal       = df.loc[ZSCORE_START:, 'pnav_smooth'].clip(upper=ZSCORE_CAP)
    pnav_mu   = cal.mean()
    pnav_std  = cal.std()
    df['pnav_zscore']      = (df['pnav_smooth'] - pnav_mu) / pnav_std
    df['pnav_zscore_clip'] = df['pnav_zscore'].clip(-3, 3)

    print(f'\nDataFrame: {len(df)} rows | {df.index[0].date()} → {df.index[-1].date()}')
    print(f'Z-score: mu={pnav_mu:.2f}x  sigma={pnav_std:.2f}x  '
          f'current={df["pnav_zscore"].iloc[-1]:+.2f}s ({df["pnav_smooth"].iloc[-1]:.2f}x)')
    last = df.iloc[-1]
    print(f'Latest {df.index[-1].date()}: mcap ${last.mstr_mktcap/1e9:,.1f}B | BTC ${last.btc_nav/1e9:,.1f}B | '
          f'net assets ${last.net_assets/1e9:,.1f}B | mNAV net {last.pnav:.2f}x (gross {last.pnav_gross:.2f}x)')

    return df, cost_series, has_cost, pnav_mu, pnav_std


# ════════════════════════════════════════════
# REGIME HELPERS
# ════════════════════════════════════════════
def build_episodes(df, mask, gap_days=5):
    """Merge consecutive True days into (start, end) episodes."""
    dates = df.index[mask].tolist()
    if not dates:
        return []
    eps, ep_s, ep_e = [], dates[0], dates[0]
    for d in dates[1:]:
        if (d - ep_e).days <= gap_days + 1:
            ep_e = d
        else:
            eps.append((ep_s, ep_e))
            ep_s = ep_e = d
    eps.append((ep_s, ep_e))
    return eps


# ════════════════════════════════════════════
# CHART FUNCTION
# ════════════════════════════════════════════
def plot_heatmap_chart(df, cost_series, has_cost, pnav_mu, pnav_std):
    is_bull    = df['pnav_smooth'] >= BULL_THRESH
    is_bear    = df['pnav_smooth'] <= BEAR_THRESH
    bull_eps   = build_episodes(df, is_bull)
    bear_eps   = build_episodes(df, is_bear)
    regime_now = 'BULL' if is_bull.iloc[-1] else ('BEAR' if is_bear.iloc[-1] else 'NEUTRAL')
    pnav_now   = df['pnav_smooth'].iloc[-1]
    zscore_now = df['pnav_zscore'].iloc[-1]

    # Diverging colorscale: RED (bear) -> STEEL (neutral) -> GREEN (bull)
    CSCALE = [
        [0.00, RED_LIT],
        [0.30, '#7A2A2A'],
        [0.48, STEEL],
        [0.52, STEEL],
        [0.70, '#1A6A45'],
        [1.00, GREEN_LIT],
    ]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.60, 0.40],
        vertical_spacing=0.03
    )

    # ── Bull/bear vrects on BTC panel only ──────────────────
    for ep_s, ep_e in bull_eps:
        fig.add_vrect(
            x0=ep_s - pd.Timedelta(days=0.5),
            x1=ep_e + pd.Timedelta(days=0.5),
            fillcolor=hex_to_rgba(GREEN_LIT, 0.12),
            line_width=0.5, line_color=hex_to_rgba(GREEN_LIT, 0.25),
            layer='below', row=1, col=1
        )
    for ep_s, ep_e in bear_eps:
        fig.add_vrect(
            x0=ep_s - pd.Timedelta(days=0.5),
            x1=ep_e + pd.Timedelta(days=0.5),
            fillcolor=hex_to_rgba(RED_LIT, 0.10),
            line_width=0.5, line_color=hex_to_rgba(RED_LIT, 0.20),
            layer='below', row=1, col=1
        )

    # ── Production cost band (panel 1) ──────────────────────
    if has_cost:
        cost_p = cost_series.reindex(df.index, method='ffill').dropna()

        fig.add_trace(go.Scatter(
            x=list(cost_p.index) + list(cost_p.index[::-1]),
            y=list(cost_p * COST_BAND_MULT) + list(cost_p[::-1]),
            fill='toself',
            fillcolor=hex_to_rgba(GREEN_LIT, 0.08),
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip',
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=cost_p.index, y=cost_p * COST_BAND_MULT,
            mode='lines',
            name=f'Cost x{COST_BAND_MULT} (near-cost zone)',
            line=dict(color=hex_to_rgba(GREEN_LIT, 0.5), width=0.8, dash='dot'),
            hovertemplate='%{x|%Y-%m-%d}<br>Cost x' + str(COST_BAND_MULT) +
                          ': $%{y:,.0f}<extra></extra>',
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=cost_p.index, y=cost_p,
            mode='lines',
            name='BTC Production Cost (30d)',
            line=dict(color=GREEN_LIT, width=1.6),
            hovertemplate='%{x|%Y-%m-%d}<br>Cost: $%{y:,.0f}<extra></extra>',
        ), row=1, col=1)

        fig.add_annotation(
            x=cost_p.index[-1], y=cost_p.iloc[-1],
            text=f'  cost ${cost_p.iloc[-1]:,.0f}',
            showarrow=False, xanchor='left',
            font=dict(family='Courier New, monospace', size=9, color=GREEN_LIT),
        )

    # ── Panel 1: BTC price (log) ─────────────────────────────
    fig.add_trace(go.Scatter(
        x=df.index, y=df['btc_price'],
        mode='lines', name='BTC Price',
        line=dict(color=GOLD, width=1.8),
        hovertemplate='%{x|%Y-%m-%d}<br>BTC: $%{y:,.0f}<extra></extra>'
    ), row=1, col=1)

    fig.add_annotation(
        x=df.index[-1], y=df['btc_price'].iloc[-1],
        text=f'  ${df["btc_price"].iloc[-1]:,.0f}',
        showarrow=False, xanchor='left',
        font=dict(family='Courier New, monospace', size=10, color=GOLD)
    )

    # ── Panel 2: P/NAV heatmap bars ─────────────────────────
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['pnav_smooth'],
        name='mNAV (z-score heatmap)',
        customdata=np.stack([df['pnav_gross'], df['net_assets'] / 1e9], axis=-1),
        showlegend=False,
        marker=dict(
            color=df['pnav_zscore_clip'],
            colorscale=CSCALE,
            cmin=-3, cmax=3,
            colorbar=dict(
                title=dict(
                    text='z-score',
                    font=dict(family='Courier New, monospace', size=9, color=MIST),
                    side='right',
                ),
                tickfont=dict(family='Courier New, monospace', size=8, color=MIST),
                tickvals=[-3, -2, -1, 0, 1, 2, 3],
                ticktext=['-3s', '-2s', '-1s', '  0', '+1s', '+2s', '+3s'],
                len=0.38, thickness=11,
                x=1.03, y=0.0, yanchor='bottom',
                bgcolor=NAVY, bordercolor=STEEL, borderwidth=1,
            ),
            line_width=0,
        ),
        hovertemplate=(
            '%{x|%Y-%m-%d}<br>'
            'mNAV (net): %{y:.2f}x<br>'
            'mNAV (gross BTC): %{customdata[0]:.2f}x<br>'
            'Net assets: $%{customdata[1]:,.1f}B<br>'
            'z-score: %{marker.color:+.2f}s'
            '<extra></extra>'
        ),
    ), row=2, col=1)

    # mu and sigma band lines (panel 2)
    pnav_max = df['pnav_smooth'].max()
    band_lines = [
        (pnav_mu + 2*pnav_std, '+2s', GREEN_LIT, 'dot'),
        (pnav_mu + 1*pnav_std, '+1s', GREEN_LIT, 'dot'),
        (pnav_mu,              ' mu', MIST,       'dash'),
        (pnav_mu - 1*pnav_std, '-1s', RED_LIT,   'dot'),
        (1.0,                  '1x',  STEEL,      'dot'),
    ]
    for thresh, label, color, dash in band_lines:
        if 0 < thresh < pnav_max * 1.15:
            fig.add_shape(
                type='line',
                x0=df.index[0], x1=df.index[-1],
                y0=thresh, y1=thresh,
                line=dict(color=color, width=0.7, dash=dash),
                row=2, col=1
            )
            fig.add_annotation(
                x=df.index[-1], y=thresh,
                text=f'  {label}', showarrow=False, xanchor='left',
                font=dict(family='Courier New, monospace', size=8, color=color),
                row=2, col=1
            )

    # Current value annotation (panel 2)
    c_now = GREEN_LIT if zscore_now > 0.5 else (RED_LIT if zscore_now < -0.5 else MIST)
    fig.add_annotation(
        x=df.index[-1], y=pnav_now,
        text=f'  {pnav_now:.2f}x ({zscore_now:+.1f}s)',
        showarrow=False, xanchor='left',
        font=dict(family='Courier New, monospace', size=9, color=c_now),
        row=2, col=1
    )

    # ── Layout ──────────────────────────────────────────────
    cost_note = '' if has_cost else '  (cost unavailable)'
    subtitle = f'{regime_now}  ·  mNAV: {pnav_now:.2f}x  ·  Z: {zscore_now:+.1f}σ  ·  net of debt & preferred'
    alphaline_layout(fig, f'MSTR mNAV — Z-Score Heatmap{cost_note}', subtitle=subtitle, height=CHART_HEIGHT)
    fig.update_layout(
        bargap=0,
        showlegend=True,
        legend=dict(bgcolor='rgba(10,22,40,0.0)', bordercolor='rgba(0,0,0,0)', borderwidth=0,
                    font=dict(size=9, color=MIST),
                    orientation='h', x=0.5, xanchor='center',
                    y=-0.07, yanchor='top', tracegroupgap=0),
    )
    fig.update_yaxes(type='log', tickprefix='$', title_text='BTC Price (log)', row=1, col=1)
    fig.update_yaxes(title_text='mNAV (x, net)', ticksuffix='x', row=2, col=1,
                     range=[0, min(df['pnav_smooth'].max() * 1.08, ZSCORE_CAP + 1)])
    fig.update_xaxes(title_text='', row=2, col=1)

    return fig


# ════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════
if __name__ == '__main__':
    os.makedirs('docs', exist_ok=True)
    print('=== Building MSTR mNAV (net) Z-Score Heatmap + Production Cost ===')
    df, cost_series, has_cost, pnav_mu, pnav_std = build_dataframe()
    fig = plot_heatmap_chart(df, cost_series, has_cost, pnav_mu, pnav_std)
    fig.write_html(OUTPUT_PATH, include_plotlyjs='cdn', config={'responsive': True})
    print(f'Saved: {OUTPUT_PATH}')
