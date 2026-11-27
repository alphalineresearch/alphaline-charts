"""
generate_eth_etf_flows.py
Alphaline Research — ETH Spot ETF Flows: are investors moving into or out of ether?

Same chart as generate_btc_etf_flows.py (it reuses that script's build and chart code):
  1. ETH weekly candles (log) with swing high / low labels, and the ETF average cost
     basis: inflows add ETH at that day's price, outflows remove it at the running
     average cost (flows since launch).
  2. 13-week rolling net flows into US spot ether ETFs, $B, stacked ETHA (BlackRock)
     vs. all other ETFs, with the total as a white line.
Price-panel shading: green while 13-week net flow is positive, red when negative;
deeper when the flow is both large and broad.

Farside reports flows in US$ millions per fund; each day's flow is converted to ETH at
that day's ETH-USD close. Seed capital and Grayscale's pre-conversion ETHE / ETH holdings
are not flows and are left out.

Data sources (no API keys required):
  - Farside Investors: daily US spot ether ETF flows by fund (farside.co.uk)
  - Yahoo Finance: ETH-USD daily OHLC (via yfinance)

Writes:
  docs/eth_etf_flows.html

Usage:
    python scripts/generate_eth_etf_flows.py
"""

import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alphaline_theme import write_chart                                    # noqa: E402
from generate_btc_etf_flows import (_get, fetch_price, build_from_flows,    # noqa: E402
                                    plot_etf_flows)

# ════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════
OUTPUT_PATH = os.path.join('docs', 'eth_etf_flows.html')
FLOWS_URL   = 'https://farside.co.uk/ethereum-etf-flow-all-data/'
PRICE_START = '2024-06-01'
LEAD_FUND   = 'ETHA'                 # BlackRock

ETH_CFG = dict(
    asset='ETH', unit='ETH', lead='ETHA (BlackRock)',
    cb_min=250_000,                  # cost basis drawn once the ETFs hold this much net ETH
    cb_method='avg',                 # average cost: the net-dollar version is meaningless after
                                     # ETH's 65% fall (redemptions at low prices inflate it to ~$4.7k)
    swing=0.35,                      # ETH swings more: label 35% moves
    full_in=6.0,                     # $B of 13-wk inflows for full-strength green
    full_out=2.5,                    # $B of 13-wk outflows for full-strength red
    ticks=[1_000, 1_500, 2_000, 3_000, 4_000, 5_000, 7_000, 10_000],
)


# ════════════════════════════════════════════
# DATA
# ════════════════════════════════════════════
def _num(cell):
    """'(484.1)' → -484.1, '1,022.5' → 1022.5, '-' or '' → 0 (fund not trading yet)."""
    c = cell.replace(',', '').replace('*', '').strip()
    if c in ('', '-'):
        return 0.0
    neg = c.startswith('(') and c.endswith(')')
    v = float(c.strip('()'))
    return -v if neg else v


def fetch_flows_usd():
    """Daily net flows, US$ millions: total and the lead fund (ETHA)."""
    print('  Fetching ETH ETF flows (Farside)...')
    page = _get(FLOWS_URL)
    i = page.find('<table')
    table = page[i:page.find('</table>', i)]
    cells = lambda row: [re.sub(r'<[^>]+>', '', c).strip()
                         for c in re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', row, re.S)]
    rows = [cells(r) for r in re.findall(r'<tr[^>]*>(.*?)</tr>', table, re.S)]
    header = rows[0]
    if LEAD_FUND not in header or header[-1] != 'Total':
        raise RuntimeError(f'Farside table layout changed: {header}')
    lead_i = header.index(LEAD_FUND)

    out = []
    for r in rows[1:]:
        if len(r) != len(header):
            continue
        try:
            date = pd.to_datetime(r[0], format='%d %b %Y')
        except (ValueError, TypeError):
            continue                                   # Fee / Seed / Total / summary rows
        if all(c in ('', '-') for c in r[1:-1]):
            continue                                   # today's row before data is in
        out.append((date, _num(r[-1]), _num(r[lead_i])))
    df = pd.DataFrame(out, columns=['date', 'total', 'lead']).set_index('date').sort_index()
    df = df[~df.index.duplicated(keep='last')]
    if len(df) < 250:
        raise RuntimeError(f'only {len(df)} days of ETH ETF flows parsed — page layout changed?')
    df['other'] = df['total'] - df['lead']
    print(f'    {len(df)} days, {df.index.min():%Y-%m-%d} → {df.index.max():%Y-%m-%d}, '
          f'net ${df["total"].sum() / 1e3:,.1f}B')
    return df


def build_dataframe():
    usd_m = fetch_flows_usd()
    px = fetch_price('ETH-USD', PRICE_START)
    close = px['close'].reindex(usd_m.index, method='ffill')
    flows_eth = usd_m.mul(1e6).div(close, axis=0)              # coins
    return build_from_flows(flows_eth, px, ETH_CFG, usd=usd_m / 1e3)


# ════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════
if __name__ == '__main__':
    os.makedirs('docs', exist_ok=True)
    print('=== Building ETH ETF Flows ===')
    data = build_dataframe()
    fig, subtitle = plot_etf_flows(data, ETH_CFG)
    write_chart(fig, OUTPUT_PATH,
                title='ETH Spot ETF Flows',
                subtitle=subtitle,
                note='Bars: 13-week rolling net creations − redemptions in US spot ether ETFs (since-launch total until Oct 2024). '
                     'Shading: green while 13-week net flow is positive, red when negative; '
                     'deeper when the flow is both large and broad (most days in the same direction). '
                     'Flow breadth: share of the past 20 trading days with net inflows. '
                     'Gold line: ETF average cost basis. Inflows add ETH at that day\'s price, outflows remove it '
                     'at the running average cost (flows since launch).',
                source='Farside Investors · Yahoo Finance',
                default='All', min_plot_height=520)
    print(f'Saved: {OUTPUT_PATH}')
