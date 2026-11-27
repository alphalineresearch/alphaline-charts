"""
generate_rsi_top_confluence.py
Alphaline Research — RSI Top Confluence: the top-side twin of generate_rsi_confluence.py.
Flags when a run of Daily, Weekly, Monthly and Annual RSI all overheated together ends,
for BTC, ETH and MSTR.

Overheat rule (one-sided, so readings hotter than the floor still count):
    dist = sqrt( (max(0, 75-D)/8)^2 + (max(0, 75-W)/6)^2 + (max(0, 75-M)/5)^2 + (max(0, 65-A)/3)^2 )
    (MSTR runs hotter: 78 / 78 / 78 / 68)
    overheated ≤ 1.0 · days within 30 of each other form one run · signal = the run's last day

Unlike washouts, "all four hot" holds for weeks through parabolic rallies, so the top tends
to come as the run ends rather than when it starts.

Data sources (no API keys required):
  - CoinMetrics community API: BTC (from 2010) and ETH (from 2015) daily prices
  - Yahoo Finance: MSTR daily closes

Writes three files to docs/:
  docs/btc_rsi_top_confluence.html
  docs/eth_rsi_top_confluence.html
  docs/mstr_rsi_top_confluence.html

Usage:
    python generate_rsi_top_confluence.py
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from alphaline_theme import apply_theme, write_chart, date_long, pct
from generate_rsi_confluence import (
    LEGS, SCALE, EPISODE_GAP, FWD_DAYS, ASSETS, OUT_DIR, DOMAINS_COLLAPSED, DOMAINS_EXPANDED,
    GOLD, TEXT_2, STEEL, NAVY, NAVY_MID,
    hex_to_rgba, build_episodes, fetch_coinmetrics, fetch_yahoo, build_dataframe,
)

# ════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════
SETUP          = {'Daily': 75, 'Weekly': 75, 'Monthly': 75, 'Annual': 65}   # overheat floors
SETUP_BY_ASSET = {'MSTR': {'Daily': 78, 'Weekly': 78, 'Monthly': 78, 'Annual': 68}}
HOT_DIST       = 1.0
RED_LIT        = '#D64444'

# ════════════════════════════════════════════
# SIGNALS
# ════════════════════════════════════════════
def heat_distance(d: pd.DataFrame, setup) -> pd.Series:
    """How far short of the overheat floors, each leg scaled; 0 when all four are at or above."""
    return np.sqrt(sum((np.clip(setup[c] - d[c], 0, None) / SCALE[c]) ** 2 for c in LEGS))

def find_runs(df: pd.DataFrame, setup, since=None):
    """Overheat runs with forward returns from the day each one ended.
    A run whose last hot day is within EPISODE_GAP of today could still restart, so it is
    left out of the history and returned as `current`."""
    d    = df if since is None else df.loc[since:]
    dist = heat_distance(d, setup)
    runs = build_episodes(list(d.index[dist <= HOT_DIST]))
    current = None
    if runs and runs[-1][-1] >= d.index[-1] - pd.Timedelta(days=EPISODE_GAP):
        r = runs.pop()
        current = dict(start=r[0], end=r[-1])

    rows = []
    for r in runs:
        exit_ = r[-1]
        c0    = d.close[exit_]
        t_end = exit_ + pd.Timedelta(days=365)
        row = dict(start=r[0], end=exit_, close=c0, days=len(r),
                   from_high=(c0 / d.close.loc[r[0]:exit_].max() - 1) * 100,
                   low=(df.close.loc[exit_:t_end].min() / c0 - 1) * 100,
                   **{c: d.loc[exit_, c] for c in LEGS})
        for lbl, n in FWD_DAYS:
            t = exit_ + pd.Timedelta(days=n)
            row[lbl] = (df.close.asof(t) / c0 - 1) * 100 if t <= df.index[-1] else np.nan
        rows.append(row)
    return d, dist, pd.DataFrame(rows), current

def _status(d, dist, current):
    if dist.iloc[-1] <= HOT_DIST:
        return 'hot'
    return 'exit' if current else 'none'

STATUS_WORD = {'hot': 'OVERHEATED', 'exit': 'OVERHEAT EXIT', 'none': 'NO SIGNAL'}

# ════════════════════════════════════════════
# CHART — same layout as the bottom charts, in red
# ════════════════════════════════════════════
def _marker(status, size):
    if status == 'exit':
        return dict(color=RED_LIT, size=size, line=dict(color=NAVY, width=2))
    if status == 'hot':
        return dict(color=NAVY, size=size - 1, line=dict(color=RED_LIT, width=2))
    return dict(color=STEEL, size=size, line=dict(color=TEXT_2, width=1.5))

def _hover_text(e, label):
    return '<br>'.join([
        f'<b>Overheat exit · {date_long(e.end)}</b>',
        f'Run {date_long(e.start)} → {date_long(e.end)} ({e.days} hot days) · '
        + ('exit at the run high' if e.from_high > -0.5 else f'exit {pct(e.from_high)} vs run high'),
        f'{label} ${e.close:,.2f}  ·  ' + '  ·  '.join(f'{c} {e[c]:.1f}' for c in LEGS),
        'After: ' + '  ·  '.join(f'{lbl} {pct(e[lbl]) if pd.notna(e[lbl]) else "n/a"}'
                                 for lbl, _ in FWD_DAYS),
        f'Deepest drop in the next year: {pct(e.low)}',
    ])

def plot_top_confluence(d, dist, runs, current, label, price_color, setup):
    fig = make_subplots(rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.0)
    today = d.iloc[-1]
    status = _status(d, dist, current)

    # ── Overheat runs shaded full height ───────────────────────────
    spans = [(e.start, e.end) for _, e in runs.iterrows()]
    if current:
        spans.append((current['start'], current['end']))
    for start, end in spans:
        fig.add_shape(type='rect', xref='x', yref='paper', y0=0, y1=1, layer='below',
                      x0=start - pd.Timedelta(days=1), x1=end + pd.Timedelta(days=1),
                      fillcolor=hex_to_rgba(RED_LIT, 0.14), line_width=0)

    # ── Row 1: price, past exits, today ────────────────────────────
    fig.add_trace(go.Scatter(
        x=d.index, y=d['close'], mode='lines', name=f'{label} price',
        line=dict(color=price_color, width=1.6),
        hovertemplate='%{x|%b %-d, %Y} · ' + label + ' $%{y:,.2f}<extra></extra>'
    ), row=1, col=1)

    if not runs.empty:
        fig.add_trace(go.Scatter(
            x=runs['end'], y=runs['close'], mode='markers', name='Overheat exit (signal)',
            marker=_marker('exit', 10), meta='legend-dot',
            hovertext=[_hover_text(e, label) for _, e in runs.iterrows()],
            hovertemplate='%{hovertext}<extra></extra>'
        ), row=1, col=1)

    now_line = {'hot': 'Overheated · run still going',
                'exit': f'Overheat exit · last hot day {date_long(current["end"]) if current else ""}',
                'none': 'No signal'}[status]
    fig.add_trace(go.Scatter(
        x=[d.index[-1]], y=[today['close']], mode='markers', name='Today',
        marker=_marker(status, 10), showlegend=False,
        hovertemplate=(f'<b>Today · {now_line}</b> · distance {dist.iloc[-1]:.2f}<br>'
                       + label + ' $%{y:,.2f}  ·  '
                       + '  ·  '.join(f'{c} {today[c]:.1f}' for c in LEGS) + '<extra></extra>')
    ), row=1, col=1)

    # ── Row 2: one lit row per RSI when it is within a scale step of its floor ──
    z_leg   = np.array([(np.clip(setup[c] - d[c], 0, None) / SCALE[c]).to_numpy() for c in LEGS])
    in_zone = (z_leg <= 1).astype(float)
    all_four_not_hot = (in_zone.sum(axis=0) == 4) & (dist > HOT_DIST).to_numpy()
    in_zone[:, all_four_not_hot] = 0.6              # each close, combined distance just outside
    fig.add_trace(go.Heatmap(
        x=d.index, y=LEGS[::-1], z=in_zone[::-1],
        customdata=np.array([d[c].to_numpy() for c in LEGS[::-1]]),
        colorscale=[[0, 'rgba(0,0,0,0)'], [0.6, hex_to_rgba(RED_LIT, 0.45)], [1, RED_LIT]],
        zmin=0, zmax=1, showscale=False, ygap=2, name='Overheated',
        hovertemplate='%{x|%b %-d, %Y} · %{y} RSI %{customdata:.1f}<extra></extra>'
    ), row=2, col=1)

    # ── Rows 3-6: RSI detail with the overheat band (hidden until toggled) ──
    rsi_trace_ids = []
    first_rsi_shape = len(fig.layout.shapes)
    for r, c in enumerate(LEGS, start=3):
        fig.add_hrect(y0=setup[c] - SCALE[c], y1=100,
                      fillcolor=hex_to_rgba(GOLD, 0.14), line_width=0, row=r, col=1)
        fig.add_hline(y=50, line_dash='dot', line_color=STEEL, line_width=0.8, row=r, col=1)
        rsi_trace_ids.append(len(fig.data))
        fig.add_trace(go.Scatter(
            x=d.index, y=d[c], mode='lines', name=f'{c} RSI',
            line=dict(color=TEXT_2, width=1.1), showlegend=False,
            hovertemplate='%{x|%b %-d, %Y} · ' + c + ' RSI %{y:.1f}<extra></extra>'
        ), row=r, col=1)
    rsi_shapes = range(first_rsi_shape, len(fig.layout.shapes))

    apply_theme(fig)
    fig.update_yaxes(type='log', tickformat='$,.0f', title_text='Price (log)', row=1, col=1)
    fig.update_yaxes(showgrid=False, fixedrange=True, row=2, col=1)
    titles = {'Daily': 'Daily', 'Weekly': 'Weekly', 'Monthly': 'Monthly', 'Annual': 'Annual (52w)'}
    for r, c in enumerate(LEGS, start=3):
        lo, hi = d[c].min(), d[c].max()
        pad = (hi - lo) * 0.06
        fig.update_yaxes(title_text=titles[c], range=[lo - pad, hi + pad], nticks=4, row=r, col=1)
    fig.update_xaxes(rangeslider_visible=False, type='date')   # bottom axis's traces start hidden

    # ── Toggle: collapsed by default ───────────────────────────────
    def state(expanded):
        vis = [True] * len(fig.data)
        for i in rsi_trace_ids:
            vis[i] = expanded
        doms = DOMAINS_EXPANDED if expanded else DOMAINS_COLLAPSED
        lay = {f'yaxis{"" if i == 0 else i + 1}.domain': dom for i, dom in enumerate(doms)}
        lay.update({f'yaxis{i}.visible': expanded for i in range(3, 7)})
        lay.update({f'shapes[{i}].visible': expanded for i in rsi_shapes})
        return [{'visible': vis}, lay]

    collapsed_traces, collapsed_layout = state(False)
    for i, v in enumerate(collapsed_traces['visible']):
        fig.data[i].visible = v
    fig.update_layout(collapsed_layout)
    fig.update_layout(
        xaxis_range=[d.index[0], d.index[-1] + (d.index[-1] - d.index[0]) * 0.03],
        updatemenus=[dict(
            type='buttons', direction='right', x=0.0, y=1.0, xanchor='left', yanchor='top',
            pad=dict(l=6, t=6), showactive=False,
            bgcolor=NAVY_MID, bordercolor=GOLD, borderwidth=1, font=dict(color=TEXT_2, size=12),
            buttons=[dict(label='Show RSI detail', method='update',
                          args=state(True), args2=state(False))],
        )],
    )
    return fig

def subtitle_for(d, dist, runs, current, label):
    today = d.iloc[-1]
    sub = (f'<b>{STATUS_WORD[_status(d, dist, current)]}</b> · distance {dist.iloc[-1]:.2f}  ·  '
           + '  ·  '.join(f'{c} {today[c]:.1f}' for c in LEGS)
           + f'  ·  {len(runs)} past exits')
    yr = runs['1Y'].dropna() if not runs.empty else pd.Series(dtype=float)
    if len(yr):
        sub += (f'  ·  1Y later: {(yr < 0).sum()} of {len(yr)} lower, '
                f'median {pct(yr.median(), 0)}')
    return sub

# ════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════
if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    print('=== Building RSI Top Confluence Charts ===')

    def note_for(setup):
        floors = ' · '.join(f'{c} ≥{setup[c]:g}' for c in LEGS)
        return (f'Red bands = Daily, Weekly, Monthly and Annual RSI all overheated together ({floors}, '
                f'each scaled ±{"/".join(str(SCALE[c]) for c in LEGS)}, combined distance ≤ {HOT_DIST:.1f}). '
                f'Hot days within {EPISODE_GAP} days form one run; the red dot marks the day a run ended, '
                'which is where tops have tended to sit. Strip rows light when each RSI is near its floor. '
                'Weekly, monthly and annual read the open candle; annual = RSI(52) on weekly candles. '
                'Hover a dot for what followed.')

    for source, ident, label, weekly_rule, color, since in ASSETS:
        print(f'\nFetching {label} ({source})...')
        close = fetch_coinmetrics(ident) if source == 'coinmetrics' else fetch_yahoo(ident)
        df = build_dataframe(close, weekly_rule)
        setup = SETUP_BY_ASSET.get(label, SETUP)
        d, dist, runs, current = find_runs(df, setup, since)
        print(f'  {label}: {STATUS_WORD[_status(d, dist, current)]} | distance {dist.iloc[-1]:.2f} | '
              + ', '.join(f'{c} {d[c].iloc[-1]:.1f}' for c in LEGS))
        for _, e in runs.iterrows():
            print(f'    exit {e.end.date()} ({e.days:3d} hot days, {e.from_high:+.0f}% vs high)  ' + '  '.join(
                f'{lbl} {e[lbl]:+.0f}%' if pd.notna(e[lbl]) else f'{lbl} n/a' for lbl, _ in FWD_DAYS)
                + f'  low {e.low:.0f}%')

        fig = plot_top_confluence(d, dist, runs, current, label, color, setup)
        title = f'{label} RSI Top Confluence' + (' (since BTC treasury, Aug 2020)' if since else '')
        source_txt = 'CoinMetrics' if source == 'coinmetrics' else 'Yahoo Finance'
        path = os.path.join(OUT_DIR, f'{label.lower()}_rsi_top_confluence.html')
        write_chart(fig, path, title=title, subtitle=subtitle_for(d, dist, runs, current, label),
                    note=note_for(setup), source=source_txt, default='All', min_plot_height=520)
        print(f'  Saved: {path}')

    print('\nDone.')
