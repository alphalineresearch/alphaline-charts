"""
tweet_images.py
Alphaline Research — PNGs for X threads, built from the same chart files the
website shows (docs/*.html), so charts are never screenshotted by hand.

Usage:
    python scripts/tweet_images.py "<thread folder>\\images.json"

Writes each image next to images.json (1.png, 2.png, ...). Format:

    { "images": [
        { "file": "1.png",
          "chart": "btc_risk_score_history.html",   # a file in docs/
          "range": "all",            # all | 5y | 3y | 1y | ytd | ["2020-01-01", "2026-10-25"]
          "show": ["BTC Price", "Composite Score (14d avg)"],   # legend names; omit = all
          "title": "BTC Opportunity Signal",                    # optional
          "subtitle": "Composite score, 0–100.",                # optional
          "format": "wide" }         # optional: portrait (default, 1080×1350) | wide (1600×900)
    ] }

Rendered at scale 2 (portrait = 2160×2700 actual) and sized for phones: at
1080 wide the title is 58px, subtitle 32px, ticks and legend 30px, and the
x axis carries at most 5 ticks.

Styling only: reads the chart HTML, never changes it or any numbers.
"""

import base64
import html
import io
import json
import math
import pathlib
import re
import sys
import tempfile

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from alphaline_theme import (GOLD, GOOGLE_FONTS, GRID, NAVY, NAVY_MID, TEXT_2, TEXT_3, WHITE,  # noqa: E402
                             _axis_keys, _refit_y, _run_detached)

DOCS = pathlib.Path(__file__).resolve().parent.parent / 'docs'
SCALE = 2

# CSS px for each format. Portrait follows the phone brief; wide keeps the
# original ≥18px floor at 1600 wide.
FORMATS = {
    'portrait': dict(w=1080, h=1350, title=58, sub=32, tick=30, legend=30, foot=26, pad=44, line=3.0),
    'wide':     dict(w=1600, h=900,  title=48, sub=24, tick=20, legend=20, foot=18, pad=36, line=2.2),
}
MAX_X_TICKS = 5


# ════════════════════════════════════════════
# READ THE CHART
# ════════════════════════════════════════════
def _decode(v):
    """plotly 6 writes arrays as {'dtype', 'bdata'[, 'shape']}; turn them back into lists."""
    if isinstance(v, dict):
        if 'bdata' in v and 'dtype' in v:
            a = np.frombuffer(base64.b64decode(v['bdata']), dtype=np.dtype(v['dtype']))
            if 'shape' in v:
                a = a.reshape([int(n) for n in str(v['shape']).split(',')])
            return a.tolist()
        return {k: _decode(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_decode(x) for x in v]
    return v


def read_chart(name):
    page = (DOCS / name).read_text(encoding='utf-8')
    s = page[page.index('Plotly.newPlot('):]
    dec = json.JSONDecoder()
    j = s.index('[{')
    data, n = dec.raw_decode(s[j:])
    layout, _ = dec.raw_decode(s[j + n:].lstrip(', \n'))
    m = re.search(r'var C = (\{.*?\})\s*,\s*gd\s*=', page)
    cfg = json.loads(m.group(1)) if m else {}

    def text(cls):
        m = re.search(rf'<(?:h1|div)[^>]*class="{cls}"[^>]*>(.*?)</(?:h1|div)>', page, re.S)
        return html.unescape(re.sub(r'<[^>]+>', '', m.group(1))).strip() if m else ''
    m = re.search(r'<h1>(.*?)</h1>', page, re.S)
    title = html.unescape(m.group(1)).strip() if m else ''
    return go.Figure(data=_decode(data), layout=_decode(layout)), cfg, title, text('al-sub')


def _clean(name):
    return re.sub(r'^█\s*', '', name or '').strip()


def _money(v):
    for div, unit in ((1e6, 'M'), (1e3, 'k')):
        if v >= div:
            return f'${v / div:g}{unit}'
    return f'${v:g}'


def _rgb(c):
    m = re.match(r'rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)', c or '')
    if m:
        return tuple(int(x) for x in m.groups())
    m = re.fullmatch(r'#([0-9a-fA-F]{6})', c or '')
    return tuple(int(m.group(1)[i:i + 2], 16) for i in (0, 2, 4)) if m else None


# ════════════════════════════════════════════
# BUILD ONE IMAGE
# ════════════════════════════════════════════
def _window(rng, cfg, first):
    end = pd.Timestamp(cfg.get('end') or pd.Timestamp.today().normalize())
    if isinstance(rng, list):
        return pd.Timestamp(rng[0]), pd.Timestamp(rng[1])
    r = (rng or 'all').lower()
    if r == 'all':
        start = pd.Timestamp(cfg['all'][0]) if cfg.get('all') else first
    elif r == 'ytd':
        start = pd.Timestamp(year=end.year, month=1, day=1)
    elif re.fullmatch(r'\d+y', r):
        start = end - pd.DateOffset(years=int(r[:-1]))
    else:
        raise SystemExit(f'unknown range {rng!r}: use all | 5y | 3y | 1y | ytd | [start, end]')
    # same 3% run-on past the last date as the range buttons (not the chart's
    # projection room, which leaves an empty strip at the right)
    return start, end + max((end - start) * 0.03, pd.Timedelta(days=1))


def _x_ticks(start, end):
    years = (end - start).days / 365.25
    if years < 2:
        return dict(nticks=MAX_X_TICKS)
    step = max(1, math.ceil(years / MAX_X_TICKS))
    y0 = start.year if start <= pd.Timestamp(year=start.year, month=1, day=1) else start.year + 1
    vals = [pd.Timestamp(year=y, month=1, day=1) for y in range(y0, end.year + 1, step)]
    return dict(tickmode='array', tickvals=[v.strftime('%Y-%m-%d') for v in vals],
                ticktext=[str(v.year) for v in vals])


def build(spec):
    fig, cfg, chart_title, chart_sub = read_chart(spec['chart'])
    F = FORMATS[spec.get('format', 'portrait')]
    L = fig.layout
    traces = list(fig.data)

    names = [_clean(t.name) for t in traces if t.name]
    show = spec.get('show')
    if show is not None:
        unknown = [s for s in show if _clean(s) not in names]
        if unknown:
            raise SystemExit(f'{spec["file"]}: no trace named {unknown} in {spec["chart"]}.\n'
                             f'Traces: {sorted(set(names))}')
        want = {_clean(s) for s in show}
    else:
        want = set(names)

    # an unnamed trace (a band's edge line, say) goes with the named one before it
    keep, owner = [], None
    for t in traces:
        if t.name:
            owner = _clean(t.name)
        keep.append(owner in want if owner else show is None)

    # legend-only placeholders (x all None) key shapes drawn in their colour,
    # e.g. the recency ribbon: hide the key, hide its shapes
    dropped_rgb = set()
    for t, k in zip(traces, keep):
        x = list(t.x) if t.x is not None else []
        if not k and x and all(v is None for v in x):
            dropped_rgb.add(_rgb(t.marker.color if t.marker else None))

    for t, k in zip(traces, keep):
        t.visible = True if k else False
        t.showlegend = False
        t.hoverinfo = 'skip'
        if k and getattr(t, 'line', None) is not None and t.line.width and getattr(t, 'mode', None) and 'lines' in t.mode:
            t.line.width = max(t.line.width * 1.6, F['line'] * (0.5 if (t.opacity or 1) < 1 else 1))

    # ── panels: drop the ones with nothing left, close the gap ──
    x_keys, y_keys = _axis_keys(fig, 'xaxis'), _axis_keys(fig, 'yaxis')
    used_y = {'yaxis' + (t.yaxis or 'y')[1:] for t, k in zip(traces, keep) if k}
    used_x = {'xaxis' + (t.xaxis or 'x')[1:] for t, k in zip(traces, keep) if k}
    for ya in y_keys:
        if L[ya].overlaying and ya in used_y:
            used_y.add('yaxis' + L[ya].overlaying[1:])
    gone = {k for k in y_keys if k not in used_y} | {k for k in x_keys if k not in used_x}
    for k in gone:
        L[k].visible = False

    def ref_gone(r):
        if not r or r == 'paper':
            return False
        r = r.split()[0]
        return ('xaxis' if r[0] == 'x' else 'yaxis') + r[1:] in gone

    shapes = []
    for s in L.shapes or []:
        if ref_gone(s.xref) or ref_gone(s.yref):
            continue
        if s.type == 'rect' and _rgb(s.fillcolor) in dropped_rgb:
            continue
        shapes.append(s)
    L.shapes = shapes
    L.annotations, L.images, L.updatemenus, L.sliders = (), (), (), ()

    panels = {}
    for k in y_keys:
        if k not in gone and not L[k].overlaying:
            panels.setdefault(tuple(L[k].domain) if L[k].domain else (0.0, 1.0), []).append(k)
    spans = sorted(panels)
    plot_h = F['h'] * 0.68                      # rough plot height, for a px gap between panels
    gap = 36 / plot_h
    scale = (1 - gap * (len(spans) - 1)) / sum(b - a for a, b in spans)
    y0 = 0.0
    for a, b in spans:
        y1 = min(y0 + (b - a) * scale, 1.0)
        for k in panels[(a, b)]:
            L[k].domain = [y0, y1]
        y0 = y1 + gap

    # ── window + y fit (Python twin of the page's refit) ──
    first = min((pd.Timestamp(v) for t in traces if t.x is not None for v in t.x[:1] if v), default=None)
    start, stop = _window(spec.get('range', 'all'), cfg, first)
    for k in x_keys:
        L[k].range = [start.strftime('%Y-%m-%d'), stop.strftime('%Y-%m-%d')]
        L[k].autorange = False
    fixed_y = [k for k in y_keys if L[k].range is not None and L[k].autorange is not True]
    for k in y_keys:
        if k not in fixed_y:
            L[k].range = None
    _refit_y(fig, L, fixed_y, cfg.get('fitY', []))

    # ── phone-sized axes ──
    ticks = _x_ticks(start, stop)
    lowest = min((k for k in y_keys if k not in gone and not L[k].overlaying),
                 key=lambda k: L[k].domain[0])
    bottom_x = 'xaxis' + (L[lowest].anchor or 'x')[1:]
    for k in x_keys:
        L[k].update(tickfont=dict(size=F['tick'], color=TEXT_2), title_text=None, showspikes=False,
                    showticklabels=(k == bottom_x), ticks='', **ticks)
    for k in y_keys:
        L[k].update(tickfont=dict(size=F['tick'], color=TEXT_2), title_text=None,
                    showspikes=False, ticks='')
        rng = L[k].range
        if L[k].type != 'log' and rng is not None and list(rng) == [0, 100]:
            L[k].update(tickmode='array', tickvals=[0, 20, 40, 60, 80, 100], range=[0, 101])
        elif L[k].type != 'log':
            L[k].nticks = 6
        if L[k].type == 'log' and L[k].range is not None:
            # one tick per decade: $100 · $1k · $10k · $100k
            a, b = L[k].range
            vals = [10 ** i for i in range(math.ceil(a), math.floor(b) + 1)]
            if len(vals) < 3:                            # short window: add the 1-2-5 steps
                vals = sorted(v * m for v in [10 ** i for i in range(math.floor(a), math.ceil(b) + 1)]
                              for m in (1, 2, 5) if a <= math.log10(v * m) <= b)
            L[k].update(tickmode='array', tickvals=vals,
                        ticktext=[_money(v) for v in vals])
    L.update(title=None, showlegend=False, hovermode=False, dragmode=False,
             paper_bgcolor=NAVY, plot_bgcolor=NAVY_MID,
             font=dict(family='"IBM Plex Sans", system-ui, sans-serif', color=TEXT_2, size=F['tick']),
             margin=dict(l=8, r=8, t=F['tick'] // 2 + 4, b=8, pad=6, autoexpand=True))

    # ── legend entries (kept named traces, chart order, one per legend group) ──
    legend, seen = [], set()
    for t, k in zip(traces, keep):
        if not (k and t.name):
            continue
        key = _clean(t.name)
        if key in seen:
            continue
        seen.add(key)
        if t.type == 'candlestick':
            legend.append(('box', t.increasing.line.color or TEXT_3, key))
            continue
        fill = getattr(t, 'fill', None) not in (None, 'none')
        mode = getattr(t, 'mode', None)
        if t.type == 'bar' or fill or (mode and 'lines' not in mode):
            c = t.fillcolor if fill else (t.marker.color if t.marker else None)
            # lift faint fills so the key reads against the dark page
            rgb = _rgb(c) or (143, 162, 178)
            legend.append(('box', f'rgb{rgb}', key))
        else:
            dash = t.line.dash if t.line and t.line.dash not in (None, 'solid') else ''
            legend.append(('line' + (' dash' if dash else ''), (t.line.color if t.line else None) or TEXT_3, key))

    return dict(fig=fig, F=F, legend=legend,
                title=spec.get('title') or chart_title,
                subtitle=spec.get('subtitle', chart_sub))


# ════════════════════════════════════════════
# RENDER
# ════════════════════════════════════════════
_PAGE = """<!DOCTYPE html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="__FONTS__">
<script src="__JS__"></script>
<style>
html,body{margin:0;background:__NAVY__;overflow:hidden}
#f{width:__W__px;height:__H__px;display:flex;flex-direction:column;box-sizing:border-box;
   padding:__PAD__px __PADX__px __PADB__px;font-family:"IBM Plex Sans",system-ui,sans-serif;color:__T2__}
h1{font:700 __TS__px/1.08 "Cormorant Garamond",Georgia,serif;color:__WHITE__;margin:0 0 10px;letter-spacing:.005em}
.sub{font-size:__SS__px;line-height:1.3;color:__T2__;margin:0 0 18px}
#g{flex:1 1 auto;min-height:0;margin:0 -6px}
.lg{display:flex;flex-wrap:wrap;gap:8px 30px;font-size:__LS__px;line-height:1.25;color:__T2__;margin:16px 0 0}
.lg span{display:inline-flex;align-items:center;gap:12px;white-space:nowrap}
.lg i{display:inline-block;flex:0 0 auto}
.lg i.line{width:34px;height:0;border-top:4px solid}
.lg i.line.dash{border-top-style:dashed}
.lg i.box{width:22px;height:22px}
.ft{display:flex;justify-content:space-between;align-items:center;gap:16px;margin-top:22px;
    font-size:__FS__px;color:__T3__}
.mk{display:inline-flex;align-items:center;gap:12px;color:__GOLD__;font-weight:600;letter-spacing:.12em;text-transform:uppercase}
.mk i{width:__MKS__px;height:__MKS__px;background:__GOLD__;display:inline-block}
</style></head><body><div id="f">
<h1>__TITLE__</h1>__SUB__
<div id="g"></div>
<div class="lg">__LEGEND__</div>
<div class="ft"><span class="mk"><i></i>Alphaline Research</span><span>alphalineresearch.com · data to __ASOF__</span></div>
</div>
<script>
var fig = __FIG__;
Promise.all([document.fonts.load('700 40px "Cormorant Garamond"'), document.fonts.load('400 20px "IBM Plex Sans"'),
             document.fonts.load('600 20px "IBM Plex Sans"')]).catch(function(){}).then(function () {
  var g = document.getElementById('g');
  fig.layout.width = g.clientWidth; fig.layout.height = g.clientHeight;
  Plotly.newPlot(g, fig.data, fig.layout, {staticPlot: true}).then(function () { document.title = 'ready'; });
});
</script></body></html>"""


def _as_of():
    try:
        d = json.loads((DOCS / 'manifest.json').read_text(encoding='utf-8'))['signals']['asOf']
        return pd.Timestamp(d).strftime('%b %-d, %Y') if sys.platform != 'win32' else \
            pd.Timestamp(d).strftime('%b %#d, %Y')
    except Exception:  # noqa: BLE001
        return pd.Timestamp.today().strftime('%b %d, %Y')


def render(img, out):
    from choreographer.browsers.chromium import Chromium      # installed with kaleido
    exe = Chromium.find_browser(skip_local=False)
    if not exe:
        raise SystemExit('no Chrome found (install kaleido, or Chrome)')
    F = img['F']
    js = pathlib.Path(plotly.__file__).parent / 'package_data' / 'plotly.min.js'
    legend = ''.join(f'<span><i class="{kind}" style="{"background" if kind == "box" else "border-top-color"}:'
                     f'{html.escape(color)}"></i>{html.escape(name)}</span>'
                     for kind, color, name in img['legend'])
    sub = f'<p class="sub">{html.escape(img["subtitle"])}</p>' if img['subtitle'] else ''
    rep = {
        '__FONTS__': GOOGLE_FONTS, '__JS__': js.as_uri(), '__NAVY__': NAVY, '__W__': F['w'], '__H__': F['h'],
        '__PAD__': F['pad'], '__PADX__': F['pad'] - 6, '__PADB__': round(F['pad'] * 0.7),
        '__T2__': TEXT_2, '__T3__': TEXT_3, '__WHITE__': WHITE, '__GOLD__': GOLD,
        '__TS__': F['title'], '__SS__': F['sub'], '__LS__': F['legend'], '__FS__': F['foot'],
        '__MKS__': round(F['foot'] * 0.6),
        '__TITLE__': html.escape(img['title']), '__SUB__': sub, '__LEGEND__': legend, '__ASOF__': _as_of(),
        '__FIG__': img['fig'].to_json().replace('</', '<\\/'),
    }
    page = _PAGE
    for k, v in rep.items():
        page = page.replace(k, str(v))
    W, H = F['w'], F['h']
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        p, shot = pathlib.Path(tmp, 'img.html'), pathlib.Path(tmp, 'img.png')
        p.write_text(page, encoding='utf-8')
        for attempt in range(5):            # Chrome on Windows sometimes quits at start-up
            _run_detached([exe, '--headless', '--disable-gpu', '--no-sandbox', '--hide-scrollbars',
                           '--allow-file-access-from-files', f'--force-device-scale-factor={SCALE}',
                           f'--user-data-dir={pathlib.Path(tmp, f"profile{attempt}")}',
                           f'--window-size={W},{H + 200}', '--virtual-time-budget=15000',
                           f'--screenshot={shot}', p.as_uri()])
            if shot.exists():
                break
        else:
            raise SystemExit('headless Chrome produced no screenshot')
        from PIL import Image
        # the window is taller than its viewport on Windows, so shoot tall and crop
        Image.open(shot).crop((0, 0, W * SCALE, H * SCALE)).save(out, 'PNG')


def main(path):
    path = pathlib.Path(path)
    spec = json.loads(path.read_text(encoding='utf-8'))
    for s in spec['images']:
        if s.get('format', 'portrait') not in FORMATS:
            raise SystemExit(f'{s["file"]}: format must be one of {list(FORMATS)}')
        out = path.parent / s['file']
        render(build(s), out)
        print(f'  wrote {out.name}  ({s.get("format", "portrait")})')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
