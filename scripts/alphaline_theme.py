"""
alphaline_theme.py
Alphaline Research — shared chart theme for every generate_*.py script.

Usage:
    from alphaline_theme import apply_theme, write_chart, hover

    fig = ...build traces...
    apply_theme(fig)
    write_chart(fig, OUTPUT_PATH,
                title='ETH Model Price',
                subtitle='ETH $3,950 · Model $4,210 · Z (0.42σ)',
                note='Combined TVL model: stablecoins + DeFi + RWA',
                source='Yahoo Finance · DeFiLlama',
                default='5Y')

The page is a flex column that fills the window (or the website's iframe):
HTML header (title, live reading, range buttons) → Plotly chart → legend →
HTML footer (Alphaline mark + source). Inside an iframe the title is hidden,
because the website shows its own.

Styling only: nothing here touches data, trace names or output file names.
"""

import html
import json
import re

import pandas as pd
import plotly.graph_objects as go


# Plotly skips add_vrect / add_hline / ... on any subplot that has no traces
# *yet* (exclude_empty_subplots=True). Most scripts add their shading before
# the traces, so regime shading and threshold lines silently never drew.
# Every chart script imports this module, so flip the default once here.
def _shapes_on_empty_subplots(method):
    def wrapper(self, *args, exclude_empty_subplots=False, **kwargs):
        return method(self, *args, exclude_empty_subplots=exclude_empty_subplots, **kwargs)
    wrapper.__doc__ = method.__doc__
    return wrapper


for _m in ('add_vrect', 'add_hrect', 'add_vline', 'add_hline'):
    setattr(go.Figure, _m, _shapes_on_empty_subplots(getattr(go.Figure, _m)))

# ════════════════════════════════════════════
# BRAND (see CLAUDE.md)
# ════════════════════════════════════════════
NAVY      = '#0A1628'
NAVY_MID  = '#102240'
GOLD      = '#D4A843'
GOLD_LIT  = '#ECC96A'
GOLD_DIM  = '#8A6B25'
WHITE     = '#F8FAFB'
TEXT_2    = '#B4C2CD'
TEXT_3    = '#8FA2B2'
STEEL     = '#374D61'
GREEN     = '#2ABF7A'
RED       = '#E86A6A'
RED_DEEP  = '#D64444'
ETH_BLUE  = '#8FAEE6'
GRID      = 'rgba(212,168,67,0.06)'

# Two spellings of the same font stack. The chart is drawn with the first, then
# switched to the second once the web fonts have loaded, so Plotly re-measures
# its text (it caches text sizes by style string).
FONT_STACK      = 'IBM Plex Sans, system-ui, -apple-system, Segoe UI, sans-serif'
FONT_STACK_LOAD = '"IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif'

MIN_FONT   = 12
TICK_FONT  = 12
AXIS_TITLE_FONT = 12
HOVER_FONT = 13

HOVER_DATE = '%{x|%b %-d, %Y}'

RANGE_BUTTONS = ['YTD', '1Y', '3Y', '5Y', 'All']

GOOGLE_FONTS = (
    'https://fonts.googleapis.com/css2?'
    'family=Cormorant+Garamond:wght@700'
    '&family=IBM+Plex+Mono:wght@400;500'
    '&family=IBM+Plex+Sans:wght@400;500;600&display=swap'
)


# ════════════════════════════════════════════
# HELPERS FOR CHART SCRIPTS
# ════════════════════════════════════════════
def hover(label, value, extra=''):
    """Standard hover: 'Oct 19, 2026 · BTC $81,562'.
    value is the Plotly format, e.g. '$%{y:,.0f}' or '%{y:+.2f}σ'."""
    label = f'{label} ' if label else ''
    return f'{HOVER_DATE} · {label}{value}{extra}<extra></extra>'


def pct(v, d=1):
    """+4.5% / (3.9%) — finance-style negatives."""
    return f'({abs(v):.{d}f}%)' if v < 0 else f'+{v:.{d}f}%'


def sigma(v, d=2):
    """+1.20σ / (0.42σ)."""
    return f'({abs(v):.{d}f}σ)' if v < 0 else f'+{v:.{d}f}σ'


def date_long(ts):
    ts = pd.Timestamp(ts)
    return f'{ts:%b} {ts.day}, {ts.year}'


# ════════════════════════════════════════════
# LAYOUT
# ════════════════════════════════════════════
_OLD_HOVER_DATE = re.compile(r'%\{x\|%Y-%m-%d\}(<br>|\s*\|\s*|\s+)')
_OTHER_DATE_BR  = re.compile(r'(%\{x\|[^}]*\})<br>')      # e.g. monthly '%b %Y'
_HOVER_COLON    = re.compile(r':\s+(?=\$?%\{)')


def _restyle_hover(tpl):
    if not tpl or not isinstance(tpl, str):
        return tpl
    tpl = _OLD_HOVER_DATE.sub(HOVER_DATE + ' · ', tpl)
    tpl = _OTHER_DATE_BR.sub(r'\1 · ', tpl)
    return _HOVER_COLON.sub(' ', tpl)


def _axis_keys(fig, prefix):
    keys = [k for k in fig.layout.to_plotly_json() if re.fullmatch(prefix + r'\d*', k)]
    return keys or [prefix]


def apply_theme(fig, legend=True):
    """Brand layout for any figure (single panel or subplots). Call after the
    traces are added; chart-specific axis settings can follow."""
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=NAVY, plot_bgcolor=NAVY_MID,
        autosize=True, height=None, width=None,
        title=None,
        font=dict(family=FONT_STACK, color=TEXT_3, size=TICK_FONT),
        margin=dict(l=12, r=16, t=24, b=8, pad=0),
        hoverlabel=dict(bgcolor=NAVY_MID, bordercolor=GOLD,
                        font=dict(size=HOVER_FONT, color=WHITE), align='left'),
        # The legend is drawn in HTML under the chart by write_chart (Plotly's
        # own clips and won't wrap well); showlegend just says whether to.
        showlegend=legend,
        modebar=dict(bgcolor='rgba(0,0,0,0)', color=TEXT_3, activecolor=GOLD),
    )
    fig.update_xaxes(gridcolor=GRID, gridwidth=1, zeroline=False, linecolor=STEEL,
                     showline=False, automargin=True,
                     tickfont=dict(size=TICK_FONT, color=TEXT_3),
                     title_font=dict(size=AXIS_TITLE_FONT, color=TEXT_3),
                     showspikes=True, spikecolor=TEXT_3, spikethickness=1,
                     spikedash='dot', spikemode='across', spikesnap='cursor')
    fig.update_yaxes(gridcolor=GRID, gridwidth=1, zeroline=False, linecolor=STEEL,
                     showline=False, automargin=True,
                     tickfont=dict(size=TICK_FONT, color=TEXT_3),
                     title_font=dict(size=AXIS_TITLE_FONT, color=TEXT_3),
                     title_standoff=8)
    return _finish(fig)


def _finish(fig):
    """Rules that must also catch settings a script makes after apply_theme;
    run again by write_chart."""
    # 'Date' under a time axis says nothing
    for k in _axis_keys(fig, 'xaxis'):
        ax = fig.layout[k]
        if ax.title and ax.title.text and ax.title.text.strip().lower() == 'date':
            ax.title.text = None

    # Annotations: inherit the brand font, nothing under 12px
    for a in fig.layout.annotations or []:
        a.font.family = None
        a.font.size = max(a.font.size or MIN_FONT, MIN_FONT)

    for tr in fig.data:
        # plotly_dark outlines every bar; on daily bars the outline swamps the colour
        if tr.type == 'bar' and tr.marker.line.width is None:
            tr.marker.line.width = 0
        if getattr(tr, 'hovertemplate', None):
            tr.hovertemplate = _restyle_hover(tr.hovertemplate)
        if getattr(tr, 'textfont', None) is not None and tr.textfont.size:
            tr.textfont.size = max(tr.textfont.size, MIN_FONT)
            tr.textfont.family = None
    return fig


# ════════════════════════════════════════════
# OUTPUT
# ════════════════════════════════════════════
def _date_extent(fig):
    lo = hi = None
    for tr in fig.data:
        x = getattr(tr, 'x', None)
        if x is None or not len(x):
            continue
        try:
            xs = pd.to_datetime(pd.Series(list(x)), errors='coerce').dropna()
        except Exception:
            continue
        if len(xs):
            lo = xs.min() if lo is None else min(lo, xs.min())
            hi = xs.max() if hi is None else max(hi, xs.max())
    return lo, hi


def _iso(v):
    return pd.Timestamp(v).strftime('%Y-%m-%d')


def write_chart(fig, path, *, title, subtitle='', note='', source='',
                ranges=True, default='5Y', end=None, min_plot_height=360, fit_y=()):
    """Write a branded, self-contained chart page.

    ranges   — show the YTD/1Y/3Y/5Y/All buttons (False for short-window charts).
    default  — window on load: one of RANGE_BUTTONS.
    end      — last real data date the windows count back from. Defaults to the
               latest x in the figure; pass it when a projection runs past today.
    fit_y    — y axes ('yaxis3' etc.) fitted to the data even though they hold
               an area filled to zero (by default those keep a zero baseline).
    The 'All' button restores the x range the script set; without one it runs
    from the first date to `end` (+3%).
    """
    _finish(fig)
    layout = fig.layout
    x_keys = _axis_keys(fig, 'xaxis')
    y_keys = _axis_keys(fig, 'yaxis')

    # Axes with a range set by the script (score 0–100 etc.) stay fixed
    fixed_y = [k for k in y_keys if layout[k].range is not None]

    want_legend = layout.showlegend is not False
    layout.showlegend = False

    first, last = _date_extent(fig)
    end = pd.Timestamp(end) if end is not None else last
    all_range = None
    if layout.xaxis.range is not None:
        all_range = [_iso(v) for v in layout.xaxis.range]
    elif first is not None and end is not None:
        all_range = [_iso(first), _iso(end + (end - first) * 0.03)]

    if not ranges:
        default = None
    if default and end is not None:
        win = all_range if default == 'All' else [_iso(v) for v in _window(default, end)]
        if win:
            for k in x_keys:
                layout[k].range = win

    cfg = dict(
        ranges=RANGE_BUTTONS if ranges else [],
        default=default,
        end=_iso(end) if end is not None else None,
        all=all_range,
        fixedY=fixed_y,
        fitY=list(fit_y),
        xKeys=x_keys,
        legend=want_legend,
        fontLoaded=FONT_STACK_LOAD,
    )
    div = fig.to_html(
        full_html=False, include_plotlyjs='cdn', default_height='100%', default_width='100%',
        config=dict(responsive=True, displaylogo=False,
                    modeBarButtonsToRemove=['select2d', 'lasso2d', 'autoScale2d'],
                    toImageButtonOptions=dict(format='png', scale=2)),
        post_script=_POST_SCRIPT.replace('__CFG__', json.dumps(cfg)),
    )

    buttons = ''.join(
        f'<button type="button" data-r="{b}"{" class=on" if b == default else ""}>{b}</button>'
        for b in cfg['ranges'])
    page = _PAGE.format(
        title=html.escape(title),
        fonts=GOOGLE_FONTS,
        css=_CSS.replace('__MINH__', str(min_plot_height)),
        h1=html.escape(title),
        sub=subtitle,
        note=f'<div class="al-note">{note}</div>' if note else '',
        buttons=f'<nav class="al-ranges" aria-label="Date range">{buttons}</nav>' if buttons else '',
        plot=div,
        source=f'Source: {html.escape(source)}' if source else '',
    )
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(page)


def _window(name, end):
    end = pd.Timestamp(end)
    if name == 'YTD':
        start = pd.Timestamp(year=end.year, month=1, day=1)
    else:
        start = end - pd.DateOffset(years=int(name[:-1]))
    return start, end + (end - start) * 0.03


# ════════════════════════════════════════════
# PAGE TEMPLATE
# ════════════════════════════════════════════
_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · Alphaline Research</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="{fonts}">
<script>
(function () {{
  // Embed mode = inside the website's iframe (title hidden; the site shows it).
  // Not in embed mode: standalone, true full screen, or when the site posts
  // {{alphaline: 'fullscreen', on: true}} from its maximise overlay.
  // ?embed=1 / ?embed=0 force either way.
  var q = new URLSearchParams(location.search), e = q.get('embed'), inFrame, told = null;
  try {{ inFrame = window.self !== window.top; }} catch (err) {{ inFrame = true; }}
  function check() {{
    var full = told !== null ? told
      : (window.innerHeight >= screen.height - 2 && window.innerWidth >= screen.width - 2);
    var on = e === '1' || (e !== '0' && inFrame && !full);
    document.documentElement.classList.toggle('embed', on);
  }}
  check();
  window.addEventListener('resize', check);
  window.addEventListener('message', function (ev) {{
    var d = ev.data || {{}};
    if (d.alphaline === 'fullscreen') {{ told = !!d.on; check(); window.dispatchEvent(new Event('resize')); }}
  }});
}})();
</script>
<style>{css}</style>
</head>
<body>
<header class="al-head">
  <div class="al-titles">
    <h1>{h1}</h1>
    <div class="al-sub">{sub}</div>
    {note}
  </div>
  {buttons}
</header>
<main class="al-plot">
{plot}
</main>
<div class="al-legend" role="group" aria-label="Series (click to show or hide)"></div>
<footer class="al-foot">
  <a class="al-mark" href="https://alphalineresearch.com" target="_blank" rel="noopener"><i></i>Alphaline Research</a>
  <span class="al-src">{source}</span>
</footer>
</body>
</html>
"""

_CSS = """
:root { --navy:#0A1628; --navy-mid:#102240; --gold:#D4A843; --gold-l:#ECC96A; --gold-d:#8A6B25;
        --white:#F8FAFB; --t2:#B4C2CD; --t3:#8FA2B2; --steel:#374D61; }
* { box-sizing: border-box; }
html, body { height: 100%; margin: 0; background: var(--navy); color: var(--t2);
             font-family: 'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif; }
body { display: flex; flex-direction: column; overflow: hidden; }
.al-head { display: flex; flex-wrap: wrap; align-items: flex-start; justify-content: space-between;
           gap: 8px 24px; padding: 16px 20px 4px; }
.al-titles { min-width: 0; flex: 1 1 320px; }
h1 { margin: 0 0 4px; font-family: 'Cormorant Garamond', Georgia, serif; font-weight: 700;
     font-size: 22px; line-height: 1.15; color: var(--white); letter-spacing: .005em; }
.al-sub { font-size: 13px; line-height: 1.45; color: var(--t2); }
.al-sub b { color: var(--white); font-weight: 600; }
.al-note { font-size: 12px; line-height: 1.4; color: var(--t3); margin-top: 2px; }
.al-ranges { display: flex; gap: 0; flex: 0 0 auto; margin-top: 2px; }
.al-ranges button { font: 500 12px/1 'IBM Plex Sans', system-ui, sans-serif; letter-spacing: .04em;
  color: var(--t2); background: transparent; border: 1px solid var(--steel); border-radius: 0;
  padding: 7px 11px; margin-left: -1px; cursor: pointer; min-width: 40px; }
.al-ranges button:first-child { margin-left: 0; }
.al-ranges button:hover { color: var(--gold-l); border-color: var(--gold-d); position: relative; z-index: 1; }
.al-ranges button.on { color: var(--navy); background: var(--gold); border-color: var(--gold);
  position: relative; z-index: 2; }
.al-ranges button:focus-visible { outline: 2px solid var(--gold-l); outline-offset: 1px; }
.al-plot { flex: 1 1 auto; min-height: min(__MINH__px, 50vh); position: relative; padding: 0 8px; }
.al-plot > div { height: 100%; }
.al-legend { display: flex; flex-wrap: wrap; gap: 2px 18px; padding: 2px 20px 2px 24px; }
.al-legend:empty { display: none; }
.al-legend button { display: inline-flex; align-items: center; gap: 7px; padding: 3px 0; border: 0;
  background: none; cursor: pointer; font: 400 12px/1.3 'IBM Plex Sans', system-ui, sans-serif;
  color: var(--t2); text-align: left; }
.al-legend button:hover { color: var(--white); }
.al-legend button.off { opacity: .4; }
.al-legend button.off span { text-decoration: line-through; }
.al-legend i { flex: 0 0 auto; display: inline-block; width: 18px; height: 0; border-top: 2px solid; }
.al-legend i.box { width: 11px; height: 11px; border: 0; }
.al-legend i.glyph { width: auto; height: auto; border: 0; font-style: normal; font-size: 11px; line-height: 1; }
.al-foot { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center;
           gap: 2px 16px; padding: 4px 20px 10px; font-size: 12px; color: var(--t3); }
.al-mark { display: inline-flex; align-items: center; gap: 7px; color: var(--gold); text-decoration: none;
           font-weight: 600; letter-spacing: .12em; text-transform: uppercase; font-size: 12px; }
.al-mark i { width: 9px; height: 9px; background: var(--gold); display: inline-block; }
.al-mark:hover { color: var(--gold-l); }
.embed h1 { display: none; }
.embed .al-head { padding-top: 10px; }
.modebar-btn path { fill: var(--t3) !important; }
@media (max-width: 640px) {
  .al-head { padding: 12px 12px 2px; gap: 8px; }
  .al-titles { flex-basis: 100%; }
  h1 { font-size: 20px; }
  .al-ranges { width: 100%; }
  .al-ranges button { flex: 1 1 0; padding: 8px 0; }
  .al-plot { padding: 0 2px; }
  .al-legend { padding: 2px 12px; gap: 0 14px; }
  .al-foot { padding: 4px 12px 8px; }
  .modebar-container { display: none !important; }
}
"""

# Runs after Plotly.newPlot (plotly.py substitutes {plot_id}).
_POST_SCRIPT = r"""
(function () {
  var C = __CFG__, gd = document.getElementById('{plot_id}');
  if (!gd || !window.Plotly) return;
  var DAY = 864e5, busy = false;

  // ── y refit: fit every auto y-axis to the data inside the visible x window ──
  function axName(id) { return (id.charAt(0) === 'x' ? 'xaxis' : 'yaxis') + id.slice(1); }
  // log axes: the densest 1-2-5 style tick set that still reads cleanly
  function logTicks(a, b) {
    var max = mq.matches ? 5 : 7, best = null;
    var sets = [[1], [1, 2, 5], [1, 1.5, 2, 3, 5, 7], [1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9]];
    for (var s = 0; s < sets.length; s++) {
      var vals = [];
      for (var k = Math.floor(a); k <= Math.ceil(b); k++) sets[s].forEach(function (m) {
        var v = m * Math.pow(10, k), l = Math.log10(v);
        if (l >= a && l <= b) vals.push(+v.toPrecision(6));
      });
      if (vals.length > max && best) break;
      best = vals;
      if (vals.length > max) break;
    }
    if (best.length > max) {            // many decades: every n-th power of ten
      var step = Math.ceil(best.length / max);
      best = best.filter(function (v, i) { return i % step === 0; });
    }
    return best;
  }

  function refit() {
    var fl = gd._fullLayout, cd = gd.calcdata || [], ext = {};
    gd._fullData.forEach(function (tr, i) {
      if (tr.visible !== true || !cd[i]) return;
      var ya = axName(tr.yaxis || 'y'), xa = fl[axName(tr.xaxis || 'x')];
      if (C.fixedY.indexOf(ya) >= 0 || !fl[ya] || !xa || !xa.range) return;
      if (fl[ya].type === 'category') return;
      var lo = xa.r2l(xa.range[0]), hi = xa.r2l(xa.range[1]), isLog = fl[ya].type === 'log';
      var e = ext[ya] || (ext[ya] = { min: Infinity, max: -Infinity, log: isLog });
      var isBar = tr.type === 'bar';
      // areas filled down to zero (or stacked) keep their zero baseline
      if (!isLog && (tr.stackgroup || tr.fill === 'tozeroy') && C.fitY.indexOf(ya) < 0) e.zero = true;
      cd[i].forEach(function (p) {
        var x = isBar ? p.p : p.x, y;
        if (x == null || x < lo || x > hi) return;
        if (isBar) {
          y = (p.b || 0) + p.s;
          if (!isLog) { e.min = Math.min(e.min, p.b || 0); e.max = Math.max(e.max, p.b || 0); }
        } else { y = p.y; }
        if (y == null || !isFinite(y) || (isLog && y <= 0)) return;
        if (y < e.min) e.min = y;
        if (y > e.max) e.max = y;
      });
    });
    var upd = {};
    Object.keys(ext).forEach(function (ya) {
      var e = ext[ya];
      if (!isFinite(e.min) || !isFinite(e.max)) return;
      var a = e.log ? Math.log10(e.min) : e.min, b = e.log ? Math.log10(e.max) : e.max;
      if (e.zero && a >= 0) a = 0;
      var pad = (b - a) * 0.06 || (e.log ? 0.05 : Math.abs(a) * 0.05 || 1);
      // fitted axes get extra room below for labels under the lows
      var padLo = C.fitY.indexOf(ya) >= 0 ? pad * 3 : pad;
      upd[ya + '.range'] = [e.zero && a === 0 ? 0 : a - padLo, b + pad];
      upd[ya + '.autorange'] = false;
      if (e.log) {
        // our own tick values; any hand-set tick labels no longer line up
        upd[ya + '.tickmode'] = 'array';
        upd[ya + '.tickvals'] = logTicks(a - pad, b + pad);
        upd[ya + '.ticktext'] = null;
      }
    });
    if (Object.keys(upd).length) { busy = true; Plotly.relayout(gd, upd).then(function () { busy = false; }); }
  }

  // ── range buttons ──
  var btns = [].slice.call(document.querySelectorAll('.al-ranges button'));
  function mark(r) { btns.forEach(function (b) { b.classList.toggle('on', b.getAttribute('data-r') === r); }); }
  function windowFor(r) {
    var end = new Date(C.end + 'T00:00:00Z').getTime(), start;
    if (r === 'YTD') start = Date.UTC(new Date(end).getUTCFullYear(), 0, 1);
    else { var d = new Date(end); d.setUTCFullYear(d.getUTCFullYear() - parseInt(r, 10)); start = d.getTime(); }
    var stop = end + Math.max((end - start) * 0.03, DAY);
    return [new Date(start).toISOString().slice(0, 10), new Date(stop).toISOString().slice(0, 10)];
  }
  function setRange(r) {
    var upd = {};
    C.xKeys.forEach(function (k) {
      if (r === 'All') {
        if (C.all) upd[k + '.range'] = C.all.slice(); else upd[k + '.autorange'] = true;
      } else upd[k + '.range'] = windowFor(r);
    });
    mark(r);
    busy = true;
    Plotly.relayout(gd, upd).then(function () { busy = false; refit(); });
  }
  btns.forEach(function (b) { b.addEventListener('click', function () { setRange(b.getAttribute('data-r')); }); });

  gd.on('plotly_relayout', function (ev) {
    if (busy) return;
    var xs = Object.keys(ev || {}).some(function (k) { return /^xaxis\d*\.(range|autorange)/.test(k) || /^xaxis\d*$/.test(k); });
    if (!xs) return;
    mark(null);
    refit();
  });

  // ── HTML legend: one entry per trace (or legend group), click to toggle ──
  function buildLegend() {
    var box = document.querySelector('.al-legend');
    if (!box || !C.legend) return;
    var seen = {};
    gd._fullData.forEach(function (tr, i) {
      if (!tr.showlegend || !tr.name) return;
      var grp = tr.legendgroup, key = grp ? 'g:' + grp + ':' + tr.name : 'i:' + i;
      if (seen[key]) return;
      seen[key] = 1;
      var idx = gd._fullData.map(function (t, j) { return (grp ? t.legendgroup === grp : j === i) ? j : -1; })
                            .filter(function (j) { return j >= 0; });
      var b = document.createElement('button'), sw = document.createElement('i'), t = document.createElement('span');
      b.type = 'button';
      var fill = tr.fill && tr.fill !== 'none', mk = tr.mode && tr.mode.indexOf('lines') < 0;
      var sym = mk && tr.marker && String(tr.marker.symbol || '');
      if (sym && /triangle-(up|down)/.test(sym)) {
        // marker series: show the actual glyph (colours can vary per point)
        var mc = tr.marker.color;
        sw.className = 'glyph';
        sw.textContent = sym.indexOf('down') >= 0 ? '▼' : '▲';
        sw.style.color = typeof mc === 'string' ? mc : (mc && mc[0]) || '#8FA2B2';
      } else if (fill || mk || tr.type === 'bar') {
        sw.className = 'box';
        var c = fill ? tr.fillcolor : tr.marker && tr.marker.color;
        sw.style.background = typeof c === 'string' ? c : (c && typeof c[0] === 'string' ? c[0] : '#8FA2B2');
      } else {
        sw.style.borderTopColor = tr.line && tr.line.color || '#8FA2B2';
        if (tr.line && tr.line.dash && tr.line.dash !== 'solid') sw.style.borderTopStyle = tr.line.dash === 'dot' ? 'dotted' : 'dashed';
        if (tr.opacity < 1) sw.style.opacity = tr.opacity;
      }
      t.textContent = tr.name.replace(/^█\s*/, '');   // some names carry their own block glyph
      b.appendChild(sw); b.appendChild(t);
      function sync() { b.classList.toggle('off', gd._fullData[idx[0]].visible !== true); b.setAttribute('aria-pressed', b.classList.contains('off') ? 'false' : 'true'); }
      b.addEventListener('click', function () {
        var on = gd._fullData[idx[0]].visible === true;
        busy = true;
        Plotly.restyle(gd, { visible: on ? 'legendonly' : true }, idx).then(function () { busy = false; sync(); refit(); });
      });
      sync();
      box.appendChild(b);
    });
  }
  // this script runs while the page is still parsing; the legend box comes after it
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', buildLegend);
  else buildLegend();

  // ── phones: fewer ticks ──
  var mq = window.matchMedia('(max-width: 640px)');
  function ticks() {
    var upd = {}, n = mq.matches ? 4 : null;
    C.xKeys.forEach(function (k) { upd[k + '.nticks'] = n; });
    Object.keys(gd._fullLayout).forEach(function (k) { if (/^yaxis\d*$/.test(k)) upd[k + '.nticks'] = mq.matches ? 5 : null; });
    busy = true;
    return Plotly.relayout(gd, upd).then(function () { busy = false; });
  }
  function onBreakpoint() { ticks().then(function () { refit(); }); }
  if (mq.addEventListener) mq.addEventListener('change', onBreakpoint); else mq.addListener(onBreakpoint);

  // ── once the web fonts are in, redraw so Plotly measures them ──
  var fontsIn = document.fonts && document.fonts.load
    ? Promise.all([document.fonts.load('12px "IBM Plex Sans"'), document.fonts.load('600 12px "IBM Plex Sans"')]).catch(function () {})
    : Promise.resolve();

  // Plotly only watches the window; the header can change height on its own
  // (fonts arriving, embed toggling, subtitle wrapping), so watch the box.
  if (window.ResizeObserver) {
    var lastH = 0, lastW = 0, raf = 0;
    new ResizeObserver(function () {
      var h = gd.clientHeight, w = gd.clientWidth;
      if (Math.abs(h - lastH) < 1 && Math.abs(w - lastW) < 1) return;
      lastH = h; lastW = w;
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(function () { Plotly.Plots.resize(gd); });
    }).observe(gd);
  }

  ticks().then(function () { refit(); });
  fontsIn.then(function () {
    busy = true;
    Plotly.relayout(gd, { 'font.family': C.fontLoaded }).then(function () { busy = false; });
  });
})();
"""
