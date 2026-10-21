"""Fill page_template.html with cache/page_data.json → prefs_research.html (artifact body)
and treasury_preferreds.html (standalone page to open in a browser)."""
import json, os
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, 'cache', 'page_data.json')))
N = d['now']
day = lambda s: date.fromisoformat(s).strftime('%b %#d, %Y' if os.name == 'nt' else '%b %-d, %Y')
sgn = lambda v: f"{'+' if v >= 0 else '−'}{abs(v) * 100:.1f}%"
rep = {
    '__STYLE__': open(os.path.join(HERE, 'style.css.html'), encoding='utf-8').read(),
    '__ASOF__': day(N['date']), '__BREAK__': day(N['strd_break_date']),
    '__BTC50__': sgn(N['btc_vs_50']), '__STRD50__': sgn(N['strd_vs_50']),
    '__SPCHG__': f"{N['strd_spread_chg20']:.2f}", '__STRDSP__': f"{N['strd_spread']:.1f}",
    '__SPPEAK__': f"{N['strd_spread_peak']:.1f}", '__STRC__': f"{N['strc']:.2f}", '__RATE__': f"{N['strc_rate']:.0f}",
    '__TNX__': f"{N['tnx']:.2f}", '__BMNP__': f"{N['bmnp']:.2f}", '__BMNPLOW__': f"{N['bmnp_low']:.2f}",
}
s = open(os.path.join(HERE, 'page_template.html'), encoding='utf-8').read()
for k, v in rep.items():
    s = s.replace(k, v)
s = s.replace('__DATA__', json.dumps(d, separators=(',', ':')))
assert '__' not in s.replace('__proto__', ''), [x for x in s.split() if '__' in x][:5]
open(os.path.join(HERE, 'prefs_research.html'), 'w', encoding='utf-8').write(s)
open(os.path.join(HERE, 'treasury_preferreds.html'), 'w', encoding='utf-8').write(
    '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width,initial-scale=1">\n</head>\n<body>\n' + s + '\n</body>\n</html>\n')
print('ok', len(s))
