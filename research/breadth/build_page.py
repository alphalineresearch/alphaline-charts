"""Fill page_template.html with cache/page_data.json → breadth_research.html"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, 'cache', 'page_data.json')))
L = d['latest']
pct = lambda v: f"{'+' if v >= 0 else '−'}{abs(v) * 100:,.0f}%"
rep = {
    '__WEEKS__': f"{d['meta']['weeks']:,}", '__COINS__': f"{d['meta']['coins_total']:,}",
    '__ASOF__': __import__('datetime').date.fromisoformat(L['date']).strftime('%b %-d, %Y').replace('%-d', '') if os.name != 'nt' else __import__('datetime').date.fromisoformat(L['date']).strftime('%b %#d, %Y'),
    '__CWTOTAL__': f"{abs(d['alts_btc_total']) * 100:.0f}%", '__CWUP26__': f"{d['alts_btc_up_share_26'] * 100:.0f}%",
    '__T3TOTAL__': pct(d['total3_btc_total']), '__T3XTOTAL__': pct(d['total3x_btc_total']),
    '__NOW_MA__': str(L['pct_above_ma']), '__NOW_PEAK__': str(L['pct_above_ma_peak_8w']),
    '__NOW_MA50__': str(L['ma50']), '__NOW_MA100__': str(L['ma100']), '__NOW_MA200__': str(L['ma200']),
    '__NOW_N200__': str(L['n200']),
    '__NOW_YOY__': str(L['pct_up_yoy']), '__NOW_HL__': f"{L['net_hi_lo']:+d}" if L['net_hi_lo'] else '0',
    '__NOW_DOM__': f"{L['btc_dom']:.1f}", '__NOW_OFF__': f"{L['btc_off_high']:+d}".replace('-', '−'),
}
s = open(os.path.join(HERE, 'page_template.html'), encoding='utf-8').read()
for k, v in rep.items():
    s = s.replace(k, v)
d['sslw'] = json.load(open(os.path.join(HERE, 'cache', 'sslw_page.json')))
s = s.replace('__DATA__', json.dumps({k: d[k] for k in ('series', 'buckets', 'events', 'baseline', 'multi_rho', 'analogs', 'sslw')}, separators=(',', ':')))
assert '__' not in s.replace('__proto__', ''), [x for x in s.split() if '__' in x][:5]
open(os.path.join(HERE, 'breadth_research.html'), 'w', encoding='utf-8').write(s)
print('ok', len(s))
