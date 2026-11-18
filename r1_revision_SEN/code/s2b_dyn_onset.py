# -*- coding: utf-8 -*-
"""Dynamic-scenario delays from the onset and the extended held-out reading.

Table 7 (dynamic column), the pairs of Figure 11(b), and the alarms added when the
held-out segment of ds5/ds6 is extended to the onset (Table A1, Appendix D).
Run from the repository root (the working folder that also holds the data folders); see --help.
"""
import argparse
import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEF_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
SCORES = ('qr', 'ae', 'tcn', 'cn0', 'wscm')
RULES = (('cusum', 0.02), ('k_blk', 0.02), ('k_iid', 0.02), ('k_p95', None))
# Window of the common-mode event near 97 s (Appendix D).
EVENT = {'ds5_r2': (96.8, 98.7), 'ds6_r2': (96.5, 98.4)}
HO_END = {'ds5_r2': None, 'ds6_r2': 96.4}


def fl(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


def read(p):
    if not os.path.exists(p):
        raise SystemExit(f'[STOP] 找不到 {p}')
    with open(p, encoding='utf-8-sig', newline='') as fh:
        return list(csv.DictReader(fh))


def write(p, rows):
    if not rows:
        return
    with open(p, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def rankavg(x):
    x = np.asarray(x, float)
    o = np.argsort(x, kind='mergesort')
    r = np.empty(len(x))
    xs = x[o]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return r


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 4 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float('nan'), len(x)
    return float(np.corrcoef(rankavg(x), rankavg(y))[0, 1]), len(x)


def times(s):
    """'97.401;97.402;...(+5)' -> (listed times, number not listed)."""
    s = (s or '').strip()
    if not s:
        return [], 0
    more = 0
    if '(+' in s:
        s, tail = s.split('...(+')
        more = int(tail.rstrip(')'))
    return [float(x) for x in s.split(';') if x.strip() not in ('', '...')], more


def is_beta(r, b):
    return (b is None and r['rule'] == 'k_p95') or (b is not None and abs(fl(r['beta']) - b) < 1e-9)


def fmt(v):
    return '--' if not np.isfinite(v) else (f'{v:.3f}' if v < 1 else (f'{v:.2f}' if v < 10 else f'{v:.1f}'))


# ---------------------------------------------------------------------------
# Table 7 (dynamic column), Figure 11(b), extended held-out reading (Table A1)
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=DEF_ROOT)
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    s1 = os.path.join(root, 'r1_revision_SEN', 's1')
    s2 = os.path.join(root, 'r1_revision_SEN', 's2')
    star = {k: read(os.path.join(s1, f'rev_{k}_star.csv')) for k in SCORES}
    dr = {(r['score'], r['star']): 100 * fl(r['post_dr@5e-02']) for r in read(os.path.join(s2, 's2_star.csv'))}
    B = ['[s2b] v1.0  只读 S1 rev_*_star.csv 与 S2 s2_star.csv；动态类 = ds5、ds6 共 10 颗星']


    B.append('== 1. 新表 7 动态列：onset 起算时延 中位 [最大]（s），β = 0.02；漏检 = onset 后无报警 ==')
    t7 = []
    for k in SCORES:
        cells = []
        for rule, b in RULES:
            rr = [r for r in star[k] if r['rule'] == rule and is_beta(r, b) and r['group'] == 'dynamic']
            d = np.array([fl(r['delay_onset_s']) for r in rr])
            ok = d[np.isfinite(d)]
            med = float(np.median(ok)) if len(ok) else float('nan')
            mx = float(np.max(ok)) if len(ok) else float('nan')
            miss = int(np.sum(~np.isfinite(d)))
            na = [int(fl(r.get('n_alarm_act_onset'))) for r in rr if np.isfinite(fl(r.get('n_alarm_act_onset')))]
            t7.append(dict(score=k, rule=rule, n=len(rr), median_s=med, max_s=mx, missed=miss,
                           stars_alarm_act_onset=sum(x > 0 for x in na)))
            cells.append(f'{rule} {fmt(med)} [{fmt(mx)}]' + (f' 漏{miss}' if miss else ''))
        B.append(f'  {k:5s} ' + ' | '.join(cells))
    write(os.path.join(s2, 's2b_table7_dyn.csv'), t7)


    B.append('== 2. 图 11(b)：检测率（α=0.05，S2）vs CUSUM onset 时延（β=0.02）==')
    pairs, per = [], []
    for k in SCORES:
        xs, ys = [], []
        for r in star[k]:
            if r['rule'] != 'cusum' or not is_beta(r, 0.02) or r['group'] != 'dynamic':
                continue
            d, x = fl(r['delay_onset_s']), dr.get((k, r['star']), float('nan'))
            if np.isfinite(d) and d > 0 and np.isfinite(x):
                xs.append(x); ys.append(d)
                pairs.append(dict(score=k, star=r['star'], dr05=x, cusum_onset_delay_s=d))
        if xs:
            per.append((k, len(xs), float(np.median(xs)), float(np.median(ys)), float(np.max(ys))))
    B.append('  ' + '  '.join(f'{k}: n={n} DR中位 {m:.1f}% 时延中位 {fmt(d)}[{fmt(mx)}]s' for k, n, m, d, mx in per))
    if per:
        mdl = [p[3] for p in per]
        B.append(f'  跨分数 CUSUM 时延中位 {min(mdl):.3f}–{max(mdl):.3f} s')
    r5 = spearman([p['dr05'] for p in pairs], [p['cusum_onset_delay_s'] for p in pairs])
    p4 = [p for p in pairs if p['score'] != 'tcn']
    r4 = spearman([p['dr05'] for p in p4], [p['cusum_onset_delay_s'] for p in p4])
    B.append(f'  Spearman ρ_S：五种 {r5[0]:+.3f} (n={r5[1]})；四种(无TCN) {r4[0]:+.3f} (n={r4[1]})；稿件(开启口径) −0.05 (n=40)')
    write(os.path.join(s2, 's2b_fig11b_pairs.csv'), pairs)


    B.append('== 3. 留出段延到 onset 后增加的 CUSUM 报警：事件窗内 / 窗外到 onset（报警次数，按星合计）==')
    B.append('   事件窗：ds5 96.8–98.7 s，ds6 96.5–98.4 s；"未列出"= csv 只存前 10 个时刻的余数')
    for k in SCORES:
        cells = []
        for b in (0.1, 0.02):
            cnt = {}
            for r in star[k]:
                if r['rule'] != 'cusum' or not is_beta(r, b) or r['group'] != 'dynamic':
                    continue
                sc = r['scenario']
                lo, hi = EVENT[sc]
                ho_end = HO_END[sc] if HO_END[sc] is not None else -1.0
                ts, more = times(r.get('ho0_times'))
                c = cnt.setdefault(sc, [0, 0, 0, 0])
                for t in ts:
                    if t <= ho_end:
                        continue
                    if lo <= t <= hi:
                        c[0] += 1
                    else:
                        c[1] += 1
                c[2] += more
                da = fl(r.get('det_raw_s'))
                if np.isfinite(da):
                    c[3] = c[3] or []
                    c[3].append(da)
            cells.append(f'β={b:g}: ' + '；'.join(
                f'{sc[:3]} 窗内 {v[0]} 窗外 {v[1]}' + (f' 未列出 {v[2]}' if v[2] else '')
                + (f' 首报中位 {np.median(v[3]):.2f}s' if isinstance(v[3], list) and v[3] else '')
                for sc, v in sorted(cnt.items())))
        B.append(f'  {k:5s} ' + ' | '.join(cells))
    with open(os.path.join(s2, 's2b_brief.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(B) + '\n')
    print('\n'.join(B))


if __name__ == '__main__':
    main()
