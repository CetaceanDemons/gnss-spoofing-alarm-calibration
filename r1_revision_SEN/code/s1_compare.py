# -*- coding: utf-8 -*-
"""Checks the S1 rerun against the original run and extracts the static delays (Table 7).

Only the onset-related columns of ds3 may differ; everything else must be identical.
Run from the repository root (the working folder that also holds the data folders); see --help.
"""
import argparse
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DEF_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
SCORES = (('qr', 'QR-LSTM', 'results_rev'), ('ae', 'LSTM-AE', 'results_rev'), ('tcn', 'TCN', 'results_rev'),
          ('cn0', 'C/N0 proxy', 'results_c'), ('wscm', 'WSCM', 'results_c'))
HO0 = ['ho0_' + k for k in ('len_s', 'n_alarm', 'n_event', 'cov', 'nwin', 'frac', 'lo95', 'up95', 'verdict', 'times')]
# Columns allowed to differ between the original run and the rerun (ds3 only).
EXPECT = set(['onset_raw_s', 'ho0_raw_end_s', 'det_onset_raw_s', 'delay_onset_s', 'n_alarm_act_onset'] + HO0)
POOL_EXPECT_COLS = set(['ho0_' + k for k in ('n_stars', 'total_s', 'n_event', 'frac', 'lo95', 'up95', 'verdict')]
                       + ['delay_onset_med', 'delay_onset_min', 'delay_onset_max'])
POOL_EXPECT_GROUPS = ('static', 'all')
IGNORE = {'regress'}
RULES = (('cusum', '0.02', 'CUSUM, block h'), ('k_blk', '0.02', 'K=5, block gamma'),
         ('k_iid', '0.02', 'K=5, indep. gamma'), ('k_p95', '', 'K=5, 95th pct.'))


def read(path):
    if not os.path.exists(path):
        return None
    with open(path, newline='', encoding='utf-8-sig') as fh:
        return list(csv.DictReader(fh))


def same(a, b):
    if a == b:
        return True
    try:
        x, y = float(a), float(b)
        return abs(x - y) <= 1e-9 * max(1.0, abs(x), abs(y))
    except (TypeError, ValueError):
        return False


def fnum(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return float('nan')


def cfg_lines(path):
    if not os.path.exists(path):
        return ['(无日志)']
    with open(path, encoding='utf-8', errors='replace') as fh:
        return [ln.rstrip() for ln in fh if ln.startswith('[cfg]')][:3]


# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------
def cmp_star(old, new):
    ko = {(r['star'], r['rule'], r['K'], r['beta']): r for r in old}
    kn = {(r['star'], r['rule'], r['K'], r['beta']): r for r in new}
    miss = sorted(set(ko) ^ set(kn))
    bad, exp_changed = [], 0
    for k in sorted(set(ko) & set(kn)):
        ro, rn = ko[k], kn[k]
        for c in ro:
            if c in IGNORE or same(ro[c], rn.get(c)):
                continue
            if ro['scenario'] == 'ds3' and c in EXPECT:
                exp_changed += 1
            else:
                bad.append((k, c, ro[c], rn.get(c)))
    return len(ko), miss, bad, exp_changed


def cmp_pool(old, new):
    key = lambda r: (r['rule'], r['beta'], r['group'])
    ko, kn = {key(r): r for r in old}, {key(r): r for r in new}
    bad = []
    for k in sorted(set(ko) & set(kn)):
        for c in ko[k]:
            if same(ko[k][c], kn[k].get(c)):
                continue
            if not (k[2] in POOL_EXPECT_GROUPS and c in POOL_EXPECT_COLS):
                bad.append((k, c, ko[k][c], kn[k].get(c)))
    return bad, sorted(set(ko) ^ set(kn))


def cell(r, a='delay_onset_med', b='delay_onset_max'):
    if r is None:
        return '—'
    return f'{fnum(r.get(a)):.3f} [{fnum(r.get(b)):.3f}]'


# ---------------------------------------------------------------------------
# Static delays for Table 7
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=DEF_ROOT)
    ap.add_argument('--new-dir', default=None, help=r'默认 <root>\r1_revision_SEN\s1')
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    nd = os.path.abspath(a.new_dir or os.path.join(root, 'r1_revision_SEN', 's1'))
    B, Dt = [], []
    B.append('[s1_compare] v1.0  新目录 ' + nd)
    B.append('== 1. 新旧逐列比对（只允许 ds3 的 onset 口径列变化） ==')
    verdict_ok = True
    tab7, pre, ds3 = {}, {}, {}
    for sc, name, sub in SCORES:
        od = os.path.join(root, 're_ii', sub)
        so, sn = read(os.path.join(od, f'rev_{sc}_star.csv')), read(os.path.join(nd, f'rev_{sc}_star.csv'))
        po, pn = read(os.path.join(od, f'rev_{sc}_pooled.csv')), read(os.path.join(nd, f'rev_{sc}_pooled.csv'))
        if not (so and sn and po and pn):
            B.append(f'  {name:11s} 缺文件：旧 star {bool(so)} 新 star {bool(sn)} 旧 pooled {bool(po)} 新 pooled {bool(pn)}')
            verdict_ok = False
            continue
        n, miss, bad, ec = cmp_star(so, sn)
        pbad, pmiss = cmp_pool(po, pn)
        ok = not miss and not bad and not pbad and not pmiss and ec > 0
        verdict_ok &= ok
        B.append(f'  {name:11s} star 行 {n}，ds3 预期列变化 {ec} 处，意外差异 {len(bad)}，行不配对 {len(miss)}；'
                 f'pooled 意外差异 {len(pbad)}，行不配对 {len(pmiss)}  → {"OK" if ok else "需查"}')
        if ec == 0:
            B.append('      [WARN] ds3 没有任何列变化：onset 的修改可能没有生效')
        Dt.append(f'### {name}\n[cfg 旧] ' + ' | '.join(cfg_lines(os.path.join(od, f'rev_{sc}_log.txt'))))
        Dt.append('[cfg 新] ' + ' | '.join(cfg_lines(os.path.join(nd, f'rev_{sc}_log.txt'))))
        for k in miss[:20]:
            Dt.append(f'  star 行不配对: {k}')
        for k, c, x, y in bad[:40]:
            Dt.append(f'  star 意外差异: {k} 列 {c}: {x} -> {y}')
        for k in pmiss[:20]:
            Dt.append(f'  pooled 行不配对: {k}')
        for k, c, x, y in pbad[:40]:
            Dt.append(f'  pooled 意外差异: {k} 列 {c}: {x} -> {y}')

        P = lambda rows, rule, beta, g: next((r for r in rows if r['rule'] == rule and r['beta'] == beta
                                              and r['group'] == g), None)
        for rule, beta, lab in RULES:
            tab7[(name, lab)] = (cell(P(po, rule, beta, 'static')), cell(P(pn, rule, beta, 'static')))
            cnt = lambda rows: sum(1 for r in rows if r['rule'] == rule and r['beta'] == beta
                                   and r['group'] == 'static' and fnum(r['n_alarm_act_onset']) > 0)
            pre[(name, lab)] = (cnt(so), cnt(sn))
        for r_o in so:
            if r_o['scenario'] == 'ds3' and r_o['rule'] in ('cusum', 'k_blk') and r_o['beta'] == '0.02':
                r_n = next((x for x in sn if x['star'] == r_o['star'] and x['rule'] == r_o['rule']
                            and x['beta'] == '0.02'), {})
                ds3[(name, r_o['rule'], r_o['star'])] = (fnum(r_o['delay_onset_s']), fnum(r_n.get('delay_onset_s')))
    B.append(f'  总判定：{"PASS（只有预期列变化）" if verdict_ok else "FAIL（见 s1_compare_detail.txt）"}')

    B.append('\n== 2. 新表 7 Static 列（β = 0.02；onset 口径；中位 [最大]，s）  旧 → 新 ==')
    for (name, lab), (o, nw) in tab7.items():
        B.append(f'  {name:11s} {lab:18s} {o:>17s} → {nw}')
    B.append('\n== 3. 静态类 7 颗星中"开启到 onset 之间已报警"的星数（β = 0.02；cusum/k_blk/k_iid/p95）  旧 → 新 ==')
    for name in [s[1] for s in SCORES]:
        B.append(f'  {name:11s} ' + '   '.join(f'{pre[(name, lab)][0]}→{pre[(name, lab)][1]}'
                                              for _, _, lab in RULES if (name, lab) in pre))
    B.append('\n== 4. ds3 逐星时延（onset 口径，β = 0.02，s）  旧 → 新 ==')
    for name in [s[1] for s in SCORES]:
        for rule in ('cusum', 'k_blk'):
            its = sorted((st, v) for (nm, ru, st), v in ds3.items() if nm == name and ru == rule)
            if its:
                B.append(f'  {name:11s} {rule:6s} ' + '；'.join(f'{st} {o:+.3f}→{nw:+.3f}' for st, (o, nw) in its))
    with open(os.path.join(nd, 's1_brief.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(B) + '\n')
    with open(os.path.join(nd, 's1_compare_detail.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(Dt) + '\n')
    print('\n'.join(B))
    print(f'\n[s1_compare] 写出 {os.path.join(nd, "s1_brief.txt")}（≤60 行，回传它）')


if __name__ == '__main__':
    main()
