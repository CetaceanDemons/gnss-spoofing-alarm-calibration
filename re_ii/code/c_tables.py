"""Collects the result files into the tables of the paper (csv and LaTeX).

Produces Tables 2, 3, 4, 5, A2, A3 and A5. Tables 7 and 8 of the revised paper come
from r1_revision_SEN/code (s1_compare.py, s2b_dyn_onset.py, s2_detrate.py).
Run from the repository root (the working folder that also holds the data folders); see --help.
"""
import os, csv, argparse, math

# ---------------------------------------------------------------------------
# Inputs and labels
# ---------------------------------------------------------------------------
# Table names used below: T1 = Table 5, T1g = Table A3, T1p = Table 3, T3 = Table 2,
# T4 = Table A5, T5 = Table 4, T6 = Table A2. T2 and T1o are the delay and per-instant
# tables of the first submission; their revised versions come from r1_revision_SEN/code.
SCORES = (('qr', 'QR-LSTM', ('re_ii', 'results_rev', 'rev_qr_pooled.csv')),
          ('ae', 'LSTM-AE', ('re_ii', 'results_rev', 'rev_ae_pooled.csv')),
          ('tcn', 'TCN', ('re_ii', 'results_rev', 'rev_tcn_pooled.csv')),
          ('cn0', 'C/N$_0$ proxy', ('re_ii', 'results_c', 'rev_cn0_pooled.csv')),
          ('wscm', 'WSCM', ('re_ii', 'results_c', 'rev_wscm_pooled.csv')))
OLD = {'qr': ('results_v2b_1e-5', 'far_mc_v2_1e-5_lstm.csv'),
       'ae': ('results_prerun_nb1000', 'pre_ae_nb1000_lstm.csv'),
       'cn0': ('results_v2b_1e-5', 'sqm_v2_1e-5_cn0.csv'),
       'wscm': ('results_v2b_1e-5', 'sqm_v2_1e-5_wscm.csv')}
QR_ALT = ('results_prerun_nb1000', 'pre_qr_nb1000_lstm.csv')
ALPHAS = (('1e-01', 0.1), ('5e-02', 0.05), ('1e-02', 0.01))
BETAS = (0.1, 0.05, 0.02)
GROUPS = ('static', 'dynamic', 'complex')
GROUP_OF = {'ds3': 'static', 'ds4_r2': 'static', 'ds5_r2': 'dynamic', 'ds6_r2': 'dynamic', 'ds7': 'complex', 'ds8': 'complex'}
RULE_TEX = {'cusum': 'CUSUM, block $h$', 'k_blk': '$K=5$, block $\\gamma$', 'k_iid': '$K=5$, indep.\\ $\\gamma$',
            'k_p95': '$K=5$, 95th pct.'}
C1B_TEX = {'qr': 'QR-LSTM score', 'cn0_db': 'C/N$_0$ proxy (dB)', 'cn0_fold': 'C/N$_0$ proxy, folded',
           'wscm_fold': 'WSCM, folded'}
MU1_TEX = {'p1sd': '$\\mu_0+\\sigma_0$', 'p3dB': '$+3$\\,dB', 'm16dB': '$-16$\\,dB'}
ARM_TEX = {'ana': 'analytic $\\ln N_{\\mathrm{fa}}$', 'iid': 'i.i.d.\\ bootstrap', 'blk': 'block bootstrap'}
ENC_TEX = dict((k, v) for k, v, _ in SCORES)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def rows_of(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8-sig', newline='') as fh:
        return list(csv.DictReader(fh))


def flt(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


def bkey(v):
    x = flt(v)
    return None if math.isnan(x) else round(x, 6)


def fin(v):
    return v is not None and isinstance(v, float) and math.isfinite(v)


def index(rows, *cols):
    out = {}
    for r in rows or []:
        k = tuple(bkey(r[c]) if c in ('beta', 'level') else (int(flt(r[c])) if c in ('T_ms', 'K') else r[c])
                  for c in cols)
        out.setdefault(k, r)
    return out


def letter(verdict, nev, up, ref):
    if verdict == 'violated':
        return 'E'
    if verdict == 'verified':
        return 'W'
    if verdict == 'inconclusive':
        return 'U' if (nev == 0 and up > ref) else 'C'
    return '-'


def pct(v):
    if not fin(v):
        return '--'
    x = 100.0 * v
    if x == 0:
        return '0'
    return f'{x:.1f}' if x >= 1 else f'{x:.2f}'


def dly(v):
    if not fin(v):
        return '--'
    return f'{v:.3f}' if v < 1 else f'{v:.1f}'


def f(v, d=2):
    return f'{v:.{d}f}' if fin(v) else '--'


def cell(r, pre='ho'):
    """Table cell from a pooled row: (text, verdict, rate, events)."""
    if r is None:
        return '--', '-', float('nan'), 0
    fr, nev, up, ref = flt(r[f'{pre}_frac']), int(flt(r[f'{pre}_n_event']) or 0), flt(r[f'{pre}_up95']), flt(r['ref'])
    L = letter(r[f'{pre}_verdict'], nev, up, ref)
    return f'{pct(fr)} ({nev}) {L}', L, fr, nev


def latex(cap, lab, spec, head, body, note=None):
    L = ['\\begin{table}[H]', f'\\caption{{{cap}\\label{{{lab}}}}}', '\\centering', '\\small',
         f'\\begin{{tabular}}{{{spec}}}', '\\toprule']
    L += [' & '.join(h) + '\\\\' for h in head]
    L.append('\\midrule')
    L += ['\\midrule' if r == 'MID' else ' & '.join(r) + '\\\\' for r in body]
    L += ['\\bottomrule', '\\end{tabular}']
    if note:
        L += ['', f'\\noindent{{\\footnotesize {note}}}']
    L.append('\\end{table}')
    return '\n'.join(L)


def save_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f'{v:.6g}' if isinstance(v, float) else v) for k, v in r.items()})


# ---------------------------------------------------------------------------
# Values checked against the manuscript (self-test)
# ---------------------------------------------------------------------------
EXP_T1 = [('qr', 'cusum', '4.8 W', '2.5 W', '1.2 C'), ('qr', 'k_blk', '7.1 C', '2.7 W', '1.3 C'),
          ('qr', 'k_iid', '11.2 C', '7.4 E', '3.3 C'), ('ae', 'cusum', '6.8 W', '2.9 W', '1.2 C'),
          ('ae', 'k_iid', '15.4 E', '8.9 E', '4.8 E'), ('tcn', 'cusum', '5.8 W', '2.0 W', '1.2 C'),
          ('tcn', 'k_blk', '5.8 W', '1.2 W', '0.29 W'), ('tcn', 'k_iid', '16.7 E', '11.4 E', '5.4 E'),
          ('cn0', 'cusum', '9.5 C', '5.9 C', '1.2 C'), ('cn0', 'k_blk', '14.2 E', '8.2 E', '3.5 E'),
          ('cn0', 'k_iid', '30.7 E', '24.9 E', '18.7 E'), ('wscm', 'cusum', '4.1 W', '1.9 W', '0.58 W'),
          ('wscm', 'k_blk', '8.5 C', '4.6 C', '2.3 C'), ('wscm', 'k_iid', '13.9 E', '7.8 E', '4.4 E')]
EXP_DELAY = [('qr', 'cusum', 0.02, 'static', '0.163'), ('qr', 'cusum', 0.02, 'dynamic', '0.101'),
             ('qr', 'cusum', 0.02, 'complex', '13.1'), ('qr', 'k_blk', 0.02, 'dynamic', '0.103'),
             ('qr', 'k_blk', 0.02, 'complex', '18.5'), ('qr', 'k_iid', 0.02, 'dynamic', '0.046')]
EXP_C1B = [('qr', T, v) for T, v in ((1, '0.08'), (20, '0.20'), (100, '0.52'), (1000, '0.61'))] + \
          [('cn0_db', T, v) for T, v in ((1, '0.61'), (20, '0.76'), (100, '0.91'), (1000, '0.72'))]
EXP_VIF = [('qr', '12'), ('cn0_db', '260'), ('cn0_fold', '127'), ('wscm_fold', '12')]
EXP_C1P = [('ana', '31.2', '30.5', '28.7'), ('iid', '32.8', '32.0', '31.8'), ('blk', '18.9', '10.6', '4.9')]
EXP_C2 = {('qr', 3): ('1.59', '5.9', '2.9'), ('qr', 5): ('1.17', '4.2', '3.3'), ('qr', 10): ('1.03', '1.6', '3.4'),
          ('cn0_fold', 3): ('8.59', '11.1', '4.2'), ('cn0_fold', 5): ('2.74', '10.6', '5.3'),
          ('cn0_fold', 10): ('1.26', '6.9', '5.8'), ('wscm_fold', 3): ('1.43', '2.1', '1.9'),
          ('wscm_fold', 5): ('1.11', '1.7', '2.0'), ('wscm_fold', 10): ('1.02', '1.4', '2.0')}


def near(v, s):
    d = len(s.split('.')[1]) if '.' in s else 0
    tol = 0.5 * 10 ** (-d) + 1e-9
    if d == 0:
        tol = max(tol, 0.01 * abs(float(s)))
    return fin(v) and abs(v - float(s)) <= tol


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='.')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    D = os.path.abspath(a.dir)
    P = lambda *p: os.path.join(D, *p)
    prefix = os.path.splitext(a.out)[0]
    os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
    brief, tex, chk = [], [], []

    def B(s=''):
        print(s)
        brief.append(s)

    def C(name, ok, got=''):
        chk.append((name, ok, got))

    B('[version] c_tables.py v1.0.1（2026-10-25）')
    pooled = {}
    for k, lab, p in SCORES:
        rows = rows_of(P(*p))
        if rows is None:
            B(f'[WARN] 找不到 {os.path.join(*p)}')
        pooled[k] = index(rows, 'rule', 'beta', 'group')

    # ---------------- T1 / T1g
    t1 = []
    for k, lab, _ in SCORES:
        for rule in ('cusum', 'k_blk', 'k_iid'):
            for g in ('all',) + GROUPS:
                for b in BETAS:
                    r = pooled[k].get((rule, round(b, 6), g))
                    txt, L, fr, nev = cell(r)
                    t1.append(dict(score=k, rule=rule, group=g, beta=b, ho_frac=fr, ho_n_event=nev,
                                   ho_lo95=flt(r['ho_lo95']) if r else float('nan'),
                                   ho_up95=flt(r['ho_up95']) if r else float('nan'),
                                   ho_total_s=flt(r['ho_total_s']) if r else float('nan'), letter=L, text=txt))
    save_csv(prefix + '_T1.csv', t1)
    t1i = {(r['score'], r['rule'], r['group'], r['beta']): r for r in t1}
    for k, rule, *ex in EXP_T1:
        for b, e in zip(BETAS, ex):
            v, L = e.split()
            r = t1i[(k, rule, 'all', b)]
            C(f'T1 {k} {rule} β={b}', near(100 * r['ho_frac'], v) and r['letter'] == L, r['text'])
    body = []
    for k, lab, _ in SCORES:
        for j, rule in enumerate(('cusum', 'k_blk', 'k_iid')):
            body.append([lab if j == 0 else '', RULE_TEX[rule]] + [t1i[(k, rule, 'all', b)]['text'] for b in BETAS])
        body.append('MID')
    tex.append(latex('Windowed false-alarm rate on the held-out segments, pooled over 14 satellites ($\\times10^{-2}$; '
                     'alarm events in parentheses)', 'tab:T1', '@{}llccc@{}',
                     [['Score', 'Decision rule', '$\\beta=10^{-1}$', '$5\\times10^{-2}$', '$2\\times10^{-2}$']], body[:-1],
                     'W, C, E, U: within, consistent with, exceeding, unresolved against the target (Section~X).'))
    body = []
    for k, lab, _ in SCORES:
        for j, rule in enumerate(('cusum', 'k_blk', 'k_iid')):
            for i, g in enumerate(GROUPS):
                body.append([lab if (j == 0 and i == 0) else '', RULE_TEX[rule] if i == 0 else '', g]
                            + [t1i[(k, rule, g, b)]['text'] for b in BETAS])
        body.append('MID')
    tex.append(latex('Held-out windowed false-alarm rate by scenario class ($\\times10^{-2}$)', 'tab:T1g',
                     '@{}lllccc@{}', [['Score', 'Decision rule', 'Class', '$\\beta=10^{-1}$', '$5\\times10^{-2}$',
                                       '$2\\times10^{-2}$']], body[:-1]))

    # ---------------- T1p
    t1p = []
    for k, lab, _ in SCORES:
        for g in ('all',) + GROUPS:
            r = pooled[k].get(('k_p95', None, g))
            if r is None:
                continue
            ref = flt(r['ref'])
            t1p.append(dict(score=k, group=g, ref=ref, cal_frac=flt(r['cal_frac']), cal_n_event=int(flt(r['cal_n_event'])),
                            D=flt(r['cal_frac']) / ref, D_lo=flt(r['cal_lo95']) / ref, D_up=flt(r['cal_up95']) / ref,
                            ho_text=cell(r)[0], ho_frac=flt(r['ho_frac']), ho_up95=flt(r['ho_up95'])))
    save_csv(prefix + '_T1p.csv', t1p)
    t1pi = {(r['score'], r['group']): r for r in t1p}
    for k, e in (('qr', '355'), ('ae', '280')):
        r = t1pi.get((k, 'dynamic'))
        C(f'T1p {k} dynamic 标定段 D≈{e}', r is not None and abs(r['D'] - float(e)) <= 0.03 * float(e),
          f'{r["D"]:.1f}' if r else 'none')
    body = [[lab] + [f'{t1pi[(k, g)]["D"]:.0f} [{t1pi[(k, g)]["D_lo"]:.0f}, {t1pi[(k, g)]["D_up"]:.0f}]'
                     if (k, g) in t1pi else '--' for g in ('static', 'dynamic', 'complex', 'all')]
            + [t1pi[(k, 'all')]['ho_text'] if (k, 'all') in t1pi else '--'] for k, lab, _ in SCORES]
    tex.append(latex('Consecutive-count rule at the 95th percentile ($K=5$): delivered over independence reference '
                     'on the calibration segments, and pooled held-out rate', 'tab:T1p', '@{}lccccc@{}',
                     [['Score', 'Static', 'Dynamic', 'Complex', 'All', 'Held-out ($\\times10^{-2}$)']], body,
                     'Calibration-segment ratio with one-sided 95\\% bounds; reference $2.97\\times10^{-4}$ per window.'))


    t1o, old_star = [], {}
    for k, lab, _ in SCORES:
        if k not in OLD:
            t1o.append(dict(score=k, group='all', n=0))
            continue
        rows = rows_of(P(*OLD[k]))
        if rows is None:
            B(f'[WARN] 找不到 {os.path.join(*OLD[k])}')
            continue
        rows = [r for r in rows if r['scenario'] != 'ds2']
        old_star[k] = {r['star']: r for r in rows}
        for g in ('all',) + GROUPS:
            sel = [r for r in rows if g == 'all' or GROUP_OF.get(r['scenario']) == g]
            d = dict(score=k, group=g, n=len(sel))
            for s, al in ALPHAS:
                v = [flt(r[f'post_dr@{s}']) for r in sel]
                v = [x for x in v if math.isfinite(x)]
                d[f'dr_mean@{al}'] = sum(v) / len(v) if v else float('nan')
                d[f'dr_min@{al}'] = min(v) if v else float('nan')
            t1o.append(d)
    save_csv(prefix + '_T1o.csv', t1o)
    t1oi = {(r['score'], r['group']): r for r in t1o}
    for k in old_star:
        C(f'T1o {k} 剔除 ds2 后星数 = 21', len(old_star[k]) == 21, str(len(old_star[k])))
    body = []
    for k, lab, _ in SCORES:
        r = t1oi.get((k, 'all'))
        body.append([lab] + ([f'{100 * r[f"dr_mean@{al}"]:.2f} [{100 * r[f"dr_min@{al}"]:.2f}]' for _, al in ALPHAS]
                             if r and r['n'] else ['--'] * 3))
    tex.append(latex('Per-instant detection rate after onset at a nominal per-epoch false-alarm rate $\\alpha$ '
                     '(\\%; mean over 21 satellites, minimum in brackets)', 'tab:T1o', '@{}lccc@{}',
                     [['Score', '$\\alpha=10^{-1}$', '$5\\times10^{-2}$', '$10^{-2}$']], body,
                     'Earlier evaluation protocol, with its own clean window; no such run exists for the TCN.'))


    alt = rows_of(P(*QR_ALT))
    if alt is not None and 'qr' in old_star:
        nm = nd = 0
        for r in alt:
            o = old_star['qr'].get(r['star'])
            if o is None:
                continue
            ok = all(abs(flt(r[f'post_dr@{s}']) - flt(o[f'post_dr@{s}'])) < 1e-4 for s, _ in ALPHAS)
            nm += ok
            nd += not ok
        C('②QR 旧标准 far_mc_v2 与 pre_qr_nb1000 的 post_dr 逐星一致', nd == 0 and nm == 21, f'{nm} 一致 / {nd} 不一致')


    t2 = []
    for k, lab, _ in SCORES:
        for rule in ('cusum', 'k_blk', 'k_iid', 'k_p95'):
            for b in (BETAS if rule != 'k_p95' else (None,)):
                for g in ('all',) + GROUPS:
                    r = pooled[k].get((rule, None if b is None else round(b, 6), g))
                    if r is None:
                        continue
                    t2.append(dict(score=k, rule=rule, beta=b if b is not None else '', group=g,
                                   n_det=int(flt(r['n_det'])), n_miss=int(flt(r['n_miss'])), n_stars=int(flt(r['n_stars'])),
                                   med=flt(r['delay_onset_med']), mx=flt(r['delay_onset_max']),
                                   act_med=flt(r['delay_act_med']), act_max=flt(r['delay_act_max'])))
    save_csv(prefix + '_T2.csv', t2)
    t2i = {(r['score'], r['rule'], r['beta'], r['group']): r for r in t2}
    for k, rule, b, g, e in EXP_DELAY:
        r = t2i.get((k, rule, b, g))
        C(f'T2 {k} {rule} β={b} {g} 时延中位 {e}', r is not None and near(r['med'], e), dly(r['med']) if r else 'none')
    body = []
    for k, lab, _ in SCORES:
        for j, rule in enumerate(('cusum', 'k_blk', 'k_iid', 'k_p95')):
            bb = 0.02 if rule != 'k_p95' else ''
            body.append([lab if j == 0 else '', RULE_TEX[rule]]
                        + [f'{dly(t2i[(k, rule, bb, g)]["med"])} [{dly(t2i[(k, rule, bb, g)]["mx"])}]'
                           if (k, rule, bb, g) in t2i else '--' for g in GROUPS])
        body.append('MID')
    nmiss = sum(r['n_miss'] for r in t2 if r['group'] == 'all')
    tex.append(latex('Detection delay after the published onset at $\\beta=2\\times10^{-2}$ (s; median [maximum])',
                     'tab:T2', '@{}llccc@{}', [['Score', 'Decision rule', 'Static', 'Dynamic', 'Complex']], body[:-1],
                     f'Missed attacks over all rows: {nmiss}.'))

    # ---------------- T3 C1b
    c1b = index(rows_of(P('re_ii', 'results_c', 'c1p_c1b_pooled.csv')), 'score', 'T_ms', 'group')
    t3 = []
    for sc in C1B_TEX:
        for T in (1, 20, 100, 1000):
            ra = c1b.get((sc, T, 'all'))
            if ra is None:
                continue
            d = dict(score=sc, T_ms=T, vif=flt(ra['vif_med']), D_b05=flt(ra['b05_D']), b05_nev=int(flt(ra['b05_nev'])),
                     b05_letter=letter(ra['b05_verdict'], int(flt(ra['b05_nev'])), flt(ra['b05_up95']), flt(ra['b05_nom'])),
                     L_rule=flt(ra['L_rule_med']), N_blocks=flt(ra['N_blocks_med']))
            for g in ('all',) + GROUPS:
                rg = c1b.get((sc, T, g))
                d[f'acf1_{g}'] = flt(rg['acf1_med']) if rg else float('nan')
            t3.append(d)
    save_csv(prefix + '_T3.csv', t3)
    t3i = {(r['score'], r['T_ms']): r for r in t3}
    for sc, T, e in EXP_C1B:
        r = t3i.get((sc, T))
        C(f'T3 {sc} T={T} ρ1 中位 {e}', r is not None and near(r['acf1_all'], e), f(r['acf1_all']) if r else 'none')
    for sc, e in EXP_VIF:
        r = t3i.get((sc, 1000))
        C(f'T3 {sc} T=1000 VIF {e}', r is not None and near(r['vif'], e), f(r['vif'], 1) if r else 'none')
    body = []
    for sc, lab in C1B_TEX.items():
        for j, T in enumerate((1, 20, 100, 1000)):
            r = t3i.get((sc, T))
            if r:
                body.append([lab if j == 0 else '', str(T), f(r['acf1_all']), f(r['acf1_static']), f(r['acf1_dynamic']),
                             f(r['vif'], 1) if T > 1 else '1', f'{f(r["D_b05"], 1)} {r["b05_letter"]}'])
        body.append('MID')
    tex.append(latex('Serial correlation of non-overlapping block means on the calibration segments (medians over 19 '
                     'satellites)', 'tab:T3', '@{}lrccccc@{}',
                     [['Series', '$T$ (ms)', '$\\rho_1$ all', 'static', 'dynamic', 'VIF', '$D$ at $0.05$']], body[:-1],
                     'VIF: variance of block means over $\\sigma^2/T$. $D$: consecutive-count rule ($K=5$) delivered over '
                     'its independence reference of $0.05$ per one-second window.'))


    c1p = index(rows_of(P('re_ii', 'results_c', 'c1p_c1p_pooled.csv')), 'T_ms', 'mu1', 'arm', 'beta', 'group')
    t4 = []
    for T in (20, 1000):
        for mu in MU1_TEX:
            for arm in ARM_TEX:
                for b in BETAS:
                    r = c1p.get((T, mu, arm, round(b, 6), 'all'))
                    if r is None:
                        continue
                    nev = int(flt(r['ho_n_event']) or 0)
                    t4.append(dict(T_ms=T, mu1=mu, arm=arm, beta=b, ho_frac=flt(r['ho_frac']), ho_n_event=nev,
                                   letter=letter(r['ho_verdict'], nev, flt(r['ho_up95']), b), h_med=flt(r['h_med']),
                                   n_degenerate=int(flt(r['n_degenerate'])), n_det=int(flt(r['n_det'])),
                                   n_stars=int(flt(r['n_stars'])), onset_med=flt(r['delay_onset_med']),
                                   ho_blocks=flt(r['ho_blocks_total'])))
    save_csv(prefix + '_T4.csv', t4)
    t4i = {(r['T_ms'], r['mu1'], r['arm'], r['beta']): r for r in t4}
    for arm, *ex in EXP_C1P:
        for b, e in zip(BETAS, ex):
            r = t4i.get((20, 'p1sd', arm, b))
            C(f'T4 T=20 p1sd {arm} β={b} {e}%', r is not None and near(100 * r['ho_frac'], e),
              pct(r['ho_frac']) if r else 'none')
    body = []
    for T in (20, 1000):
        for mu in MU1_TEX:
            for j, arm in enumerate(ARM_TEX):
                rs = [t4i.get((T, mu, arm, b)) for b in BETAS]
                r5, r2 = rs[1], rs[2]
                body.append([f'{T}' if (mu == 'p1sd' and j == 0) else '', MU1_TEX[mu] if j == 0 else '', ARM_TEX[arm]]
                            + [f'{pct(r["ho_frac"])} ({r["ho_n_event"]}) {r["letter"]}' if r else '--' for r in rs]
                            + [f(r5['h_med'], 2) if r5 else '--', str(r5['n_degenerate']) if r5 else '--',
                               f'{r2["n_det"]}/{r2["n_stars"]}' if r2 else '--', dly(r2['onset_med']) if r2 else '--'])
            body.append('MID')
    tex.append(latex('Gaussian log-likelihood-ratio CUSUM on block means of the C/N$_0$ proxy: pooled held-out rate '
                     '($\\times10^{-2}$) under three ways of setting $h$', 'tab:T4', '@{}lllccccccc@{}',
                     [['$T_{\\mathrm{sp}}$ (ms)', '$\\mu_1$', 'Threshold', '$\\beta=10^{-1}$', '$5\\times10^{-2}$',
                       '$2\\times10^{-2}$', '$h$', 'Degen.', 'Detected', 'Delay (s)']], body[:-1],
                     '$h$ and degenerate count at $\\beta=5\\times10^{-2}$; detections and median onset delay at '
                     '$\\beta=2\\times10^{-2}$. Degenerate: the statistic never rises above zero on the calibration segment.'))

    # ---------------- T5 C2
    c2a = index(rows_of(P('re_ii', 'results_c', 'c2_k3510_pooled.csv')), 'score', 'K', 'mode', 'level', 'group')
    c2b = index(rows_of(P('re_ii', 'results_c', 'c2_pooled.csv')), 'score', 'K', 'mode', 'level', 'group')
    c2 = c2a or c2b
    t5 = []
    for sc in ('qr', 'cn0_fold', 'wscm_fold'):
        for K in (3, 5, 10):
            for g in ('all',) + GROUPS:
                r = c2.get((sc, K, 'beta', 0.05, g))
                if r is None:
                    continue
                t5.append(dict(score=sc, K=K, group=g, r_over_p=flt(r['r_over_p_med']), pred_exact=flt(r['pred_exact']),
                               pred_approx=flt(r['pred_approx_med']), D=flt(r['D']), D_lo=flt(r['D_lo']),
                               D_up=flt(r['D_up']), verdict=r['verdict'], resolvable=r['resolvable_3q']))
                rb = c2b.get((sc, K, 'beta', 0.05, g)) if c2b else None
                if g == 'all' and rb is not None and c2a:
                    C(f'③c2 与 c2_k3510 一致 {sc} K={K}', abs(flt(rb['D']) - flt(r['D'])) < 1e-6,
                      f'{flt(rb["D"]):.3f} vs {flt(r["D"]):.3f}')
    save_csv(prefix + '_T5.csv', t5)
    t5i = {(r['score'], r['K'], r['group']): r for r in t5}
    for (sc, K), (rp, pe, dd) in EXP_C2.items():
        r = t5i.get((sc, K, 'all'))
        C(f'T5 {sc} K={K}', r is not None and near(r['r_over_p'], rp) and near(r['pred_exact'], pe) and near(r['D'], dd),
          f'{f(r["r_over_p"])}/{f(r["pred_exact"], 1)}/{f(r["D"], 1)}' if r else 'none')
    body = []
    for sc in ('qr', 'cn0_fold', 'wscm_fold'):
        for j, K in enumerate((3, 5, 10)):
            r = t5i.get((sc, K, 'all'))
            if r:
                body.append([C1B_TEX[sc] if j == 0 else '', str(K), f(r['r_over_p']), f(r['pred_exact'], 1),
                             f'{f(r["D"], 1)} [{f(r["D_lo"], 1)}, {f(r["D_up"], 1)}]',
                             {'violated': 'E', 'inconclusive': 'C', 'verified': 'W'}.get(r['verdict'], '-')])
        body.append('MID')
    tex.append(latex('Excess of the consecutive-count rule set for $\\beta=5\\times10^{-2}$ under independence: '
                     'two-state Markov prediction against measurement on the calibration segments', 'tab:T5',
                     '@{}lrcccc@{}', [['Score', '$K$', '$r/p$', 'Predicted $D$', 'Measured $D$ [95\\%]', '']], body[:-1],
                     '$r$: probability that an exceedance follows an exceedance; $p$: marginal exceedance probability '
                     '(medians over satellites).'))

    # ---------------- T6 C3
    c3 = (rows_of(P('re_ii', 'results_c', 'c3_pooled.csv')) or []) + (rows_of(P('re_ii', 'results_c', 'c3_tcn_pooled.csv')) or [])
    c3i = index(c3, 'encoder', 'rule', 'beta', 'group')
    t6 = []
    for k, lab, _ in SCORES:
        for rule in ('cusum', 'k_blk', 'k_iid', 'k_p95'):
            for b in (BETAS if rule != 'k_p95' else (None,)):
                r = c3i.get((k, rule, None if b is None else round(b, 6), 'all'))
                if r is None:
                    continue
                nev = int(flt(r['n_event']) or 0)
                t6.append(dict(encoder=k, rule=rule, beta=b if b is not None else '', frac=flt(r['frac']), n_event=nev,
                               ratio=flt(r['ratio_union_single']), ref_union=flt(r['ref_union']),
                               Ls=letter(r['verdict_single'], nev, flt(r['up95']), flt(r['decl_single'])),
                               Lu=letter(r['verdict_union'], nev, flt(r['up95']), flt(r['ref_union']))))
    save_csv(prefix + '_T6.csv', t6)
    t6i = {(r['encoder'], r['rule'], r['beta']): r for r in t6}
    for k, _, _ in SCORES:
        vs = [t6i[(k, 'cusum', b)]['Lu'] for b in BETAS if (k, 'cusum', b) in t6i]
        C(f'T6 {k} cusum 对 1−(1−β)^n 无 E', len(vs) == 3 and 'E' not in vs, ''.join(vs))
    body = []
    for k, lab, _ in SCORES:
        for j, rule in enumerate(('cusum', 'k_blk', 'k_iid')):
            body.append([lab if j == 0 else '', RULE_TEX[rule]]
                        + [f'{pct(r["frac"])} ({r["ratio"]:.1f}) {r["Ls"]}/{r["Lu"]}' if r else '--'
                           for r in (t6i.get((k, rule, b)) for b in BETAS)])
        body.append('MID')
    tex.append(latex('Receiver-level false-alarm rate: union of alarms over the satellites of each recording, '
                     'pooled over four recordings ($\\times10^{-2}$)', 'tab:T6', '@{}llccc@{}',
                     [['Score', 'Decision rule', '$\\beta=10^{-1}$', '$5\\times10^{-2}$', '$2\\times10^{-2}$']], body[:-1],
                     'Parentheses: union over mean single-satellite rate. Letters: verdict against the single-satellite '
                     '$\\beta$ / against $1-(1-\\beta)^n$.'))

    with open(prefix + '_tables.tex', 'w', encoding='utf-8') as fh:
        fh.write('\n\n'.join(tex) + '\n')

    # ---------------- brief
    nok = sum(ok for _, ok, _ in chk)
    B(f'== 自检：{nok}/{len(chk)} 通过' + ('' if nok == len(chk) else '；不通过的项：'))
    for name, ok, got in chk:
        if not ok:
            B(f'  [DIFF] {name}  本次={got}')
    for name, ok, got in chk:
        if name.startswith(('②', '③', 'T1o')):
            B(f'  {name}：{"OK" if ok else "DIFF"}  {got}')
    B('\n== 新数字 1：AE 主误报（留出段合并，×10⁻²）==')
    for rule in ('cusum', 'k_blk', 'k_iid'):
        B(f'  AE {rule:6s} ' + ' | '.join(t1i[("ae", rule, "all", b)]['text'] for b in BETAS))
    B('\n== 新数字 2：k_p95 标定段 D（静/动/复/全）与留出段 ==')
    for k, lab, _ in SCORES:
        B(f'  {k:5s} ' + '  '.join(f'{t1pi[(k, g)]["D"]:7.1f}' if (k, g) in t1pi else '      -'
                                   for g in ('static', 'dynamic', 'complex', 'all'))
          + f'   ho {t1pi[(k, "all")]["ho_text"] if (k, "all") in t1pi else "-"}')
    B('\n== 新数字 3：旧标准逐时刻检测率（%，21 颗星均值 [最小]；α = 0.1 / 0.05 / 0.01）==')
    for k, lab, _ in SCORES:
        r = t1oi.get((k, 'all'))
        B(f'  {k:5s} ' + ('  '.join(f'{100 * r[f"dr_mean@{al}"]:6.2f} [{100 * r[f"dr_min@{al}"]:6.2f}]' for _, al in ALPHAS)
                          if r and r['n'] else '—'))
        for g in GROUPS:
            rg = t1oi.get((k, g))
            if rg and rg['n']:
                B(f'        {g:8s} ' + '  '.join(f'{100 * rg[f"dr_mean@{al}"]:6.2f}' for _, al in ALPHAS))
    B('\n== 新数字 4：时延 β=0.02，onset 中位 [最大] / act 中位 [最大]（s；静/动/复）==')
    for k, lab, _ in SCORES:
        for rule in ('cusum', 'k_blk'):
            B(f'  {k:5s} {rule:6s} ' + '  '.join(
                f'{dly(t2i[(k, rule, 0.02, g)]["med"])} [{dly(t2i[(k, rule, 0.02, g)]["mx"])}] / '
                f'{dly(t2i[(k, rule, 0.02, g)]["act_med"])} [{dly(t2i[(k, rule, 0.02, g)]["act_max"])}]'
                if (k, rule, 0.02, g) in t2i else '-' for g in GROUPS))
    B(f'  漏检合计 {nmiss}')
    B('\n== 新数字 5：接收机级（单星判定/并集判定；β = 0.1 / 0.05 / 0.02）==')
    for k, lab, _ in SCORES:
        B(f'  {k:5s} ' + ' | '.join(f'{rule} ' + ' '.join(f'{t6i[(k, rule, b)]["Ls"]}/{t6i[(k, rule, b)]["Lu"]}'
                                                          if (k, rule, b) in t6i else '-/-' for b in BETAS)
                                    for rule in ('cusum', 'k_blk', 'k_iid')))
    B(f'\n[saved] {prefix}_T1/_T1p/_T1o/_T2/_T3/_T4/_T5/_T6.csv、{prefix}_tables.tex（{len(tex)} 张表）、{prefix}_brief.txt')
    with open(prefix + '_brief.txt', 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(brief) + '\n')
