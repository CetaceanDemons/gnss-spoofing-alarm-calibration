# -*- coding: utf-8 -*-
"""Robustness of the dynamic-scenario results to the assumed activation instant.
Paper: Appendix D and Table A4; summarised in Section 4.6.

The activation of the spoofer in ds5/ds6 is not documented (Appendix D). In the pipeline the
activation instant t_a enters in two places only, through far_calib_v32.segments():
  - the held-out segment ends 0.1 s before t_a (ds6; ds5 has none);
  - the calibration segment ends 0.1 s before t_a when the clean-labelled span runs past it (ds5).
Delays are counted from the onset (Table 1), not from t_a.

For every dynamic satellite and score, t_a is moved over a grid from the event near 97 s to the
onset. The pipeline of rev_main.run_star() is rerun unchanged whenever the calibration segment
changes (new limits); otherwise the existing limits are re-evaluated on the new segments with
rev_main.evaluate(). For each t_a the script records the limits, the delays from the onset, the
delays with the onset reference moved by -/+ --shift seconds, the first alarm at or after t_a,
the alarms between t_a and the onset, and the held-out reading.
The first grid point is the activation instant of Table 1; its rows must reproduce
r1_revision_SEN/s1/rev_<score>_star.csv.

Run from the repository root (J:\\Database\\TEXBAT\\processed_Mao):
    python r1_revision_SEN/code/s8_dyn_activation_scan.py                        (all, ~1 h)
    python r1_revision_SEN/code/s8_dyn_activation_scan.py --scores qr --stars ds5_r2_prn9   (timing test)
Outputs: r1_revision_SEN/s8/ (s8_rows.csv, s8_brief.txt, s8_log.txt).
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

VERSION = 's8_dyn_activation_scan.py v1.0 (2026-11-23)'
HERE = os.path.dirname(os.path.abspath(__file__))
DEF_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))

SCORES = ('qr', 'ae', 'tcn', 'cn0', 'wscm')
DYN = ('ds5_r2', 'ds6_r2')
# Alternative activation instants (s); the instant of Table 1 is always evaluated first.
GRID = {'ds5_r2': (97.5, 98.5, 99.5, 100.5, 101.5, 102.1),
        'ds6_r2': (97.5, 98.5, 99.5, 100.5, 101.5, 102.5, 103.5, 104.4)}
RULES = ('cusum', 'k_blk', 'k_iid', 'k_p95')
VERD = {'verified': 'W', 'violated': 'E', 'inconclusive': 'C', 'too_short': '-'}


def fl(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


def read(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8-sig', newline='') as fh:
        return list(csv.DictReader(fh))


def setup(root):
    for p in (root, os.path.join(root, 're_ii', 'code')):
        if p not in sys.path:
            sys.path.insert(0, p)
    import far_calib_v32 as fc       # noqa: E402
    import rev_main as rm            # noqa: E402
    import rev_sqm as rq             # noqa: E402
    return fc, rm, rq


# ---------------------------------------------------------------------------
# Alarm function of a calibrated row (as built inside rev_main.run_star)
# ---------------------------------------------------------------------------
def arrays(S, seg):
    cm = seg['cmask']
    Sc = np.ascontiguousarray(S[cm])
    mu, sd = float(Sc.mean()), float(Sc.std())
    Z = np.ascontiguousarray((S - mu) / sd)
    Z32 = np.ascontiguousarray(Z.astype(np.float32).astype(np.float64))
    return dict(S=S, Sc=Sc, Z=Z, Zc=np.ascontiguousarray(Z[cm]), Z32=Z32, Zc32=np.ascontiguousarray(Z32[cm]))


def rule_fn(rm, fc, r, A, K):
    """Alarm function, full series, calibration series and reference rate of one row."""
    rule = r['rule']
    if rule == 'cusum':
        h = fl(r.get('thr_z'))
        if not np.isfinite(h):
            return None
        return (lambda X, _h=h: fc.alarm_idx(X, rm.K0, _h)), A['Z'], A['Zc'], fl(r['beta'])
    if rule == 'k_blk':
        g = fl(r['thr_z'])
        return (lambda X, _g=g: rm.kr.krun_idx(X, _g, K)), A['Z32'], A['Zc32'], fl(r['beta'])
    g = fl(r['thr'])   # k_iid, k_p95: threshold on the raw score
    return (lambda X, _g=g: rm.kr.krun_idx(X, _g, K)), A['S'], A['Sc'], fl(r['nominal_iid'])


def first_at_or_after(idx, i0):
    j = int(np.searchsorted(idx, i0, side='left'))
    return int(idx[j]) if j < len(idx) else None


def extras(rm, fc, r, A, seg, scen, K, shift):
    """Delays with the onset reference moved by -/+ shift seconds."""
    f = rule_fn(rm, fc, r, A, K)
    out = dict(delay_onset_m_shift_s=float('nan'), delay_onset_p_shift_s=float('nan'))
    if f is None:
        return out
    fn, X, _, _ = f
    full = np.asarray(fn(X), dtype=np.int64)
    raw = seg['raw']
    for tag, s in (('m', -shift), ('p', +shift)):
        i_ref = seg['oi'] + int(round(s * 1000))
        ia = first_at_or_after(full, i_ref)
        out[f'delay_onset_{tag}_shift_s'] = (raw(ia) - (rm.AUTH[scen] + s)) if ia is not None else float('nan')
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--root', default=DEF_ROOT, help='processed_Mao 根目录（默认为脚本上两级）')
    ap.add_argument('--scores', nargs='+', default=list(SCORES), choices=list(SCORES))
    ap.add_argument('--stars', nargs='+', default=None, help='只跑这些星，例 ds5_r2_prn9')
    ap.add_argument('--betas', nargs='+', type=float, default=[0.1, 0.05, 0.02])
    ap.add_argument('--shift', type=float, default=0.13, help='onset 参考前后移动的秒数')
    ap.add_argument('--K', type=int, default=5)
    ap.add_argument('--nboot', type=int, default=2000)
    ap.add_argument('--seg', type=float, default=80.0)
    ap.add_argument('--m', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--guard-s', dest='guard_s', type=float, default=0.1)
    ap.add_argument('--out-dir', default=None, help=r'默认 <root>\r1_revision_SEN\s8')
    ap.add_argument('--force-recal', action='store_true', help='每个开启时刻都完整重跑（核对"门限沿用"用，慢）')
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    out = os.path.abspath(a.out_dir or os.path.join(root, 'r1_revision_SEN', 's8'))
    os.makedirs(out, exist_ok=True)
    os.chdir(root)
    fc, rm, rq = setup(root)
    import sqm_beta_neweval_v2 as sq   # noqa: E402
    a.pertap_map, a.e_idx, a.l_idx = {}, sq.DEFAULT_E_IDX, sq.DEFAULT_L_IDX
    log = fc.Tee(os.path.join(out, 's8_log.txt'))
    log(f'[version] {VERSION}')
    log(f'[numba] {"ON" if rm.HAVE_NUMBA else "OFF（会很慢）"}')
    rm.BETAS_ALL = list(a.betas)
    rm.BETAS_MAIN = list(a.betas)
    ACT0 = dict(fc.ACTIVATION_RAW)
    log(f'[cfg] β = {a.betas}  K = {a.K}  nboot = {a.nboot}  seg = {a.seg} s  seed = {a.seed}  guard = {a.guard_s} s  '
        f'shift = ±{a.shift} s')
    log(f'[cfg] 表 1 开启时刻 ds5 {ACT0["ds5_r2"]} / ds6 {ACT0["ds6_r2"]}；onset {rm.AUTH["ds5_r2"]} / {rm.AUTH["ds6_r2"]}')
    log(f'[cfg] 备选开启时刻：ds5 {GRID["ds5_r2"]}；ds6 {GRID["ds6_r2"]}')
    gidx = int(round(a.guard_s * 1000))
    load_orig = rm.load_any
    allrows, regress, t_all = [], [], time.time()

    for k in a.scores:
        # ---- score files of the dynamic satellites (as in rev_main / rev_sqm)
        if k in ('cn0', 'wscm'):
            rm.load_any = rq.make_loader(k, a, log, {})
            items = rm.qr_items(root, a.stars)
        else:
            rm.load_any = load_orig
            if k == 'qr':
                items = rm.qr_items(root, a.stars)
            else:
                sub, s1, s2 = rm.DEFAULT_DUAL[k]
                items = rm.dual_items(os.path.join(root, sub, s1), os.path.join(root, sub, s2), a.stars, log)
        items = [it for it in items if it[0].split('_prn')[0] in DYN]
        log(f'\n######## {k}: {len(items)} 颗动态星 ' + ', '.join(b for b, _, _ in items))
        ref = {}
        for r in read(os.path.join(root, 'r1_revision_SEN', 's1', f'rev_{k}_star.csv')) or []:
            ref[(r['star'], r['rule'], round(fl(r['beta']), 12) if r['rule'] != 'k_p95' else None)] = r

        for item in items:
            base, path, _ = item
            scen = base.split('_prn')[0]
            cache = {}
            for t_a in (ACT0[scen],) + GRID[scen]:
                fc.ACTIVATION_RAW[scen] = t_a          # same dict object as rev_main.ACT
                S, lab, _, _ = rm.load_any(path, k)
                S = np.ascontiguousarray(np.asarray(S, dtype=np.float64))
                seg = fc.segments(S, lab, scen, gidx)
                raw = seg['raw']
                key = (seg['c0'], seg['c1'])
                t0 = time.time()
                if key not in cache or a.force_recal:
                    rows, _, _ = rm.run_star(item, a, log, {'cal': {}, 'k': {}}, [], k)
                    cache[key] = [dict(r) for r in rows]
                    recal = True
                else:
                    recal = False
                A = arrays(S, seg)
                new = []
                for r0 in cache[key]:
                    if r0.get('rule') not in RULES:
                        continue
                    r = dict(r0)
                    if not recal:
                        f = rule_fn(rm, fc, r0, A, a.K)
                        r.update(ho_raw_end_s=raw(seg['hb_act']), act_raw_s=t_a)
                        if f is not None:
                            fn, X, Xc, rr = f
                            r.update(rm.evaluate(fn, X, Xc, seg, rr, a.m, scen))
                    r.update(extras(rm, fc, r, A, seg, scen, a.K, a.shift))
                    r.update(score=k, t_a=t_a, recal=recal, baseline=(t_a == ACT0[scen]),
                             cal_end_raw_s=raw(seg['c1']), ho_end_raw_s=raw(seg['hb_act']),
                             ho_len_s_now=(seg['hb_act'] - seg['ho_a']) / 1000.0)
                    new.append(r)
                    # Regression at the activation instant of Table 1
                    if t_a == ACT0[scen]:
                        kb = (base, r['rule'], round(fl(r['beta']), 12) if r['rule'] != 'k_p95' else None)
                        q = ref.get(kb)
                        if q is None:
                            regress.append((k, base, r['rule'], r['beta'], 'no_ref'))
                        else:
                            a1, b1 = fl(r.get('thr_z')), fl(q.get('thr_z'))
                            ok_h = (np.isnan(a1) and np.isnan(b1)) or abs(a1 - b1) <= 1e-4 * max(abs(a1), abs(b1), 1e-9)
                            d1, d2 = fl(r.get('delay_onset_s')), fl(q.get('delay_onset_s'))
                            ok_d = (np.isnan(d1) and np.isnan(d2)) or abs(d1 - d2) <= 5e-4
                            ok_e = int(fl(r.get('ho_n_event'))) == int(fl(q.get('ho_n_event')))
                            regress.append((k, base, r['rule'], r['beta'],
                                            'match' if (ok_h and ok_d and ok_e) else
                                            f'MISMATCH thr_z {a1:.6g}/{b1:.6g} delay {d1}/{d2} ho_ev '
                                            f'{r.get("ho_n_event")}/{q.get("ho_n_event")}'))
                allrows += new
                log(f'  [{k} {base}] t_a = {t_a:6.1f} s  标定段到 {raw(seg["c1"]):.2f} s  '
                    f'{"重新标定" if recal else "门限沿用"}  {time.time() - t0:.1f} s')
            fc.ACTIVATION_RAW[scen] = ACT0[scen]
    rm.load_any = load_orig

    # ---- rows
    keys = []
    for r in allrows:
        for kk in r:
            if kk not in keys:
                keys.append(kk)
    with open(os.path.join(out, 's8_rows.csv'), 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        w.writerows(allrows)

    # ---- brief
    B = [f'[{VERSION}]  总耗时 {(time.time() - t_all) / 60:.1f} min',
         f'β = {a.betas}；onset：ds5 {rm.AUTH["ds5_r2"]} s，ds6 {rm.AUTH["ds6_r2"]} s；表 1 开启时刻：ds5 {ACT0["ds5_r2"]}，ds6 {ACT0["ds6_r2"]}']
    nm = sum(1 for x in regress if x[4] == 'match')
    bad = [x for x in regress if x[4] != 'match']
    B.append(f'== 0. 表 1 开启时刻处与 S1（rev_*_star.csv）的核对：{nm} 行一致，{len(bad)} 行不一致或无参照 ==')
    for x in bad[:40]:
        B.append(f'  {x[0]:5s} {x[1]:14s} {x[2]:6s} β={x[3]}  {x[4]}')

    def sel(score, rule, beta=None, scen=None, t_a=None):
        out_ = []
        for r in allrows:
            if r['score'] != score or r['rule'] != rule:
                continue
            if beta is not None and abs(fl(r['beta']) - beta) > 1e-12:
                continue
            if scen is not None and not r['star'].startswith(scen):
                continue
            if t_a is not None and r['t_a'] != t_a:
                continue
            out_.append(r)
        return out_

    def md(v):
        v = np.asarray([x for x in v if np.isfinite(x)])
        return (f'{np.median(v):.3f}[{np.max(v):.3g}]' if len(v) else '—')

    B.append('\n== 1. ds5：标定段终点与 β=0.02 门限（thr_z 与表 1 开启时刻之比的中位数 [最小, 最大]；'
             'thr_z 是标准化单位，标准化本身随标定段变，只作参考）==')
    for k in a.scores:
        for rule in ('cusum', 'k_blk'):
            line = []
            for i, t_a in enumerate((ACT0['ds5_r2'],) + GRID['ds5_r2']):
                rr = sel(k, rule, 0.02, 'ds5_r2', t_a)
                b0 = {r['star']: fl(r['thr_z']) for r in sel(k, rule, 0.02, 'ds5_r2', ACT0['ds5_r2'])}
                rat = [fl(r['thr_z']) / b0[r['star']] for r in rr if r['star'] in b0 and b0[r['star']] != 0]
                ce = sorted(set(round(fl(r['cal_end_raw_s']), 2) for r in rr))
                line.append(f't_a {t_a}: 终点 {ce[0] if ce else "—"} 比值 '
                            + (f'{np.median(rat):.2f}[{min(rat):.2f},{max(rat):.2f}]' if rat else '—'))
            B.append(f'  {k:5s} {rule:6s} ' + ' | '.join(line))

    for rule in RULES:
        bt = None if rule == 'k_p95' else 0.02
        B.append(f'\n== 2. onset 起算时延，{rule}{"" if bt is None else " β=0.02"}：中位[最大]（s）；列 = 开启时刻 ==')
        B.append('  分数   范围  ' + '  '.join(f'{t:>14}' for t in ['表1'] + [f'{x}' for x in GRID['ds6_r2']]))
        for k in a.scores:
            for scope, scn in (('10星', None), ('ds5', 'ds5_r2'), ('ds6', 'ds6_r2')):
                cells = []
                for t6 in (None,) + GRID['ds6_r2']:
                    rr = []
                    for sc in DYN:
                        if scn is not None and sc != scn:
                            continue
                        if t6 is None:
                            ta = ACT0[sc]
                        else:
                            g = GRID[sc]
                            ta = t6 if t6 in g else (max([x for x in g if x <= t6]) if any(x <= t6 for x in g) else None)
                            if sc == 'ds5_r2' and t6 > max(g):
                                ta = max(g)
                        if ta is None:
                            continue
                        rr += sel(k, rule, bt, sc, ta)
                    cells.append(md([fl(r['delay_onset_s']) for r in rr]))
                B.append(f'  {k:5s} {scope:4s} ' + '  '.join(f'{c:>14}' for c in cells))
    B.append('  注：ds5 的网格到 102.1 s（onset）为止，比它晚的列沿用 102.1 s 的结果。')

    B.append(f'\n== 3. onset 参考移动 ±{a.shift} s：CUSUM β=0.02，表 1 开启时刻，10 颗星中位[最大]（s）==')
    for k in a.scores:
        rr = [r for sc in DYN for r in sel(k, 'cusum', 0.02, sc, ACT0[sc])]
        B.append(f'  {k:5s} −{a.shift}: {md([fl(r["delay_onset_m_shift_s"]) for r in rr])}   '
                 f'0: {md([fl(r["delay_onset_s"]) for r in rr])}   +{a.shift}: {md([fl(r["delay_onset_p_shift_s"]) for r in rr])}')
    for k in a.scores:
        rr = [r for sc in DYN for r in sel(k, 'k_blk', 0.02, sc, ACT0[sc])]
        B.append(f'  {k:5s} k_blk −{a.shift}: {md([fl(r["delay_onset_m_shift_s"]) for r in rr])}   '
                 f'0: {md([fl(r["delay_onset_s"]) for r in rr])}   +{a.shift}: {md([fl(r["delay_onset_p_shift_s"]) for r in rr])}')

    B.append('\n== 4. ds6 留出段（80 s 到 t_a−0.1 s）合并读数：比率×10⁻² (事件) 判定；β = 0.1 / 0.05 / 0.02 ==')
    for k in a.scores:
        for rule in ('cusum', 'k_blk'):
            cells = []
            for t_a in (ACT0['ds6_r2'],) + GRID['ds6_r2']:
                cc = []
                for beta in a.betas:
                    rr = sel(k, rule, beta, 'ds6_r2', t_a)
                    nev = int(sum(fl(r.get('ho_n_event')) for r in rr))
                    cov = int(sum(fl(r.get('ho_cov')) for r in rr))
                    nwin = int(sum(fl(r.get('ho_nwin')) for r in rr))
                    lo, up = fc.win_rate_bounds(nev, cov, nwin, a.m)
                    v = VERD.get(fc.verdict(lo, up, beta, nwin), '?')
                    if nev == 0 and v == 'C':
                        v = 'U'
                    cc.append(f'{100 * cov / nwin:.1f}({nev}){v}' if nwin > 0 else '—')
                cells.append(f't_a {t_a}: ' + '/'.join(cc))
            B.append(f'  {k:5s} {rule:6s} ' + ' | '.join(cells))

    B.append('\n== 5. t_a 到 onset 之间的报警（CUSUM β=0.02，各星合计）与 t_a 后首报（s，中位）==')
    for k in a.scores:
        cells = []
        for t6 in (None,) + GRID['ds6_r2']:
            rr = []
            for sc in DYN:
                ta = ACT0[sc] if t6 is None else (t6 if t6 in GRID[sc] else None)
                if ta is not None:
                    rr += sel(k, 'cusum', 0.02, sc, ta)
            n = int(sum(fl(r.get('n_alarm_act_onset')) for r in rr if np.isfinite(fl(r.get('n_alarm_act_onset')))))
            da = [fl(r.get('delay_act_s')) for r in rr]
            cells.append(f'{"表1" if t6 is None else t6}: {n} 次, 首报 {md(da)}')
        B.append(f'  {k:5s} ' + ' | '.join(cells))
    B.append(f'\n[saved] {out}')
    with open(os.path.join(out, 's8_brief.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(B) + '\n')
    log('\n'.join(B))
    log.close()


if __name__ == '__main__':
    main()
