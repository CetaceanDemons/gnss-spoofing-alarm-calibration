# -*- coding: utf-8 -*-
"""Calibration and verification on TEXBAT cleanStatic and OAKBAT cleanDynamic (Table 6).

Arm A calibrates on 80 s, arm B on 200 s; verification on the rest of each recording,
lower targets, and transfer of CUSUM limits between the two receivers. Uses
rev_main.run_star with the segmentation of the long recordings.
Run from the repository root (the working folder that also holds the data folders); see --help.
"""
import argparse
import csv
import glob
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEF_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
W = 100
LONG = ('TEX_cS', 'OAK_cD')
GROUP = {'TEX_cS': 'long_tex', 'OAK_cD': 'long_oak'}
GROUPS = ('all', 'long_tex', 'long_oak')
# Arm A: 80-s calibration as in the paper; arm B: 200-s calibration.
ARMS = {'A': 80.0, 'B': 200.0}
TEX_VER_A_RAW = 110.0
K_BETAS = [1e-1, 5e-2, 2e-2, 1e-2, 5e-3]
METHODS = ('qr', 'ae', 'tcn', 'cn0', 'wscm')
NAME = {'qr': 'QR-LSTM', 'ae': 'LSTM-AE', 'tcn': 'TCN', 'cn0': 'C/N0', 'wscm': 'WSCM'}
OLD_STAR = {'qr': ('results_rev', 'rev_qr_star.csv'), 'ae': ('results_rev', 'rev_ae_star.csv'),
            'tcn': ('results_rev', 'rev_tcn_star.csv'), 'cn0': ('results_c', 'rev_cn0_star.csv'),
            'wscm': ('results_c', 'rev_wscm_star.csv')}
# Targets for the transfer of CUSUM limits between receivers.
TR_BETAS = (1e-1, 5e-2, 2e-2)
BIN_S, BIN_BETA = 30, 1e-1
SEGCFG = {}


# ---------------------------------------------------------------------------
# Long-recording segmentation plugged into rev_main.run_star
# ---------------------------------------------------------------------------
def setup(root):
    code = os.path.join(root, 're_ii', 'code')
    for p in (root, code):
        if p not in sys.path:
            sys.path.insert(0, p)
    import rev_main as rm
    import far_calib_v32 as fc
    import sqm_beta_neweval_v2 as sq
    orig_seg, orig_grp = fc.segments, fc.group_of

    def seg_long(S, lab, scen, gidx):
        if scen not in SEGCFG:
            return orig_seg(S, lab, scen, gidx)
        asr, cal_n, ver_a = SEGCFG[scen]
        n = len(S)
        cm = np.zeros(n, dtype=bool)
        cm[:cal_n] = True
        return dict(c0=0, c1=cal_n, contiguous=True, cmask=cm, n_lab0=n, truncated=False, ho_a=ver_a, ho_b=n,
                    hb_act=n, oi=n, act_i=n, asr=asr, raw=lambda i, _a=asr: (i + W + _a) / 1000.0)

    fc.segments = seg_long
    fc.group_of = lambda scen: GROUP.get(scen) or orig_grp(scen)
    for s in LONG:
        fc.ACTIVATION_RAW[s] = float('nan')
        fc.AUTH[s] = float('nan')
    rm.BETAS_MAIN = list(K_BETAS)
    return rm, fc, sq


def set_arm(arm, items):
    SEGCFG.clear()
    for _, path, _ in items:
        d = np.load(path, allow_pickle=True)
        scen, asr = str(d['scenario']), int(d['align_start_ms'])
        cal_n = int(round(ARMS[arm] * 1000)) - W
        ver_a = cal_n
        if arm == 'A' and scen == 'TEX_cS':
            ver_a = int(round(TEX_VER_A_RAW * 1000)) - W - asr
        SEGCFG[scen] = (asr, cal_n, ver_a)


def make_loader(method, fc, sq):
    def load(path, enc=None):
        d = np.load(path, allow_pickle=True)
        scen, prn, lab = str(d['scenario']), int(d['prn']), np.asarray(d['labels'])
        if method == 'qr':
            S = d['per'].astype(np.float64).sum(1)
        elif method in ('ae', 'tcn'):
            S = d['S_' + method].astype(np.float64)
        else:
            metric = (d['cn0'].astype(np.float64) if method == 'cn0'
                      else sq.sqm_wscm_EL(d['taps_raw'].astype(np.float64)))
            cm = fc.segments(metric, lab, scen, 100)['cmask']
            S = np.abs(metric - float(np.median(metric[cm])))
        if len(S) != len(lab) or len(S) == 0:
            raise ValueError(f'{os.path.basename(path)}: {method} 分数为空或长度不符（s3a 是否跑了这个模型？）')
        return S, lab, scen, prn
    return load


def zscore(S, seg):
    Sc = S[seg['cmask']]
    return np.ascontiguousarray((S - Sc.mean()) / Sc.std())


def vletter(v, n_ev):
    return {'verified': 'W', 'violated': 'E', 'too_short': 'S'}.get(v, 'U' if n_ev == 0 else 'C')


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float('nan')


def cell(rate, n, v):
    return f'{rate * 100:.2g}({n}){vletter(v, n)}' if np.isfinite(rate) else '—'


def spearman(x, y):
    if len(x) < 3:
        return float('nan')
    rk = lambda v: np.argsort(np.argsort(v, kind='mergesort'), kind='mergesort').astype(float)
    a, b = rk(np.asarray(x)), rk(np.asarray(y))
    if a.std() == 0 or b.std() == 0:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------------------
# Verification, lower targets and transfer between receivers (Section 4.5, Table 6)
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=DEF_ROOT)
    ap.add_argument('--s3-dir', default=None, help=r'默认 <root>\r1_revision_SEN\s3')
    ap.add_argument('--arms', nargs='+', default=['A', 'B'], choices=list(ARMS))
    ap.add_argument('--methods', nargs='+', default=list(METHODS), choices=list(METHODS))
    ap.add_argument('--nboot', type=int, default=2000)
    ap.add_argument('--seg', type=float, default=80.0)
    a = ap.parse_args(argv)
    root = os.path.abspath(a.root)
    s3 = os.path.abspath(a.s3_dir or os.path.join(root, 'r1_revision_SEN', 's3'))
    rm, fc, sq = setup(root)
    files = sorted(glob.glob(os.path.join(s3, 'scores', 'long_*_prn*.npz')))
    items = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        items.append((f'{d["scenario"]}_prn{int(d["prn"])}', f, f'{d["scenario"]}_prn{int(d["prn"])}'))
    if not items:
        raise SystemExit(f'[STOP] {os.path.join(s3, "scores")} 里没有 long_*.npz，先跑 s3a_infer_long.py')
    ns = SimpleNamespace(guard_s=0.1, m=1000, K=5, nboot=a.nboot, seg=a.seg, seed=42, pertap_map={})
    m = ns.m
    B = [f'[s3b] v1.0  {time.strftime("%Y-%m-%d %H:%M:%S")}  星 {len(items)}：{", ".join(b for b, _, _ in items)}',
         f'  nboot={a.nboot} seg={a.seg} s；CUSUM β={rm.BETAS_ALL}；K=5 β={K_BETAS}；单位 ×10⁻²，括号内为事件数']
    pools, rows_by = {}, {}
    t_all = time.time()
    for arm in a.arms:
        set_arm(arm, items)
        od = os.path.join(s3, f'arm{arm}')
        os.makedirs(od, exist_ok=True)
        for method in a.methods:
            loader = make_loader(method, fc, sq)
            rm.load_any = loader
            log = fc.Tee(os.path.join(od, f'long_{method}_log.txt'))
            log(f'[s3b] arm {arm}（标定 {ARMS[arm]:.0f} s）method {method}  SEGCFG={SEGCFG}')
            rows = []
            for it in items:
                rr, _, _ = rm.run_star(it, ns, log, {'cal': {}, 'k': {}}, [], method)
                rows += rr
            pool = rm.pooled(rows, m, GROUPS)
            rm.save_csv(os.path.join(od, f'long_{method}_star.csv'), rows, rm.STAR_FIELDS)
            rm.save_csv(os.path.join(od, f'long_{method}_pooled.csv'), pool, rm.POOL_FIELDS)
            log.close() if hasattr(log, 'close') else None
            pools[(arm, method)] = pool
            rows_by[(arm, method)] = rows
            print(f'[s3b] arm {arm} {method} 完成，累计 {(time.time() - t_all) / 60:.1f} min')


    B.append('\n== 0. 核验数据量与可分辨下限 3m/N（合并） ==')
    for arm in a.arms:
        set_arm(arm, items)
        tot = {g: 0 for g in GROUPS}
        for _, f, _ in items:
            d = np.load(f, allow_pickle=True)
            scen = str(d['scenario'])
            n = len(d['labels'])
            s = (n - SEGCFG[scen][2]) / 1000.0
            tot['all'] += s
            tot[GROUP[scen]] += s
        B.append(f'  arm {arm}（标定 {ARMS[arm]:.0f} s，q≈{1000 / (ARMS[arm] * 1000 - W):.1e}）：' +
                 '；'.join(f'{g} {tot[g]:.0f} s → {3 * m / (tot[g] * 1000):.1e}' for g in GROUPS if tot[g] > 0))

    def pline(pool, rule, betas, g):
        out = []
        for b in betas:
            r = next((p for p in pool if p['rule'] == rule and p['group'] == g and np.isfinite(p['beta'])
                      and abs(p['beta'] - b) < 1e-12), None)
            out.append('—' if r is None or r.get('ho_verdict') in (None, 'none') else
                       cell(r['ho_frac'], r['ho_n_event'], r['ho_verdict']))
        return '  '.join(f'{x:>11s}' for x in out)

    B.append('\n== 1. CUSUM（块自助 h），全部长录音合并的核验率 ==')
    for arm in a.arms:
        B.append(f'  arm {arm}   β: ' + '  '.join(f'{b:>11.0e}' for b in rm.BETAS_ALL))
        for mth in a.methods:
            B.append(f'  {NAME[mth]:8s}       ' + pline(pools[(arm, mth)], 'cusum', rm.BETAS_ALL, 'all'))
    B.append('\n== 2. K=5（块自助 γ），合并 ==')
    for arm in a.arms:
        B.append(f'  arm {arm}   β: ' + '  '.join(f'{b:>11.0e}' for b in K_BETAS))
        for mth in a.methods:
            B.append(f'  {NAME[mth]:8s}       ' + pline(pools[(arm, mth)], 'k_blk', K_BETAS, 'all'))
    arm_g = 'B' if 'B' in a.arms else a.arms[0]
    bg = (2e-2, 1e-2, 5e-3)
    B.append(f'\n== 3. CUSUM 分接收机（arm {arm_g}；β = {", ".join(f"{b:.0e}" for b in bg)}） ==')
    for mth in a.methods:
        B.append(f'  {NAME[mth]:8s} TEX ' + pline(pools[(arm_g, mth)], 'cusum', bg, 'long_tex')
                 + '  | OAK ' + pline(pools[(arm_g, mth)], 'cusum', bg, 'long_oak'))


    tr_rows = []
    if 'A' in a.arms:
        set_arm('A', items)
        B.append('\n== 4. CUSUM 门限迁移（Z 单位中位数 h；目标端只用自己 A 臂标定段的均值、标准差） ==')
        B.append('  方向                     β: ' + '  '.join(f'{b:>11.0e}' for b in TR_BETAS))
        for mth in a.methods:
            loader = make_loader(mth, fc, sq)
            old = os.path.join(root, 're_ii', *OLD_STAR[mth])
            src = {}
            if os.path.exists(old):
                with open(old, newline='', encoding='utf-8-sig') as fh:
                    rs = list(csv.DictReader(fh))
                for b in TR_BETAS:
                    hs = [fnum(r['thr']) for r in rs if r['rule'] == 'cusum' and abs(fnum(r['beta']) - b) < 1e-12
                          and r['scenario'] in ('ds3', 'ds4_r2')]
                    if hs:
                        src[('TEX静态7星→OAK', b)] = float(np.median(hs))
            for b in TR_BETAS:
                hs = [r['thr'] for r in rows_by[('A', mth)] if r['rule'] == 'cusum' and r['scenario'] == 'OAK_cD'
                      and abs(r['beta'] - b) < 1e-12 and np.isfinite(r.get('thr', np.nan))]
                if hs:
                    src[('OAK→TEX cleanStatic', b)] = float(np.median(hs))
            for direction, tgt in (('TEX静态7星→OAK', 'OAK_cD'), ('OAK→TEX cleanStatic', 'TEX_cS')):
                cells = []
                for b in TR_BETAS:
                    h = src.get((direction, b))
                    if h is None:
                        cells.append('—')
                        continue
                    ev = cov = nwin = 0
                    for base, f, _ in items:
                        S, lab, scen, _ = loader(f)
                        if scen != tgt:
                            continue
                        seg = fc.segments(S, lab, scen, 100)
                        Z = zscore(S, seg)
                        res = fc.ho_eval(lambda X, _h=h: fc.alarm_idx(X, fc.K0, _h), Z, seg['ho_a'], seg['hb_act'], [], m)
                        ev, cov, nwin = ev + res['n_event'], cov + res['cov'], nwin + res['nwin']
                    lo, up = fc.win_rate_bounds(ev, cov, nwin, m)
                    v = fc.verdict(lo, up, b, nwin)
                    rate = cov / nwin if nwin > 0 else float('nan')
                    tr_rows.append(dict(method=mth, direction=direction, beta=b, h_src=h, n_event=ev, cov=cov,
                                        nwin=nwin, rate=rate, lo95=lo, up95=up, verdict=v))
                    cells.append(cell(rate, ev, v))
                B.append(f'  {NAME[mth]:8s} {direction:16s} ' + '  '.join(f'{x:>11s}' for x in cells))
        with open(os.path.join(s3, 's3b_transfer.csv'), 'w', newline='', encoding='utf-8-sig') as fh:
            w = csv.DictWriter(fh, fieldnames=['method', 'direction', 'beta', 'h_src', 'n_event', 'cov', 'nwin',
                                               'rate', 'lo95', 'up95', 'verdict'])
            w.writeheader()
            w.writerows(tr_rows)


        B.append(f'\n== 5. A 臂 CUSUM（β = {BIN_BETA:g}）每 {BIN_S} s 的含报警窗口占比（×10⁻²）：最小/中位/最大，与时间的秩相关 ==')
        bin_rows = []
        for mth in a.methods:
            loader = make_loader(mth, fc, sq)
            line = []
            for g, tgt in (('TEX', 'TEX_cS'), ('OAK', 'OAK_cD')):
                acc = {}
                for base, f, _ in items:
                    S, lab, scen, _ = loader(f)
                    if scen != tgt:
                        continue
                    r = next((x for x in rows_by[('A', mth)] if x['star'] == base and x['rule'] == 'cusum'
                              and abs(x['beta'] - BIN_BETA) < 1e-12), None)
                    if r is None or not np.isfinite(r.get('thr', np.nan)):
                        continue
                    seg = fc.segments(S, lab, scen, 100)
                    idx = np.asarray(fc.alarm_idx(zscore(S, seg), fc.K0, r['thr']), dtype=np.int64)
                    k, b0 = 0, seg['ho_a']
                    while b0 + BIN_S * 1000 <= len(S):
                        c, nw = fc.window_cov(idx, b0, b0 + BIN_S * 1000, m)
                        e = fc.n_events(idx[(idx >= b0) & (idx < b0 + BIN_S * 1000)], m)
                        t = acc.setdefault(k, [0, 0, 0])
                        t[0] += c
                        t[1] += nw
                        t[2] += e
                        k, b0 = k + 1, b0 + BIN_S * 1000
                ks = sorted(acc)
                rates = [acc[k][0] / acc[k][1] for k in ks if acc[k][1] > 0]
                for k in ks:
                    bin_rows.append(dict(method=mth, group=g, bin=k, t0_rel_s=k * BIN_S, cov=acc[k][0], nwin=acc[k][1],
                                         n_event=acc[k][2], rate=acc[k][0] / acc[k][1] if acc[k][1] else float('nan')))
                if rates:
                    line.append(f'{g} {min(rates) * 100:.1f}/{np.median(rates) * 100:.1f}/{max(rates) * 100:.1f}'
                                f'（{len(rates)} 段，ρ={spearman(range(len(rates)), rates):+.2f}）')
            B.append(f'  {NAME[mth]:8s} ' + '；'.join(line))
        with open(os.path.join(s3, 's3b_bins.csv'), 'w', newline='', encoding='utf-8-sig') as fh:
            w = csv.DictWriter(fh, fieldnames=['method', 'group', 'bin', 't0_rel_s', 'cov', 'nwin', 'n_event', 'rate'])
            w.writeheader()
            w.writerows(bin_rows)

    B.append(f'\n[s3b] 完成，用时 {(time.time() - t_all) / 60:.1f} min。回传 s3a_brief.txt 和本文件')
    with open(os.path.join(s3, 's3b_brief.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(B) + '\n')
    print('\n'.join(B))


if __name__ == '__main__':
    main()
