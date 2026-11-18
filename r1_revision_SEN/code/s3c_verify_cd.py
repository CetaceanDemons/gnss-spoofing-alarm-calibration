# -*- coding: utf-8 -*-
"""Calibration and verification on TEXBAT cleanDynamic (Table 6).

Verification starts at 107.0 s, after the part of the drive used by ds5 and ds6.
Arms and procedure as in s3b_verify_long.py.
Run from the repository root (the working folder that also holds the data folders); see --help.
"""
import argparse
import csv
import glob
import json
import math
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
SCEN = 'TEX_cD'
GROUP = {SCEN: 'long_texdyn'}
GROUPS = ('all', 'long_texdyn')
ARMS = {'A': 80.0, 'B': 200.0}
K_BETAS = [1e-1, 5e-2, 2e-2, 1e-2, 5e-3]
MAIN_B = (1e-1, 5e-2, 2e-2)
LOW_B = (1e-2, 5e-3, 3e-3, 1e-3)
METHODS = ('qr', 'ae', 'tcn', 'cn0', 'wscm')
NAME = {'qr': 'QR-LSTM', 'ae': 'LSTM-AE', 'tcn': 'TCN', 'cn0': 'C/N0', 'wscm': 'WSCM'}
OLD_DIR = {'qr': 'results_rev', 'ae': 'results_rev', 'tcn': 'results_rev', 'cn0': 'results_c', 'wscm': 'results_c'}
DYN_DS = ('ds5_r2', 'ds6_r2')
BIN_S, BIN_BETA, STAR_BETA = 30, 1e-1, 5e-2
SEGCFG = {}


# ---------------------------------------------------------------------------
# Long-recording segmentation plugged into rev_main.run_star
# ---------------------------------------------------------------------------
def setup(root):
    code = os.path.join(root, 're_ii', 'code')
    for p in (root, code, HERE):
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
    auth_ds = {ds: float(fc.AUTH[ds]) for ds in DYN_DS}
    fc.ACTIVATION_RAW[SCEN] = float('nan')
    fc.AUTH[SCEN] = float('nan')
    rm.BETAS_MAIN = list(K_BETAS)
    return rm, fc, sq, auth_ds


# Verification starts after the part of the drive that ds5/ds6 replay (107.0 s).
def ver_start(s3, auth_ds, log):
    """End of the part of cleanDynamic used by ds5/ds6, rounded up to 0.1 s."""
    p = os.path.join(s3, 'cd_offset.json')
    if not os.path.exists(p):
        raise SystemExit(f'[STOP] 缺 {p}：先跑 s3a_infer_long.py v1.2')
    off = json.load(open(p, encoding='utf-8'))
    Ls, offs, ends = {}, [], []
    for ds in DYN_DS:
        good = [q for q in off.get(ds, {}).get('pairs', []) if q['resid_std_hz'] < 1.0]
        if len(good) < 2:
            raise SystemExit(f'[STOP] {ds} 可靠的匹配星不足 2 颗（残差 <1 Hz），把 s3a_cd_brief.txt 发回来')
        Ls[ds] = float(np.median([q['L'] for q in good]))
        offs += [q['dop_offset_hz'] for q in good]
        ends.append(Ls[ds] + auth_ds[ds])
        log(f'  {ds}: L = {Ls[ds]:.3f} s（{len(good)} 星，极差 {max(q["L"] for q in good) - min(q["L"] for q in good):.3f} s）'
            f'，onset {auth_ds[ds]:.1f} s → 已用段止于 cleanDynamic {Ls[ds] + auth_ds[ds]:.2f} s')
    return math.ceil(max(ends) * 10) / 10.0, Ls, float(np.median(offs))


def set_arm(arm, items, ver_raw):
    SEGCFG.clear()
    for _, path, _ in items:
        d = np.load(path, allow_pickle=True)
        scen, asr = str(d['scenario']), int(d['align_start_ms'])
        t0 = float(d['t0_rec_s']) if 't0_rec_s' in d.files and np.isfinite(d['t0_rec_s']) else asr / 1000.0
        cal_n = int(round(ARMS[arm] * 1000)) - W
        ver_a = max(cal_n, int(round((ver_raw - t0) * 1000)) - W)
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
            raise ValueError(f'{os.path.basename(path)}: {method} 分数为空或长度不符')
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
    return f'{rate * 100:.2g}({int(n)}){vletter(v, n)}' if np.isfinite(rate) else '—'


def spearman(x, y):
    if len(x) < 3:
        return float('nan')
    rk = lambda v: np.argsort(np.argsort(v, kind='mergesort'), kind='mergesort').astype(float)
    a, b = rk(np.asarray(x)), rk(np.asarray(y))
    if a.std() == 0 or b.std() == 0:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline='', encoding='utf-8-sig') as fh:
        return list(csv.DictReader(fh))


def prow(pool, rule, b, g):
    return next((p for p in pool if p['rule'] == rule and str(p['group']) == g
                 and (np.isnan(b) and not np.isfinite(fnum(p['beta'])) or abs(fnum(p['beta']) - b) < 1e-12)), None)


def pcell(r, iid=False):
    if r is None or r.get('ho_verdict') in (None, '', 'none'):
        return '—'
    rate, n = fnum(r['ho_frac']), fnum(r['ho_n_event'])
    s = cell(rate, n, r['ho_verdict'])
    ref = fnum(r.get('ref'))
    if iid and np.isfinite(ref) and ref > 0:
        s += f'×{rate / ref:.3g}'
    return s


# ---------------------------------------------------------------------------
# Check of the ds5/ds6 offsets
# ---------------------------------------------------------------------------
def dop_check(root, Ls, off_hz, log):
    import s3a_infer_long as s3a
    cd = {}
    for ch in s3a.CD_CHS:
        prn, t, x = s3a._dop_series(os.path.join(root, s3a.CD_DIR), ch)
        cd[prn] = (t, x)
    for ds, L in Ls.items():
        parts = []
        for ch in range(5):
            prn, t, x = s3a._dop_series(os.path.join(root, ds), ch)
            if prn not in cd:
                continue
            tc, y = s3a._bin(t, x, 0.0, 100.0, 1.0)
            tg, g = s3a._bin(cd[prn][0], cd[prn][1], 0.0, 445.0, 1.0)
            ok = np.isfinite(g)
            gi = np.interp(tc + L, tg[ok], g[ok], left=np.nan, right=np.nan)
            r = np.abs(y - gi - off_hz)
            tcd = tc + L
            a = np.nanmedian(r[tcd < 21.0]) if np.isfinite(r[tcd < 21.0]).any() else float('nan')
            b = np.nanmedian(r[(tcd >= 21.0) & (tcd < 99.0)])
            parts.append(f'PRN{prn} {a:.1f}/{b:.2f}')
        log(f'  {ds}（L={L:.3f}）: ' + '  '.join(parts))


# ---------------------------------------------------------------------------
# Verification on TEXBAT cleanDynamic (Section 4.5, Table 6)
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=DEF_ROOT)
    ap.add_argument('--s3-dir', default=None, help=r'默认 <root>\r1_revision_SEN\s3')
    ap.add_argument('--arms', nargs='+', default=['A', 'B'], choices=list(ARMS))
    ap.add_argument('--methods', nargs='+', default=list(METHODS), choices=list(METHODS))
    ap.add_argument('--exclude-prn', nargs='*', type=int, default=[])
    ap.add_argument('--nboot', type=int, default=2000)
    ap.add_argument('--seg', type=float, default=80.0)
    ap.add_argument('--skip-dop', action='store_true', help='不做 0b 多普勒一致性（不读 .dat）')
    a = ap.parse_args(argv)
    root = os.path.abspath(a.root)
    s3 = os.path.abspath(a.s3_dir or os.path.join(root, 'r1_revision_SEN', 's3'))
    out = os.path.join(s3, 'cd')
    os.makedirs(out, exist_ok=True)
    rm, fc, sq, auth_ds = setup(root)
    B = []

    def blog(s):
        print(s)
        B.append(s)

    files = sorted(glob.glob(os.path.join(s3, 'scores', f'long_{SCEN}_prn*.npz')))
    items = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        prn = int(d['prn'])
        if prn in a.exclude_prn:
            continue
        items.append((f'{SCEN}_prn{prn}', f, f'{SCEN}_prn{prn}'))
    if not items:
        raise SystemExit(f'[STOP] {os.path.join(s3, "scores")} 里没有 long_{SCEN}_prn*.npz，先跑 s3a_infer_long.py v1.2')
    blog(f'[s3c] v1.0  {time.strftime("%Y-%m-%d %H:%M:%S")}  星 {len(items)}：{", ".join(b[len(SCEN) + 1:] for b, _, _ in items)}'
         + (f'（排除 PRN {a.exclude_prn}）' if a.exclude_prn else ''))
    blog(f'  nboot={a.nboot} seg={a.seg} s；动态模型分数；单位 ×10⁻²，括号内为事件数；W 核验通过 C 不确定 U 无事件 E 超标')
    blog('\n== 0. 核验起点（避开 ds5/ds6 已用段）与数据量 ==')
    ver_raw, Ls, off_hz = ver_start(s3, auth_ds, blog)
    m = 1000
    for arm in a.arms:
        set_arm(arm, items, ver_raw)
        tot = 0.0
        for _, f, _ in items:
            d = np.load(f, allow_pickle=True)
            tot += (len(d['labels']) - SEGCFG[SCEN][2]) / 1000.0
        asr, cal_n, ver_a = SEGCFG[SCEN]
        t0 = float(np.load(items[0][1], allow_pickle=True)['t0_rec_s'])
        blog(f'  arm {arm}：标定 {t0 + W / 1000:.1f}–{t0 + (cal_n + W) / 1000:.1f} s，核验 {t0 + (ver_a + W) / 1000:.1f} s 起，'
             f'合计 {tot:.0f} s → 可分辨下限 3m/N ≈ {3 * m / (tot * 1000):.1e}；标定分辨率 q ≈ {m / cal_n:.1e}')
    if not a.skip_dop:
        blog(f'\n== 0b. 多普勒一致性：|ds − cd(t+L) − ({off_hz:+.2f} Hz)| 中位，cd 时间 <21 s / 21–99 s（Hz） ==')
        try:
            dop_check(root, Ls, off_hz, blog)
        except Exception as e:
            blog(f'  [跳过] {type(e).__name__}: {e}')

    ns = SimpleNamespace(guard_s=0.1, m=m, K=5, nboot=a.nboot, seg=a.seg, seed=42, pertap_map={})
    pools, rows_by = {}, {}
    t_all = time.time()
    for arm in a.arms:
        set_arm(arm, items, ver_raw)
        od = os.path.join(out, f'arm{arm}')
        os.makedirs(od, exist_ok=True)
        for method in a.methods:
            rm.load_any = make_loader(method, fc, sq)
            log = fc.Tee(os.path.join(od, f'cd_{method}_log.txt'))
            log(f'[s3c] arm {arm}（标定 {ARMS[arm]:.0f} s）method {method}  SEGCFG={SEGCFG}')
            rows = []
            for it in items:
                rr, _, _ = rm.run_star(it, ns, log, {'cal': {}, 'k': {}}, [], method)
                rows += rr
            pool = rm.pooled(rows, m, GROUPS)
            rm.save_csv(os.path.join(od, f'cd_{method}_star.csv'), rows, rm.STAR_FIELDS)
            rm.save_csv(os.path.join(od, f'cd_{method}_pooled.csv'), pool, rm.POOL_FIELDS)
            log.close() if hasattr(log, 'close') else None
            pools[(arm, method)] = pool
            rows_by[(arm, method)] = rows
            print(f'[s3c] arm {arm} {method} 完成，累计 {(time.time() - t_all) / 60:.1f} min')


    arm1 = 'A' if 'A' in a.arms else a.arms[0]
    blog(f'\n== 1. 标定段相关结构（块长 L 中位[最小,最大]；ρ1 中位[最小,最大]）：cleanDynamic arm {arm1} | 论文 ds5/ds6 ==')
    for mth in a.methods:
        cur = {}
        for r in rows_by[(arm1, mth)]:
            cur[r['star']] = (fnum(r['L']), fnum(r['rho1']))
        old = {}
        for r in read_csv(os.path.join(root, 're_ii', OLD_DIR[mth], f'rev_{mth}_star.csv')):
            if r['scenario'] in DYN_DS:
                old[r['star']] = (fnum(r['L']), fnum(r['rho1']))

        def summ(dct):
            if not dct:
                return '—'
            Lv = np.array([v[0] for v in dct.values()])
            rv = np.array([v[1] for v in dct.values()])
            return (f'L {np.median(Lv):.0f}[{Lv.min():.0f},{Lv.max():.0f}] ρ1 {np.median(rv):.3f}'
                    f'[{rv.min():.3f},{rv.max():.3f}]（{len(dct)} 星）')
        blog(f'  {NAME[mth]:8s} {summ(cur)} | {summ(old)}')


    blog('\n== 2. 合并核验率，β = 0.1 / 0.05 / 0.02：CUSUM | K=5 块自助 γ | K=5 独立 γ（×=核验率/独立假设参照值）| K=5 p95 ==')
    for arm in a.arms:
        blog(f'  arm {arm}（标定 {ARMS[arm]:.0f} s）')
        for mth in a.methods:
            P = pools[(arm, mth)]
            c = '  '.join(pcell(prow(P, 'cusum', b, 'all')) for b in MAIN_B)
            k = '  '.join(pcell(prow(P, 'k_blk', b, 'all')) for b in MAIN_B)
            i = '  '.join(pcell(prow(P, 'k_iid', b, 'all'), iid=True) for b in MAIN_B)
            p = pcell(prow(P, 'k_p95', float('nan'), 'all'))
            blog(f'   {NAME[mth]:8s} {c} | {k} | {i} | {p}')

    blog(f'\n== 3. 低 β：CUSUM β = {"/".join(f"{b:g}" for b in LOW_B)} | K=5 块自助 γ β = 0.01/0.005 ==')
    for arm in a.arms:
        for mth in a.methods:
            P = pools[(arm, mth)]
            c = '  '.join(pcell(prow(P, 'cusum', b, 'all')) for b in LOW_B)
            k = '  '.join(pcell(prow(P, 'k_blk', b, 'all')) for b in (1e-2, 5e-3))
            blog(f'  {arm} {NAME[mth]:8s} {c} | {k}')


    blog('\n== 4. 对照：论文 ds5/ds6 留出段（re_ii 旧 pooled，dynamic 组），同口径 CUSUM | 块自助 γ | 独立 γ ==')
    for mth in a.methods:
        P = read_csv(os.path.join(root, 're_ii', OLD_DIR[mth], f'rev_{mth}_pooled.csv'))
        if not P:
            blog(f'  {NAME[mth]:8s} 缺旧 pooled CSV')
            continue
        c = '  '.join(pcell(prow(P, 'cusum', b, 'dynamic')) for b in MAIN_B)
        k = '  '.join(pcell(prow(P, 'k_blk', b, 'dynamic')) for b in MAIN_B)
        i = '  '.join(pcell(prow(P, 'k_iid', b, 'dynamic'), iid=True) for b in MAIN_B)
        blog(f'   {NAME[mth]:8s} {c} | {k} | {i}')


    blog(f'\n== 6. 逐星 arm {arm1}，β = {STAR_BETA:g}：CUSUM 核验率 | 独立 γ 核验率/参照值 ==')
    for mth in a.methods:
        cs, ii = [], []
        for base, _, _ in items:
            prn = base.split('prn')[1]
            rc = next((r for r in rows_by[(arm1, mth)] if r['star'] == base and r['rule'] == 'cusum'
                       and abs(fnum(r['beta']) - STAR_BETA) < 1e-12), None)
            ri = next((r for r in rows_by[(arm1, mth)] if r['star'] == base and r['rule'] == 'k_iid'
                       and abs(fnum(r['beta']) - STAR_BETA) < 1e-12), None)
            cs.append(f'{prn}:{fnum(rc["ho_frac"]) * 100:.1f}' if rc else f'{prn}:—')
            ref = fnum(ri.get('nominal_iid')) if ri else float('nan')
            ii.append(f'{prn}:×{fnum(ri["ho_frac"]) / ref:.2g}' if ri and np.isfinite(ref) and ref > 0 else f'{prn}:—')
        blog(f'  {NAME[mth]:8s} {" ".join(cs)} | {" ".join(ii)}')


    bin_rows = []
    if 'A' in a.arms:
        set_arm('A', items, ver_raw)
        blog(f'\n== 7. arm A CUSUM（β = {BIN_BETA:g}）每 {BIN_S} s 含报警窗口占比（×10⁻²）：最小/中位/最大，与时间的秩相关 ==')
        for mth in a.methods:
            loader = make_loader(mth, fc, sq)
            acc = {}
            for base, f, _ in items:
                S, lab, scen, _ = loader(f)
                r = next((x for x in rows_by[('A', mth)] if x['star'] == base and x['rule'] == 'cusum'
                          and abs(fnum(x['beta']) - BIN_BETA) < 1e-12), None)
                if r is None or not np.isfinite(fnum(r.get('thr'))):
                    continue
                seg = fc.segments(S, lab, scen, 100)
                idx = np.asarray(fc.alarm_idx(zscore(S, seg), fc.K0, fnum(r['thr'])), dtype=np.int64)
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
                bin_rows.append(dict(method=mth, bin=k, t0_rel_s=k * BIN_S, cov=acc[k][0], nwin=acc[k][1],
                                     n_event=acc[k][2], rate=acc[k][0] / acc[k][1] if acc[k][1] else float('nan')))
            if rates:
                blog(f'  {NAME[mth]:8s} {min(rates) * 100:.1f}/{np.median(rates) * 100:.1f}/{max(rates) * 100:.1f}'
                     f'（{len(rates)} 段，ρ={spearman(range(len(rates)), rates):+.2f}）')
        with open(os.path.join(out, 's3c_bins.csv'), 'w', newline='', encoding='utf-8-sig') as fh:
            w = csv.DictWriter(fh, fieldnames=['method', 'bin', 't0_rel_s', 'cov', 'nwin', 'n_event', 'rate'])
            w.writeheader()
            w.writerows(bin_rows)

    blog(f'\n[s3c] 完成，用时 {(time.time() - t_all) / 60:.1f} min。回传 s3\\cd\\s3c_brief.txt')
    with open(os.path.join(out, 's3c_brief.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(B) + '\n')


if __name__ == '__main__':
    main()
