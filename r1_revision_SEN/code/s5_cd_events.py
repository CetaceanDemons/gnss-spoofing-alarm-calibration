# -*- coding: utf-8 -*-
"""Common-mode events of the clean drive (Appendix D).

Checks whether the events near 97 s and 58 s of ds5/ds6 appear at the same instants of
TEXBAT cleanDynamic and counts the common-mode events of the clean recording.
Run from the repository root (the working folder that also holds the data folders); see --help.
"""
import argparse
import csv
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEF_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
W, K0 = 100, 4.0

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
CD_CAL = (21.1, 101.0)
TARGETS = (('58 s 事件', 60.6, 1.5), ('97 s 事件（开启特征）', 99.97, 1.0))
METHODS = ('qr', 'ae', 'tcn', 'wscm')
DYN = ('ds5_r2', 'ds6_r2')
G_MIN, GAP, NMIN = 10.0, 0.3, 4


# ---------------------------------------------------------------------------
# CUSUM trace without reset and its excursions
# ---------------------------------------------------------------------------
def gtrace(Z):
    g = np.empty(len(Z))
    s = 0.0
    for i, z in enumerate(Z):
        s = s + z - K0
        if s < 0.0:
            s = 0.0
        g[i] = s
    return g


def peaks(g, t):
    """Peak time and height of each positive excursion of the CUSUM without reset."""
    pos = g > 0
    if not pos.any():
        return []
    d = np.diff(np.concatenate([[0], pos.astype(np.int8), [0]]))
    st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    out = []
    for a, b in zip(st, en):
        j = a + int(np.argmax(g[a:b]))
        out.append((float(t[j]), float(g[j])))
    return out


def zs(S, cal):
    Sc = S[cal]
    return (S - Sc.mean()) / Sc.std()


def cn0_step(cdb, t, t_ev):
    pre = cdb[(t >= t_ev - 3.0) & (t < t_ev)]
    post = cdb[(t >= t_ev) & (t < t_ev + 2.0)]
    return float(np.median(post) - np.median(pre)) if len(pre) and len(post) else float('nan')


def to_db(c):
    c = np.asarray(c, dtype=np.float64)
    return 10.0 * np.log10(np.maximum(c, 1e-12))


# ---------------------------------------------------------------------------
# Events near 58 s and 97 s in ds5, ds6 and the clean drive (Appendix D)
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=DEF_ROOT)
    ap.add_argument('--out-dir', default=None)
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    s3 = os.path.join(root, 'r1_revision_SEN', 's3')
    out = os.path.abspath(a.out_dir or os.path.join(root, 'r1_revision_SEN', 's5'))
    os.makedirs(out, exist_ok=True)
    for p in (os.path.join(root, 're_ii', 'code'), root):
        if p not in sys.path:
            sys.path.insert(0, p)
    import far_calib_v32 as fc
    import far_mc_beta_neweval_v2 as v2
    import sqm_beta_neweval_v2 as sq
    import rev_main as rm
    t_start = time.time()
    B = []

    def blog(s):
        print(s)
        B.append(s)

    off = json.load(open(os.path.join(s3, 'cd_offset.json'), encoding='utf-8'))
    L = {}
    for ds in DYN:
        good = [q for q in off.get(ds, {}).get('pairs', []) if q['resid_std_hz'] < 1.0]
        L[ds] = float(np.median([q['L'] for q in good]))
    blog(f'[s5] v1.0 {time.strftime("%Y-%m-%d %H:%M:%S")}  L = ' + '，'.join(f'{d} {L[d]:.3f} s' for d in DYN)
         + f'（t_cd = t_ds + L）；k0 = {K0:g}；g 为不重置轨迹')


    recs = {}
    files = sorted(glob.glob(os.path.join(s3, 'scores', 'long_TEX_cD_prn*.npz')))
    if not files:
        raise SystemExit(f'[STOP] {s3}\\scores 里没有 long_TEX_cD_prn*.npz')
    lst = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        n = len(d['labels'])
        t0 = float(d['t0_rec_s']) if np.isfinite(d['t0_rec_s']) else int(d['align_start_ms']) / 1000.0
        t = t0 + (np.arange(n) + W) / 1000.0
        cal = (t >= CD_CAL[0]) & (t < CD_CAL[1])
        w = sq.sqm_wscm_EL(d['taps_raw'].astype(np.float64))
        S = {'qr': d['per'].astype(np.float64).sum(1), 'ae': d['S_ae'].astype(np.float64),
             'tcn': d['S_tcn'].astype(np.float64), 'wscm': np.abs(w - np.median(w[cal]))}
        G = {k: gtrace(zs(v, cal)) for k, v in S.items() if len(v) == n}
        lst.append((int(d['prn']), t, G, to_db(d['cn0'])))
    recs['cleanDynamic'] = lst
    pert = {os.path.basename(f)[len('pertap_'):-len('_dyn.npz')]: f for _, f, _ in [(b, f, k) for b, f, k in rm.qr_items(root, None)]
            if f.endswith('_dyn.npz')}
    for ds in DYN:
        lst = []
        dual = {}
        for k in ('ae', 'tcn'):
            sub, _, s2 = rm.DEFAULT_DUAL[k]
            for f in glob.glob(os.path.join(root, sub, s2, f'{ds}_prn*_*scores_dual_dyn.npz')):
                prn = int(os.path.basename(f).split('_prn')[1].split('_')[0])
                dual[(k, prn)] = f
        for key, f in sorted(pert.items()):
            if not key.startswith(ds + '_prn'):
                continue
            S_qr, lab, scen, prn = fc.load_star(f)
            seg = fc.segments(S_qr, lab, scen, 100)
            n, asr, cal = len(S_qr), seg['asr'], seg['cmask']
            t = (np.arange(n) + 100 + asr) / 1000.0 + L[ds]
            taps, cn0, _, _, _, _ = sq.load_star_v2(f)
            w = sq.sqm_wscm_EL(taps)
            S = {'qr': S_qr, 'wscm': np.abs(w - np.median(w[cal]))}
            for k in ('ae', 'tcn'):
                if (k, prn) in dual:
                    s_, _, _, _ = v2.load_S_scores(dual[(k, prn)])
                    if len(s_) == n:
                        S[k] = np.asarray(s_, dtype=np.float64)
            G = {k: gtrace(zs(v, cal)) for k, v in S.items()}
            lst.append((prn, t, G, to_db(cn0) if cn0 is not None else None))
        recs[ds] = lst
    blog('[数据] ' + '；'.join(f'{r}: PRN ' + ','.join(str(p) for p, *_ in v) for r, v in recs.items()))


    blog('== 1. 关键时刻（cleanDynamic 时间）窗口内逐星 g 峰值：中位 [最小, 最大]；C/N0 台阶中位（dB）==')
    rows = []
    for name, tc, hw in TARGETS:
        blog(f'-- {name}：cd {tc - hw:.2f}–{tc + hw:.2f} s')
        for rec, lst in recs.items():
            cells = []
            for k in METHODS:
                v, tp = [], []
                for prn, t, G, _ in lst:
                    if k not in G:
                        continue
                    msk = (t >= tc - hw) & (t <= tc + hw)
                    if msk.any():
                        j = int(np.argmax(np.where(msk, G[k], -1.0)))
                        v.append(G[k][j])
                        tp.append(t[j])
                if v:
                    cells.append(f'{k} {np.median(v):6.1f} [{min(v):5.1f},{max(v):6.1f}]')
                    rows.append(dict(target=name, record=rec, method=k, g_med=float(np.median(v)), g_min=min(v),
                                     g_max=max(v), t_peak_med_cd=float(np.median(tp)), n=len(v)))

            stp = []
            for prn, t, G, cdb in lst:
                if cdb is None:
                    continue
                msk = (t >= tc - hw) & (t <= tc + hw)
                gq = np.where(msk, G['qr'], -1.0)
                te = t[int(np.argmax(gq))] if gq.max() > 0 else tc
                stp.append(cn0_step(cdb, t, te))
            tpk = [r['t_peak_med_cd'] for r in rows if r['target'] == name and r['record'] == rec and r['method'] == 'qr'
                   and r['g_med'] >= 1.0]
            blog(f'  {rec:12s} ' + ' | '.join(cells) + f' | C/N0 台阶 {np.nanmedian(stp):+.2f} dB'
                 + (f' | QR 峰时刻中位 cd {tpk[0]:.2f} s' if tpk else ' | QR 无偏移'))


    blog(f'== 2. cleanDynamic 全程共模事件（QR 的 g 峰 ≥{G_MIN:g}，≥{NMIN}/6 星在 ±{GAP} s 内）；按 QR 峰值中位排序 ==')
    pk = []
    for prn, t, G, cdb in recs['cleanDynamic']:
        for tp, gp in peaks(G['qr'], t):
            if gp >= G_MIN:
                pk.append((tp, prn, gp))
    pk.sort()
    ev, cur = [], []
    for p in pk:
        if cur and p[0] - cur[-1][0] > GAP:
            ev.append(cur)
            cur = []
        cur.append(p)
    if cur:
        ev.append(cur)
    cm = []
    for e in ev:
        prns = {p[1] for p in e}
        if len(prns) >= NMIN:
            tm = float(np.median([p[0] for p in e]))
            gq = float(np.median([max(p[2] for p in e if p[1] == q) for q in prns]))
            other = {}
            for k in ('ae', 'wscm'):
                vv = []
                for prn, t, G, _ in recs['cleanDynamic']:
                    if k in G:
                        msk = (t >= tm - 0.5) & (t <= tm + 0.5)
                        vv.append(float(G[k][msk].max()) if msk.any() else 0.0)
                other[k] = float(np.median(vv)) if vv else float('nan')
            stp = float(np.nanmedian([cn0_step(c, t, tm) for _, t, _, c in recs['cleanDynamic']]))
            cm.append(dict(t_cd=tm, n_stars=len(prns), g_qr=gq, g_ae=other['ae'], g_wscm=other['wscm'], cn0_step_db=stp,
                           in_cal=CD_CAL[0] <= tm < CD_CAL[1]))
    blog(f'  共 {len(cm)} 个共模事件；在 A 臂标定段内 {sum(c["in_cal"] for c in cm)} 个')
    for c in sorted(cm, key=lambda x: -x['g_qr'])[:15]:
        blog(f'  cd {c["t_cd"]:7.2f} s  星 {c["n_stars"]}/6  g中位 QR {c["g_qr"]:6.1f} AE {c["g_ae"]:6.1f} WSCM {c["g_wscm"]:6.1f}'
             f'  C/N0 台阶 {c["cn0_step_db"]:+.2f} dB' + ('  [标定段内]' if c['in_cal'] else ''))
    with open(os.path.join(out, 's5_events.csv'), 'w', newline='', encoding='utf-8-sig') as fh:
        fl = ['kind', 'target', 'record', 'method', 'g_med', 'g_min', 'g_max', 't_peak_med_cd', 'n',
              't_cd', 'n_stars', 'g_qr', 'g_ae', 'g_wscm', 'cn0_step_db', 'in_cal']
        wr = csv.DictWriter(fh, fieldnames=fl, extrasaction='ignore')
        wr.writeheader()
        for r in rows:
            wr.writerow(dict(kind='target', **r))
        for c in cm:
            wr.writerow(dict(kind='common_mode_cd', **c))
    blog(f'[done] 用时 {(time.time() - t_start) / 60:.1f} 分钟；输出 {out}')
    with open(os.path.join(out, 's5_brief.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(B) + '\n')


if __name__ == '__main__':
    main()
