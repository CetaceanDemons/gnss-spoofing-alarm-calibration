"""Autocorrelation with masked intervals (Section 4.2, Figure 4).

Tests whether the serial correlation of the dynamic satellites is caused by the
transient near 58 s: full series, the transient masked, the largest excursion masked,
and a placebo mask of the same length.
Run from the repository root (the working folder that also holds the data folders); see --help.
"""
import os, sys, csv, argparse, time
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)
import far_calib_v32 as fc

MAXLAG = 600
LAG_REPORT = (1, 10, 100, 300, 500)


# ---------------------------------------------------------------------------
# Autocorrelation with masked intervals (Section 4.2, Figure 4)
# ---------------------------------------------------------------------------
def masked_acf(x, w, maxlag):
    """Sample ACF using only pairs with both samples unmasked."""
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    nw = w.sum()
    mu = (x * w).sum() / nw
    y = (x - mu) * w
    var = (y * y).sum() / nw
    nfft = 1 << int(np.ceil(np.log2(2 * len(x))))
    Y = np.fft.rfft(y, nfft)
    W = np.fft.rfft(w, nfft)
    num = np.fft.irfft(Y * np.conj(Y), nfft)[:maxlag + 1]
    cnt = np.rint(np.fft.irfft(W * np.conj(W), nfft)[:maxlag + 1])
    return num / np.maximum(cnt, 1.0) / var, cnt


def block_len_rule(a, lo=10, hi=500, thr=0.05):
    """Smallest lag after which the ACF stays below thr, clipped to [lo, hi]."""
    above = np.flatnonzero(a[1:] >= thr)
    L = int(above[-1]) + 2 if len(above) else 1
    return int(min(max(L, lo), hi)), (L > hi)


# ---------------------------------------------------------------------------
# Per satellite: full, transient masked, largest excursion masked, placebo
# ---------------------------------------------------------------------------
def run(files, mask58, placebo, pad_s, log):
    rows, curves = [], {}
    for fi, f in enumerate(files, 1):
        star = os.path.basename(f)[len('pertap_'):-len('.npz')]
        S, lab, scen, prn = fc.load_star(f)
        seg = fc.segments(S, lab, scen, 100)
        c0, c1, asr, raw = seg['c0'], seg['c1'], seg['asr'], seg['raw']
        Sc = S[seg['cmask']]
        mu, sd = float(Sc.mean()), float(Sc.std())
        Z = (S - mu) / sd
        x = Z[c0:c1]
        base_w = seg['cmask'][c0:c1].copy()
        N = int(base_w.sum())
        rho1_farmc = float(fc.acf(Z[seg['cmask']], 1))
        L_pick = int(fc.pick_block_len(Z[seg['cmask']]))


        gtr = fc.g_trace(Z[:c1], fc.K0)
        j = c0 + int(np.argmax(gtr[c0:c1]))
        top = None
        if gtr[j] > 0:
            s0 = fc.excursion_start(gtr, j)
            e = j
            while e + 1 < c1 and gtr[e + 1] > 0:
                e += 1
            pad = int(round(pad_s * 1000))
            top = (max(c0, s0 - pad), min(c1, e + 1 + pad), raw(j), float(gtr[j]))

        def win(a_s, b_s):
            return (max(c0, fc.idx_of(a_s, asr)), min(c1, fc.idx_of(b_s, asr)))

        variants = [('full', None, '')]
        variants.append(('mask58', win(*mask58), f'raw {mask58[0]}–{mask58[1]} s'))
        if top is not None:
            variants.append(('maskTop', (top[0], top[1]),
                             f'raw {raw(top[0]):.2f}–{raw(top[1]):.2f} s（峰 {top[2]:.2f} s，g={top[3]:.1f}）'))
        variants.append(('placebo', win(*placebo), f'raw {placebo[0]}–{placebo[1]} s'))

        log(f'\n=== [{fi}/{len(files)}] {star} ({fc.group_of(scen)})  净段 {raw(c0):.2f}–{raw(c1):.2f} s  N={N / 1000:.1f} s'
            f'  far_mc: ρ1={rho1_farmc:.3f} L={L_pick}')
        for v, iv, desc in variants:
            w = base_w.copy()
            nmask = 0
            if iv is not None and iv[1] > iv[0]:
                nmask = int(w[iv[0] - c0:iv[1] - c0].sum())
                w[iv[0] - c0:iv[1] - c0] = False
            a, cnt = masked_acf(x, w, MAXLAG)
            nv = int(w.sum())
            band = 1.96 / np.sqrt(nv)
            L_rule, capped = block_len_rule(a)
            frac_pos = float(np.mean(a[1:501] > band))
            frac_neg = float(np.mean(a[1:501] < -band))
            r = dict(star=star, scenario=scen, group=fc.group_of(scen), prn=prn, N_clean_s=N / 1000.0,
                     rho1_farmc=rho1_farmc, L_pick_farmc=L_pick, variant=v, mask=desc,
                     masked_s=nmask / 1000.0, L_rule=L_rule, L_capped=capped,
                     frac_sig_pos_1_500=frac_pos, frac_sig_neg_1_500=frac_neg)
            for k in LAG_REPORT:
                r[f'acf_{k}'] = float(a[k])
            rows.append(r)
            curves[f'{star}|{v}'] = a.astype(np.float32)
            log(f'  {v:8s} 屏蔽 {nmask / 1000:5.2f} s  ρ1={a[1]:.3f}  ACF@10={a[10]:.3f} @100={a[100]:.3f} '
                f'@300={a[300]:.3f} @500={a[500]:.3f}  L={L_rule}{"(顶到上限)" if capped else ""}  '
                f'显著正 {frac_pos:.0%} / 显著负 {frac_neg:.0%}  {desc}')
    return rows, curves


FIELDS = ['star', 'scenario', 'group', 'prn', 'N_clean_s', 'rho1_farmc', 'L_pick_farmc', 'variant', 'mask',
          'masked_s'] + [f'acf_{k}' for k in LAG_REPORT] + ['L_rule', 'L_capped', 'frac_sig_pos_1_500',
                                                            'frac_sig_neg_1_500']


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', dest='npz_dir', default='.')
    ap.add_argument('--stars', nargs='+', default=None)
    ap.add_argument('--encoder', choices=['dyn', 'static'], default='dyn')
    ap.add_argument('--mask', nargs=2, type=float, default=[57.0, 60.0], help='mask58 的 raw 起止秒')
    ap.add_argument('--placebo', nargs=2, type=float, default=[30.0, 33.0], help='对照屏蔽的 raw 起止秒')
    ap.add_argument('--pad-s', dest='pad_s', type=float, default=0.5, help='maskTop 前后各加的秒数')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    base = os.path.splitext(a.out)[0]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    log = fc.Tee(base + '_log.txt')
    log('[version] e1b_acf_transient.py v1.0（2026-10-11）')
    files = fc.select_files(a.npz_dir, a.encoder, a.stars)
    log(f'[cfg] {len(files)} 颗星  mask58={a.mask}  placebo={a.placebo}  pad={a.pad_s} s  K0={fc.K0}')
    if not files:
        log('[ERROR] 没找到 pertap_*.npz，检查 --dir')
    else:
        t0 = time.time()
        rows, curves = run(files, tuple(a.mask), tuple(a.placebo), a.pad_s, log)
        with open(a.out, 'w', newline='', encoding='utf-8-sig') as fh:
            w = csv.writer(fh)
            w.writerow(FIELDS)
            for r in rows:
                w.writerow([fc.fmt(r.get(k)) for k in FIELDS])
        np.savez_compressed(base + '_acf.npz', **curves)
        log('\n==== 汇总：ρ1 / ACF@300 / 块长 L（full → mask58 → maskTop → placebo）====')
        by = {}
        for r in rows:
            by.setdefault(r['star'], {})[r['variant']] = r
        for s, d in by.items():
            def g(v, k, f='{:.3f}'):
                return f.format(d[v][k]) if v in d else '  —  '
            log(f'{s:18s} ρ1 {g("full", "acf_1")}→{g("mask58", "acf_1")}→{g("maskTop", "acf_1")}→{g("placebo", "acf_1")}'
                f' | @300 {g("full", "acf_300")}→{g("mask58", "acf_300")}→{g("maskTop", "acf_300")}→{g("placebo", "acf_300")}'
                f' | L {g("full", "L_rule", "{}")}→{g("mask58", "L_rule", "{}")}→{g("maskTop", "L_rule", "{}")}→{g("placebo", "L_rule", "{}")}')
        log(f'\n[saved] {a.out} / {base}_acf.npz / {base}_log.txt   用时 {time.time() - t0:.0f} s')
        log('[done] 把三个输出文件传回即可。')
    log.close()
