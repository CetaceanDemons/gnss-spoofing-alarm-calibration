"""Block means and the log-likelihood-ratio CUSUM (Table 2, Figure 5, Appendix G).

Part 1: non-overlapping block means of the QR score, the carrier-to-noise proxy (dB and
folded) and the folded WSCM for blocks of 1, 20, 100 and 1000 ms; lag-one
autocorrelation, variance inflation, and the count rule on block means.
Part 2: Gaussian log-likelihood-ratio CUSUM on 20-ms and 1-s block means of the
carrier-to-noise proxy, with analytic, independence and block-bootstrap thresholds
(Table A5).
Run from the repository root (the working folder that also holds the data folders); see --help.
"""
import numpy as np, os, sys, argparse, time, math

_here = os.path.dirname(os.path.abspath(__file__))
for _d in (_here, os.path.dirname(_here), os.path.dirname(os.path.dirname(_here))):
    if os.path.exists(os.path.join(_d, 'sqm_beta_neweval_v2.py')):
        if _d not in sys.path:
            sys.path.insert(0, _d)
        break
if _here not in sys.path:
    sys.path.insert(0, _here)

import rev_main as rm
import far_calib_v32 as fc
import krule_clean_far_v12 as kr
import sqm_beta_neweval_v2 as sq
from far_mc_beta_neweval import cusum_gmax

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
# TB: block lengths (ms) of the block means; TSP: block lengths of the LLR CUSUM.
# MU1: post-change means of the LLR CUSUM; the paper reports the one-sigma alternative (p1sd).
TB = (1, 20, 100, 1000)
TSP = (20, 1000)
LAGS = (1, 5, 20, 100, 1000)
SCORES = ('qr', 'cn0_db', 'cn0_fold', 'wscm_fold')
MU1 = (('m16dB', 'db', -1.6), ('p3dB', 'db', 0.3), ('p1sd', 'sd', 1.0))
ARMS = ('ana', 'iid', 'blk')
BETAS = (0.1, 0.05, 0.02)
K = 5
KMODES = ('p95', 'b05')
_PT = {}


def ptab(m):
    if m not in _PT:
        _PT[m] = kr.p_for_target(0.05, K, m)
    return _PT[m]
GROUPS = ('all', 'static', 'dynamic', 'complex')

# Values of an earlier six-satellite check, reproduced as a self-test.
PREVIEW = {
    'ds3_prn23':    (0.017, 0.108, 0.090, 0.763, 0.928, 0.919, 0.261, 0.619, 0.790),
    'ds4_r2_prn13': (0.290, 0.200, 0.182, 0.774, 0.953, 0.978, 0.143, 0.477, 0.625),
    'ds5_r2_prn15': (0.915, 0.701, 0.474, 0.899, 0.908, 0.696, 0.510, 0.768, 0.634),
    'ds6_r2_prn9':  (0.772, 0.580, 0.211, 0.898, 0.888, 0.585, 0.492, 0.671, 0.644),
    'ds7_prn3':     (0.182, 0.092, 0.088, 0.703, 0.926, 0.946, 0.184, 0.552, 0.819),
    'ds8_prn3':     (0.186, 0.104, 0.097, 0.722, 0.931, 0.945, 0.188, 0.553, 0.816),
}


# ---------------------------------------------------------------------------
# Block means and helpers
# ---------------------------------------------------------------------------
def acf_nan(x, k):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if k >= n - 2:
        return float('nan')
    x = x - x.mean()
    d = float(np.dot(x, x))
    return float(np.dot(x[:n - k], x[k:]) / d) if d > 0 else float('nan')


def block_mean(x, T):
    nb = len(x) // T
    return np.asarray(x[:nb * T], dtype=np.float64).reshape(nb, T).mean(1)


def blk_range(a, b, T):
    """Samples [a, b) -> blocks fully inside [ceil(a/T), floor(b/T))."""
    return -(-a // T), b // T


def fold(x, cm):
    x = np.asarray(x, dtype=np.float64).ravel()
    bad = ~np.isfinite(x)
    med = float(np.median(x[cm & ~bad]))
    return np.abs(np.where(bad, med, x) - med), int(bad.sum())


def make_mp0(Zmat, m):
    """As far_calib_v32.make_mp with k0 = 0."""
    cache = {}

    def mp(h):
        key = float(f'{float(h):.9g}')
        if key not in cache:
            cache[key] = fc.mpfa_batch(Zmat, 0.0, key, m)[0]
        return cache[key]
    return mp


def med(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else float('nan')


def load_all(path, gidx, log):
    S_qr, lab, scen, prn = fc.load_star(path)
    taps, cn0, lab2, _, _, _ = sq.load_star_v2(path)
    lab = np.asarray(lab)
    seg = fc.segments(S_qr, lab, scen, gidx)
    cm = seg['cmask']
    if cn0 is None:
        raise KeyError('npz 里没有 cn0 / noise_var')
    cn0 = np.asarray(cn0, dtype=np.float64).ravel()
    wscm = np.asarray(sq.get_metric('wscm', taps, cn0, sq.DEFAULT_E_IDX, sq.DEFAULT_L_IDX), dtype=np.float64)
    if not (len(cn0) == len(wscm) == len(S_qr) == len(lab)):
        raise ValueError(f'长度不一致：qr {len(S_qr)} cn0 {len(cn0)} wscm {len(wscm)} labels {len(lab)}')
    bad = ~np.isfinite(cn0) | (cn0 <= 0)
    if bad.any():
        log(f'  [WARN] cn0 有 {int(bad.sum())} 个非有限或 ≤0 的值，已填为净段中位数')
        cn0 = np.where(bad, float(np.median(cn0[cm & ~bad])), cn0)
    cf, _ = fold(cn0, cm)
    wf, nb = fold(wscm, cm)
    if nb:
        log(f'  [WARN] wscm 有 {nb} 个非有限值，已填为净段中位数')
    series = {'qr': np.asarray(S_qr, dtype=np.float64), 'cn0_db': 10.0 * np.log10(cn0),
              'cn0_fold': cf, 'wscm_fold': wf}
    return series, cn0, lab, scen, prn, seg


# ============================================================ C1b

# ---------------------------------------------------------------------------
# Serial correlation of block means (Table 2, Figure 5)
# ---------------------------------------------------------------------------
def c1b_star(base, scen, grp, series, seg, log):
    rows = []
    c0, c1 = seg['c0'], seg['c1']
    for sc in SCORES:
        x = series[sc]
        sig2 = float(np.var(x[c0:c1]))
        for T in TB:
            a, b = blk_range(c0, c1, T)
            xc = block_mean(x, T)[a:b]
            N = len(xc)
            m = max(1, 1000 // T)
            r = dict(star=base, scenario=scen, group=grp, score=sc, T_ms=T, N_blocks=N, m_blocks=m,
                     pooled_excluded=(scen in rm.POOL_EXCLUDE))
            for lag in LAGS:
                r[f'acf{lag}'] = acf_nan(xc, lag)
            r['vif'] = float(np.var(xc) / (sig2 / T)) if sig2 > 0 else float('nan')
            sd = float(xc.std())
            r['L_rule'] = int(fc.pick_block_len((xc - xc.mean()) / sd)) if (N >= 30 and sd > 0) else float('nan')
            for md in KMODES:
                p_t = 0.05 if md == 'p95' else ptab(m)
                g = float(np.quantile(xc, 1.0 - p_t))
                p_hat = float(np.mean(xc > g))
                nom = kr.nominal_iid(p_hat, K, m)
                idx = kr.krun_idx(np.ascontiguousarray(xc), g, K)
                cov, nwin = kr.window_cov(idx, 0, N, m)
                nev = kr.n_events(idx, m)
                lo, up = kr.win_rate_bounds(nev, cov, nwin, m)
                dlv = cov / nwin if nwin > 0 else float('nan')
                r.update({f'{md}_gamma': g, f'{md}_p_hat': p_hat, f'{md}_nom': nom, f'{md}_cov': cov,
                          f'{md}_nwin': nwin, f'{md}_nev': nev, f'{md}_dlv': dlv, f'{md}_lo95': lo, f'{md}_up95': up,
                          f'{md}_D': (dlv / nom if nom > 0 else float('nan')),
                          f'{md}_res': (nom >= 3.0 * m / N if N > 0 else False)})
            rows.append(r)
    return rows


def c1b_pooled(rows):
    out = []
    for sc in SCORES:
        for T in TB:
            for g in GROUPS:
                sel = [r for r in rows if r['score'] == sc and r['T_ms'] == T and not r['pooled_excluded']
                       and (g == 'all' or r['group'] == g)]
                if not sel:
                    continue
                m = sel[0]['m_blocks']
                d = dict(score=sc, T_ms=T, group=g, n_stars=len(sel), N_blocks_med=med([r['N_blocks'] for r in sel]))
                for lag in LAGS:
                    d[f'acf{lag}_med'] = med([r[f'acf{lag}'] for r in sel])
                d['acf1_min'] = min((r['acf1'] for r in sel if np.isfinite(r['acf1'])), default=float('nan'))
                d['acf1_max'] = max((r['acf1'] for r in sel if np.isfinite(r['acf1'])), default=float('nan'))
                d['vif_med'] = med([r['vif'] for r in sel])
                d['L_rule_med'] = med([r['L_rule'] for r in sel])
                for md in KMODES:
                    nw = sum(r[f'{md}_nwin'] for r in sel)
                    if nw <= 0:
                        continue
                    cov = sum(r[f'{md}_cov'] for r in sel)
                    nev = sum(r[f'{md}_nev'] for r in sel)
                    nom = sum(r[f'{md}_nom'] * r[f'{md}_nwin'] for r in sel) / nw
                    lo, up = kr.win_rate_bounds(nev, cov, nw, m)
                    d.update({f'{md}_nom': nom, f'{md}_dlv': cov / nw, f'{md}_nev': nev, f'{md}_lo95': lo,
                              f'{md}_up95': up, f'{md}_D': (cov / nw / nom if nom > 0 else float('nan')),
                              f'{md}_res': (nom >= 3.0 * m / sum(r['N_blocks'] for r in sel)),
                              f'{md}_verdict': kr.verdict(lo, up, nom, nw)})
                out.append(d)
    return out


# ---------------------------------------------------------------------------
# Gaussian log-likelihood-ratio CUSUM on block means (Appendix G, Table A5)
# ---------------------------------------------------------------------------
# Three thresholds: analytic, independent reshuffle and block bootstrap.
def c1p_star(base, scen, grp, cn0, seg, a, log):
    rows = []
    raw = seg['raw']
    for ti, T in enumerate(TSP):
        xb = block_mean(cn0, T)
        ca, cb = blk_range(seg['c0'], seg['c1'], T)
        xc = xb[ca:cb]
        N = len(xc)
        m = max(1, 1000 // T)
        seg_len = int(round(a.seg * 1000)) // T
        ho_a = -(-seg['ho_a'] // T)
        bseg = dict(ho_a=ho_a, ho_end=max(ho_a, seg['hb_act'] // T), ho0_end=max(ho_a, seg['ho_b'] // T))
        j_act = -(-(seg['act_i'] + 1) // T) - 1
        j_on = -(-(seg['oi'] + 1) // T) - 1

        def rawb(j, _T=T):
            return raw((j + 1) * _T - 1)
        mu0, s0 = float(xc.mean()), float(xc.std())
        L_b = int(fc.pick_block_len((xc - mu0) / s0)) if s0 > 0 else 1
        L_b = min(L_b, max(1, N // 2))
        rho1 = acf_nan(xc, 1)
        log(f'  C1′ T={T:4d} ms: 净段 {N} 块  ho {bseg["ho_end"] - ho_a} 块  ho0 {bseg["ho0_end"] - ho_a} 块  '
            f'μ0={mu0:.4g} σ0={s0:.4g} (CV {s0 / mu0:.3f})  ρ1={rho1:.3f}  L={L_b}')
        for mt, kind, ex in MU1:
            mu1 = mu0 * 10.0 ** ex if kind == 'db' else mu0 + ex * s0
            D = mu1 - mu0
            Zs = np.ascontiguousarray((D / s0 ** 2) * (xb - 0.5 * (mu0 + mu1)))
            Zc = np.ascontiguousarray(Zs[ca:cb])
            gm = float(cusum_gmax(Zc, 0.0))
            mps = {'iid': make_mp0(fc.synth_matrix(Zc, 1, a.nboot, seg_len, fc.star_rng(a.seed, base, 100 + 2 * ti)), m),
                   'blk': make_mp0(fc.synth_matrix(Zc, L_b, a.nboot, seg_len, fc.star_rng(a.seed, base, 101 + 2 * ti)), m)}
            for beta in BETAS:
                for arm in ARMS:
                    if arm == 'ana':
                        nfa = (1000.0 / beta) / T
                        h, info = math.log(nfa), {'status': 'analytic'}
                    else:
                        h, info = fc.calibrate_h_v3(mps[arm], gm, beta)
                    r = dict(star=base, scenario=scen, group=grp, T_ms=T, mu1=mt, arm=arm, beta=beta, m_blocks=m,
                             N_blocks=N, ho_blocks=bseg['ho_end'] - ho_a, ho0_blocks=bseg['ho0_end'] - ho_a,
                             mu0=mu0, s0=s0, rho1=rho1, L=L_b, gm=gm, status=info['status'],
                             pooled_excluded=(scen in rm.POOL_EXCLUDE))
                    if h is None:
                        rows.append(r)
                        continue
                    r['h'] = h
                    r['boot_iid_at_h'] = mps['iid'](h)
                    r['boot_blk_at_h'] = mps['blk'](h)
                    fn = (lambda X, _h=h: fc.alarm_idx(X, 0.0, _h))
                    cal = np.asarray(fn(Zc), dtype=np.int64)
                    cov, nwin = fc.window_cov(cal, 0, N, m)
                    nev = fc.n_events(cal, m)
                    lo, up = fc.win_rate_bounds(nev, cov, nwin, m)
                    r.update(cal_n_event=nev, cal_cov=cov, cal_nwin=nwin,
                             cal_frac=(cov / nwin if nwin > 0 else float('nan')), cal_lo95=lo, cal_up95=up,
                             cal_verdict=fc.verdict(lo, up, beta, nwin))
                    for v, bb in (('ho', bseg['ho_end']), ('ho0', bseg['ho0_end'])):
                        fc.put_ho(r, v, fc.ho_eval(fn, Zs, ho_a, bb, [], m), beta, rawb)
                    full = np.asarray(fn(Zs), dtype=np.int64)
                    ia = rm.first_at_or_after(full, j_act)
                    io = rm.first_at_or_after(full, j_on)
                    r.update(det=(ia is not None),
                             delay_act_s=(rawb(ia) - rm.ACT[scen] if ia is not None else float('nan')),
                             delay_onset_s=(rawb(io) - rm.AUTH[scen] if io is not None else float('nan')))
                    rows.append(r)
            sel = [r for r in rows if r['T_ms'] == T and r['mu1'] == mt and r['beta'] == 0.05 and 'h' in r]
            log(f'    μ1={mt:5s} β=0.05: ' + '  '.join(
                f'{r["arm"]} h={r["h"]:.3g} ho事件 {r.get("ho_n_event", 0)} {r.get("ho_verdict", "")[:4]} '
                f'onset {r["delay_onset_s"]:+.2f}' for r in sel))
    return rows


def c1p_pooled(rows):
    out = []
    for T in TSP:
        for mt, _, _ in MU1:
            for arm in ARMS:
                for beta in BETAS:
                    for g in GROUPS:
                        sel = [r for r in rows if r['T_ms'] == T and r['mu1'] == mt and r['arm'] == arm
                               and r['beta'] == beta and (g == 'all' or r['group'] == g)]
                        if not sel:
                            continue
                        ok = [r for r in sel if 'h' in r]
                        fa = [r for r in ok if not r['pooled_excluded']]
                        m = sel[0]['m_blocks']
                        d = dict(T_ms=T, mu1=mt, arm=arm, beta=beta, group=g, n_stars=len(sel), n_h=len(ok),
                                 n_stars_fa=len(fa), h_med=med([r['h'] for r in ok]),
                                 n_degenerate=sum(1 for r in sel if r['status'] == 'degenerate'),
                                 n_not_found=sum(1 for r in sel if r['status'] == 'not_found'),
                                 boot_iid_at_h_med=med([r['boot_iid_at_h'] for r in ok]),
                                 boot_blk_at_h_med=med([r['boot_blk_at_h'] for r in ok]))
                        for v in ('cal', 'ho', 'ho0'):
                            kn = 'cal_nwin' if v == 'cal' else f'{v}_nwin'
                            kc = 'cal_cov' if v == 'cal' else f'{v}_cov'
                            ke = 'cal_n_event' if v == 'cal' else f'{v}_n_event'
                            hs = [r for r in fa if r.get(kn, 0) > 0]
                            nw = sum(r[kn] for r in hs)
                            if nw <= 0:
                                d[f'{v}_verdict'] = 'none'
                                continue
                            cov = sum(r[kc] for r in hs)
                            nev = sum(r[ke] for r in hs)
                            lo, up = fc.win_rate_bounds(nev, cov, nw, m)
                            d.update({f'{v}_n_event': nev, f'{v}_nwin': nw, f'{v}_frac': cov / nw,
                                      f'{v}_lo95': lo, f'{v}_up95': up, f'{v}_verdict': fc.verdict(lo, up, beta, nw)})
                        d['ho_blocks_total'] = sum(r['ho_blocks'] for r in fa)
                        da = [r['delay_act_s'] for r in ok if r.get('det')]
                        do = [r['delay_onset_s'] for r in ok if np.isfinite(r.get('delay_onset_s', float('nan')))]
                        d.update(n_det=len(da), delay_act_med=med(da), delay_onset_med=med(do),
                                 delay_onset_max=(max(do) if do else float('nan')))
                        out.append(d)
    return out


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
C1B_STAR = (['star', 'scenario', 'group', 'score', 'T_ms', 'N_blocks', 'm_blocks', 'pooled_excluded']
            + [f'acf{l}' for l in LAGS] + ['vif', 'L_rule']
            + [f'{md}_{k}' for md in ('p95', 'b05') for k in ('gamma', 'p_hat', 'nom', 'nev', 'cov', 'nwin', 'dlv',
                                                            'lo95', 'up95', 'D', 'res')]
            + ['krule_v12_D', 'krule_v12_match'])
C1B_POOL = (['score', 'T_ms', 'group', 'n_stars', 'N_blocks_med'] + [f'acf{l}_med' for l in LAGS]
            + ['acf1_min', 'acf1_max', 'vif_med', 'L_rule_med']
            + [f'{md}_{k}' for md in ('p95', 'b05') for k in ('nom', 'dlv', 'nev', 'lo95', 'up95', 'D', 'res',
                                                            'verdict')])
C1P_STAR = (['star', 'scenario', 'group', 'T_ms', 'mu1', 'arm', 'beta', 'm_blocks', 'N_blocks', 'ho_blocks',
             'ho0_blocks', 'mu0', 's0', 'rho1', 'L', 'gm', 'status', 'h', 'boot_iid_at_h', 'boot_blk_at_h',
             'pooled_excluded', 'cal_n_event', 'cal_frac', 'cal_lo95', 'cal_up95', 'cal_verdict']
            + [f'{v}_{k}' for v in ('ho', 'ho0') for k in ('len', 'n_alarm', 'n_event', 'cov', 'nwin', 'frac',
                                                           'lo95', 'up95', 'verdict', 'times')]
            + ['det', 'delay_act_s', 'delay_onset_s'])
C1P_POOL = (['T_ms', 'mu1', 'arm', 'beta', 'group', 'n_stars', 'n_h', 'n_stars_fa', 'n_degenerate', 'n_not_found',
             'h_med', 'boot_iid_at_h_med', 'boot_blk_at_h_med', 'ho_blocks_total']
            + [f'{v}_{k}' for v in ('cal', 'ho', 'ho0') for k in ('n_event', 'nwin', 'frac', 'lo95', 'up95', 'verdict')]
            + ['n_det', 'delay_act_med', 'delay_onset_med', 'delay_onset_max'])


def vshort(v):
    return {'verified': 'W', 'inconclusive': 'C', 'violated': 'E', 'none': '-', 'too_short': 'S'}.get(v, '?')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', dest='npz_dir', default='.')
    ap.add_argument('--stars', nargs='+', default=None)
    ap.add_argument('--nboot', type=int, default=2000)
    ap.add_argument('--seg', type=float, default=80.0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--guard-s', dest='guard_s', type=float, default=0.1)
    ap.add_argument('--krule-csv', default=None, help=r'默认 <dir>\re_ii\results_v32\krule_v12.csv')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    D = os.path.abspath(a.npz_dir)
    prefix = os.path.splitext(a.out)[0]
    os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
    log = fc.Tee(prefix + '_log.txt')
    brief = []

    def B(s=''):
        log(s)
        brief.append(s)

    B('[version] c1p_blockcn0.py v1.0（2026-10-24）')
    B(f'[cfg] nboot={a.nboot} seg={a.seg} s seed={a.seed} guard={a.guard_s} s  numba={"ON" if rm.HAVE_NUMBA else "OFF"}'
      f'  T={list(TB)} ms  T_sp={list(TSP)} ms  K={K}  β={list(BETAS)}')
    items = rm.qr_items(D, a.stars)
    B(f'[cfg] {len(items)} 颗星' + ('' if a.stars or len(items) == 21 else '  [WARN] ≠ 21'))
    gidx = int(round(a.guard_s * 1000))
    rows_b, rows_p = [], []
    t0 = time.time()
    for i, (base, path, key) in enumerate(items, 1):
        ts = time.time()
        try:
            series, cn0, lab, scen, prn, seg = load_all(path, gidx, log)
        except Exception as e:
            log(f'[skip] {base}: {e}')
            continue
        grp = fc.group_of(scen)
        log(f'\n=== [{i}/{len(items)}] {base} ({grp})  净段 {seg["raw"](seg["c0"]):.2f}–{seg["raw"](seg["c1"]):.2f} s'
            f'  连续={seg["contiguous"]}' + ('  [不进合并]' if scen in rm.POOL_EXCLUDE else ''))
        if not seg['contiguous']:
            log('  [WARN] 净段不连续：块按 [c0, c1) 切，个别块会跨过非净样本')
        rb = c1b_star(base, scen, grp, series, seg, log)
        for r in rb:
            r['_key'] = key
        rows_b += rb
        q = {(r['score'], r['T_ms']): r for r in rb}
        log('  C1b ρ1（T=1/20/100/1000 ms）：' + '  '.join(
            f'{sc} ' + '/'.join(f'{q[(sc, T)]["acf1"]:.3f}' for T in TB) for sc in SCORES))
        rows_p += c1p_star(base, scen, grp, cn0, seg, a, log)
        log(f'  用时 {time.time() - ts:.0f} s')


    B('\n== Self-check 1: six-satellite preview（cn0dB 1ms ACF lag1/100/1000、cn0dB 块ρ1 20/100/1000、QR 块ρ1 20/100/1000；容差 0.01）==')
    idx_b = {(r['star'], r['score'], r['T_ms']): r for r in rows_b}
    for st, ref in PREVIEW.items():
        if (st, 'cn0_db', 1) not in idx_b:
            continue
        mine = (idx_b[(st, 'cn0_db', 1)]['acf1'], idx_b[(st, 'cn0_db', 1)]['acf100'], idx_b[(st, 'cn0_db', 1)]['acf1000'],
                idx_b[(st, 'cn0_db', 20)]['acf1'], idx_b[(st, 'cn0_db', 100)]['acf1'], idx_b[(st, 'cn0_db', 1000)]['acf1'],
                idx_b[(st, 'qr', 20)]['acf1'], idx_b[(st, 'qr', 100)]['acf1'], idx_b[(st, 'qr', 1000)]['acf1'])
        dmax = max(abs(x - y) for x, y in zip(mine, ref))
        B(f'  {st:14s} 最大偏差 {dmax:.3f}  {"match" if dmax < 0.01 else "DIFF"}  本次 ' + ' '.join(f'{v:.3f}' for v in mine))


    kp = a.krule_csv or os.path.join(D, 're_ii', 'results_v32', 'krule_v12.csv')
    if os.path.exists(kp):
        kref = {r['star']: r for r in rm._read_csv(kp) if r['mode'] == 'alpha' and int(rm._f(r['K'])) == 5
                and abs(rm._f(r['level']) - 0.05) < 1e-9}
        nm = nd = 0
        for r in rows_b:
            if r['score'] != 'qr' or r['T_ms'] != 1:
                continue
            ref = kref.get(r['_key'])
            if ref is None:
                r['krule_v12_match'] = 'no_ref'
                continue
            r['krule_v12_D'] = rm._f(ref['D_clean'])
            ok = rm.close(r['p95_D'], r['krule_v12_D']) or (r['p95_D'] == 0 and r['krule_v12_D'] == 0)
            r['krule_v12_match'] = 'match' if ok else 'DIFF'
            nm += ok
            nd += not ok
        B(f'== 自检②：QR T=1 的 D_clean 对 krule_v12（alpha 0.05, K=5）：{nm} 一致，{nd} 不一致'
          + ('' if nd == 0 else '  [WARN] 先把 _c1b_star.csv 里 krule_v12_match=DIFF 的行看一下'))
    else:
        B(f'== 自检②：找不到 {kp}，跳过')

    fc.save_csv(prefix + '_c1b_star.csv', rows_b, C1B_STAR)
    pb = c1b_pooled(rows_b)
    fc.save_csv(prefix + '_c1b_pooled.csv', pb, C1B_POOL)
    fc.save_csv(prefix + '_c1p_star.csv', rows_p, C1P_STAR)
    pp = c1p_pooled(rows_p)
    fc.save_csv(prefix + '_c1p_pooled.csv', pp, C1P_POOL)


    B('\n== C1b：净段不重叠块均值（合并不含 ds8；ρ1 为跨星中位数）==')
    B('score     T(ms) | ρ1 all  stat  dyn   cplx | ρ1范围(all)   | VIF中位 | L中位 | D_p95 [事件,可分辨] | D_b05 [事件,判定,可分辨]')
    pbi = {(p['score'], p['T_ms'], p['group']): p for p in pb}
    for sc in SCORES:
        for T in TB:
            pa = pbi.get((sc, T, 'all'))
            if not pa:
                continue
            rr = ' '.join(f'{pbi[(sc, T, g)]["acf1_med"]:5.2f}' if (sc, T, g) in pbi else '  -  '
                          for g in ('static', 'dynamic', 'complex'))
            B(f'{sc:9s} {T:5d} | {pa["acf1_med"]:5.2f}  {rr} | [{pa["acf1_min"]:+.2f},{pa["acf1_max"]:+.2f}] | '
              f'{pa["vif_med"]:7.1f} | {pa["L_rule_med"]:5.0f} | {pa.get("p95_D", float("nan")):7.2f} '
              f'[{pa.get("p95_nev", 0)},{"是" if pa.get("p95_res") else "否"}] | {pa.get("b05_D", float("nan")):6.2f} '
              f'[{pa.get("b05_nev", 0)},{vshort(pa.get("b05_verdict", "none"))},{"是" if pa.get("b05_res") else "否"}]')


    B('\n== C1′：cn0 块均值上的 Gaussian LLR CUSUM（ho 合并，不含 ds8；W/C/E=verified/inconclusive/violated）==')
    B('T_sp  μ1    arm | β=0.1 ho率 判  | β=0.05        | β=0.02        | h中位@.05 | 检出 | onset中位@.02 | 退化/未找到')
    ppi = {(p['T_ms'], p['mu1'], p['arm'], p['beta'], p['group']): p for p in pp}
    for T in TSP:
        for mt, _, _ in MU1:
            for arm in ARMS:
                cells = []
                for beta in BETAS:
                    p = ppi.get((T, mt, arm, beta, 'all'), {})
                    cells.append(f'{p.get("ho_frac", float("nan")):7.2e} {vshort(p.get("ho_verdict", "none"))}'
                                 f'({p.get("ho_n_event", 0)})')
                p5 = ppi.get((T, mt, arm, 0.05, 'all'), {})
                p2 = ppi.get((T, mt, arm, 0.02, 'all'), {})
                ndg = sum(ppi.get((T, mt, arm, b, 'all'), {}).get('n_degenerate', 0) for b in BETAS)
                nnf = sum(ppi.get((T, mt, arm, b, 'all'), {}).get('n_not_found', 0) for b in BETAS)
                B(f'{T:4d}  {mt:5s} {arm:3s} | ' + ' | '.join(cells) + f' | {p5.get("h_med", float("nan")):9.3g} | '
                  f'{p2.get("n_det", 0):2d}/{p2.get("n_stars", 0):<2d} | {p2.get("delay_onset_med", float("nan")):+12.3f} | '
                  f'{ndg}/{nnf}')
    nb_ho = med([p['ho_blocks_total'] for p in pp if p['T_ms'] == 1000 and p['group'] == 'all'])
    B(f'注：T_sp=1000 ms 时 ho 合并共约 {nb_ho:.0f} 块、m=1 块，统计力很弱，判定基本只能是 C；只作方向参考。')
    B('注：ana=h=ln(N_fa)；iid=逐点打乱自助；blk=块自助。ana 与 iid 差 → 解析界松紧；iid 与 blk 差 → 独立性。')
    B('注：退化 = 净段里 s 从不为正（gm=0），任何 h>0 的自助误报都是 0，h 取到 ~1e-10；此时该档对"独立性"没有分辨力。')
    B('注：C1b 的 D_b05 判定参照 = 各星独立假设参照值（≈0.05）；E 表示块均值序列仍不满足独立。')
    B(f'[saved] {prefix}_c1b_star.csv / _c1b_pooled.csv / _c1p_star.csv / _c1p_pooled.csv / _log.txt / _brief.txt')
    B(f'[done] 用时 {(time.time() - t0) / 60:.1f} 分钟')
    with open(prefix + '_brief.txt', 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(brief) + '\n')
    log.close()
