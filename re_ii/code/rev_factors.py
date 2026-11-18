"""Sensitivity analysis for the QR score (Section 4.8, Table A4).

Allowance k0 in {1, 2, 4, 8}; block length and random seed; removal of the transient
near 58 s from the calibration segment. All other settings as in rev_main.py.
Run from the repository root (the working folder that also holds the data folders); see --help.
"""
import numpy as np, os, sys, argparse, time

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)
import rev_main as rm
fc = rm.fc
import e1b_acf_transient as e1b

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
DYN = rm.DYN_SCENES
SEEDS = [42, 123, 456, 789, 2024]


# ---------------------------------------------------------------------------
# Calibration helpers (block bootstrap with gaps)
# ---------------------------------------------------------------------------
def make_mp(M, m, k0):
    cache = {}

    def mp(h):
        key = round(float(h), 9)
        if key not in cache:
            cache[key] = fc.mpfa_batch(M, k0, key, m)[0]
        return cache[key]
    return mp


def pick_from_acf(a, thresh=0.05, lag_max=500, floor=10):
    """Block length from a precomputed ACF (Algorithm 1, step 2)."""
    below = 0
    for k in range(1, lag_max + 1):
        if abs(a[k]) < thresh:
            below += 1
            if below >= 3:
                return max(k - 2, floor)
        else:
            below = 0
    return lag_max


def synth_pieces(pieces, L, n_boot, seg_len, rng):
    """Block bootstrap with blocks confined to each piece (no block crosses a gap)."""
    Zcat = np.concatenate(pieces).astype(np.float64)
    starts, off = [], 0
    for p in pieces:
        if len(p) >= L:
            starts.append(off + np.arange(0, len(p) - L + 1))
        off += len(p)
    starts = np.concatenate(starts)
    n_blocks = int(np.ceil(seg_len / L))
    out = np.empty((n_boot, seg_len), dtype=np.float32)
    ar = np.arange(L)
    chunk = max(1, int(2e7 // (n_blocks * L)))
    for r0 in range(0, n_boot, chunk):
        r1 = min(n_boot, r0 + chunk)
        s = starts[rng.integers(0, len(starts), size=(r1 - r0, n_blocks))]
        out[r0:r1] = Zcat[(s[:, :, None] + ar).reshape(r1 - r0, -1)[:, :seg_len]]
    return out


def calib_eval(M, gm, k0, Z, Zc, seg, scen, betas, m, common, factor, level, L_used, seed, log, extra=None):
    rows = []
    mp = make_mp(M, m, k0)
    for beta in betas:
        h, info = fc.calibrate_h_v3(mp, gm, beta)
        r = dict(common, factor=factor, level=level, L_used=L_used, seed=seed, rule=f'{factor}={level}',
                 K='', beta=beta, status=info['status'], gm=gm)
        if extra:
            r.update(extra)
        if h is None:
            rows.append(r)
            continue
        r.update(thr=h, thr_z=h, h_ge_gm=(h >= 0.999 * gm))
        r.update(rm.evaluate(lambda X, _h=h: fc.alarm_idx(X, k0, _h), Z, Zc, seg, beta, m, scen))
        rows.append(r)
        log(f'  {factor}={level!s:6s} β={beta:.0e}: h={h:9.3f}{" ≥gm" if r["h_ge_gm"] else "    "} gm={gm:8.2f} '
            f'| ho {r["ho_n_event"]} 事件 {r["ho_verdict"]:12s} (ho0 {r["ho0_n_event"]}) '
            f'| onset {r["delay_onset_s"]:+.3f} s  act {r["delay_act_s"]:+.3f} s')
    return rows


# ---------------------------------------------------------------------------
# Factors varied one at a time (Section 4.8, Table A4)
# ---------------------------------------------------------------------------
# S1: k0 in {1, 2, 4, 8}; S2: block length and seed; S3: transient near 58 s masked.
def run_star(item, a, log):
    base, path, key = item
    S, lab, scen, prn = fc.load_star(path)
    S = np.ascontiguousarray(np.asarray(S, dtype=np.float64))
    gidx = int(round(a.guard_s * 1000))
    seg = fc.segments(S, lab, scen, gidx)
    cm = seg['cmask']
    Sc = S[cm]
    mu, sd = float(Sc.mean()), float(Sc.std())
    Z = np.ascontiguousarray((S - mu) / sd)
    Zc = np.ascontiguousarray(Z[cm])
    Ls = int(fc.pick_block_len(Zc))
    rho1 = float(fc.acf(Zc, 1))
    raw, m, seg_len = seg['raw'], a.m, int(a.seg * 1000)
    common = dict(star=base, scenario=scen, group=fc.group_of(scen), encoder='qr', prn=prn, N_clean_s=len(Zc) / 1000.0,
                  clean_raw_start_s=raw(seg['c0']), clean_raw_end_s=raw(seg['c1']), ho_raw_start_s=raw(seg['ho_a']),
                  ho_raw_end_s=raw(seg['hb_act']), ho0_raw_end_s=raw(seg['ho_b']), act_raw_s=rm.ACT[scen],
                  onset_raw_s=rm.AUTH[scen], L=Ls, L_capped=(Ls >= 500), rho1=rho1,
                  pooled_excluded=(scen in rm.POOL_EXCLUDE))
    log(f'\n=== {base}  L*={Ls}  ρ1={rho1:.3f}')
    rows, hvar = [], None
    dyn = scen in DYN
    M42 = None
    if 's1' in a.which or (dyn and 's2' in a.which):
        M42 = fc.synth_matrix(Zc, Ls, a.nboot, seg_len, fc.star_rng(a.seed, base, 1))
    if 's1' in a.which:
        for k0 in (1.0, 2.0, 4.0, 8.0):
            rows += calib_eval(M42, float(fc.cusum_gmax(Zc, k0)), k0, Z, Zc, seg, scen, a.betas, m, common,
                               'k0', int(k0), Ls, a.seed, log)
    if dyn and 's2' in a.which:
        gm4 = float(fc.cusum_gmax(Zc, rm.K0))
        for f in (0.25, 0.5, 1.0, 2.0):
            L = max(1, int(round(f * Ls)))
            M = M42 if L == Ls else fc.synth_matrix(Zc, L, a.nboot, seg_len, fc.star_rng(a.seed, base, 1))
            rows += calib_eval(M, gm4, rm.K0, Z, Zc, seg, scen, a.betas, m, common, 'Lmult', f, L, a.seed, log)
            del M
        hs = {}
        for sdv in SEEDS:
            M = M42 if sdv == a.seed else fc.synth_matrix(Zc, Ls, a.nboot, seg_len, fc.star_rng(sdv, base, 1))
            rr = calib_eval(M, gm4, rm.K0, Z, Zc, seg, scen, a.betas, m, common, 'seed', sdv, Ls, sdv, log)
            rows += rr
            for r in rr:
                if 'thr' in r:
                    hs.setdefault(r['beta'], []).append(r['thr'])
            del M
        hvar = [dict(star=base, beta=b, n=len(v), h_mean=float(np.mean(v)), h_std=float(np.std(v)),
                     h_relstd=float(np.std(v) / np.mean(v)), h_min=float(min(v)), h_max=float(max(v)))
                for b, v in sorted(hs.items())]
        for d in hvar:
            log(f'  [种子] β={d["beta"]:.0e}: h 均值 {d["h_mean"]:.3f}  相对标准差 {d["h_relstd"]:.3f}  '
                f'[{d["h_min"]:.3f}, {d["h_max"]:.3f}]')
    del M42
    if dyn and 's3' in a.which:
        asr = seg['asr']
        ia = max(seg['c0'], fc.idx_of(a.mask[0], asr))
        ib = min(seg['c1'], fc.idx_of(a.mask[1], asr))
        cm2 = cm.copy()
        cm2[ia:ib] = False
        mu2, sd2 = float(S[cm2].mean()), float(S[cm2].std())
        Z2 = np.ascontiguousarray((S - mu2) / sd2)
        idx = np.flatnonzero(cm2)
        cuts = np.flatnonzero(np.diff(idx) > 1) + 1
        pieces = [np.ascontiguousarray(Z2[p]) for p in np.split(idx, cuts)]
        w = cm2[seg['c0']:seg['c1']]
        acf2, _ = e1b.masked_acf(Z2[seg['c0']:seg['c1']], w, 600)
        L2 = pick_from_acf(acf2)
        gm2 = max(float(fc.cusum_gmax(p, rm.K0)) for p in pieces)
        M = synth_pieces(pieces, L2, a.nboot, seg_len, fc.star_rng(a.seed, base, 1))
        extra = dict(L=L2, rho1=float(acf2[1]), masked_s=(ib - ia) / 1000.0, gm_full=float(fc.cusum_gmax(Zc, rm.K0)),
                     L_full=Ls, rho1_full=rho1)
        log(f'  [S3] 剔除 raw {a.mask[0]}–{a.mask[1]} s（{(ib - ia) / 1000:.2f} s）：L {Ls}→{L2}  ρ1 {rho1:.3f}→{acf2[1]:.3f}  '
            f'gm {extra["gm_full"]:.2f}→{gm2:.2f}')
        rows += calib_eval(M, gm2, rm.K0, Z2, np.ascontiguousarray(Z2[cm2]), seg, scen, a.betas, m, common,
                           'mask58', 'yes', L2, a.seed, log, extra)
        del M
    return rows, hvar


ROW_FIELDS = ['factor', 'level', 'L_used', 'seed', 'masked_s', 'gm_full', 'L_full', 'rho1_full'] + rm.STAR_FIELDS
HVAR_FIELDS = ['star', 'beta', 'n', 'h_mean', 'h_std', 'h_relstd', 'h_min', 'h_max']

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', dest='npz_dir', default='.')
    ap.add_argument('--which', nargs='+', default=['s1', 's2', 's3'], choices=['s1', 's2', 's3'])
    ap.add_argument('--stars', nargs='+', default=None)
    ap.add_argument('--betas', nargs='+', type=float, default=[1e-1, 2e-2])
    ap.add_argument('--nboot', type=int, default=1000)
    ap.add_argument('--seg', type=float, default=80.0)
    ap.add_argument('--m', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--guard-s', dest='guard_s', type=float, default=0.1)
    ap.add_argument('--mask', nargs=2, type=float, default=[57.0, 60.0], help='S3 剔除的 raw 起止秒')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    base_out = os.path.splitext(a.out)[0]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    log = fc.Tee(base_out + '_log.txt')
    log(f'[version] rev_factors.py v1.0（2026-10-16）  which={a.which}  β={a.betas}  nboot={a.nboot}  seed={a.seed}  '
        f'mask={a.mask}')
    items = rm.qr_items(os.path.abspath(a.npz_dir), a.stars)
    log(f'[cfg] {len(items)} 颗星')
    t0 = time.time()
    rows, hv = [], []
    for i, it in enumerate(items, 1):
        log(f'[{i}/{len(items)}]')
        r, h = run_star(it, a, log)
        rows += r
        hv += (h or [])
    rm.save_csv(base_out + '_rows.csv', rows, ROW_FIELDS)
    pool = rm.pooled(rows, a.m, rm.GROUPS)
    rm.save_csv(base_out + '_pooled.csv', pool, rm.POOL_FIELDS)
    if hv:
        rm.save_csv(base_out + '_s2_hvar.csv', hv, HVAR_FIELDS)
    log('\n==== 合并（误报不含 ds8）====')
    for p in pool:
        if p['group'] not in ('all', 'dynamic'):
            continue
        log(f'{p["rule"]:14s} β={p["beta"]:.0e} {p["group"]:8s} | ho {p.get("ho_n_event", 0):3d} 事件 '
            f'{p.get("ho_frac", float("nan")):.2e} {p.get("ho_verdict", ""):12s} | ho0 {p.get("ho0_n_event", 0):3d} '
            f'{p.get("ho0_verdict", ""):12s} | onset 中位 {p.get("delay_onset_med", float("nan")):+.3f} s')
    log(f'[saved] {base_out}_rows.csv / _pooled.csv / _s2_hvar.csv / _log.txt')
    log(f'[done] 总用时 {(time.time() - t0) / 60:.1f} 分钟。把 _pooled.csv、_s2_hvar.csv、_log.txt 发回即可。')
    log.close()
