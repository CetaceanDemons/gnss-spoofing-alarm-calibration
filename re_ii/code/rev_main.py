"""Main pipeline for the learned scores (--encoder qr | ae | tcn).

For each satellite, calibrates four rules at the same targets: the CUSUM with a
block-bootstrap limit h, the count rule (K = 5) with a block-bootstrap threshold, the
count rule with its threshold set under independence, and the count rule at the 95th
percentile. Verifies each on the held-out segment and measures detection delays
(Tables 5 and 7, Figures 8 and 10). With --encoder qr it first reproduces the
limits of far_calib_v32.py and stops on any mismatch.
Run from the repository root (the working folder that also holds the data folders); see --help.
"""
import numpy as np, os, sys, re, glob, csv, argparse, time

_here = os.path.dirname(os.path.abspath(__file__))
ROOT = None
for _d in (_here, os.path.dirname(_here), os.path.dirname(os.path.dirname(_here))):
    if os.path.exists(os.path.join(_d, 'far_mc_beta_neweval.py')):
        ROOT = _d
        if _d not in sys.path:
            sys.path.insert(0, _d)
        break
if _here not in sys.path:
    sys.path.insert(0, _here)

import far_calib_v32 as fc
import krule_clean_far_v12 as kr
import far_mc_beta_neweval_v2 as v2

# ---------------------------------------------------------------------------
# Settings shared with far_calib_v32.py
# ---------------------------------------------------------------------------
K0 = fc.K0
BETAS_ALL = list(fc.BETAS)
BETAS_MAIN = [1e-1, 5e-2, 2e-2]
SCENARIOS, DYN_SCENES, POOL_EXCLUDE = fc.SCENARIOS, fc.DYN_SCENES, fc.POOL_EXCLUDE
ACT = fc.ACTIVATION_RAW
AUTH = fc.AUTH
GROUPS = ('all', 'static', 'dynamic', 'complex')

# Delays reported in the conference paper [3]; used only as a regression check.
FIRST_PAPER_T2 = {'ds3': 0.089, 'ds4_r2': 0.179, 'ds5_r2': 0.077, 'ds6_r2': 0.004, 'ds7': 45.180, 'ds8': 19.386}
DEFAULT_DUAL = {'ae': ('lstm_ae_results', 'seed42_dual', 'seed42_dual_dyn'),
                'tcn': ('tcn_results', 'seed42_dual', 'seed42_dual_dyn')}

try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False


# ---------------------------------------------------------------------------
# Count rule calibrated on the block-bootstrap sequences (Algorithm 1, step 8)
# ---------------------------------------------------------------------------
def _kmpfa_np(X, gamma, K, m):
    n_boot, seg_len = X.shape
    nwin = seg_len - m + 1
    run = np.zeros(n_boot, dtype=np.int64)
    last_end = np.full(n_boot, -1, dtype=np.int64)
    covered = np.zeros(n_boot, dtype=np.int64)
    for t in range(seg_len):
        run = np.where(X[:, t] > gamma, run + 1, 0)
        fired = run >= K
        if fired.any():
            run[fired] = 0
            s0 = np.maximum(max(0, t - m + 1), last_end[fired] + 1)
            s1 = min(t, nwin - 1)
            add = np.maximum(0, s1 - s0 + 1)
            covered[fired] += add
            last_end[fired] = np.where(add > 0, s1, last_end[fired])
    return covered.sum() / float(n_boot * nwin)


if HAVE_NUMBA:
    @njit(cache=True)
    def _kmpfa_nb(X, gamma, K, m):
        n_boot, seg_len = X.shape
        nwin = seg_len - m + 1
        covered = 0
        for b in range(n_boot):
            run = 0
            last_end = -1
            for t in range(seg_len):
                if X[b, t] > gamma:
                    run += 1
                    if run >= K:
                        run = 0
                        s0 = t - m + 1
                        if s0 < 0:
                            s0 = 0
                        if s0 <= last_end:
                            s0 = last_end + 1
                        s1 = t
                        if s1 > nwin - 1:
                            s1 = nwin - 1
                        if s1 >= s0:
                            covered += s1 - s0 + 1
                            last_end = s1
                else:
                    run = 0
        return covered / (n_boot * nwin)

    def kmpfa(X, gamma, K, m):
        return _kmpfa_nb(X, float(gamma), int(K), int(m))
else:
    kmpfa = _kmpfa_np


def calibrate_gamma_k(Xmat, cand, K, m, beta, cache=None):
    """Smallest count-rule threshold with share <= beta on the block-bootstrap matrix."""
    cache = {} if cache is None else cache

    def f(j):
        if j not in cache:
            cache[j] = kmpfa(Xmat, float(cand[j]), K, m)
        return cache[j]
    n = len(cand)
    hi = n - 1
    lo = n // 2
    if f(lo) <= beta:
        hi, lo = lo, 0
        if f(0) <= beta:
            return 0, f(0), 'degenerate'
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if f(mid) <= beta:
            hi = mid
        else:
            lo = mid
    return hi, f(hi), ('saturated' if hi == n - 1 else 'ok')


# ---------------------------------------------------------------------------
# Score files
# ---------------------------------------------------------------------------
def qr_items(npz_dir, stars):
    out = []
    for f in fc.select_files(npz_dir, 'dyn', None):
        key = os.path.basename(f)[len('pertap_'):-len('.npz')]
        base = key[:-len('_dyn')] if key.endswith('_dyn') else key
        if stars and not any(s in base for s in stars):
            continue
        out.append((base, f, key))
    return out


_DUAL_RE = re.compile(r'^(?P<scen>ds\d(?:_r2)?)_prn(?P<prn>\d+)_.*scores_dual(?:_dyn)?\.npz$')


def dual_items(static_dir, dyn_dir, stars, log):
    """Score files from the static folder (ds3/ds4/ds7/ds8) and the dynamic folder (ds5/ds6)."""
    out = []
    for d, want_dyn in ((static_dir, False), (dyn_dir, True)):
        if not os.path.isdir(d):
            raise SystemExit(f'[ERROR] 分数目录不存在：{d}')
        for f in sorted(glob.glob(os.path.join(d, '*scores_dual*.npz'))):
            mt = _DUAL_RE.match(os.path.basename(f))
            if not mt or mt.group('scen') not in SCENARIOS:
                continue
            if (mt.group('scen') in DYN_SCENES) != want_dyn:
                continue
            base = f"{mt.group('scen')}_prn{mt.group('prn')}"
            if stars and not any(s in base for s in stars):
                continue
            out.append((base, f, base))
    if not stars and len(out) != 21:
        log(f'[WARN] 找到 {len(out)} 颗星，预期 21：' + ', '.join(b for b, _, _ in out))
    return out


def load_any(path, encoder):
    if encoder == 'qr':
        return fc.load_star(path)
    return v2.load_S_scores(path)


# ---------------------------------------------------------------------------
# Regression check against the limits of far_calib_v32.py
# ---------------------------------------------------------------------------
def _read_csv(path):
    with open(path, encoding='utf-8-sig', newline='') as fh:
        return list(csv.DictReader(fh))


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float('nan')


def load_regress(cal_path, k_path, log):
    reg = {'cal': {}, 'k': {}}
    if cal_path and os.path.exists(cal_path):
        for r in _read_csv(cal_path):
            reg['cal'][(r['star'], r['calib'], round(_f(r['beta']), 12))] = r
        log(f'[回归] 读入 {cal_path}：{len(reg["cal"])} 行')
    else:
        log(f'[回归] 找不到 {cal_path}，CUSUM 门限无法回归核对')
    if k_path and os.path.exists(k_path):
        for r in _read_csv(k_path):
            reg['k'][(r['star'], r['mode'], int(_f(r['K'])), round(_f(r['level']), 12))] = r
        log(f'[回归] 读入 {k_path}：{len(reg["k"])} 行')
    else:
        log(f'[回归] 找不到 {k_path}，K 规则门限无法回归核对')
    return reg


def close(a, b, rel=2e-5):
    """Relative tolerance 2e-5 (csv values have six significant digits)."""
    if not (np.isfinite(a) and np.isfinite(b)):
        return False
    return abs(a - b) <= rel * max(abs(a), abs(b), 1e-12)


# ---------------------------------------------------------------------------
# Detection delay (Section 4.6)
# ---------------------------------------------------------------------------
# Delay = first alarm at or after the onset, minus the onset.
def first_at_or_after(idx, i0):
    j = int(np.searchsorted(idx, i0, side='left'))
    return int(idx[j]) if j < len(idx) else None


def evaluate(alarm_fn, X, Xc, seg, ref, m, scen):
    raw = seg['raw']
    N = len(Xc)
    cal_idx = np.asarray(alarm_fn(Xc), dtype=np.int64)
    cov, nwin = fc.window_cov(cal_idx, 0, N, m)
    nev = fc.n_events(cal_idx, m)
    lo, up = fc.win_rate_bounds(nev, cov, nwin, m)
    r = dict(ref=ref, cal_n_event=nev, cal_cov=cov, cal_nwin=nwin,
             cal_frac=(cov / nwin if nwin > 0 else float('nan')), cal_lo95=lo, cal_up95=up,
             cal_verdict=fc.verdict(lo, up, ref, nwin))
    for v, bb in (('ho', seg['hb_act']), ('ho0', seg['ho_b'])):
        fc.put_ho(r, v, fc.ho_eval(alarm_fn, X, seg['ho_a'], bb, [], m), ref, raw)
    full = np.asarray(alarm_fn(X), dtype=np.int64)
    ia = first_at_or_after(full, seg['act_i'])
    io = first_at_or_after(full, seg['oi'])
    r.update(det=(ia is not None),
             det_raw_s=(raw(ia) if ia is not None else float('nan')),
             delay_act_s=(raw(ia) - ACT[scen] if ia is not None else float('nan')),
             det_onset_raw_s=(raw(io) if io is not None else float('nan')),
             delay_onset_s=(raw(io) - AUTH[scen] if io is not None else float('nan')),
             n_alarm_act_onset=int(np.sum((full >= seg['act_i']) & (full < seg['oi']))))
    return r


# ---------------------------------------------------------------------------
# Per satellite: four rules at the same targets (Tables 5 and 7, Figure 8)
# ---------------------------------------------------------------------------
# CUSUM with block-bootstrap h; K = 5 with block-bootstrap gamma; K = 5 with gamma set
# under independence; K = 5 at the 95th percentile of the calibration segment.
def run_star(item, a, log, reg, mism, enc):
    base, path, key = item
    t0 = time.time()
    S, lab, scen, prn = load_any(path, enc)
    S = np.ascontiguousarray(np.asarray(S, dtype=np.float64))
    if not np.any(lab == 0):
        log(f'[skip] {base}: 没有 labels==0 的净段')
        return [], None, None
    gidx = int(round(a.guard_s * 1000))
    seg = fc.segments(S, lab, scen, gidx)
    raw = seg['raw']
    cm = seg['cmask']
    Sc = np.ascontiguousarray(S[cm])
    N = len(Sc)
    if N < 4000:
        log(f'[skip] {base}: 净段太短 ({N} 点)')
        return [], None, None
    mu, sd = float(Sc.mean()), float(Sc.std())
    Z = np.ascontiguousarray((S - mu) / sd)
    Zc = np.ascontiguousarray(Z[cm])
    L = int(fc.pick_block_len(Zc))
    rho1 = float(fc.acf(Zc, 1))
    gm = float(fc.cusum_gmax(Zc, K0))
    grp = fc.group_of(scen)
    m, K = a.m, a.K
    common = dict(star=base, scenario=scen, group=grp, encoder=enc, prn=prn, N_clean_s=N / 1000.0,
                  clean_raw_start_s=raw(seg['c0']), clean_raw_end_s=raw(seg['c1']),
                  ho_raw_start_s=raw(seg['ho_a']), ho_raw_end_s=raw(seg['hb_act']), ho0_raw_end_s=raw(seg['ho_b']),
                  act_raw_s=ACT[scen], onset_raw_s=AUTH[scen], L=L, L_capped=(L >= 500), rho1=rho1, gm=gm,
                  pooled_excluded=(scen in POOL_EXCLUDE))
    log(f'\n=== {base} ({grp}, {enc})  净段 {raw(seg["c0"]):.2f}–{raw(seg["c1"]):.2f} s  N={N / 1000:.1f} s'
        f'  ho {raw(seg["ho_a"]):.2f}–{raw(seg["hb_act"]):.2f} s  ρ1={rho1:.3f}  L={L}'
        f'{"(顶到上限)" if L >= 500 else ""}  gm={gm:.2f}' + ('  [不进误报合并]' if scen in POOL_EXCLUDE else ''))

    Mblk = fc.synth_matrix(Zc, L, a.nboot, int(a.seg * 1000), fc.star_rng(a.seed, base, 1))
    mp = fc.make_mp(Mblk, m)
    rows = []
    do_h_reg = (enc == 'qr' and a.nboot == 2000 and a.seg == 80.0 and a.seed == 42 and m == 1000
                and abs(a.guard_s - 0.1) < 1e-12 and reg['cal'])
    do_k_reg = (enc == 'qr' and m == 1000 and abs(a.guard_s - 0.1) < 1e-12 and K == 5 and reg['k'])

    # ---- (a) CUSUM
    for beta in BETAS_ALL:
        h, info = fc.calibrate_h_v3(mp, gm, beta)
        r = dict(common, rule='cusum', K='', beta=beta, status=info['status'])
        if h is None:
            rows.append(r)
            log(f'  cusum β={beta:.0e}: h 未找到 ({info["status"]})')
            continue
        r.update(thr=h, thr_z=h, h_ge_gm=(h >= 0.999 * gm))
        r.update(evaluate(lambda X, _h=h: fc.alarm_idx(X, K0, _h), Z, Zc, seg, beta, m, scen))
        if do_h_reg:
            ref = reg['cal'].get((key, 'blk', round(beta, 12)))
            if ref is None:
                r['regress'] = 'no_ref'
            else:
                ok = (close(h, _f(ref['h'])) and int(_f(ref['ho_n_event'])) == r['ho_n_event']
                      and int(_f(ref['ho0_n_event'])) == r['ho0_n_event'])
                r['regress'] = 'match' if ok else 'MISMATCH'
                if not ok:
                    mism.append(f'{base} cusum β={beta:g}: h {h:.6g} vs {ref["h"]}；ho 事件 {r["ho_n_event"]} vs '
                                f'{ref["ho_n_event"]}；ho0 事件 {r["ho0_n_event"]} vs {ref["ho0_n_event"]}')
        rows.append(r)


    Z32 = np.ascontiguousarray(Z.astype(np.float32).astype(np.float64))
    Zc32 = np.ascontiguousarray(Z32[cm])
    cand = np.unique(Zc.astype(np.float32)).astype(np.float64)
    kfun = (lambda _g: (lambda X: kr.krun_idx(X, _g, K)))
    kcache = {}
    for beta in BETAS_MAIN:

        p_t = kr.p_for_target(beta, K, m)
        g = float(np.quantile(Sc, 1.0 - p_t))
        p_hat = float(np.mean(Sc > g))
        nom = kr.nominal_iid(p_hat, K, m)
        r = dict(common, rule='k_iid', K=K, beta=beta, status='ok', thr=g, thr_z=(g - mu) / sd, p_hat=p_hat, nominal_iid=nom)
        r.update(evaluate(kfun(g), S, Sc, seg, nom, m, scen))
        if do_k_reg:
            ref = reg['k'].get((key, 'beta', 5, round(beta, 12)))
            if ref is None:
                r['regress'] = 'no_ref'
            else:
                ok = close(g, _f(ref['gamma'])) and int(_f(ref['ho_n_event'])) == r['ho_n_event']
                r['regress'] = 'match' if ok else 'MISMATCH'
                if not ok:
                    mism.append(f'{base} k_iid β={beta:g}: γ {g:.6g} vs {ref["gamma"]}；ho 事件 {r["ho_n_event"]} vs {ref["ho_n_event"]}')
        rows.append(r)

        j, fj, st = calibrate_gamma_k(Mblk, cand, K, m, beta, kcache)
        gz = float(cand[j])
        p_hat_c = float(np.mean(Zc32 > gz))
        r = dict(common, rule='k_blk', K=K, beta=beta, status=st, thr=mu + sd * gz, thr_z=gz, p_hat=p_hat_c,
                 nominal_iid=kr.nominal_iid(p_hat_c, K, m), boot_mpfa=fj)
        r.update(evaluate(kfun(gz), Z32, Zc32, seg, beta, m, scen))
        rows.append(r)
    # (d) p95
    g = float(np.quantile(Sc, 0.95))
    p_hat = float(np.mean(Sc > g))
    nom = kr.nominal_iid(p_hat, K, m)
    r = dict(common, rule='k_p95', K=K, beta=float('nan'), status='ok', thr=g, thr_z=(g - mu) / sd, p_hat=p_hat, nominal_iid=nom)
    r.update(evaluate(kfun(g), S, Sc, seg, nom, m, scen))
    if do_k_reg:
        ref = reg['k'].get((key, 'alpha', 5, 0.05))
        if ref is None:
            r['regress'] = 'no_ref'
        else:
            ok = close(g, _f(ref['gamma'])) and int(_f(ref['ho_n_event'])) == r['ho_n_event']
            r['regress'] = 'match' if ok else 'MISMATCH'
            if not ok:
                mism.append(f'{base} k_p95: γ {g:.6g} vs {ref["gamma"]}；ho 事件 {r["ho_n_event"]} vs {ref["ho_n_event"]}')
    rows.append(r)
    del Mblk, mp


    asr = v2.ALIGN_START_MS.get(scen, 0)
    if scen in v2.ABS_CALIB_SCENARIOS:
        lo_i = max(0, int(round(v2.CALIB_RAW_S[0] * 1000 - 100 - asr)))
        hi_i = min(len(S), int(round(v2.CALIB_RAW_S[1] * 1000 - 100 - asr)))
        cm2 = np.zeros(len(S), dtype=bool)
        cm2[lo_i:hi_i] = True
    else:
        cm2 = (lab == 0)
    g2 = float(np.quantile(S[cm2], 0.95))
    a5 = kr.krun_idx(S, g2, 5)
    chk = dict(star=base, scenario=scen, encoder=enc, gamma_v2=g2,
               first5_raw_s=(raw(int(a5[0])) if len(a5) else float('nan')),
               mao_alarm_s=(raw(int(a5[0])) - AUTH[scen] if len(a5) else float('nan')),
               first_paper_t2_s=FIRST_PAPER_T2.get(scen, float('nan')))


    xchk = None
    if enc != 'qr' and a.pertap_map.get(base):
        S2, lab2, _, _ = fc.load_star(a.pertap_map[base])
        seg2 = fc.segments(S2, lab2, scen, gidx)
        same = (len(S2) == len(S) and all(seg2[k] == seg[k] for k in ('c0', 'c1', 'ho_a', 'ho_b', 'hb_act', 'oi', 'act_i')))
        xchk = 'same' if same else 'DIFF'
        if not same:
            log(f'  [WARN] 与 pertap 不一致：长度 {len(S)} vs {len(S2)}；c0/c1/ho_a/hb_act '
                f'{seg["c0"]}/{seg["c1"]}/{seg["ho_a"]}/{seg["hb_act"]} vs '
                f'{seg2["c0"]}/{seg2["c1"]}/{seg2["ho_a"]}/{seg2["hb_act"]}')
    for r in rows:
        r['xcheck_pertap'] = xchk

    for r in rows:
        if r['rule'] == 'cusum' and r['beta'] not in BETAS_MAIN:
            continue
        if 'thr' not in r:
            continue
        tag = r['rule'] + (f' β={r["beta"]:.0e}' if np.isfinite(r['beta']) else '')
        log(f'  {tag:16s} thr={r["thr"]:9.4f} {r["status"]:9s} 净段 {r["cal_frac"]:.2e}({r["cal_n_event"]}) '
            f'| ho {r["ho_n_event"]} 事件 {r["ho_frac"]:.2e} {r["ho_verdict"]:12s} (ho0 {r["ho0_n_event"]}) '
            f'| 时延 act {r["delay_act_s"]:+.3f} s  onset {r["delay_onset_s"]:+.3f} s'
            + (f'  [{r["regress"]}]' if r.get('regress') else ''))
    log(f'  天然校验(第一篇口径) γ={g2:.4f}  首次连续5次 {chk["first5_raw_s"]:.3f} s → 相对 onset {chk["mao_alarm_s"]:+.3f} s'
        + (f'  (第一篇表 2 该场景代表星 {FIRST_PAPER_T2[scen]:+.3f} s)' if scen in FIRST_PAPER_T2 else '')
        + f'   用时 {time.time() - t0:.0f} s')
    return rows, chk, xchk


# ---------------------------------------------------------------------------
# Pooling and output
# ---------------------------------------------------------------------------
def pooled(rows, m, keys):
    out = []
    combos = sorted({(r['rule'], (r['beta'] if np.isfinite(r['beta']) else -1.0)) for r in rows if 'thr' in r},
                    key=lambda x: (x[0], -x[1]))
    for rule, b in combos:
        for g in keys:
            sel = [r for r in rows if 'thr' in r and r['rule'] == rule
                   and (r['beta'] if np.isfinite(r['beta']) else -1.0) == b
                   and (g == 'all' or r['group'] == g or r['scenario'] == g)]
            if not sel:
                continue
            fa = [r for r in sel if r['scenario'] not in POOL_EXCLUDE]
            d = dict(rule=rule, beta=(b if b > 0 else float('nan')), group=g, n_stars=len(sel), n_stars_fa=len(fa))
            cnw = sum(r['cal_nwin'] for r in fa)
            ref = (b if rule in ('cusum', 'k_blk') else
                   (sum(r['ref'] * r['cal_nwin'] for r in fa) / cnw if cnw > 0 else float('nan')))
            d['ref'] = ref
            if cnw > 0:
                ccov = sum(r['cal_cov'] for r in fa)
                cev = sum(r['cal_n_event'] for r in fa)
                lo, up = fc.win_rate_bounds(cev, ccov, cnw, m)
                d.update(cal_frac=ccov / cnw, cal_n_event=cev, cal_lo95=lo, cal_up95=up,
                         cal_verdict=fc.verdict(lo, up, ref, cnw))
            for v in ('ho', 'ho0'):
                hs = [r for r in fa if r.get(f'{v}_nwin', 0) > 0]
                hnw = sum(r[f'{v}_nwin'] for r in hs)
                if hnw <= 0:
                    d[f'{v}_verdict'] = 'none'
                    continue
                hcov = sum(r[f'{v}_cov'] for r in hs)
                hev = sum(r[f'{v}_n_event'] for r in hs)
                lo, up = fc.win_rate_bounds(hev, hcov, hnw, m)
                d.update({f'{v}_n_stars': len(hs), f'{v}_total_s': sum(r[f'{v}_len'] for r in hs) / 1000.0,
                          f'{v}_n_event': hev, f'{v}_frac': hcov / hnw, f'{v}_lo95': lo, f'{v}_up95': up,
                          f'{v}_verdict': fc.verdict(lo, up, ref, hnw)})
            da = [r['delay_act_s'] for r in sel if r['det']]
            do = [r['delay_onset_s'] for r in sel if np.isfinite(r['delay_onset_s'])]
            d.update(n_det=len(da), n_miss=len(sel) - len(da))
            if da:
                d.update(delay_act_med=float(np.median(da)), delay_act_min=float(min(da)), delay_act_max=float(max(da)))
            if do:
                d.update(delay_onset_med=float(np.median(do)), delay_onset_min=float(min(do)),
                         delay_onset_max=float(max(do)))
            out.append(d)
    return out


STAR_FIELDS = ['star', 'scenario', 'group', 'encoder', 'prn', 'N_clean_s', 'clean_raw_start_s', 'clean_raw_end_s',
               'ho_raw_start_s', 'ho_raw_end_s', 'ho0_raw_end_s', 'act_raw_s', 'onset_raw_s', 'L', 'L_capped', 'rho1',
               'gm', 'pooled_excluded', 'xcheck_pertap', 'rule', 'K', 'beta', 'status', 'thr', 'thr_z', 'h_ge_gm',
               'p_hat', 'nominal_iid', 'boot_mpfa', 'ref', 'cal_n_event', 'cal_cov', 'cal_nwin', 'cal_frac',
               'cal_lo95', 'cal_up95', 'cal_verdict'] + \
              [f'{v}_{k}' for v in ('ho', 'ho0') for k in ('len_s', 'n_alarm', 'n_event', 'cov', 'nwin', 'frac',
                                                          'lo95', 'up95', 'verdict', 'times')] + \
              ['det', 'det_raw_s', 'delay_act_s', 'det_onset_raw_s', 'delay_onset_s', 'n_alarm_act_onset', 'regress']
POOL_FIELDS = ['rule', 'beta', 'group', 'n_stars', 'n_stars_fa', 'ref', 'cal_frac', 'cal_n_event', 'cal_lo95',
               'cal_up95', 'cal_verdict'] + \
              [f'{v}_{k}' for v in ('ho', 'ho0') for k in ('n_stars', 'total_s', 'n_event', 'frac', 'lo95', 'up95',
                                                          'verdict')] + \
              ['n_det', 'n_miss', 'delay_act_med', 'delay_act_min', 'delay_act_max',
               'delay_onset_med', 'delay_onset_min', 'delay_onset_max']
CHK_FIELDS = ['star', 'scenario', 'encoder', 'gamma_v2', 'first5_raw_s', 'mao_alarm_s', 'v2_csv_mao_alarm_s',
              'v2_match', 'first_paper_t2_s']


def save_csv(path, rows, fields):
    with open(path, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.writer(fh)
        w.writerow(fields)
        for r in rows:
            w.writerow([fc.fmt(r.get(k)) for k in fields])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', dest='npz_dir', default='.', help=r'processed_Mao 根目录')
    ap.add_argument('--encoder', choices=['qr', 'ae', 'tcn'], default='qr')
    ap.add_argument('--scores-dir', default=None, help='非 qr：静态场景分数目录（默认按 --encoder 推）')
    ap.add_argument('--scores-dir-dyn', default=None, help='非 qr：动态场景分数目录（默认按 --encoder 推）')
    ap.add_argument('--stars', nargs='+', default=None)
    ap.add_argument('--K', type=int, default=5)
    ap.add_argument('--nboot', type=int, default=2000)
    ap.add_argument('--seg', type=float, default=80.0)
    ap.add_argument('--m', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--guard-s', dest='guard_s', type=float, default=0.1)
    ap.add_argument('--regress-cal', default=None, help=r'默认 <dir>\re_ii\results_v32\calib_v32_dyn.csv')
    ap.add_argument('--regress-k', default=None, help=r'默认 <dir>\re_ii\results_v32\krule_v12.csv')
    ap.add_argument('--v2-csv', default=None,
                    help=r'天然校验参照；qr 默认 <dir>\results_v2b_1e-5\far_mc_v2_1e-5_lstm.csv，'
                         r'ae 默认 <dir>\results_prerun_nb1000\pre_ae_nb1000_lstm.csv')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    D = os.path.abspath(a.npz_dir)
    base_out = os.path.splitext(a.out)[0]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    log = fc.Tee(base_out + '_log.txt')
    log('[version] rev_main.py v1.0（2026-10-16）')
    log(f'[numba] {"ON" if HAVE_NUMBA else "OFF（会很慢）"}')
    log(f'[cfg] encoder={a.encoder}  K={a.K}  K0={K0}  m={a.m}  nboot={a.nboot}  seg={a.seg} s  seed={a.seed}  '
        f'guard={a.guard_s} s')
    log(f'[cfg] CUSUM β={BETAS_ALL}；K 规则 β={BETAS_MAIN}；ACTIVATION_RAW={ACT}')

    if a.encoder == 'qr':
        items = qr_items(D, a.stars)
        a.pertap_map = {}
    else:
        sub, s1, s2 = DEFAULT_DUAL[a.encoder]
        sd_ = a.scores_dir or os.path.join(D, sub, s1)
        sdd = a.scores_dir_dyn or os.path.join(D, sub, s2)
        log(f'[cfg] 分数目录：静态 {sd_}  动态 {sdd}')
        items = dual_items(sd_, sdd, a.stars, log)
        a.pertap_map = {b: f for b, f, _ in qr_items(D, None)}
    log(f'[cfg] {len(items)} 颗星：' + ', '.join(b for b, _, _ in items))
    if not items:
        log('[ERROR] 没找到星，检查 --dir / --encoder / --stars')
        log.close()
        sys.exit(1)

    rp = os.path.join(D, 're_ii', 'results_v32')
    reg = (load_regress(a.regress_cal or os.path.join(rp, 'calib_v32_dyn.csv'),
                        a.regress_k or os.path.join(rp, 'krule_v12.csv'), log)
           if a.encoder == 'qr' else {'cal': {}, 'k': {}})
    if a.encoder == 'qr' and a.nboot != 2000:
        log('[回归] nboot ≠ 2000：CUSUM 门限 h 不做回归（只核 K 规则 γ）')

    t_all = time.time()
    rows, chks, mism, xdiff = [], [], [], []
    for i, it in enumerate(items, 1):
        log(f'[{i}/{len(items)}]')
        rr, chk, xc = run_star(it, a, log, reg, mism, a.encoder)
        rows += rr
        if chk:
            chks.append(chk)
        if xc == 'DIFF':
            xdiff.append(it[0])


    v2p = a.v2_csv or (os.path.join(D, 'results_v2b_1e-5', 'far_mc_v2_1e-5_lstm.csv') if a.encoder == 'qr' else
                       os.path.join(D, 'results_prerun_nb1000', 'pre_ae_nb1000_lstm.csv'))
    col = 'mao_alarm_s@5e-02'
    if os.path.exists(v2p):
        ref = {r['star']: r for r in _read_csv(v2p)}
        nm = 0
        for c in chks:
            rv = ref.get(c['star'], {}).get(col)
            c['v2_csv_mao_alarm_s'] = _f(rv)
            if rv is None or rv == '':
                c['v2_match'] = 'no_ref'
            else:
                c['v2_match'] = 'match' if abs(_f(rv) - c['mao_alarm_s']) < 1.5e-3 else 'DIFF'
                nm += c['v2_match'] == 'DIFF'
        log(f'\n[天然校验] 对照 {v2p} 的 {col}：{len(chks) - nm}/{len(chks)} 颗星一致（容差 1.5 ms）')
    else:
        log(f'\n[天然校验] 找不到 {v2p}，只输出本口径数值')

    bad = a.encoder == 'qr' and len(mism) > 0
    n_cmp = sum(1 for r in rows if r.get('regress') in ('match', 'MISMATCH'))
    if a.encoder == 'qr':
        log(f'\n==== 回归核对：比对 {n_cmp} 项，不一致 {len(mism)} 项 ====')
        for s in mism[:40]:
            log('  ' + s)
        if n_cmp == 0:
            log('  [WARN] 一项都没比上（参照 CSV 缺失或参数不同），结果未经回归核对')
    if xdiff:
        log(f'[WARN] {len(xdiff)} 颗星与 pertap 长度/段边界不一致：{xdiff}')
    tag = '_UNVERIFIED' if bad else ''
    if bad:
        log('[STOP] 回归不一致：结果文件名加 _UNVERIFIED，先把日志发回来，不要用这批数字')
    save_csv(base_out + tag + '_star.csv', rows, STAR_FIELDS)
    pool = pooled(rows, a.m, GROUPS)
    save_csv(base_out + tag + '_pooled.csv', pool, POOL_FIELDS)
    save_csv(base_out + tag + '_pooled_scen.csv', pooled(rows, a.m, sorted({r['scenario'] for r in rows})), POOL_FIELDS)
    save_csv(base_out + tag + '_check.csv', chks, CHK_FIELDS)

    log('\n==== 合并（误报不含 ds8；时延含 ds8）====')
    log(f'{"rule":6s} {"β":>7s} {"group":8s} {"参照":>9s} | {"净段":>9s} {"判定":12s} | {"ho事件":>5s} {"ho率":>9s} '
        f'{"判定":12s} | {"检出":>5s} {"时延act中位":>11s} {"范围":>19s} | {"onset中位":>9s}')
    for p in pool:
        if p['rule'] == 'cusum' and p['beta'] not in BETAS_MAIN:
            continue
        b = f'{p["beta"]:.0e}' if np.isfinite(p['beta']) else '  p95'
        rng = (f'[{p["delay_act_min"]:+.3f},{p["delay_act_max"]:+.3f}]' if 'delay_act_min' in p else '—')
        log(f'{p["rule"]:6s} {b:>7s} {p["group"]:8s} {p["ref"]:9.2e} | {p.get("cal_frac", float("nan")):9.2e} '
            f'{p.get("cal_verdict", ""):12s} | {p.get("ho_n_event", 0):5d} {p.get("ho_frac", float("nan")):9.2e} '
            f'{p.get("ho_verdict", ""):12s} | {p["n_det"]:2d}/{p["n_stars"]:<2d} '
            f'{p.get("delay_act_med", float("nan")):+11.3f} {rng:>19s} | {p.get("delay_onset_med", float("nan")):+9.3f}')
    nc = sum(1 for r in rows if r['rule'] == 'cusum' and r['beta'] == BETAS_ALL[0] and r['L_capped'])
    log(f'\n[诊断] 块长顶到 500 的星：{nc} 颗（若是非 qr 编码器的大多数星，先停下来讨论块长上限）')
    log(f'[saved] {base_out}{tag}_star.csv / _pooled.csv / _pooled_scen.csv / _check.csv / _log.txt')
    log(f'[done] 总用时 {(time.time() - t_all) / 60:.1f} 分钟。把这五个文件发回即可。')
    log.close()
