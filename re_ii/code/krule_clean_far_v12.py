"""Count rule (K consecutive exceedances of a per-epoch threshold) on clean data.

Thresholds are set either under independence for a windowed target beta, or at the
(1-alpha) quantile of the calibration segment (95th percentile for alpha = 0.05).
Computes the exact window false-alarm probability under independence by a Markov
chain (Eq. (12)), the delivered rate on the calibration segment, the rate after
shuffling, and the held-out verification of far_calib_v32.py (Figure 6, Table 3).
Run from the repository root (the working folder that also holds the data folders); see --help.
"""
import numpy as np, glob, os, argparse, csv, math, time, zlib

import sys

_here = os.path.dirname(os.path.abspath(__file__))
for _d in (_here, os.path.dirname(_here), os.path.dirname(os.path.dirname(_here))):
    if os.path.exists(os.path.join(_d, 'far_mc_beta_neweval.py')):
        if _d not in sys.path:
            sys.path.insert(0, _d)
        break

from far_mc_beta_neweval import AUTH, ALIGN_START_MS

try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False

# ---------------------------------------------------------------------------
# Targets and scenarios (as in far_calib_v32.py)
# ---------------------------------------------------------------------------
BETAS = [1e-1, 5e-2, 3e-2, 2e-2]
ALPHAS = [0.05, 0.01]
SCENARIOS = ('ds3', 'ds4_r2', 'ds5_r2', 'ds6_r2', 'ds7', 'ds8')
DYN_SCENES = ('ds5_r2', 'ds6_r2')
GROUPS = ('all', 'static', 'dynamic', 'complex')
VARIANTS = ('ho0', 'ho')

ACTIVATION_RAW = {'ds3': 114.1, 'ds4_r2': 109.9, 'ds5_r2': 96.8, 'ds6_r2': 96.5, 'ds7': 110.0, 'ds8': 110.0}
POOL_EXCLUDE = ('ds8',)


# ---------------------------------------------------------------------------
# Helpers, segmentation and bounds (as in far_calib_v32.py)
# ---------------------------------------------------------------------------
class Tee:
    def __init__(self, path):
        self.f = open(path, 'w', encoding='utf-8')

    def __call__(self, *args):
        s = ' '.join(str(a) for a in args)
        print(s, flush=True)
        self.f.write(s + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def group_of(scen):
    if scen in ('ds3', 'ds4_r2'):
        return 'static'
    if scen in DYN_SCENES:
        return 'dynamic'
    return 'complex'


def fmt(v):
    if v is None:
        return ''
    if isinstance(v, (bool, np.bool_)):
        return 'yes' if v else 'no'
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f'{float(v):.6g}' if np.isfinite(v) else ''
    return str(v)


def star_rng(seed, star, k):
    base = star[:-len('_dyn')] if star.endswith('_dyn') else star
    return np.random.default_rng([int(seed), zlib.crc32(base.encode('utf-8')), int(k)])


def parse_excludes(specs):
    out = {}
    for s in specs or []:
        try:
            scen, a, b = s.rsplit(':', 2)
            a, b = float(a), float(b)
        except ValueError:
            raise SystemExit(f'[ERROR] --exclude 格式应为 场景:起点秒:终点秒，收到 {s!r}')
        if b <= a:
            raise SystemExit(f'[ERROR] --exclude 终点必须大于起点：{s!r}')
        out.setdefault(scen, []).append((a, b))
    return out


def select_files(npz_dir, encoder, stars):
    allf = sorted(glob.glob(os.path.join(npz_dir, '**', 'pertap_*.npz'), recursive=True))
    seen = {}
    for f in allf:
        seen.setdefault(os.path.basename(f), f)
    out = []
    for b in sorted(seen):
        core = b[len('pertap_'):-len('.npz')]
        is_dyn = core.endswith('_dyn')
        base = core[:-len('_dyn')] if is_dyn else core
        scen = base.split('_prn')[0]
        if scen not in SCENARIOS:
            continue
        if scen in DYN_SCENES:
            if is_dyn != (encoder == 'dyn'):
                continue
        else:
            if is_dyn or encoder == 'static':
                continue
        if stars and not any(s in b for s in stars):
            continue
        out.append(seen[b])
    return out


def load_star(f):
    d = np.load(f, allow_pickle=True)
    per = d['per'].astype(np.float64)
    lab = np.asarray(d['labels'])
    scen = str(d['scenario'])
    prn = int(d['prn'])
    S = np.ascontiguousarray(per.sum(1))
    if len(lab) != len(S):
        raise ValueError(f'labels 长度 {len(lab)} 与分数长度 {len(S)} 不一致')
    return S, lab, scen, prn


def idx_of(raw_s, asr):
    return int(round(raw_s * 1000 - 100 - asr))


def segments(S, lab, scen, gidx):
    """Same segmentation as far_calib_v32.segments."""
    asr = ALIGN_START_MS.get(scen, 0)
    n = len(S)
    act_i = min(idx_of(ACTIVATION_RAW[scen], asr), n)
    cut = act_i - gidx
    lab0 = np.flatnonzero(lab == 0)
    idx_c = lab0[lab0 < cut]
    if len(idx_c) == 0:
        raise ValueError('按开启时刻截短后净段为空')
    c0, c1 = int(idx_c[0]), int(idx_c[-1]) + 1
    contiguous = (c1 - c0) == len(idx_c)
    cmask = np.zeros(n, dtype=bool)
    cmask[idx_c] = True
    oi = min(idx_of(AUTH[scen], asr), n)
    ho_a, ho_b = c1, oi
    sp = np.flatnonzero(lab == 1)
    sp_after = sp[sp >= ho_a]
    if len(sp_after) and sp_after[0] < ho_b:
        ho_b = int(sp_after[0])
    ho_b = max(ho_a, ho_b)
    hb_act = max(ho_a, min(ho_b, cut))

    def raw(i):
        return (i + 100 + asr) / 1000.0

    return dict(c0=c0, c1=c1, contiguous=contiguous, cmask=cmask, n_lab0=len(lab0),
                truncated=(len(idx_c) < len(lab0)), ho_a=ho_a, ho_b=ho_b, hb_act=hb_act,
                oi=oi, act_i=act_i, asr=asr, raw=raw)


def window_cov(idx, a, b, m):
    nwin = (b - a) - m + 1
    if nwin <= 0:
        return 0, 0
    cov = 0
    last_end = a - 1
    for t in idx:
        t = int(t)
        if t < a or t >= b:
            continue
        s0 = max(a, t - m + 1)
        s1 = min(t, b - m)
        if s0 <= last_end:
            s0 = last_end + 1
        if s1 >= s0:
            cov += s1 - s0 + 1
            last_end = s1
    return cov, nwin


def n_events(idx, m):
    if len(idx) == 0:
        return 0
    n = 1
    prev = int(idx[0])
    for t in idx[1:]:
        t = int(t)
        if t - prev >= m:
            n += 1
        prev = t
    return n


def _pois_cdf(n, lam):
    if lam <= 0:
        return 1.0
    logs = [-lam + k * math.log(lam) - math.lgamma(k + 1) for k in range(n + 1)]
    mx = max(logs)
    return min(1.0, math.exp(mx) * sum(math.exp(v - mx) for v in logs))


def pois_upper(n, conf=0.95):
    lo, hi = 0.0, max(10.0, 3.0 * (n + 5))
    while _pois_cdf(n, hi) > 1 - conf:
        hi *= 2.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _pois_cdf(n, mid) > 1 - conf:
            lo = mid
        else:
            hi = mid
    return hi


def pois_lower(n, conf=0.95):
    if n == 0:
        return 0.0
    lo, hi = 0.0, float(n)
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _pois_cdf(n - 1, mid) > conf:
            lo = mid
        else:
            hi = mid
    return lo


def win_rate_bounds(n_ev, cov, nwin, m, conf=0.95):
    """Same bounds as far_calib_v32.win_rate_bounds."""
    if nwin <= 0:
        return float('nan'), float('nan')
    cbar = cov / float(n_ev) if n_ev > 0 else float(m)
    lu = pois_upper(n_ev, conf)
    ll = pois_lower(n_ev, conf)
    return min(1.0, ll * cbar / nwin), min(1.0, lu * cbar / nwin)


def verdict(lo95, up95, ref, nwin):
    if nwin <= 0:
        return 'too_short'
    if lo95 > ref:
        return 'violated'
    if up95 <= ref:
        return 'verified'
    return 'inconclusive'


def ho_eval(alarm_fn, X, a, b, ex_idx, m):
    """Held-out verification of the count rule, as far_calib_v32.ho_eval."""
    segs, cur = [], a
    for e0, e1 in sorted(ex_idx):
        if e1 <= cur or e0 >= b:
            continue
        if e0 > cur:
            segs.append((cur, e0))
        cur = max(cur, e1)
    if cur < b:
        segs.append((cur, b))
    tot = dict(len=0, n_alarm=0, n_event=0, cov=0, nwin=0)
    times = []
    full = None
    for s0, s1 in segs:
        if s0 == a:
            if full is None:
                full = np.asarray(alarm_fn(X[:b]), dtype=np.int64)
            idx = full[(full >= s0) & (full < s1)]
        else:
            idx = np.asarray(alarm_fn(X[s0:s1]), dtype=np.int64) + s0
        c, w = window_cov(idx, s0, s1, m)
        tot['len'] += s1 - s0
        tot['n_alarm'] += len(idx)
        tot['n_event'] += n_events(idx, m)
        tot['cov'] += c
        tot['nwin'] += w
        times.extend(int(i) for i in idx)
    lo95, up95 = win_rate_bounds(tot['n_event'], tot['cov'], tot['nwin'], m)
    tot.update(frac=(tot['cov'] / tot['nwin'] if tot['nwin'] > 0 else float('nan')),
               lo95=lo95, up95=up95, times=times)
    return tot


def put_ho(r, v, res, ref, raw):
    r[f'{v}_len'] = res['len']
    r[f'{v}_len_s'] = res['len'] / 1000.0
    for k in ('n_alarm', 'n_event', 'cov', 'nwin', 'frac', 'lo95', 'up95'):
        r[f'{v}_{k}'] = res[k]
    r[f'{v}_verdict'] = verdict(res['lo95'], res['up95'], ref, res['nwin'])
    t = res['times']
    s = ';'.join(f'{raw(i):.3f}' for i in t[:10])
    if len(t) > 10:
        s += f';...(+{len(t) - 10})'
    r[f'{v}_times'] = s


# ---------------------------------------------------------------------------
# Count rule: K consecutive exceedances, reset after each alarm (Eq. (9))
# ---------------------------------------------------------------------------
def _krun_py(X, gamma, K):
    out = []
    run = 0
    for t in range(X.shape[0]):
        if X[t] > gamma:
            run += 1
            if run >= K:
                out.append(t)
                run = 0
        else:
            run = 0
    return np.asarray(out, dtype=np.int64)


if HAVE_NUMBA:
    @njit(cache=True)
    def _krun_nb(X, gamma, K):
        n = X.shape[0]
        out = np.empty(n, dtype=np.int64)
        c = 0
        run = 0
        for t in range(n):
            if X[t] > gamma:
                run += 1
                if run >= K:
                    out[c] = t
                    c += 1
                    run = 0
            else:
                run = 0
        return out[:c]

    def krun_idx(X, gamma, K):
        return _krun_nb(np.ascontiguousarray(X, dtype=np.float64), float(gamma), int(K))
else:
    krun_idx = _krun_py


# Independence reference, Eq. (12): Markov chain on the current run length.
def nominal_iid(p, K, m):
    """Probability of at least one K-run alarm in m epochs under independence (Markov chain, Eq. (12)).
    """
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    T = np.zeros((K, K))
    T[:, 0] = 1.0 - p
    for r in range(K - 1):
        T[r, r + 1] = p
    pi = np.array([p ** r for r in range(K)], dtype=np.float64)
    pi /= pi.sum()
    v = pi @ np.linalg.matrix_power(T, m)
    return float(min(1.0, max(0.0, 1.0 - v.sum())))


# Per-epoch exceedance probability whose independence reference equals beta.
def p_for_target(beta, K, m):
    lo, hi = 1e-9, 1.0 - 1e-9
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if nominal_iid(mid, K, m) < beta:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Per satellite (Figure 6, Table 3)
# ---------------------------------------------------------------------------
def run(files, betas, alphas, Ks, m, R, seed, encoder, guard_s, log):
    gidx = int(round(guard_s * 1000))
    ptab = {(K, b): p_for_target(b, K, m) for K in Ks for b in betas}
    log('[独立假设下的预算] mode=beta：(K, β) -> 逐时刻超限概率 p')
    for (K, b), p in sorted(ptab.items()):
        log(f'   K={K:2d}  β={b:.0e}  ->  p={p:.4f}')
    log('[独立假设下的名义值] mode=alpha：(K, α) -> 名义窗口误报')
    for K in Ks:
        for al in alphas:
            log(f'   K={K:2d}  α={al:g}  ->  名义 {nominal_iid(al, K, m):.2e}')
    rows = []
    for fi, f in enumerate(files, 1):
        star = os.path.basename(f)[len('pertap_'):-len('.npz')]
        try:
            S, lab, scen, prn = load_star(f)
        except Exception as e:
            log(f'[skip] {star}: 读取失败 {e}')
            continue
        if not np.any(lab == 0):
            log(f'[skip] {star}: 没有 labels==0 的净段')
            continue
        seg = segments(S, lab, scen, gidx)
        raw, asr = seg['raw'], seg['asr']
        Sc = np.ascontiguousarray(S[seg['cmask']])
        N = len(Sc)
        q_res = m / float(N)
        ho_a, ho_b, oi, hb_act = seg['ho_a'], seg['ho_b'], seg['oi'], seg['hb_act']
        variants = {'ho0': ho_b, 'ho': hb_act}
        grp = group_of(scen)
        rng = star_rng(seed, star, 4)
        log(f'\n=== [{fi}/{len(files)}] {star} ({grp})  净段 {N / 1000:.1f} s（其中 S=0 占 {np.mean(Sc == 0):.1%}）'
            + (f'  [已截短，labels==0 原有 {seg["n_lab0"] / 1000:.1f} s]' if seg['truncated'] else '')
            + f'  ho {raw(ho_a):.2f}–{raw(hb_act):.2f} s；ho0 到 {raw(ho_b):.2f} s'
            + ('  [不进合并]' if scen in POOL_EXCLUDE else ''))
        for K in Ks:
            levels = [('beta', b, ptab[(K, b)]) for b in betas] + [('alpha', al, al) for al in alphas]
            for mode, lev, p_t in levels:
                gamma = float(np.quantile(Sc, 1.0 - p_t))
                p_hat = float(np.mean(Sc > gamma))
                nom = nominal_iid(p_hat, K, m)
                idx = krun_idx(Sc, gamma, K)
                cov, nwin = window_cov(idx, 0, N, m)
                dlv = cov / nwin if nwin > 0 else float('nan')
                nev_c = n_events(idx, m)
                c_lo, c_up = win_rate_bounds(nev_c, cov, nwin, m)
                sh = []
                for _ in range(R):
                    ci, nw = window_cov(krun_idx(rng.permutation(Sc), gamma, K), 0, N, m)
                    sh.append(ci / nw if nw > 0 else float('nan'))
                dsh = float(np.mean(sh)) if sh else float('nan')
                r = dict(star=star, scenario=scen, group=grp, encoder=encoder, prn=prn,
                         N_clean_s=N / 1000.0, q_res=q_res, mode=mode, K=K, level=lev, p_target=p_t,
                         gamma=gamma, p_hat=p_hat, nominal_iid=nom, dlv_clean=dlv,
                         clean_lo95=c_lo, clean_up95=c_up,
                         D_clean=(dlv / nom if nom > 0 else float('nan')), dlv_shuffle=dsh,
                         D_shuffle=(dsh / nom if nom > 0 else float('nan')),
                         n_alarm_clean=len(idx), n_event_clean=nev_c, clean_cov=cov,
                         clean_nwin=nwin, resolvable_3q=(nom >= 3 * q_res),
                         guard_s=guard_s, act_raw_s=ACTIVATION_RAW[scen], clean_truncated=seg['truncated'],
                         N_clean_label_s=seg['n_lab0'] / 1000.0, pooled_excluded=(scen in POOL_EXCLUDE),
                         ho_raw_start_s=raw(ho_a), ho0_raw_end_s=raw(ho_b), ho_raw_end_s=raw(hb_act))

                def fn(X, _g=gamma, _K=K):
                    return krun_idx(X, _g, _K)
                for v, bb in variants.items():
                    put_ho(r, v, ho_eval(fn, S, ho_a, bb, [], m), nom, raw)
                rows.append(r)
                tag = f'β={lev:.0e}' if mode == 'beta' else f'α={lev:g}'
                log(f'  K={K:2d} {tag:>8s}: γ={gamma:9.4f} p̂={p_hat:.4f}  名义(独立)={nom:.2e}  '
                    f'实际={dlv:.2e} [{c_lo:.1e}, {c_up:.1e}] D={r["D_clean"]:.2f}  打乱={dsh:.2e}  '
                    f'净段事件 {nev_c}  | 留出 ho: 事件 {r["ho_n_event"]} 窗口误报 {r["ho_frac"]:.2e} '
                    f'[{r["ho_lo95"]:.1e}, {r["ho_up95"]:.1e}] {r["ho_verdict"]}'
                    f'  (ho0 事件 {r["ho0_n_event"]})')
    return rows


# ---------------------------------------------------------------------------
# Pooling and output
# ---------------------------------------------------------------------------
def pooled(rows, m, keys):
    out = []
    lv = sorted({(r['mode'], r['K'], r['level']) for r in rows}, key=lambda x: (x[0] != 'beta', x[1], -x[2]))
    for mode, K, lev in lv:
        for g in keys:
            sel = [r for r in rows if r['mode'] == mode and r['K'] == K and r['level'] == lev
                   and (g == 'all' or r['group'] == g or r['scenario'] == g)
                   and r['scenario'] not in POOL_EXCLUDE]
            if not sel:
                continue
            cnw = sum(r['clean_nwin'] for r in sel)
            ccov = sum(r['clean_cov'] for r in sel)
            cev = sum(r['n_event_clean'] for r in sel)
            cN = sum(r['N_clean_s'] for r in sel) * 1000.0
            nom = sum(r['nominal_iid'] * r['clean_nwin'] for r in sel) / cnw if cnw > 0 else float('nan')
            dlv = ccov / cnw if cnw > 0 else float('nan')
            c_lo, c_up = win_rate_bounds(cev, ccov, cnw, m)
            d = dict(mode=mode, K=K, level=lev, group=g, n_stars=len(sel), nominal_iid=nom,
                     dlv_clean=dlv, clean_n_event=cev, clean_lo95=c_lo, clean_up95=c_up,
                     D_clean=(dlv / nom if nom > 0 else float('nan')))
            for v in VARIANTS:
                hs = [r for r in sel if r.get(f'{v}_nwin', 0) > 0]
                hnw = sum(r[f'{v}_nwin'] for r in hs)
                hcov = sum(r[f'{v}_cov'] for r in hs)
                hev = sum(r[f'{v}_n_event'] for r in hs)
                hL = sum(r[f'{v}_len'] for r in hs)
                lo95, up95 = win_rate_bounds(hev, hcov, hnw, m)
                d.update({f'{v}_total_s': hL / 1000.0, f'{v}_n_event': hev,
                          f'{v}_frac': (hcov / hnw if hnw > 0 else float('nan')),
                          f'{v}_lo95': lo95, f'{v}_up95': up95,
                          f'{v}_verdict': (verdict(lo95, up95, nom, hnw) if hnw > 0 else 'none')})
            out.append(d)
    return out


VAR_FIELDS = ['len_s', 'n_alarm', 'n_event', 'nwin', 'frac', 'lo95', 'up95', 'verdict', 'times']
FIELDS = ['star', 'scenario', 'group', 'encoder', 'prn', 'N_clean_s', 'q_res', 'mode', 'K', 'level',
          'p_target', 'gamma', 'p_hat', 'nominal_iid', 'dlv_clean', 'clean_lo95', 'clean_up95', 'D_clean',
          'dlv_shuffle', 'D_shuffle', 'n_alarm_clean', 'n_event_clean', 'resolvable_3q', 'guard_s', 'act_raw_s',
          'clean_truncated', 'N_clean_label_s', 'pooled_excluded', 'ho_raw_start_s', 'ho0_raw_end_s',
          'ho_raw_end_s'] + \
         [f'{v}_{k}' for v in VARIANTS for k in VAR_FIELDS]
POOL_FIELDS = ['mode', 'K', 'level', 'group', 'n_stars', 'nominal_iid', 'dlv_clean', 'clean_n_event',
               'clean_lo95', 'clean_up95', 'D_clean'] + \
              [f'{v}_{k}' for v in VARIANTS for k in ('total_s', 'n_event', 'frac', 'lo95', 'up95', 'verdict')]


def save_csv(path, rows, fields):
    with open(path, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.writer(fh)
        w.writerow(fields)
        for r in rows:
            w.writerow([fmt(r.get(k)) for k in fields])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', dest='npz_dir', default='.')
    ap.add_argument('--stars', nargs='+', default=None)
    ap.add_argument('--encoder', choices=['dyn', 'static'], default='dyn')
    ap.add_argument('--K', nargs='+', type=int, default=[5])
    ap.add_argument('--betas', nargs='+', type=float, default=None)
    ap.add_argument('--alphas', nargs='*', type=float, default=None,
                    help='mode=alpha 的 α（γ = 净段 1−α 分位）；默认 0.05 0.01；只写 --alphas 不跟值 = 不跑 alpha 档')
    ap.add_argument('--m', type=int, default=1000)
    ap.add_argument('--shuffles', type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--guard-s', dest='guard_s', type=float, default=0.1)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    base_out = os.path.splitext(a.out)[0]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    log = Tee(base_out + '_log.txt')
    betas = a.betas or BETAS
    alphas = ALPHAS if a.alphas is None else a.alphas
    log('[version] krule_clean_far_v12.py  v1.2（2026-09-29）')
    log(f'[numba] {"ON" if HAVE_NUMBA else "OFF（纯 Python，也能跑，只是慢一些）"}')
    log(f'[cfg] K={a.K}  betas={betas}  alphas={alphas}  m={a.m} ms  shuffles={a.shuffles}  encoder={a.encoder}')
    log(f'[cfg] guard={a.guard_s} s  ACTIVATION_RAW={ACTIVATION_RAW}  不进合并={POOL_EXCLUDE}')
    files = select_files(a.npz_dir, a.encoder, a.stars)
    log(f'[cfg] {len(files)} 颗星：' + ', '.join(os.path.basename(f)[7:-4] for f in files))
    if not files:
        log('[ERROR] 没找到匹配的 pertap_*.npz，检查 --dir / --stars / --encoder')
    else:
        t0 = time.time()
        rows = run(files, betas, alphas, a.K, a.m, a.shuffles, a.seed, a.encoder, a.guard_s, log)
        save_csv(a.out, rows, FIELDS)
        pool = pooled(rows, a.m, GROUPS)
        save_csv(base_out + '_pooled.csv', pool, POOL_FIELDS)
        scens = sorted({r['scenario'] for r in rows})
        save_csv(base_out + '_pooled_scen.csv', pooled(rows, a.m, scens), POOL_FIELDS)
        log('\n==== 合并结果（净段：实际/名义；留出段 ho：相对名义值的判定）====')
        log(f'{"mode":>5s} {"K":>3s} {"level":>7s} {"group":>8s} {"星数":>4s} {"名义":>9s} {"实际":>9s} {"D":>7s} '
            f'{"净段事件":>5s} | {"留出s":>7s} {"事件":>5s} {"窗口误报":>9s} {"上界95":>9s}  判定')
        for p in pool:
            log(f'{p["mode"]:>5s} {p["K"]:3d} {p["level"]:7.0e} {p["group"]:>8s} {p["n_stars"]:4d} '
                f'{p["nominal_iid"]:9.2e} {p["dlv_clean"]:9.2e} {p["D_clean"]:7.2f} {p["clean_n_event"]:5d} | '
                f'{p["ho_total_s"]:7.1f} {p["ho_n_event"]:5d} {p["ho_frac"]:9.2e} {p["ho_up95"]:9.2e}  {p["ho_verdict"]}')
        log(f'\n[saved] {a.out}（{len(rows)} 行） / {base_out}_pooled.csv / {base_out}_pooled_scen.csv / {base_out}_log.txt')
        log(f'[done] 用时 {(time.time() - t0) / 60:.1f} 分钟。把本次所有输出文件打包传回即可。')
    log.close()
