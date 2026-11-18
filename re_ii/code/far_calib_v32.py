"""Alarm layer: segmentation, block-bootstrap calibration and held-out verification.

For each satellite and target beta: splits the recording into the calibration
segment and the held-out segment (Table 1); calibrates the CUSUM limit h by an
order-preserving block bootstrap (blk) and, for comparison, by pointwise shuffling
(iid) (Section 3.3, Algorithm 1); checks self-consistency on fresh resamples
(Figure A2); and verifies the limit on the held-out segment with one-sided 95%
Poisson bounds on the event count (Eq. (15)). Writes per-satellite and pooled csv
files.
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

from far_mc_beta_neweval import acf, pick_block_len, cusum_gmax, K0, AUTH, ALIGN_START_MS

try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False

# ---------------------------------------------------------------------------
# Targets, scenarios and activation instants
# ---------------------------------------------------------------------------
# Windowed targets per one-second window; Table A1 reports all eight for the QR score.
BETAS = [1e-1, 5e-2, 3e-2, 2e-2, 1e-2, 5e-3, 3e-3, 1e-3]
SCENARIOS = ('ds3', 'ds4_r2', 'ds5_r2', 'ds6_r2', 'ds7', 'ds8')
DYN_SCENES = ('ds5_r2', 'ds6_r2')
GROUPS = ('all', 'static', 'dynamic', 'complex')
VARIANTS = ('ho0', 'ho')

# Held-out segments end 0.1 s before these instants (Table 1): the activation signature
# for ds3/ds4 (Lemmenes et al., 2016), the start of the overlay for ds7/ds8, and for ds5/ds6
# the common-mode event near 97 s, a benign event of the drive (Appendix D).
ACTIVATION_RAW = {'ds3': 114.1, 'ds4_r2': 109.9, 'ds5_r2': 96.8, 'ds6_r2': 96.5, 'ds7': 110.0, 'ds8': 110.0}

# ds8 repeats ds7 before 110 s and is left out of pooled false-alarm figures.
POOL_EXCLUDE = ('ds8',)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class Tee:
    """Writes screen output also to <out>_log.txt."""
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
    """Independent random stream per satellite and purpose; static and _dyn versions of a satellite share it.
    """
    base = star[:-len('_dyn')] if star.endswith('_dyn') else star
    return np.random.default_rng([int(seed), zlib.crc32(base.encode('utf-8')), int(k)])


def parse_excludes(specs):
    """'ds6_r2:96.5:98.0' -> {'ds6_r2': [(96.5, 98.0)]} (recording seconds)."""
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


# ---------------------------------------------------------------------------
# Score files and segmentation (Table 1)
# ---------------------------------------------------------------------------
def select_files(npz_dir, encoder, stars):
    """dyn: static files for ds3/ds4/ds7/ds8 and _dyn files for ds5/ds6; static: static files of ds5/ds6.
    """
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
    S = per.sum(1)
    if len(lab) != len(S):
        raise ValueError(f'labels 长度 {len(lab)} 与分数长度 {len(S)} 不一致')
    return S, lab, scen, prn


def idx_of(raw_s, asr):
    """Recording seconds -> array index (raw = (i + 100 + asr) / 1000)."""
    return int(round(raw_s * 1000 - 100 - asr))


def segments(S, lab, scen, gidx):
    """Calibration segment and the two held-out segments (ending before activation, or at the onset).
    """
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


def load_cn0_db(f, n):
    """C/N0 proxy in dB from the npz, or None."""
    d = np.load(f, allow_pickle=True)
    if 'cn0' not in d.files:
        return None
    c = np.asarray(d['cn0'], dtype=np.float64).ravel()
    if len(c) != n:
        return None
    if np.nanmedian(c) > 1000.0:
        c = 10.0 * np.log10(np.maximum(c, 1e-9))
    return c


# ---------------------------------------------------------------------------
# Block bootstrap and calibration of h (Section 3.3, Algorithm 1)
# ---------------------------------------------------------------------------
def synth_matrix(Zc, L, n_boot, seg_len, rng):
    """Block resampling (L = 1 is pointwise shuffling): random blocks of length L joined end to end.
    """
    Zc = np.asarray(Zc, dtype=np.float64)
    nZ = len(Zc)
    L = int(max(1, min(L, nZ)))
    n_blocks = int(np.ceil(seg_len / L))
    out = np.empty((n_boot, seg_len), dtype=np.float32)
    off = np.arange(L)
    per_row = n_blocks * L
    chunk = max(1, int(2e7 // per_row))
    for r0 in range(0, n_boot, chunk):
        r1 = min(n_boot, r0 + chunk)
        starts = rng.integers(0, nZ - L + 1, size=(r1 - r0, n_blocks))
        idx = (starts[:, :, None] + off[None, None, :]).reshape(r1 - r0, per_row)[:, :seg_len]
        out[r0:r1] = Zc[idx]
    return out


def _mpfa_batch_np(Zmat, k0, h, m):
    """NumPy fallback without numba: (share of windows with an alarm, number of alarms)."""
    n_boot, seg_len = Zmat.shape
    nwin = seg_len - m + 1
    g = np.zeros(n_boot)
    last_end = np.full(n_boot, -1, dtype=np.int64)
    covered = np.zeros(n_boot, dtype=np.int64)
    nal = 0
    for t in range(seg_len):
        g += Zmat[:, t] - k0
        np.maximum(g, 0.0, out=g)
        fired = g >= h
        if fired.any():
            g[fired] = 0.0
            nal += int(fired.sum())
            s0 = np.maximum(max(0, t - m + 1), last_end[fired] + 1)
            s1 = min(t, nwin - 1)
            add = np.maximum(0, s1 - s0 + 1)
            covered[fired] += add
            last_end[fired] = np.where(add > 0, s1, last_end[fired])
    return covered.sum() / float(n_boot * nwin), nal


def _alarm_idx_py(Z, k0, h):
    out = []
    g = 0.0
    for t in range(Z.shape[0]):
        g += float(Z[t]) - k0
        if g < 0.0:
            g = 0.0
        if g >= h:
            out.append(t)
            g = 0.0
    return np.asarray(out, dtype=np.int64)


def _g_trace_py(Z, k0):
    out = np.empty(Z.shape[0])
    g = 0.0
    for t in range(Z.shape[0]):
        g += float(Z[t]) - k0
        if g < 0.0:
            g = 0.0
        out[t] = g
    return out


if HAVE_NUMBA:
    @njit(cache=True)
    def _mpfa_batch_nb(Zmat, k0, h, m):
        n_boot, seg_len = Zmat.shape
        nwin = seg_len - m + 1
        covered = 0
        nal = 0
        for b in range(n_boot):
            g = 0.0
            last_end = -1
            for t in range(seg_len):
                g += Zmat[b, t] - k0
                if g < 0.0:
                    g = 0.0
                if g >= h:
                    g = 0.0
                    nal += 1
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
        return covered / (n_boot * nwin), nal

    @njit(cache=True)
    def _alarm_idx_nb(Z, k0, h):
        n = Z.shape[0]
        out = np.empty(n, dtype=np.int64)
        c = 0
        g = 0.0
        for t in range(n):
            g += Z[t] - k0
            if g < 0.0:
                g = 0.0
            if g >= h:
                out[c] = t
                c += 1
                g = 0.0
        return out[:c]

    @njit(cache=True)
    def _g_trace_nb(Z, k0):
        n = Z.shape[0]
        out = np.empty(n)
        g = 0.0
        for t in range(n):
            g += Z[t] - k0
            if g < 0.0:
                g = 0.0
            out[t] = g
        return out

    def mpfa_batch(Zmat, k0, h, m):
        return _mpfa_batch_nb(Zmat, float(k0), float(h), int(m))

    def alarm_idx(Z, k0, h):
        return _alarm_idx_nb(np.ascontiguousarray(Z, dtype=np.float64), float(k0), float(h))

    def g_trace(Z, k0):
        return _g_trace_nb(np.ascontiguousarray(Z, dtype=np.float64), float(k0))
else:
    mpfa_batch = _mpfa_batch_np
    alarm_idx = _alarm_idx_py
    g_trace = _g_trace_py


def top_peaks(g, sep, k=3):
    """Highest k excursions of the CUSUM without reset; peaks closer than sep keep only the highest.
    """
    g = np.asarray(g, dtype=np.float64)
    pos = g > 0
    if not pos.any():
        return []
    starts = np.flatnonzero(pos & ~np.r_[False, pos[:-1]])
    ends = np.flatnonzero(pos & ~np.r_[pos[1:], False]) + 1
    mx = np.maximum.reduceat(g, starts)
    out = []
    for j in np.argsort(-mx):
        s0, s1 = int(starts[j]), int(ends[j])
        i = s0 + int(np.argmax(g[s0:s1]))
        if all(abs(i - q) >= sep for q, _ in out):
            out.append((i, float(g[i])))
            if len(out) >= k:
                break
    return out


def make_mp(Zmat, m):
    """Cached windowed false-alarm share as a function of h on one resampled matrix."""
    cache = {}

    def mp(h):
        key = round(float(h), 9)
        if key not in cache:
            cache[key] = mpfa_batch(Zmat, K0, key, m)[0]
        return cache[key]
    return mp


# Bracket [0.3 gmax, 8 gmax]. The lower end is halved while it still meets beta, so the
# bisection cannot stop at the bracket (Algorithm 1, steps 5-7).
def calibrate_h_v3(mp, gm, beta, tol_rel=1e-3, max_iter=60):
    """Smallest h with share <= beta; returns (h or None, info) with status ok / not_found / degenerate.
    """
    info = {'n_down': 0, 'n_up': 0, 'status': 'ok'}
    gm = gm if gm > 0 else 1.0
    lo, hi = 0.3 * gm, 8.0 * gm
    while mp(hi) > beta:
        if info['n_up'] >= 6:
            info['status'] = 'not_found'
            return None, info
        hi *= 2.0
        info['n_up'] += 1
    while mp(lo) <= beta:
        if info['n_down'] >= 30:
            info['status'] = 'degenerate'
            return lo, info
        hi = lo
        lo *= 0.5
        info['n_down'] += 1
    for _ in range(max_iter):
        if hi - lo <= tol_rel * gm:
            break
        mid = 0.5 * (lo + hi)
        if mp(mid) <= beta:
            hi = mid
        else:
            lo = mid
    return hi, info


# ---------------------------------------------------------------------------
# Held-out verification (Section 3.5, Eq. (15))
# ---------------------------------------------------------------------------
# Alarms closer than m form one event; the bounds are Poisson bounds on the event count.
def window_cov(idx, a, b, m):
    """Number of length-m windows in [a, b) that contain an alarm, and the number of windows."""
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
    """Alarms less than m apart form one event."""
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
    """One-sided 95% bounds of the windowed rate from Poisson bounds on the event count."""
    if nwin <= 0:
        return float('nan'), float('nan')
    cbar = cov / float(n_ev) if n_ev > 0 else float(m)
    lu = pois_upper(n_ev, conf)
    ll = pois_lower(n_ev, conf)
    return min(1.0, ll * cbar / nwin), min(1.0, lu * cbar / nwin)


# verified, violated and inconclusive are W, E and C in the tables.
def verdict(lo95, up95, ref, nwin):
    if nwin <= 0:
        return 'too_short'
    if lo95 > ref:
        return 'violated'
    if up95 <= ref:
        return 'verified'
    return 'inconclusive'


def ho_eval(alarm_fn, Z, a, b, ex_idx, m):
    """Held-out verification on [a, b), continuing the CUSUM state from the calibration segment."""
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
                full = np.asarray(alarm_fn(Z[:b]), dtype=np.int64)
            idx = full[(full >= s0) & (full < s1)]
        else:
            idx = np.asarray(alarm_fn(Z[s0:s1]), dtype=np.int64) + s0
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
# Checks of the activation instants
# ---------------------------------------------------------------------------
def excursion_start(g, j):
    s0 = j
    while s0 > 0 and g[s0 - 1] > 0:
        s0 -= 1
    return s0


def cn0_step(cdb, s0, pre=3000, post=2000):
    if cdb is None:
        return float('nan'), float('nan'), float('nan')
    a, b = cdb[max(0, s0 - pre):s0], cdb[s0:s0 + post]
    if len(a) == 0 or len(b) == 0:
        return float('nan'), float('nan'), float('nan')
    mb, ma = float(np.median(a)), float(np.median(b))
    return mb, ma, ma - mb


def activation_check(star, scen, gtr, cdb, seg, log):
    out = []
    raw, asr = seg['raw'], seg['asr']
    for tag, lo_s, hi_i in (('act', 90.0, seg['oi']), ('e58', 55.0, idx_of(62.0, asr))):
        lo, hi = max(0, idx_of(lo_s, asr)), min(len(gtr), hi_i)
        if hi - lo <= 0:
            continue
        j = lo + int(np.argmax(gtr[lo:hi]))
        if not gtr[j] > 0:
            continue
        s0 = excursion_start(gtr, j)
        mb, ma, st = cn0_step(cdb, s0)
        warn = (tag == 'act' and s0 < seg['act_i'])
        out.append(dict(star=star, scenario=scen, event=tag, start_raw_s=raw(s0), peak_raw_s=raw(j),
                        g_peak=float(gtr[j]), cn0_before_db=mb, cn0_after_db=ma, cn0_step_db=st,
                        start_before_activation=warn, activation_raw_s=ACTIVATION_RAW[scen]))
        log(f'    [开启核验] {tag}: 偏移段起点 {raw(s0):.3f} s，峰 {raw(j):.3f} s，g={gtr[j]:.1f}，'
            f'C/N0 {mb:.2f} → {ma:.2f} dB（台阶 {st:+.2f} dB）'
            + ('  [WARN] 起点早于 ACTIVATION_RAW，需把开启时刻再提前' if warn else ''))
    return out


# ---------------------------------------------------------------------------
# Per satellite: calibration, self-consistency (Figure A2), held-out verification
# ---------------------------------------------------------------------------
def run(files, betas, m, n_boot, seg_s, seed, encoder, guard_s, log):
    seg_len = int(seg_s * 1000)
    gidx = int(round(guard_s * 1000))
    rows, act_rows = [], []
    for fi, f in enumerate(files, 1):
        t0 = time.time()
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
        Sc = S[seg['cmask']]
        N = len(Sc)
        if N < 4000:
            log(f'[skip] {star}: 净段太短 ({N} 点)')
            continue
        mu, sd = float(Sc.mean()), float(Sc.std())
        Z = np.ascontiguousarray((S - mu) / sd)
        Zc = np.ascontiguousarray(Z[seg['cmask']])
        L = int(pick_block_len(Zc))
        rho1 = float(acf(Zc, 1))
        gm = float(cusum_gmax(Zc, K0))
        q_res = m / float(N)
        ho_a, ho_b, oi, hb_act = seg['ho_a'], seg['ho_b'], seg['oi'], seg['hb_act']
        variants = {'ho0': ho_b, 'ho': hb_act}
        grp = group_of(scen)


        gtr = g_trace(Z[:max(oi, seg['c1'])], K0)
        pk = top_peaks(gtr, sep=m, k=3)

        def seg_tag(i):
            if i < seg['c1']:
                return 'clean'
            if i < hb_act:
                return 'ho'
            if i < oi:
                return 'act->onset'
            return 'post'
        peaks_txt = ';'.join(f'{raw(i):.3f}:{v:.1f}:{seg_tag(i)}' for i, v in pk)

        log(f'\n=== [{fi}/{len(files)}] {star}  ({grp})  prn={prn}')
        log(f'    净段 raw {raw(seg["c0"]):.2f}–{raw(seg["c1"]):.2f} s  N={N / 1000:.1f} s  连续={seg["contiguous"]}'
            + (f'  [已截短，labels==0 原有 {seg["n_lab0"] / 1000:.1f} s]' if seg['truncated'] else '')
            + f'  |  开启 {ACTIVATION_RAW[scen]:.1f} s / AUTH onset {AUTH[scen]:.1f} s'
            + f'  |  ho {raw(ho_a):.2f}–{raw(hb_act):.2f} s ({(hb_act - ho_a) / 1000:.1f} s)'
            + f'；ho0 到 {raw(ho_b):.2f} s ({(ho_b - ho_a) / 1000:.1f} s)'
            + ('  [不进合并：110 s 前与 ds7 为同一份数据]' if scen in POOL_EXCLUDE else ''))
        log(f'    L={L}{"(顶到上限)" if L >= 500 else ""}  rho1={rho1:.3f}  gm={gm:.2f}  q_res=m/N={q_res:.2e}')
        log('    g 峰（不重置）: ' + ' | '.join(f'{raw(i):.3f} s  {v:.1f}  [{seg_tag(i)}]' for i, v in pk))
        if hb_act - ho_a < m:
            log('    [注意] ho 留出段不足一个窗口：判为 too_short，不进合并')
        if scen in DYN_SCENES:
            act_rows += activation_check(star, scen, gtr, load_cn0_db(f, len(S)), seg, log)

        rng = {k: star_rng(seed, star, k) for k in (1, 2, 3)}
        mats = {'blk': synth_matrix(Zc, L, n_boot, seg_len, rng[1]),
                'iid': synth_matrix(Zc, 1, n_boot, seg_len, rng[2])}
        Vmat = synth_matrix(Zc, L, n_boot, seg_len, rng[3])
        mps = {k: make_mp(v, m) for k, v in mats.items()}

        base = dict(star=star, scenario=scen, group=grp, encoder=encoder, prn=prn,
                    N_clean_s=N / 1000.0, clean_raw_start_s=raw(seg['c0']),
                    clean_raw_end_s=raw(seg['c1']), clean_contiguous=seg['contiguous'],
                    q_res=q_res, L=L, L_capped=(L >= 500), rho1=rho1, gm=gm, peaks=peaks_txt,
                    guard_s=guard_s, act_raw_s=ACTIVATION_RAW[scen], clean_truncated=seg['truncated'],
                    N_clean_label_s=seg['n_lab0'] / 1000.0, pooled_excluded=(scen in POOL_EXCLUDE),
                    ho_raw_start_s=raw(ho_a), ho0_raw_end_s=raw(ho_b), ho_raw_end_s=raw(hb_act))

        for beta in betas:
            for tag in ('blk', 'iid'):
                h, info = calibrate_h_v3(mps[tag], gm, beta)
                r = dict(base)
                r.update(beta=beta, resolvable_3q=(beta >= 3 * q_res), calib=tag, h=h,
                         status=info['status'], n_down=info['n_down'], n_up=info['n_up'])
                if h is None:
                    for v in VARIANTS:
                        r[f'{v}_verdict'] = 'none'
                    rows.append(r)
                    log(f'  β={beta:.0e} {tag}: h 未找到 ({info["status"]})')
                    continue
                boot = mpfa_batch(Vmat, K0, h, m)[0]
                cal_idx = alarm_idx(Zc, K0, h)
                r.update(h_ge_gm=(h >= 0.999 * gm), n_alarm_cal=len(cal_idx),
                         n_event_cal=n_events(cal_idx, m), boot_dlv=boot, D_boot=boot / beta)

                def fn(X, _h=h):
                    return alarm_idx(X, K0, _h)
                for v, bb in variants.items():
                    put_ho(r, v, ho_eval(fn, Z, ho_a, bb, [], m), beta, raw)
                rows.append(r)
                log(f'  β={beta:.0e} {tag}: h={h:9.3f} (下探{info["n_down"]}次{", ≥gm" if r["h_ge_gm"] else ""})'
                    f'  净段事件 {r["n_event_cal"]:3d}  自洽 {boot:.2e} (D={boot / beta:.2f})'
                    f'  | 留出 ho: 事件 {r["ho_n_event"]} 窗口误报 {r["ho_frac"]:.2e} '
                    f'[{r["ho_lo95"]:.1e}, {r["ho_up95"]:.1e}] {r["ho_verdict"]}'
                    f'  (ho0 事件 {r["ho0_n_event"]})'
                    + (f'  ho0 时刻 {r["ho0_times"]}' if r['ho0_times'] else ''))
        del mats, Vmat, mps
        log(f'    用时 {time.time() - t0:.0f} s')
    return rows, act_rows


# ---------------------------------------------------------------------------
# Pooling over satellites (Tables 5 and A1)
# ---------------------------------------------------------------------------
# Satellites are treated as independent; common-mode events make the bounds optimistic.
def pooled(rows, m, keys):
    """Pools held-out results over satellites (treated as independent)."""
    out = []
    betas = sorted({r['beta'] for r in rows}, reverse=True)
    for v in VARIANTS:
        for beta in betas:
            for tag in ('blk', 'iid'):
                for g in keys:
                    sel = [r for r in rows if r['beta'] == beta and r['calib'] == tag
                           and r.get(f'{v}_nwin', 0) > 0 and r['scenario'] not in POOL_EXCLUDE
                           and (g == 'all' or r['group'] == g or r['scenario'] == g)]
                    if not sel:
                        continue
                    cov = sum(r[f'{v}_cov'] for r in sel)
                    nwin = sum(r[f'{v}_nwin'] for r in sel)
                    nev = sum(r[f'{v}_n_event'] for r in sel)
                    L = sum(r[f'{v}_len'] for r in sel)
                    lo95, up95 = win_rate_bounds(nev, cov, nwin, m)
                    out.append(dict(variant=v, beta=beta, calib=tag, group=g, n_stars=len(sel),
                                    total_ho_s=L / 1000.0, n_event=nev, frac=cov / nwin,
                                    lo95=lo95, up95=up95, verdict=verdict(lo95, up95, beta, nwin)))
    return out


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
BASE_FIELDS = ['star', 'scenario', 'group', 'encoder', 'prn', 'N_clean_s', 'clean_raw_start_s',
               'clean_raw_end_s', 'clean_contiguous', 'q_res', 'L', 'L_capped', 'rho1', 'gm', 'peaks',
               'guard_s', 'act_raw_s', 'clean_truncated', 'N_clean_label_s', 'pooled_excluded',
               'ho_raw_start_s', 'ho0_raw_end_s', 'ho_raw_end_s',
               'beta', 'resolvable_3q', 'calib', 'h', 'status', 'n_down', 'n_up', 'h_ge_gm',
               'n_alarm_cal', 'n_event_cal', 'boot_dlv', 'D_boot']
VAR_FIELDS = ['len_s', 'n_alarm', 'n_event', 'nwin', 'frac', 'lo95', 'up95', 'verdict', 'times']
FIELDS = BASE_FIELDS + [f'{v}_{k}' for v in VARIANTS for k in VAR_FIELDS]
ACT_FIELDS = ['star', 'scenario', 'event', 'activation_raw_s', 'start_raw_s', 'peak_raw_s', 'g_peak',
              'cn0_before_db', 'cn0_after_db', 'cn0_step_db', 'start_before_activation']
POOL_FIELDS = ['variant', 'beta', 'calib', 'group', 'n_stars', 'total_ho_s', 'n_event', 'frac',
               'lo95', 'up95', 'verdict']


def save_csv(path, rows, fields):
    with open(path, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.writer(fh)
        w.writerow(fields)
        for r in rows:
            w.writerow([fmt(r.get(k)) for k in fields])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', dest='npz_dir', default='.',
                    help=r'npz 根目录（递归找 pertap_*.npz），本机为 J:\Database\TEXBAT\processed_Mao')
    ap.add_argument('--stars', nargs='+', default=None, help='只跑文件名含这些子串的星')
    ap.add_argument('--encoder', choices=['dyn', 'static'], default='dyn',
                    help='dyn=主结果（ds5/ds6 用动态编码器，21 颗）；static=换编码器检查（ds5/ds6 静态编码器，10 颗）')
    ap.add_argument('--nboot', type=int, default=2000)
    ap.add_argument('--m', type=int, default=1000, help='窗口长度（ms）')
    ap.add_argument('--seg', type=float, default=80.0, help='每条合成序列秒数')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--betas', nargs='+', type=float, default=None)
    ap.add_argument('--guard-s', dest='guard_s', type=float, default=0.1,
                    help='[修5] 保护间隔（秒）：留出段终点 = onset − guard；0 等于 v3.0 口径')
    ap.add_argument('--out', required=True, help='结果 csv 路径')
    a = ap.parse_args()

    base_out = os.path.splitext(a.out)[0]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    log = Tee(base_out + '_log.txt')
    betas = a.betas or BETAS
    log('[version] far_calib_v32.py  v3.2（2026-09-29）')
    log(f'[numba] {"ON" if HAVE_NUMBA else "OFF（会很慢，建议 conda install numba，或先用 --nboot 200）"}')
    log(f'[cfg] K0={K0}  m={a.m} ms  nboot={a.nboot}  seg={a.seg} s  seed={a.seed}  encoder={a.encoder}')
    log(f'[cfg] betas={betas}')
    log(f'[cfg] guard={a.guard_s} s  ACTIVATION_RAW={ACTIVATION_RAW}  不进合并={POOL_EXCLUDE}')
    files = select_files(a.npz_dir, a.encoder, a.stars)
    log(f'[cfg] {len(files)} 颗星：' + ', '.join(os.path.basename(f)[7:-4] for f in files))
    if not files:
        log('[ERROR] 没找到匹配的 pertap_*.npz，检查 --dir / --stars / --encoder')
    else:
        t_all = time.time()
        rows, act_rows = run(files, betas, a.m, a.nboot, a.seg, a.seed, a.encoder, a.guard_s, log)
        if act_rows:
            save_csv(base_out + '_activation.csv', act_rows, ACT_FIELDS)
        save_csv(a.out, rows, FIELDS)
        pool = pooled(rows, a.m, GROUPS)
        save_csv(base_out + '_pooled.csv', pool, POOL_FIELDS)
        scens = sorted({r['scenario'] for r in rows})
        save_csv(base_out + '_pooled_scen.csv', pooled(rows, a.m, scens), POOL_FIELDS)

        log('\n==== 留出段合并检验（各星当作独立合并；同场景共模事件会让界偏乐观）====')
        log('  ho = 终点为开启时刻 − 保护间隔（主结果）；ho0 = 终点为 AUTH onset（旧口径，对照）；ds8 不进合并')
        log(f'{"variant":>7s} {"beta":>7s} {"calib":>5s} {"group":>8s} {"星数":>4s} {"总时长s":>8s} {"事件":>5s} '
            f'{"窗口误报":>9s} {"下界95":>9s} {"上界95":>9s}  判定')
        for p in pool:
            log(f'{p["variant"]:>7s} {p["beta"]:7.0e} {p["calib"]:>5s} {p["group"]:>8s} {p["n_stars"]:4d} '
                f'{p["total_ho_s"]:8.1f} {p["n_event"]:5d} {p["frac"]:9.2e} {p["lo95"]:9.2e} {p["up95"]:9.2e}'
                f'  {p["verdict"]}')
        log(f'\n[saved] {a.out}（{len(rows)} 行）')
        log(f'[saved] {base_out}_pooled.csv / {base_out}_pooled_scen.csv / {base_out}_activation.csv / {base_out}_log.txt')
        log(f'[done] 总用时 {(time.time() - t_all) / 60:.1f} 分钟。把本次所有输出文件打包传回即可。')
    log.close()
