"""Core CUSUM utilities shared by the alarm layer.

Block length from the sample autocorrelation (Algorithm 1, step 2), reset CUSUM on
bootstrap matrices (share of length-m windows with an alarm), alarm times of the reset
CUSUM on one series, bisection for the smallest limit h whose windowed false-alarm
share is at most beta, selection of the score files, and the onset table AUTH
(Table 1). Imported by far_mc_beta_neweval_v2.py, far_calib_v32.py and
krule_clean_far_v12.py.
Run from the repository root (the working folder that also holds the data folders); see --help.
"""
import numpy as np, glob, os, argparse, csv

# ---------------------------------------------------------------------------
# Scenario constants
# ---------------------------------------------------------------------------
# Onsets in recording seconds (Table 1). ds3 still carries the value used before the
# revision; r1_revision_SEN/code/s1_onset_ds3.py sets it to 118.9 s at run time.
AUTH = {'ds2': 110.1, 'ds3': 119.6, 'ds4_r2': 113.8, 'ds5_r2': 102.1,
        'ds6_r2': 104.4, 'ds7': 110.0, 'ds8': 110.0}


# ds5 and ds8 start later, after the initial tracking transient (ms).
ALIGN_START_MS = {'ds5_r2': 21100, 'ds8': 16000}


# Calibration window of ds7/ds8 in the earlier evaluation (recording seconds).
ABS_CALIB_SCENARIOS = {'ds7', 'ds8'}
CALIB_RAW_S = (20.0, 100.0)


# CUSUM allowance k0 in standardised units (Section 3.1).
K0 = 4.0
EPS = 1e-9


# ---------------------------------------------------------------------------
# Autocorrelation and block length (Algorithm 1, step 2)
# ---------------------------------------------------------------------------
def acf(x, k):
    x = x - x.mean(); n = len(x)
    d = np.sum(x * x)
    if d == 0:
        return 0.0
    return np.sum(x[:n - k] * x[k:]) / d


def pick_block_len(Z, thresh=0.05, lag_max=500, floor=10):
    """Block length: first lag after which the ACF stays below thresh, with a lower floor."""
    below = 0
    for k in range(1, lag_max + 1):
        if abs(acf(Z, k)) < thresh:
            below += 1
            if below >= 3:
                return max(k - 2, floor)
        else:
            below = 0
    return lag_max


# ---------------------------------------------------------------------------
# Reset CUSUM, Eqs. (7)-(8)
# ---------------------------------------------------------------------------
# The share of length-m windows containing an alarm is the calibrated quantity;
# it stands in for the supremum of Eq. (10) (Section 3.3, 'What is estimated').
def cusum_alarm_flags_batch(Zmat, k0, h):
    """Reset CUSUM on each row of a (n_boot, seg_len) matrix; returns 0/1 alarm flags."""
    n_boot, seg_len = Zmat.shape
    g = np.zeros(n_boot)
    flags = np.zeros((n_boot, seg_len), dtype=np.int8)
    inc = Zmat - k0
    for t in range(seg_len):
        g += inc[:, t]
        np.maximum(g, 0.0, out=g)
        fired = g >= h
        if fired.any():
            flags[fired, t] = 1
            g[fired] = 0.0
    return flags


def cusum_alarm_times(Z, k0, h):
    """Reset CUSUM on one series; returns the indices of all alarms (g is reset to 0 after each).
    """
    g = 0.0
    alarms = []
    n = Z.shape[0]
    for i in range(n):
        g += Z[i] - k0
        if g < 0.0:
            g = 0.0
        if g >= h:
            alarms.append(i)
            g = 0.0
    return alarms


def cusum_first(Z, k0, h):
    """Index of the first g >= h, or -1."""
    g = 0.0
    for i in range(Z.shape[0]):
        g += Z[i] - k0
        if g < 0.0:
            g = 0.0
        elif g >= h:
            return i
    return -1


def cusum_gmax(Z, k0):
    g = 0.0; gm = 0.0
    for z in Z:
        g += z - k0
        if g < 0: g = 0.0
        elif g > gm: gm = g
    return gm


def mpfa_from_flags(flags, m):
    n_boot, seg_len = flags.shape
    if seg_len < m:
        return np.nan
    csum = np.cumsum(flags, axis=1)
    csum = np.concatenate([np.zeros((n_boot, 1), dtype=csum.dtype), csum], axis=1)
    hits = csum[:, m:] - csum[:, :-m]
    return float((hits >= 1).mean())


# ---------------------------------------------------------------------------
# Moving-block resampling and bisection for h (Algorithm 1, steps 3-7)
# ---------------------------------------------------------------------------
def make_synth_matrix(Zc, block_len, n_boot, seg_len, rng):
    nZ = len(Zc)
    n_blocks = int(np.ceil(seg_len / block_len))
    out = np.empty((n_boot, n_blocks * block_len), dtype=np.float64)
    for b in range(n_boot):
        starts = rng.integers(0, nZ - block_len + 1, size=n_blocks)
        cols = 0
        for s in starts:
            out[b, cols:cols + block_len] = Zc[s:s + block_len]
            cols += block_len
    return out[:, :seg_len]


def calibrate_h(Zc, k0, beta, m, block_len, n_boot, seg_len, rng, verbose=False):
    """Bisection for the smallest h with windowed false-alarm share <= beta."""
    seg_len = int(seg_len)
    gm = cusum_gmax(Zc, k0)
    Zmat = make_synth_matrix(Zc, block_len, n_boot, seg_len, rng)

    def mpfa_at(h):
        flags = cusum_alarm_flags_batch(Zmat, k0, h)
        return mpfa_from_flags(flags, m), int(flags.sum())

    lo, hi = 0.3 * gm, 8.0 * gm
    trace = []
    mpfa_hi, na_hi = mpfa_at(hi)
    trace.append((hi, mpfa_hi, na_hi))
    if mpfa_hi > beta:
        for mult in [16, 32, 64]:
            hi = mult * gm
            mpfa_hi, na_hi = mpfa_at(hi)
            trace.append((hi, mpfa_hi, na_hi))
            if mpfa_hi <= beta:
                break
        else:
            if verbose:
                for h, mp, na in trace:
                    print(f'    h={h:8.2f} MPFA={mp:.2e} ({na} al)')
            return None, trace, block_len
    mpfa_lo, na_lo = mpfa_at(lo)
    trace.append((lo, mpfa_lo, na_lo))

    for _ in range(18):
        mid = 0.5 * (lo + hi)
        mp, na = mpfa_at(mid)
        trace.append((mid, mp, na))
        if mp <= beta:
            hi = mid
        else:
            lo = mid
        if (hi - lo) / max(gm, 1e-9) < 0.01:
            break
    if verbose:
        for h, mp, na in sorted(trace):
            print(f'    h={h:8.2f}  MPFA_{m}={mp:.2e}  ({na} alarms)')
    return hi, trace, block_len


# ---------------------------------------------------------------------------
# Score files
# ---------------------------------------------------------------------------
def load_star(f):
    d = np.load(f, allow_pickle=True)
    per = d['per'].astype(np.float64); lab = d['labels']
    scen = str(d['scenario']); prn = int(d['prn']); ss = int(d['spoof_start'])
    return per, lab, scen, prn, ss


def select_files(npz_dir, ds6mode):
    """Recursively collects pertap_*.npz, removing duplicates; ds6 uses the static or the _dyn file.
    """
    allf = sorted(glob.glob(os.path.join(npz_dir, '**', 'pertap_*.npz'),
                            recursive=True))
    seen = {}
    for f in allf:
        b = os.path.basename(f)
        if b not in seen:
            seen[b] = f
    allf = sorted(seen.values(), key=lambda x: os.path.basename(x))
    others = [f for f in allf if 'ds6_r2' not in os.path.basename(f)]
    ds6_static = [f for f in allf if 'ds6_r2' in os.path.basename(f)
                  and not os.path.basename(f).endswith('_dyn.npz')]
    ds6_dyn = [f for f in allf if os.path.basename(f).endswith('_dyn.npz')]
    if ds6mode == 'dynamic':
        return others + (ds6_dyn if ds6_dyn else ds6_static)
    return others + ds6_static


# ---------------------------------------------------------------------------
# Stand-alone run of the earlier evaluation protocol
# ---------------------------------------------------------------------------
def run(beta, m_ms, n_boot, seg_s, ds6mode, seed, npz_dir,
        stars_filter=None, include_ds5=False, verbose=False,
        calib_raw=CALIB_RAW_S):
    rng = np.random.default_rng(seed)
    m = m_ms
    seg_len = int(seg_s * 1000)
    files = select_files(npz_dir, ds6mode)

    if stars_filter:
        files = [f for f in files
                 if any(s in os.path.basename(f) for s in stars_filter)]
    elif not include_ds5:
        dropped = [f for f in files if 'ds5_r2' in os.path.basename(f)]
        files = [f for f in files if 'ds5_r2' not in os.path.basename(f)]
        if dropped:
            print(f'[note] 跳过 ds5_r2 × {len(dropped)}: 本机只有静态模型 npz, '
                  f'但 ds5 是动态场景(用错模型)。要纳入: --include-ds5 或 --stars ds5_r2\n')

    if not files:
        print(f'[ERROR] 在 {npz_dir} 下没找到 pertap_*.npz。确认 --dir 指对了 '
              f'(应能找到 pertap_residual_seed42 / _dynamic / _seed42_ds78)。')
        return []

    print(f'[cfg] beta={beta:.0e}  m={m}ms  n_boot={n_boot}  seg={seg_s}s  '
          f'ds6={ds6mode}  seed={seed}')
    print(f'[cfg] {len(files)} stars  (effective MC samples/star = {n_boot*seg_len:,})')
    print(f'[cfg] dir={npz_dir}')
    print(f'[cfg] calib window: {sorted(ABS_CALIB_SCENARIOS)} 用 raw '
          f'[{calib_raw[0]:.0f},{calib_raw[1]:.0f}]s 绝对窗; 其余场景用各自 lab==0')
    print(f'[cfg] align_start(ms) = {ALIGN_START_MS} (其余=0; onset 闸门据此折算)')
    print(f'[eval] NEW standard: onset 后首报=检测延迟 + onset 前全部报警=误报\n')
    print(f'{"star":16s} {"L":>4s} {"h":>8s} | {"class":5s} {"latency":>9s} | '
          f'{"#FA":>4s}  pre-onset FA times (s)')
    print('-' * 92)

    c_lo_s, c_hi_s = calib_raw
    rows = []
    for f in files:
        per, lab, scen, prn, ss = load_star(f)
        asr_ms = ALIGN_START_MS.get(scen, 0)

        onset = AUTH[scen]

        oi = int(round(onset * 1000 - 100 - asr_ms))
        si = ss - 100

        S = per.sum(1)


        if scen in ABS_CALIB_SCENARIOS:
            lo_idx = max(0, int(round(c_lo_s * 1000 - 100 - asr_ms)))
            hi_idx = min(len(S), int(round(c_hi_s * 1000 - 100 - asr_ms)))
            cmask = np.zeros(len(S), dtype=bool)
            if hi_idx > lo_idx:
                cmask[lo_idx:hi_idx] = True
            if cmask.sum() < 10:
                cmask = (lab == 0)
        else:
            cmask = (lab == 0)

        Sc = S[cmask]
        mu, sd = Sc.mean(), Sc.std()
        Z = (S - mu) / sd
        Zc = Z[cmask]

        L = pick_block_len(Zc)
        h_star, results, _ = calibrate_h(Zc, K0, beta, m, L, n_boot, seg_len,
                                         rng, verbose=verbose)

        star = os.path.basename(f).replace('pertap_', '').replace('_dyn', '').replace('.npz', '')
        if h_star is None:
            rows.append(dict(star=star, scen=scen, prn=prn, L=L, h=None,
                             lat=None, cls='H_NOT_FOUND', sub='-',
                             n_fa=None, fa_times=[]))
            print(f'{star:16s} {L:4d} {"NF":>8s} | H_NOT_FOUND (β={beta:.0e} 未命中)')
            continue


        alarms = cusum_alarm_times(Z, K0, h_star)
        pre = [i for i in alarms if i < oi]
        post = [i for i in alarms if i >= oi]
        fa_times = [round((i + 100 + asr_ms) / 1000.0, 3) for i in pre]
        if post:
            det_i = post[0]
            lat = (det_i + 100 + asr_ms) / 1000.0 - onset
            sub = 'DET_g' if det_i < si else 'DET_s'
            cls = 'DET'
        else:
            lat, sub, cls = None, '-', 'MISS'

        rows.append(dict(star=star, scen=scen, prn=prn, L=L, h=h_star,
                         lat=lat, cls=cls, sub=sub,
                         n_fa=len(pre), fa_times=fa_times))


        if fa_times:
            shown = ','.join(f'{t:.2f}' for t in fa_times[:6])
            if len(fa_times) > 6:
                shown += f' ...(+{len(fa_times) - 6})'
        else:
            shown = '-'
        lat_disp = f'{lat:+.3f}s' if lat is not None else 'MISS'
        print(f'{star:16s} {L:4d} {h_star:8.3f} | {cls:5s} {lat_disp:>9s} | '
              f'{len(pre):>4d}  [{shown}]')
    return rows


def summarize(rows, beta, out_path=None):
    print('\n' + '=' * 78)
    print(f'汇总 @ β={beta:.0e}  (共 {len(rows)} 星)  — 新评价标准')
    print('=' * 78)
    det = sum(1 for r in rows if r['cls'] == 'DET')
    miss = sum(1 for r in rows if r['cls'] == 'MISS')
    nf = sum(1 for r in rows if r['cls'] == 'H_NOT_FOUND')
    lats = [r['lat'] for r in rows if r['cls'] == 'DET' and r['lat'] is not None]
    fa_total = sum(r['n_fa'] for r in rows if r['n_fa'] is not None)
    stars_with_fa = [r for r in rows if r['n_fa']]
    print(f'onset 后检出 {det}/{len(rows)}   漏检(MISS) {miss}   h未命中 {nf}')
    if lats:
        print(f'检测延迟(onset 后首报): 中位 {np.median(lats):+.3f}s  '
              f'均值 {np.mean(lats):+.3f}s  范围 [{min(lats):+.3f}, {max(lats):+.3f}]')
    print(f'onset 前误报总数 = {fa_total}  (分布在 {len(stars_with_fa)} 颗星)')
    print()
    print(f'{"star":16s} {"L":>4s} {"h":>8s} {"class":>6s} {"sub":>6s} '
          f'{"latency":>9s} {"#FA":>4s}  FA times (s)')
    print('-' * 96)
    for r in sorted(rows, key=lambda x: (x['scen'], x['prn'])):
        hs = f'{r["h"]:.3f}' if r['h'] is not None else '-'
        ls = f'{r["lat"]:+.3f}' if r['lat'] is not None else '-'
        nf_s = str(r['n_fa']) if r['n_fa'] is not None else '-'
        ft = ';'.join(f'{t:.2f}' for t in r['fa_times']) if r['fa_times'] else '-'
        if len(ft) > 40:
            ft = ft[:40] + '...'
        print(f'{r["star"]:16s} {r["L"]:4d} {hs:>8s} {r["cls"]:>6s} {r["sub"]:>6s} '
              f'{ls:>9s} {nf_s:>4s}  {ft}')

    if out_path:
        with open(out_path, 'w', newline='', encoding='utf-8-sig') as fh:
            w = csv.writer(fh)
            w.writerow(['star', 'scenario', 'prn', 'block_L', 'h', 'class',
                        'det_subwindow', 'latency_s', 'n_pre_onset_FA',
                        'pre_onset_FA_times_s'])
            for r in sorted(rows, key=lambda x: (x['scen'], x['prn'])):
                w.writerow([
                    r['star'], r['scen'], r['prn'], r['L'],
                    f'{r["h"]:.4f}' if r['h'] is not None else '',
                    r['cls'], r['sub'],
                    f'{r["lat"]:.4f}' if r['lat'] is not None else '',
                    r['n_fa'] if r['n_fa'] is not None else '',
                    ';'.join(f'{t:.3f}' for t in r['fa_times'])])
        print(f'\n[saved] {out_path}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--beta', type=float, default=1e-5)
    ap.add_argument('--m', type=int, default=1000, help='滑动窗 ms')
    ap.add_argument('--nboot', type=int, default=1000,
                    help='每星 block-bootstrap 条数(1e-5 建议 >=1000)')
    ap.add_argument('--seg', type=float, default=80.0, help='每条合成 clean 秒数')
    ap.add_argument('--ds6', choices=['static', 'dynamic'], default='dynamic')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--dir', dest='npz_dir', default='.',
                    help=r'npz 目录(递归); 本机指 J:\Database\TEXBAT\processed_Mao')
    ap.add_argument('--stars', nargs='+', default=None, help='只跑含这些子串的星')
    ap.add_argument('--include-ds5', action='store_true',
                    help='纳入 ds5_r2(静态模型 npz, 动态场景, 仅供 eyeball)')
    ap.add_argument('--out', default=None, help='结果 CSV 输出路径')
    ap.add_argument('--calib-start-s', type=float, default=CALIB_RAW_S[0],
                    help='ds7/ds8 绝对标定窗起点(raw 秒, 默认 20)')
    ap.add_argument('--calib-end-s', type=float, default=CALIB_RAW_S[1],
                    help='ds7/ds8 绝对标定窗终点(raw 秒, 默认 100)')
    ap.add_argument('--verbose', action='store_true')
    a = ap.parse_args()
    rows = run(a.beta, a.m, a.nboot, a.seg, a.ds6, a.seed, a.npz_dir,
               a.stars, a.include_ds5, a.verbose,
               calib_raw=(a.calib_start_s, a.calib_end_s))
    if rows:
        summarize(rows, a.beta, a.out)
