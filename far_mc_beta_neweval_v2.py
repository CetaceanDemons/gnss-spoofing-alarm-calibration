"""Extensions of far_mc_beta_neweval.py (loaded from the same folder).

Per-satellite random streams, file selection with static models for ds2/3/4/7/8 and
dynamic models for ds5/ds6, loading of the QR per-tap residuals and of the AE/TCN
scores, and the per-instant threshold gamma at level alpha with the per-instant
detection rate after onset (Table 8). Imported by the alarm-layer scripts.
Run from the repository root (the working folder that also holds the data folders); see --help.
"""
import sys
import os
import glob
import argparse
import csv
import importlib.util
import re
import zlib
import numpy as np


# ---------------------------------------------------------------------------
# Definitions shared with far_mc_beta_neweval.py (loaded from the same folder)
# ---------------------------------------------------------------------------
V1_PATH = os.environ.get(
    'FAR_MC_PATH',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'far_mc_beta_neweval.py'))
if not os.path.exists(V1_PATH):
    sys.exit(f"[ERROR] 找不到 v1 far_mc_beta_neweval.py: {V1_PATH}\n"
             f"  把它和本脚本放同目录, 或设环境变量 FAR_MC_PATH 指向它。")
_spec = importlib.util.spec_from_file_location('far_mc_beta_neweval', V1_PATH)
v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v1)


AUTH                = v1.AUTH
ALIGN_START_MS      = v1.ALIGN_START_MS
ABS_CALIB_SCENARIOS = v1.ABS_CALIB_SCENARIOS
CALIB_RAW_S         = v1.CALIB_RAW_S
K0                  = v1.K0
pick_block_len      = v1.pick_block_len
calibrate_h         = v1.calibrate_h
cusum_alarm_times   = v1.cusum_alarm_times


DYN_SCENARIOS = {'ds5_r2', 'ds6_r2'}


# ---------------------------------------------------------------------------
# Score files: static models for ds2/3/4/7/8, dynamic models for ds5/ds6
# ---------------------------------------------------------------------------
def select_files_v2(npz_dir, dyn_mode):
    """Recursively collects pertap_*.npz; ds5_r2 and ds6_r2 use the _dyn files when dyn_mode='dynamic'.
    """
    allf = sorted(glob.glob(os.path.join(npz_dir, '**', 'pertap_*.npz'), recursive=True))
    seen = {}
    for f in allf:
        b = os.path.basename(f)
        if b not in seen:
            seen[b] = f
    allf = sorted(seen.values(), key=lambda x: os.path.basename(x))

    def is_dyn(b):
        return any(s in b for s in DYN_SCENARIOS)

    statics  = [f for f in allf if not is_dyn(os.path.basename(f))]                       # ds2/3/4/7/8
    dyn_dyn  = [f for f in allf if is_dyn(os.path.basename(f))
                and os.path.basename(f).endswith('_dyn.npz')]                              # ds5/ds6 _dyn
    dyn_stat = [f for f in allf if is_dyn(os.path.basename(f))
                and not os.path.basename(f).endswith('_dyn.npz')]
    if dyn_mode == 'dynamic':
        return statics + (dyn_dyn if dyn_dyn else dyn_stat)
    return statics + dyn_stat


SCORES_GLOB = '*_scores_dual*.npz'   # AE: {scen}_prn{p}_lstm_ae_scores_dual[_dyn].npz


def select_score_files(scores_dir, scores_dir_dyn, dyn_mode):
    """AE/TCN score files from two explicit folders (not recursive, so that seeds are never mixed).
    """
    def _scan(d):
        if not d:
            return []
        if not os.path.isdir(d):
            sys.exit(f"[ERROR] 分数目录不存在: {d}")
        return sorted(glob.glob(os.path.join(d, SCORES_GLOB)))

    def is_dyn_scen(b):
        return any(s in b for s in DYN_SCENARIOS)

    main_files = _scan(scores_dir)
    statics    = [f for f in main_files if not is_dyn_scen(os.path.basename(f))]
    dyn_stat   = [f for f in main_files if is_dyn_scen(os.path.basename(f))]

    if dyn_mode != 'dynamic':

        print(f'[AE] --ds6 static: ds5/ds6 用【静态模型】分数 {len(dyn_stat)} 星 '
              f'(违反"动态场景动态模型"原则, 仅供调试)')
        return statics + dyn_stat


    if not scores_dir_dyn:
        sys.exit(
            "[ERROR] --ds6 dynamic 下必须同时给 --scores-dir-dyn。\n"
            "  原因: 动态场景(ds5_r2/ds6_r2)要用动态 AE 模型的分数, 它们不在\n"
            "        --scores-dir 里, 而在 lstm_ae_inference_dual.py --dynamic\n"
            "        产出的 seed{S}_dual_dyn\\ 目录。\n"
            "  例:  --scores-dir     \"...\\lstm_ae_results\\seed42_dual\"\n"
            "       --scores-dir-dyn \"...\\lstm_ae_results\\seed42_dual_dyn\"\n"
            "  (若确实只想用静态模型跑 ds5/ds6 做调试, 加 --ds6 static)")

    dyn_files = _scan(scores_dir_dyn)
    dyn_dyn   = [f for f in dyn_files if is_dyn_scen(os.path.basename(f))]
    if not dyn_dyn:
        sys.exit(f"[ERROR] --scores-dir-dyn 下没有 ds5_r2/ds6_r2 的分数: "
                 f"{scores_dir_dyn}\n"
                 f"  先跑: python lstm_ae_inference_dual.py --seed 42 --dynamic")


    def _seedtag(d):
        b = os.path.basename(os.path.normpath(d))
        m = re.match(r'seed(\d+)', b)
        return m.group(1) if m else None
    s1, s2 = _seedtag(scores_dir), _seedtag(scores_dir_dyn)
    if s1 and s2 and s1 != s2:
        sys.exit(f"[ERROR] 两个分数目录的模型 seed 不一致: "
                 f"seed{s1}({scores_dir}) vs seed{s2}({scores_dir_dyn})。\n"
                 f"  混用不同 seed 的分数 = 把两个模型的结果混进一张表, 方法论错误。")

    print(f'[AE] 静态场景 {len(statics)} 星 <- {scores_dir}')
    print(f'[AE] 动态场景 {len(dyn_dyn)} 星 <- {scores_dir_dyn}  (ds5/ds6 用动态模型)')
    if dyn_stat:
        print(f'[AE] 已丢弃 --scores-dir 里 ds5/ds6 的静态版 {len(dyn_stat)} 星'
              f'(用错模型, 不进正式实验)')
    if not statics:
        sys.exit(f"[ERROR] --scores-dir 里一颗【静态场景】星都没有: {scores_dir}\n"
                 f"  常见原因: 指到了父目录 lstm_ae_results\\ 而不是 seed{{S}}_dual\\。\n"
                 f"  本函数【不递归】(防止把五个 seed 的同名文件混在一起), \n"
                 f"  必须指到具体的 seed 目录本身。")


    out, seen = [], {}
    for f in statics + dyn_dyn:
        b = os.path.basename(f)
        if b in seen:
            print(f'[WARN] 两个分数目录出现同名文件 {b}, 保留 {seen[b]}, '
                  f'忽略 {f} —— 请检查目录是否指错。')
            continue
        seen[b] = f
        out.append(f)
    return sorted(out, key=lambda x: os.path.basename(x))


def star_name_from_path(f):
    """Satellite name such as 'ds7_prn3' or 'ds4_r2_prn13' from either file-name pattern."""
    b = os.path.basename(f)
    for suf in ('_lstm_ae_scores_dual_dyn.npz', '_lstm_ae_scores_dual.npz',
                '_scores_dual_dyn.npz', '_scores_dual.npz', '.npz'):
        if b.endswith(suf):
            b = b[:-len(suf)]
            break
    if b.startswith('pertap_'):
        b = b[len('pertap_'):]
    if b.endswith('_dyn'):
        b = b[:-len('_dyn')]
    return b


# One stream per satellite, so a result does not depend on which satellites run together.
def star_rng(seed, star):
    """Independent random stream per satellite: seed + crc32(name) & 0xFFFF."""
    return np.random.default_rng(seed + (zlib.crc32(star.encode('utf-8')) & 0xFFFF))


# ---------------------------------------------------------------------------
# Per-instant threshold and detection rate (Table 8)
# ---------------------------------------------------------------------------
# Alarms in the DELTA_GUARD_S seconds before the onset are not counted as false alarms.
DELTA_GUARD_S = 0.02


def calib_gamma(Sc, alpha):
    """Per-instant threshold: (1-alpha) quantile of S on the calibration segment."""
    return float(np.quantile(np.asarray(Sc, dtype=float), 1.0 - alpha))


def consecutive_alarm_idx(S, gamma, k=5):
    """Index of the k-th sample of the first run of k consecutive S > gamma, or -1."""
    run = 0
    for i in range(len(S)):
        if S[i] > gamma:
            run += 1
            if run >= k:
                return i
        else:
            run = 0
    return -1


def detrate_series(S, gamma, asr_ms, bin_ms=1000):
    """Share of S > gamma in non-overlapping bins; returns (bin centres in s, shares, decisions).
    """
    dec = (np.asarray(S, dtype=float) > gamma).astype(np.float64)
    nbins = len(dec) // bin_ms
    centers = np.empty(nbins, dtype=np.float64)
    rates   = np.empty(nbins, dtype=np.float64)
    for b in range(nbins):
        rates[b]   = dec[b * bin_ms:(b + 1) * bin_ms].mean()
        centers[b] = (b * bin_ms + bin_ms // 2 + 100 + asr_ms) / 1000.0
    return centers, rates, dec


def post_onset_detrate(dec, oi):
    """Share of epochs after onset with S > gamma."""
    if oi >= len(dec):
        return float('nan')
    return float(dec[oi:].mean())


DEFAULT_ALPHAS = [0.10, 0.05, 0.01, 1e-3, 1e-4]


def alpha_tag(alpha):
    """Fixed short text for alpha in column names."""
    return f'{alpha:.0e}'


# 5-ms aggregation, for comparison with detectors that take 5-ms inputs.
AGG_MS = 5


def aggregate_dec(dec, agg_ms=AGG_MS, how='any'):
    """Aggregates 1-ms decisions to agg_ms ('any' or 'majority')."""
    d = np.asarray(dec, dtype=np.float64)
    n = (len(d) // agg_ms) * agg_ms
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    blk = d[:n].reshape(-1, agg_ms)
    if how == 'majority':
        return (blk.mean(1) > 0.5).astype(np.float64)
    return (blk.max(1) > 0).astype(np.float64)


def detrate_series_5ms(dec, asr_ms, bin_ms=1000, agg_ms=AGG_MS):
    """Binned detection rate after 5-ms aggregation, on the same time axis."""
    d5 = aggregate_dec(dec, agg_ms)
    per_bin = bin_ms // agg_ms
    nbins = len(d5) // per_bin
    centers = np.empty(nbins, dtype=np.float64)
    rates   = np.empty(nbins, dtype=np.float64)
    for b in range(nbins):
        rates[b]   = d5[b * per_bin:(b + 1) * per_bin].mean()
        centers[b] = (b * bin_ms + bin_ms // 2 + 100 + asr_ms) / 1000.0
    return centers, rates, d5


def post_onset_detrate_5ms(dec, oi, agg_ms=AGG_MS):
    """Detection rate after onset after 5-ms aggregation."""
    d5 = aggregate_dec(dec, agg_ms)
    oi5 = oi // agg_ms
    if oi5 >= len(d5):
        return float('nan')
    return float(d5[oi5:].mean())


def eval_alphas(S, Sc, asr_ms, onset, oi, alphas):
    """Per-instant metrics for each alpha (no recalibration of h)."""
    per_alpha, payload = {}, []
    for al in alphas:
        gamma = calib_gamma(Sc, al)
        centers, rates1, dec = detrate_series(S, gamma, asr_ms)
        _, rates5, _ = detrate_series_5ms(dec, asr_ms)
        a5 = consecutive_alarm_idx(S, gamma)
        mao_alarm = ((a5 + 100 + asr_ms) / 1000.0 - onset) if a5 >= 0 else None
        per_alpha[al] = dict(gamma=gamma, mao_alarm=mao_alarm,
                             post_dr=post_onset_detrate(dec, oi),
                             post_dr5=post_onset_detrate_5ms(dec, oi))
        payload.append((al, centers, rates1, rates5))
    return per_alpha, payload


# ---------------------------------------------------------------------------
# CSV layout
# ---------------------------------------------------------------------------
def detrate_csv_header():
    return ['star', 'scenario', 'prn', 'alpha', 'time_s', 'detrate_1ms', 'detrate_5ms']


def detrate_rows_from_payload(star, scen, prn, payload):
    out = []
    for al, centers, r1, r5 in payload:
        n = min(len(centers), len(r1), len(r5))
        for i in range(n):
            out.append((star, scen, prn, alpha_tag(al),
                        round(float(centers[i]), 3),
                        round(float(r1[i]), 4), round(float(r5[i]), 4)))
    return out


def alpha_csv_cols(alphas):
    """Column names of the main csv for each alpha."""
    cols = []
    for al in alphas:
        t = alpha_tag(al)
        cols += [f'gamma@{t}', f'mao_alarm_s@{t}', f'post_dr@{t}', f'post_dr_5ms@{t}']
    return cols


def alpha_csv_vals(per_alpha, alphas):
    vals = []
    for al in alphas:
        d = per_alpha.get(al, {})
        g = d.get('gamma'); ma = d.get('mao_alarm')
        pd_ = d.get('post_dr'); pd5 = d.get('post_dr5')
        vals += [
            f'{g:.6g}' if g is not None else '',
            f'{ma:.4f}' if ma is not None else '',
            f'{pd_:.4f}' if pd_ is not None and pd_ == pd_ else '',
            f'{pd5:.4f}' if pd5 is not None and pd5 == pd5 else '']
    return vals


# ---------------------------------------------------------------------------
# Loading the scores
# ---------------------------------------------------------------------------
# Input window of the networks: the first score belongs to epoch W_WARMUP of the record.
W_WARMUP = 100


def load_S_pertap(f):
    """QR score: per-tap residuals summed over taps."""
    d = np.load(f, allow_pickle=True)
    per = d['per'].astype(np.float64)
    lab = d['labels']
    scen = str(d['scenario']); prn = int(d['prn'])
    S = per.sum(1)
    return S, lab, scen, prn


def load_S_scores(f):
    """AE/TCN score S_total_old with labels aligned to it (labels[W:])."""
    d = np.load(f, allow_pickle=True)
    if 'S_total_old' not in d.files:
        raise KeyError(f"{os.path.basename(f)} 缺 S_total_old —— "
                       f"确认 --scores-dir 指的是 lstm_ae_inference_dual.py 的产物。")
    S = d['S_total_old'].astype(np.float64)
    N = len(S)


    if 'times' in d.files:
        t = d['times']
        if len(t) and int(t[0]) != W_WARMUP:
            raise AssertionError(
                f"{os.path.basename(f)}: times[0]={int(t[0])} != W={W_WARMUP}。"
                f"npz 的时间轴语义变了 —— oi=onset*1000-100-asr_ms 这条折算不再成立, "
                f"必须先核对 dump/inference 侧再跑, 否则延迟口径会整体错位。")


    lab_raw = d['labels']
    if len(lab_raw) == N + W_WARMUP:
        lab = lab_raw[W_WARMUP:W_WARMUP + N]
    elif len(lab_raw) == N:
        lab = lab_raw
    else:
        lab = lab_raw[W_WARMUP:W_WARMUP + N]
        if len(lab) != N:
            raise ValueError(
                f"{os.path.basename(f)}: labels 长度 {len(lab_raw)} 与 S 长度 {N} "
                f"既非相等也非差 W={W_WARMUP}, 无法安全对齐。")
        print(f'[WARN] {os.path.basename(f)}: labels 长度异常 '
              f'({len(lab_raw)} vs S {N}), 已按 [W:W+N] 切齐, 请复核。')

    scen = str(d['scenario']); prn = int(d['prn'])
    return S, lab, scen, prn


# ---------------------------------------------------------------------------
# Stand-alone run for the learned scores
# ---------------------------------------------------------------------------
def run_lstm_v2(beta, m_ms, n_boot, seg_s, dyn_mode, seed, npz_dir, alphas,
                stars_filter=None, exclude_ds5=False, exclude_ds2=False, verbose=False,
                calib_raw=CALIB_RAW_S, out_prefix=None, scores_dir=None,
                scores_dir_dyn=None):

    m = m_ms
    seg_len = int(seg_s * 1000)
    alphas = list(alphas)
    a_ref = alphas[-1]


    use_scores = scores_dir is not None
    if use_scores:
        files = select_score_files(scores_dir, scores_dir_dyn, dyn_mode)
        load_star_S = load_S_scores
        src_tag = ('LSTM-AE  scores-dir=' + str(scores_dir) +
                   (('  +dyn=' + str(scores_dir_dyn)) if scores_dir_dyn else ''))
    else:
        files = select_files_v2(npz_dir, dyn_mode)
        load_star_S = load_S_pertap
        src_tag = f'QR-LSTM  pertap-dir={npz_dir}'

    if stars_filter:
        files = [f for f in files if any(s in os.path.basename(f) for s in stars_filter)]
    else:
        if exclude_ds5:
            files = [f for f in files if 'ds5_r2' not in os.path.basename(f)]
        if exclude_ds2:
            files = [f for f in files if 'ds2_' not in os.path.basename(f)]


    if not stars_filter:
        expect = 21 if exclude_ds2 else 23
        if len(files) != expect:
            got = sorted(star_name_from_path(f) for f in files)
            print(f'[WARN] 星数 {len(files)} != 预期 {expect} '
                  f'({"六场景21星, ds2已排除" if exclude_ds2 else "含ds2共23星"})。')
            print(f'       实际: {got}')
            print(f'       若是有意为之(冒烟/子集)可忽略; '
                  f'正式跑批请先确认数据是否缺失。')

    if not files:
        if use_scores:
            print(f'[ERROR] {scores_dir} 下没找到 {SCORES_GLOB}。'
                  f'确认 --scores-dir 指到了 lstm_ae_results\\seed{{S}}_dual 目录本身'
                  f'(不是它的父目录, 也不递归)。')
        else:
            print(f'[ERROR] {npz_dir} 下没找到 pertap_*.npz。确认 --dir 指对了。')
        return []

    delta_idx = int(round(DELTA_GUARD_S * 1000))
    c_lo_s, c_hi_s = calib_raw
    print(f'[LSTM v2] 编码器来源: {src_tag}')
    print(f'[LSTM v2] beta={beta:.0e} m={m}ms nboot={n_boot} seg={seg_s}s '
          f'dyn={dyn_mode} seed={seed}  ({len(files)} stars)  δguard={DELTA_GUARD_S}s')
    print(f'[LSTM v2] rng: 每星独立(seed + crc32(星名)) '
          f'—— h 与星集/星序无关, 单跑==批跑, QR批==AE批 [E4]')
    print(f'[LSTM v2] α 档位({len(alphas)}): '
          f'{", ".join(alpha_tag(a) for a in alphas)}   '
          f'(α 只定 γ, 不影响 h/β → 标定只跑一次)')
    print(f'[LSTM v2] calib: {sorted(ABS_CALIB_SCENARIOS)} 用 raw '
          f'[{c_lo_s:.0f},{c_hi_s:.0f}]s 绝对窗; 其余 lab==0')
    print(f'{"star":16s} {"L":>4s} {"h":>8s} | {"cls":4s} {"evLat":>8s} | {"#FA":>4s} | '
          f'{"MaoAlm@" + alpha_tag(a_ref):>13s} {"pDR":>6s} {"pDR5":>6s}')
    print('-' * 100)

    rows, detrate_rows = [], []
    for f in files:

        S, lab, scen, prn = load_star_S(f)
        star = star_name_from_path(f)

        asr_ms = ALIGN_START_MS.get(scen, 0)
        onset = AUTH[scen]
        oi = int(round(onset * 1000 - 100 - asr_ms))


        rng = star_rng(seed, star)


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
        h_star, _, _ = calibrate_h(Zc, K0, beta, m, L, n_boot, seg_len, rng, verbose=verbose)


        per_alpha, payload = eval_alphas(S, Sc, asr_ms, onset, oi, alphas)
        detrate_rows += detrate_rows_from_payload(star, scen, prn, payload)
        ref = per_alpha[a_ref]

        if h_star is None:
            rows.append(dict(star=star, scen=scen, prn=prn, L=L, h=None, lat=None,
                             cls='H_NOT_FOUND', n_fa=None, fa_times=[],
                             per_alpha=per_alpha))
            print(f'{star:16s} {L:4d} {"NF":>8s} | H_NOT_FOUND (β={beta:.0e} 未命中)')
            continue


        alarms = cusum_alarm_times(Z, K0, h_star)
        pre  = [i for i in alarms if i < oi - delta_idx]
        post = [i for i in alarms if i >= oi]
        fa_times = [round((i + 100 + asr_ms) / 1000.0, 3) for i in pre]
        if post:
            det_i = post[0]
            lat = (det_i + 100 + asr_ms) / 1000.0 - onset
            cls = 'DET'
        else:
            lat, cls = None, 'MISS'

        rows.append(dict(star=star, scen=scen, prn=prn, L=L, h=h_star, lat=lat, cls=cls,
                         n_fa=len(pre), fa_times=fa_times, per_alpha=per_alpha))

        latd = f'{lat:+.3f}' if lat is not None else 'MISS'
        maod = f'{ref["mao_alarm"]:+.3f}' if ref['mao_alarm'] is not None else '—'
        print(f'{star:16s} {L:4d} {h_star:8.3f} | {cls:4s} {latd:>8s} | '
              f'{len(pre):>4d} | {maod:>13s} {ref["post_dr"]:6.3f} {ref["post_dr5"]:6.3f}')


    if out_prefix:
        p = f'{out_prefix}_lstm_detrate.csv'
        with open(p, 'w', newline='', encoding='utf-8-sig') as fh:
            w = csv.writer(fh)
            w.writerow(detrate_csv_header())
            w.writerows(detrate_rows)
        print(f'[saved] {p}  ({len(detrate_rows)} 行 = 星 × α档 × 1s窗)')
    return rows


def summarize_v2(rows, beta, alphas, out_path=None, tag='LSTM v2'):
    alphas = list(alphas)
    print('\n' + '=' * 88)
    det  = sum(1 for r in rows if r['cls'] == 'DET')
    miss = sum(1 for r in rows if r['cls'] == 'MISS')
    nf   = sum(1 for r in rows if r['cls'] == 'H_NOT_FOUND')
    lats = [r['lat'] for r in rows if r['cls'] == 'DET' and r['lat'] is not None]
    fa_total = sum(r['n_fa'] for r in rows if r['n_fa'] is not None)
    print(f'[{tag}] β={beta:.0e}  (共 {len(rows)} 星)')
    print(f'  ── 腿A 事件级(β-CUSUM, 与 α 无关) ──')
    print(f'  检出 {det}/{len(rows)}  MISS {miss}  NF {nf}  | δ守卫后 onset前FA总数={fa_total}')
    if lats:
        print(f'  事件延迟: 中位 {np.median(lats):+.3f}s  均值 {np.mean(lats):+.3f}s  '
              f'[{min(lats):+.3f},{max(lats):+.3f}]')


    print(f'  ── 腿B 逐样本检出率(γ=clean (1-α) 分位) ──')
    print(f'    {"alpha":>8s} | {"postDR中位":>10s} {"范围":>17s} | '
          f'{"5ms中位":>8s} | {"MaoAlarm中位":>12s}')
    for al in alphas:
        drs  = [r['per_alpha'][al]['post_dr'] for r in rows
                if al in r.get('per_alpha', {}) and r['per_alpha'][al]['post_dr'] ==
                r['per_alpha'][al]['post_dr']]
        drs5 = [r['per_alpha'][al]['post_dr5'] for r in rows
                if al in r.get('per_alpha', {}) and r['per_alpha'][al]['post_dr5'] ==
                r['per_alpha'][al]['post_dr5']]
        mas  = [r['per_alpha'][al]['mao_alarm'] for r in rows
                if al in r.get('per_alpha', {}) and
                r['per_alpha'][al]['mao_alarm'] is not None]
        if not drs:
            continue
        rng_s = f'[{min(drs):.3f},{max(drs):.3f}]'
        m5 = f'{np.median(drs5):.3f}' if drs5 else '—'
        mm = f'{np.median(mas):+.3f}s' if mas else '无'
        print(f'    {alpha_tag(al):>8s} | {np.median(drs):10.3f} {rng_s:>17s} | '
              f'{m5:>8s} | {mm:>12s}')
    if len(alphas) > 1:
        print(f'    ⚠ 最严档 {alpha_tag(alphas[-1])}: clean 段期望越界点数已很少, '
              f'γ 估计不稳 —— 这一档是【用来证明样本量下限】的, 不作主口径。')

    if out_path:
        with open(out_path, 'w', newline='', encoding='utf-8-sig') as fh:
            w = csv.writer(fh)

            w.writerow(['star', 'scenario', 'prn', 'block_L', 'h', 'class', 'latency_s',
                        'n_pre_onset_FA', 'pre_onset_FA_times_s'] + alpha_csv_cols(alphas))
            for r in sorted(rows, key=lambda x: (x['scen'], x['prn'])):
                w.writerow([
                    r['star'], r['scen'], r['prn'], r['L'],
                    f'{r["h"]:.4f}' if r['h'] is not None else '', r['cls'],
                    f'{r["lat"]:.4f}' if r['lat'] is not None else '',
                    r['n_fa'] if r['n_fa'] is not None else '',
                    ';'.join(f'{t:.3f}' for t in r['fa_times'])]
                    + alpha_csv_vals(r.get('per_alpha', {}), alphas))
        print(f'[saved] {out_path}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--beta', type=float, default=1e-5)
    ap.add_argument('--m', type=int, default=1000, help='滑动窗 ms')
    ap.add_argument('--nboot', type=int, default=1000)
    ap.add_argument('--seg', type=float, default=80.0)
    ap.add_argument('--ds6', dest='dyn', choices=['static', 'dynamic'], default='dynamic',
                    help='动态场景(ds5/ds6)取 _dyn 还是静态')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--dir', dest='npz_dir', default='.',
                    help=r'npz 目录(递归); 本机指 J:\Database\TEXBAT\processed_Mao')
    ap.add_argument('--scores-dir', dest='scores_dir', default=None,
                    help=r'[E1] 给了则改读 LSTM-AE 分数(S_total_old), 走消融那一路; '
                         r'不给则原样读 pertap(QR-LSTM)。'
                         r'例 J:\Database\TEXBAT\processed_Mao\lstm_ae_results')
    ap.add_argument('--scores-dir-dyn', dest='scores_dir_dyn', default=None,
                    help=r'[E1] 动态场景(ds5/ds6)的 AE 分数目录, --ds6 dynamic 下必给。'
                         r'例 ...\lstm_ae_results\seed42_dual_dyn')
    ap.add_argument('--stars', nargs='+', default=None, help='只跑含这些子串的星')
    ap.add_argument('--exclude-ds5', action='store_true', help='排除 ds5_r2(默认纳入)')
    ap.add_argument('--exclude-ds2', action='store_true',
                    help='[P4] 排除 ds2(默认【不】排除, ds2 照跑)')
    ap.add_argument('--alphas', type=float, nargs='+', default=DEFAULT_ALPHAS,
                    help='[P2] 逐样本检出率虚警水平, 可多档 '
                         '(默认 0.1 0.05 0.01 1e-3 1e-4; α 只定 γ, 不影响 h/β 标定)')
    ap.add_argument('--out', default=None,
                    help='输出前缀(自动补 _lstm.csv / _lstm_detrate.csv)')
    ap.add_argument('--verbose', action='store_true')
    a = ap.parse_args()

    alphas = sorted(set(a.alphas), reverse=True)
    rows = run_lstm_v2(a.beta, a.m, a.nboot, a.seg, a.dyn, a.seed, a.npz_dir, alphas,
                       a.stars, a.exclude_ds5, a.exclude_ds2, a.verbose,
                       out_prefix=a.out, scores_dir=a.scores_dir,
                       scores_dir_dyn=a.scores_dir_dyn)
    if rows:
        summarize_v2(rows, a.beta, alphas, (f'{a.out}_lstm.csv' if a.out else None))
