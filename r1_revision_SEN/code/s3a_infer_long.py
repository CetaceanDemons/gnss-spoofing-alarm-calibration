# -*- coding: utf-8 -*-
"""Scores the long clean recordings (Section 4.5).

TEXBAT cleanStatic and OAKBAT cleanDynamic with the static networks; TEXBAT
cleanDynamic, the clean recording of the drive replayed in ds5/ds6, with the dynamic
networks. Also computes the two hand-crafted metrics and, for TEXBAT cleanDynamic,
the start of ds5 and ds6 within it by carrier-Doppler matching (Appendix D).
Model weights are not included.
Run from the repository root (the working folder that also holds the data folders); see --help.
"""
import argparse
import glob
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEF_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))      # J:\Database\TEXBAT\processed_Mao

# ---------------------------------------------------------------------------
# Recordings and settings
# ---------------------------------------------------------------------------
OAK_BASE = r'J:\Database\OAKBAT\processed_03_26\Mao_base'
W, N_TAPS, PROMPT, BATCH = 100, 21, 10, 2048
MU_LEN = 80_000
NCHK = 20_000
FS = 25_000_000
GAP = 1.5 * (FS / 1000)
REC = np.dtype([('taps', np.float32, (21,)), ('prompt_iq', np.float32, (2,)), ('timestamp', np.uint64),
                ('states', np.float32, (12,)), ('next_epoch', np.uint64), ('prn', np.int32)])
CD_DIR, CD_CHS = 'cleanDynamic_r2', (0, 1, 2, 3, 4, 5)
DS_DIRS = ('ds5_r2', 'ds6_r2')
DOP_WIN = (5.0, 90.0)
LAG_RANGE = (-60.0, 80.0)
# Static networks for TEXBAT cleanStatic and OAKBAT cleanDynamic, dynamic
# networks for TEXBAT cleanDynamic, as for the scenarios they correspond to.
SET_MODELS = {'TEX_cS': 'static', 'OAK_cD': 'static', 'TEX_cD': 'dyn'}


# ---------------------------------------------------------------------------
# Networks and scores (Section 4.1)
# ---------------------------------------------------------------------------
def load_models(paths):
    from tensorflow.keras.models import load_model
    return {k: load_model(p, compile=False) for k, p in paths.items()}


def _predict(model, X):
    return model.predict(X, batch_size=len(X), verbose=0)


def qr_swap(model, taps_n):
    """Identifies which QR output is the lower quantile from their means on the first windows."""
    X = np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(taps_n[:W + 512], (W, N_TAPS))[:512, 0])
    o0, o1 = _predict(model, X)
    return bool(np.mean(o0) > np.mean(o1))


def score_series(models, taps_raw, mu, sig, n_max=None, swap=False, log=print):
    """Returns per-tap QR residuals [N,21], S_ae [N] and S_tcn [N], N = T - W."""
    taps_n = np.ascontiguousarray(((taps_raw - mu) / sig).astype(np.float32))
    N = len(taps_n) - W
    if n_max is not None:
        N = min(N, n_max)
    view = np.lib.stride_tricks.sliding_window_view(taps_n, (W, N_TAPS))[:, 0]
    per = np.zeros((N, N_TAPS), np.float32) if 'qr' in models else None
    s_ae = np.zeros(N, np.float32) if 'ae' in models else None
    s_tcn = np.zeros(N, np.float32) if 'tcn' in models else None
    t0 = time.time()
    for s in range(0, N, BATCH):
        e = min(s + BATCH, N)
        X = np.ascontiguousarray(view[s:e])
        y = taps_n[s + W:e + W]
        if per is not None:
            o0, o1 = _predict(models['qr'], X)
            lo, hi = (o1, o0) if swap else (o0, o1)
            per[s:e] = np.maximum(0, y - hi) + np.maximum(0, lo - y)
        if s_ae is not None:
            s_ae[s:e] = ((y - _predict(models['ae'], X)) ** 2).sum(1)
        if s_tcn is not None:
            s_tcn[s:e] = ((y - _predict(models['tcn'], X)) ** 2).sum(1)
        if (s // BATCH) % 50 == 0 and s > 0:
            log(f'      {e}/{N}  {(time.time() - t0) / 60:.1f} min')
    return per, s_ae, s_tcn


# ---------------------------------------------------------------------------
# Alignment and standardisation
# ---------------------------------------------------------------------------
def pooled_mu_sigma(taps_list):
    cat = np.concatenate([x[:MU_LEN] for x in taps_list], axis=0)
    mu = cat.mean(axis=0, keepdims=True).astype(np.float32)
    sig = (cat.std(axis=0, keepdims=True) + 1e-8).astype(np.float32)
    return mu, sig


def _longest(d):
    bp = np.unique(np.concatenate(([0], np.where(np.diff(d['timestamp'].astype(np.float64)) > GAP)[0] + 1,
                                   [len(d)])))
    return max(((int(bp[i]), int(bp[i + 1])) for i in range(len(bp) - 1)), key=lambda r: r[1] - r[0])


def align_dat(sc_dir, chs, seg_log=None):
    """Aligns the channels: longest continuous stretch of each, common start rounded up to 100 ms.
    """
    ranges, used = [], []
    for ch in chs:
        f = os.path.join(sc_dir, f'trk_ch_{ch}.dat')
        if not os.path.exists(f):
            continue
        d = np.fromfile(f, dtype=REC)
        best = _longest(d)
        if seg_log is not None:
            seg_log.append((ch, int(d['prn'][best[0]]), best, len(d)))
        ranges.append(best)
        used.append(ch)
    start = int(math.ceil(max(r[0] for r in ranges) / 100.0) * 100)
    end = min(r[1] for r in ranges)
    return start, end, used


def _load_stars(sc_dir, chs, seg_log=None):
    A, E, used = align_dat(sc_dir, chs, seg_log)
    stars, t0 = [], None
    for ch in used:
        d = np.fromfile(os.path.join(sc_dir, f'trk_ch_{ch}.dat'), dtype=REC)
        stars.append((int(d['prn'][A]), d['taps'][A:E].astype(np.float32)))
        if t0 is None:
            t0 = float(d['timestamp'][A]) / FS
    return stars, A, E, t0


# ---------------------------------------------------------------------------
# The three long recordings (Section 4.5)
# ---------------------------------------------------------------------------
def tex_cs(root):
    stars, A, E, _ = _load_stars(os.path.join(root, 'cleanStatic'), (0, 1))
    return stars, A, E


def tex_cd(root, seg_log=None):
    """TEXBAT cleanDynamic: fixed channels 0-5."""
    return _load_stars(os.path.join(root, CD_DIR), CD_CHS, seg_log)


def oak_cd(oak_dir):
    stars = []
    for f in sorted(glob.glob(os.path.join(oak_dir, 'cleanDynamic_*_raw_taps.npy'))):
        prn = int(os.path.basename(f).split('_prn')[1].split('_')[0])
        stars.append((prn, np.load(f).astype(np.float32)))
    return stars


# ---------------------------------------------------------------------------
# Carrier-to-noise proxy
# ---------------------------------------------------------------------------
def cn0_block(taps_win):
    """C/N0 proxy as in dump_pertap_residual_v2: noise variance of the two outer taps over the first 80 s.
    """
    hi = max(0, MU_LEN - W)
    nb = np.concatenate([taps_win[:hi, 0], taps_win[:hi, 20]]).astype(np.float64)
    nv = float(np.var(nb)) + 1e-12
    pp = taps_win[:, PROMPT].astype(np.float64) ** 2
    return (pp / nv).astype(np.float32), pp.astype(np.float32), np.float32(nv)


# ---------------------------------------------------------------------------
# Start of ds5/ds6 within TEXBAT cleanDynamic by carrier-Doppler matching (Appendix D)
# ---------------------------------------------------------------------------
def _dop_series(sc_dir, ch):
    d = np.fromfile(os.path.join(sc_dir, f'trk_ch_{ch}.dat'), dtype=REC)
    prn = int(np.bincount(d['prn'].astype(np.int64)).argmax())
    keep = (d['prn'] == prn) & (d['states'][:, 9] > 35.0)
    return prn, d['timestamp'][keep].astype(np.float64) / FS, d['states'][keep, 1].astype(np.float64)


def _bin(t, x, t0, t1, dt):
    nb = int(round((t1 - t0) / dt))
    idx = np.floor((t - t0) / dt).astype(np.int64)
    ok = (idx >= 0) & (idx < nb)
    s = np.bincount(idx[ok], weights=x[ok], minlength=nb)
    c = np.bincount(idx[ok], minlength=nb)
    with np.errstate(invalid='ignore', divide='ignore'):
        m = s / c
    m[c < max(1, int(0.5 * dt * 1000))] = np.nan
    return t0 + (np.arange(nb) + 0.5) * dt, m


def _match(tc, y, tg, g, lags):
    """For each lag L, std and mean of y(t) - g(t + L) (carrier-Doppler matching)."""
    ok = np.isfinite(g)
    tg, g = tg[ok], g[ok]
    fy = np.isfinite(y)
    sd = np.full(len(lags), np.nan)
    mo = np.full(len(lags), np.nan)
    for k, L in enumerate(lags):
        gi = np.interp(tc + L, tg, g, left=np.nan, right=np.nan)
        r = y - gi
        m = fy & np.isfinite(r)
        if m.sum() < 0.8 * fy.sum():
            continue
        mo[k] = r[m].mean()
        sd[k] = r[m].std()
    return sd, mo


def estimate_offset(root, log):
    cd = {}
    for ch in CD_CHS:
        f = os.path.join(root, CD_DIR, f'trk_ch_{ch}.dat')
        if os.path.exists(f):
            prn, t, x = _dop_series(os.path.join(root, CD_DIR), ch)
            cd[prn] = (t, x)
    res = {}
    for ds in DS_DIRS:
        pairs = []
        for ch in range(5):
            f = os.path.join(root, ds, f'trk_ch_{ch}.dat')
            if not os.path.exists(f):
                continue
            prn, t, x = _dop_series(os.path.join(root, ds), ch)
            if prn not in cd:
                continue
            tc, y = _bin(t, x, DOP_WIN[0], DOP_WIN[1], 0.1)
            tg, g = _bin(cd[prn][0], cd[prn][1], 0.0, 445.0, 0.1)
            lags = np.arange(LAG_RANGE[0], LAG_RANGE[1] + 1e-9, 0.1)
            sd, _ = _match(tc, y, tg, g, lags)
            if not np.isfinite(sd).any():
                continue
            k = int(np.nanargmin(sd))
            L0, sd_c = float(lags[k]), float(sd[k])
            far = np.isfinite(sd) & (np.abs(lags - L0) > 2.0)
            nxt = float(np.min(sd[far])) if far.any() else float('nan')
            tc2, y2 = _bin(t, x, DOP_WIN[0], DOP_WIN[1], 0.01)
            tg2, g2 = _bin(cd[prn][0], cd[prn][1], 0.0, 445.0, 0.01)
            lags2 = np.arange(L0 - 0.15, L0 + 0.15 + 1e-9, 0.002)
            sd2, mo2 = _match(tc2, y2, tg2, g2, lags2)
            k2 = int(np.nanargmin(sd2))
            pairs.append(dict(prn=prn, L=round(float(lags2[k2]), 3), resid_std_hz=round(float(sd2[k2]), 3),
                              dop_offset_hz=round(float(mo2[k2]), 2), coarse_std_hz=round(sd_c, 3),
                              next_best_std_hz=round(nxt, 3)))
        if pairs:
            Ls = np.array([p['L'] for p in pairs])
            res[ds] = dict(L_med=round(float(np.median(Ls)), 3), L_spread=round(float(Ls.max() - Ls.min()), 3),
                           pairs=pairs)
    log('\n== [v1.2] cleanDynamic ↔ ds5/ds6 时间偏移（t_cd = t_ds + L；多普勒匹配，ds 录音 '
        f'{DOP_WIN[0]:.0f}–{DOP_WIN[1]:.0f} s）')
    for ds, r in res.items():
        log(f'  {ds}: L 中位 {r["L_med"]:.3f} s，各星极差 {r["L_spread"]:.3f} s')
        for p in r['pairs']:
            log(f'     PRN{p["prn"]:>3}  L={p["L"]:8.3f}  残差σ={p["resid_std_hz"]:.3f} Hz（粗 {p["coarse_std_hz"]:.2f}，'
                f'偏 2 s 外的次优 ≥{p["next_best_std_hz"]:.2f}）  常数频偏 {p["dop_offset_hz"]:+.2f} Hz')
    if not res:
        log('  [警告] 没有可匹配的星（检查 cleanDynamic_r2 / ds5_r2 / ds6_r2 是否都在）')
    return res


def _first(pattern):
    fs = sorted(glob.glob(pattern))
    return fs[0] if fs else pattern


# ---------------------------------------------------------------------------
# Check against the stored scores of the paper
# ---------------------------------------------------------------------------
def load_exp1(root):
    spec = importlib.util.spec_from_file_location('experiment1_zscore_ablation',
                                                  os.path.join(root, 'experiment1_zscore_ablation.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.TEXBAT_ROOT = root
    return m


def _refs(root, kind, scen, prn):
    if kind == 'static':
        return {'qr': (os.path.join(root, 'pertap_residual_seed42_ds78', f'pertap_{scen}_prn{prn}.npz'), 'per'),
                'ae': (_first(os.path.join(root, 'lstm_ae_results', 'seed42_dual', f'{scen}_prn{prn}_*scores_dual.npz')),
                       'S_total_old'),
                'tcn': (_first(os.path.join(root, 'tcn_results', 'seed42_dual', f'{scen}_prn{prn}_*scores_dual.npz')),
                        'S_total_old')}
    return {'qr': (os.path.join(root, 'pertap_residual_dynamic', f'pertap_{scen}_prn{prn}_dyn.npz'), 'per'),
            'ae': (_first(os.path.join(root, 'lstm_ae_results', 'seed42_dual_dyn', f'{scen}_prn{prn}_*scores_dual_dyn.npz')),
                   'S_total_old'),
            'tcn': (_first(os.path.join(root, 'tcn_results', 'seed42_dual_dyn', f'{scen}_prn{prn}_*scores_dual_dyn.npz')),
                    'S_total_old')}


def identity_check(root, models, swap, log, kind='static'):
    """Checks the networks against stored scores of one satellite (ds7 static, ds6_r2 dynamic)."""
    scen = 'ds7' if kind == 'static' else 'ds6_r2'
    exp1 = load_exp1(root)
    cfg = exp1.SCENARIO_CONFIG[scen]
    alignment, _ = exp1.align_scenario(scen, cfg)
    mu, sig = exp1.compute_local_mu_sigma(scen, cfg, alignment)
    ch = cfg['channels'][0]
    d = np.fromfile(os.path.join(root, cfg['dir'], f'trk_ch_{ch}.dat'), dtype=exp1.record_dtype)
    prn = int(d['prn'][alignment['start']])
    taps = d['taps'][alignment['start']:alignment['start'] + NCHK + W].astype(np.float32)
    per, s_ae, s_tcn = score_series(models, taps, mu, sig, n_max=NCHK, swap=swap)
    ref = _refs(root, kind, scen, prn)
    new = {'qr': None if per is None else per.sum(1), 'ae': s_ae, 'tcn': s_tcn}
    ok = True
    for k in models:
        p, fld = ref[k]
        if not os.path.exists(p):
            log(f'  [检查] {k}: 找不到参照 {p}')
            ok = False
            continue
        r = np.load(p, allow_pickle=True)[fld][:NCHK].astype(np.float64)
        r = r.sum(1) if r.ndim == 2 else r
        x = new[k].astype(np.float64)
        dif = np.abs(x - r)
        rel = float(dif.max() / (np.abs(r).max() + 1e-12))
        good = rel < 1e-3
        ok &= good
        log(f'  [检查·{kind}] {k}: {scen}_prn{prn} 前 {NCHK / 1000:.0f} s  最大相对差 {rel:.2e}  '
            f'均值 新 {x.mean():.5f} / 旧 {r.mean():.5f}  → {"一致" if good else "不一致"}')
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=DEF_ROOT)
    ap.add_argument('--oak-dir', default=os.path.join(OAK_BASE, 'oakbat_single_star_v2'))
    ap.add_argument('--model-qr', default=os.path.join(OAK_BASE, 'qr_lstm_results_v2_mod1', 'qr_lstm_v1_seed42.keras'))
    ap.add_argument('--model-ae', default=os.path.join(OAK_BASE, 'lstm_ae_results', 'lstm_ae_seed42.keras'))
    ap.add_argument('--model-tcn', default=os.path.join(OAK_BASE, 'tcn_results', 'tcn_seed42.keras'))
    ap.add_argument('--dyn-model-qr', default=os.path.join(OAK_BASE, 'dynamic_results_expB', 'qr_lstm_dynamic.keras'))
    ap.add_argument('--dyn-model-ae', default=os.path.join(OAK_BASE, 'lstm_ae_results', 'lstm_ae_dynamic_seed42.keras'))
    ap.add_argument('--dyn-model-tcn', default=os.path.join(OAK_BASE, 'tcn_results', 'tcn_dynamic_seed42.keras'))
    ap.add_argument('--models', nargs='+', default=['qr', 'ae', 'tcn'], choices=['qr', 'ae', 'tcn'])
    ap.add_argument('--sets', nargs='+', default=['TEX_cD'], choices=['TEX_cS', 'OAK_cD', 'TEX_cD'])
    ap.add_argument('--out-dir', default=None, help=r'默认 <root>\r1_revision_SEN\s3')
    ap.add_argument('--check-only', action='store_true')
    ap.add_argument('--skip-check', action='store_true', help='跳过模型一致性检查（不建议）')
    a = ap.parse_args(argv)
    root = os.path.abspath(a.root)
    out = os.path.abspath(a.out_dir or os.path.join(root, 'r1_revision_SEN', 's3'))
    sdir = os.path.join(out, 'scores')
    os.makedirs(sdir, exist_ok=True)
    brief = 's3a_cd_brief.txt' if 'TEX_cD' in a.sets else 's3a_brief.txt'
    B = []

    def log(s):
        print(s)
        B.append(s)

    kinds = sorted({SET_MODELS[s] for s in a.sets})                 # 'dyn' / 'static'
    log(f'[s3a] v1.2  {time.strftime("%Y-%m-%d %H:%M:%S")}  root={root}  sets={a.sets}  模型组={kinds}')
    allp = {'static': {'qr': a.model_qr, 'ae': a.model_ae, 'tcn': a.model_tcn},
            'dyn': {'qr': a.dyn_model_qr, 'ae': a.dyn_model_ae, 'tcn': a.dyn_model_tcn}}
    mpaths = {kd: {k: allp[kd][k] for k in a.models} for kd in kinds}
    need = [p for kd in kinds for p in mpaths[kd].values()] + [os.path.join(root, 'experiment1_zscore_ablation.py')]
    if 'TEX_cS' in a.sets or 'static' in kinds:
        need += [os.path.join(root, 'cleanStatic', 'trk_ch_0.dat'), os.path.join(root, 'cleanStatic', 'trk_ch_1.dat')]
    if 'OAK_cD' in a.sets:
        need.append(a.oak_dir)
    if 'TEX_cD' in a.sets:
        need += [os.path.join(root, CD_DIR, f'trk_ch_{c}.dat') for c in CD_CHS]
        need += [os.path.join(root, ds, 'trk_ch_0.dat') for ds in DS_DIRS]
    miss = [p for p in need if not os.path.exists(p)]
    for p in need:
        log(f'  [路径] {"OK  " if os.path.exists(p) else "缺失"} {p}')
    if miss:
        log(f'[STOP] 有路径缺失，把 {brief} 发回来')
        _write(out, B, brief)
        sys.exit(1)
    if 'OAK_cD' in a.sets:
        log(f'  [路径] OAKBAT cleanDynamic 文件 {len(glob.glob(os.path.join(a.oak_dir, "cleanDynamic_*_raw_taps.npy")))} 个')

    if 'TEX_cD' in a.sets:
        off = estimate_offset(root, log)
        with open(os.path.join(out, 'cd_offset.json'), 'w', encoding='utf-8') as fh:
            json.dump(off, fh, ensure_ascii=False, indent=1)
        log(f'  [写出] {os.path.join(out, "cd_offset.json")}')

    M, SW = {}, {}
    for kd in kinds:
        M[kd] = load_models(mpaths[kd])
        SW[kd] = False
        if 'qr' in M[kd]:
            if kd == 'static':
                stars, _, _ = tex_cs(root)
            else:
                stars, _, _, _ = tex_cd(root)
            mu0, sig0 = pooled_mu_sigma([x for _, x in stars])
            SW[kd] = qr_swap(M[kd]['qr'], ((stars[0][1] - mu0) / sig0).astype(np.float32))
            log(f'  [QR·{kd}] 输出顺序：{"交换（out1 是下分位）" if SW[kd] else "正常（out0 是下分位）"}')
        if not a.skip_check:
            if not identity_check(root, M[kd], SW[kd], log, kind=kd):
                log(f'[STOP] {kd} 模型一致性检查未通过：模型文件可能被换过，或参照文件不对。把 {brief} 发回来，先不要往下跑')
                _write(out, B, brief)
                sys.exit(1)
            log(f'  [检查·{kd}] passed: the models reproduce the stored scores')
    if a.check_only:
        _write(out, B, brief)
        return

    t_all = time.time()
    for set_name in a.sets:
        kd = SET_MODELS[set_name]
        models, swap = M[kd], SW[kd]
        t0s = None
        if set_name == 'TEX_cS':
            stars, A, E = tex_cs(root)
            asr = A
            log(f'\n== TEXBAT cleanStatic：对齐 [{A}, {E})（{(E - A) / 1000:.1f} s），星 {[p for p, _ in stars]}')
        elif set_name == 'TEX_cD':
            seg = []
            stars, A, E, t0s = tex_cd(root, seg)
            asr = A
            log(f'\n== TEXBAT cleanDynamic：对齐 [{A}, {E})（{(E - A) / 1000:.1f} s，录音时间 {t0s:.3f} s 起），'
                f'星 {[p for p, _ in stars]}，动态模型')
            for ch, prn, (s0, s1), n in seg:
                log(f'     ch{ch} PRN{prn:>3} 最长连续段 [{s0}, {s1}) / 共 {n} 条')
        else:
            stars = oak_cd(a.oak_dir)
            asr = 0
            log(f'\n== OAKBAT cleanDynamic：星 {[p for p, _ in stars]}，长度 {[len(x) for _, x in stars]}')
        mu, sig = pooled_mu_sigma([x for _, x in stars])
        log(f'  场景内合并 μ/σ（前 80 s）：prompt μ={mu[0, PROMPT]:.4g} σ={sig[0, PROMPT]:.4g}')
        for prn, taps in stars:
            t0 = time.time()
            log(f'  -- {set_name}_prn{prn}：{len(taps) / 1000:.1f} s')
            per, s_ae, s_tcn = score_series(models, taps, mu, sig, swap=swap)
            N = len(taps) - W
            tw = taps[W:W + N]
            cn0, pp, nv = cn0_block(tw)
            f = os.path.join(sdir, f'long_{set_name}_prn{prn}.npz')
            np.savez_compressed(
                f, per=(per if per is not None else np.zeros((N, 0), np.float32)),
                S_ae=(s_ae if s_ae is not None else np.zeros(0, np.float32)),
                S_tcn=(s_tcn if s_tcn is not None else np.zeros(0, np.float32)),
                taps_raw=tw.astype(np.float32), cn0=cn0, prompt_pow=pp, noise_var=nv,
                labels=np.zeros(N, np.int8), times=np.arange(W, W + N), scenario=set_name, prn=prn,
                spoof_start=N + W, align_start_ms=asr, mu=mu, sig=sig, qr_swap=swap,
                model_set=kd, t0_rec_s=(np.nan if t0s is None else t0s),
                models=np.array([f'{k}={v}' for k, v in mpaths[kd].items()]))
            cal = slice(0, MU_LEN - W)
            msg = '  '.join(f'{k} 均值 {v[cal].mean():.4f}' for k, v in
                            (('QR', None if per is None else per.sum(1)), ('AE', s_ae), ('TCN', s_tcn))
                            if v is not None and len(v))
            log(f'     存 {os.path.basename(f)}  N={N}  {msg}  用时 {(time.time() - t0) / 60:.1f} min')
    log(f'\n[s3a] 完成，用时 {(time.time() - t_all) / 60:.1f} min。把 {brief} 和 cd_offset.json 发回来，再写 s3b 的 TEX_cD 部分')
    _write(out, B, brief)


def _write(out, B, name='s3a_brief.txt'):
    with open(os.path.join(out, name), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(B) + '\n')


if __name__ == '__main__':
    main()
