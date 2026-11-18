"""Hand-crafted metrics on the pipeline of far_mc_beta_neweval_v2.py.

Computes the carrier-to-noise proxy (prompt power over the noise variance of the two
outer taps) and the WSCM from the stored correlator taps, folds each about the median
of the calibration segment, and evaluates them with the same calibration and
per-instant detection rate as the learned scores. Imported by rev_sqm.py and the
scripts of r1_revision_SEN/code.
Run from the repository root (the working folder that also holds the data folders); see --help.
"""
import sys
import os
import argparse
import csv
import importlib.util
import numpy as np


# ---------------------------------------------------------------------------
# Definitions shared with far_mc_beta_neweval_v2.py (loaded from the same folder)
# ---------------------------------------------------------------------------
FM2_PATH = os.environ.get(
    'FAR_MC_V2_PATH',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'far_mc_beta_neweval_v2.py'))
if not os.path.exists(FM2_PATH):
    sys.exit(f"[ERROR] 找不到 far_mc_beta_neweval_v2.py: {FM2_PATH}\n"
             f"  把它和本脚本放同目录, 或设环境变量 FAR_MC_V2_PATH 指向它。")
_spec = importlib.util.spec_from_file_location('far_mc_beta_neweval_v2', FM2_PATH)
fm2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fm2)


AUTH                = fm2.AUTH
ALIGN_START_MS      = fm2.ALIGN_START_MS
ABS_CALIB_SCENARIOS = fm2.ABS_CALIB_SCENARIOS
CALIB_RAW_S         = fm2.CALIB_RAW_S
K0                  = fm2.K0
DELTA_GUARD_S       = fm2.DELTA_GUARD_S
pick_block_len      = fm2.pick_block_len
calibrate_h         = fm2.calibrate_h
cusum_alarm_times   = fm2.cusum_alarm_times
select_files_v2     = fm2.select_files_v2


star_rng            = fm2.star_rng
star_name_from_path = fm2.star_name_from_path
calib_gamma         = fm2.calib_gamma
detrate_series      = fm2.detrate_series
consecutive_alarm_idx = fm2.consecutive_alarm_idx
post_onset_detrate  = fm2.post_onset_detrate

DEFAULT_ALPHAS      = fm2.DEFAULT_ALPHAS
alpha_tag           = fm2.alpha_tag
eval_alphas         = fm2.eval_alphas
detrate_csv_header  = fm2.detrate_csv_header
detrate_rows_from_payload = fm2.detrate_rows_from_payload
alpha_csv_cols      = fm2.alpha_csv_cols
alpha_csv_vals      = fm2.alpha_csv_vals
summarize_v2        = fm2.summarize_v2


# ---------------------------------------------------------------------------
# Hand-crafted metrics, Eqs. (5)-(6)
# ---------------------------------------------------------------------------
# Tap 10 is the prompt; taps 8 and 12 are the early and late correlators of the WSCM.
PROMPT_IDX    = 10
DEFAULT_E_IDX = 8
DEFAULT_L_IDX = 12
WSCM_K        = 5
WSCM_ALPHA    = np.array([0.438, 0.219, 0.145, 0.110, 0.088])
EPS = 1e-10


def sqm_ratio(taps, e_idx, l_idx):
    return (taps[:, e_idx] + taps[:, l_idx]) / (2.0 * taps[:, PROMPT_IDX] + EPS)


def sqm_delta(taps, e_idx, l_idx):
    return (taps[:, e_idx] - taps[:, l_idx]) / (2.0 * taps[:, PROMPT_IDX] + EPS)


def sqm_wscm_EL(taps):
    t = taps.astype(np.float64)
    out = np.zeros(len(t), dtype=np.float64)
    for k in range(1, WSCM_K + 1):
        left  = t[:, PROMPT_IDX - k]
        right = t[:, PROMPT_IDX + k]
        out += WSCM_ALPHA[k - 1] * (left ** 2 - right ** 2)
    return out


# ---------------------------------------------------------------------------
# Taps, metric series and folding about the calibration median
# ---------------------------------------------------------------------------
def load_star_v2(f):
    d = np.load(f, allow_pickle=True)
    if 'taps_raw' not in d.files:
        raise KeyError(f"{os.path.basename(f)} 缺 taps_raw — 需用加了 I/Q dump 的 npz。")
    taps = d['taps_raw'].astype(np.float64)
    lab  = d['labels']
    scen = str(d['scenario']); prn = int(d['prn']); ss = int(d['spoof_start'])

    if 'cn0' in d.files:
        cn0 = d['cn0'].astype(np.float64)
    elif 'noise_var' in d.files:
        nv = float(d['noise_var'])
        cn0 = taps[:, PROMPT_IDX] ** 2 / (nv + EPS)
    else:
        cn0 = None
    return taps, cn0, lab, scen, prn, ss


def get_metric(method, taps, cn0, e_idx, l_idx):
    """Raw two-sided metric series; folded later as |metric - median(calibration)|."""
    if method == 'cn0':
        if cn0 is None:
            raise KeyError("cn0 不可得(npz 无 cn0 也无 noise_var)。")
        return cn0
    if method == 'ratio':
        return sqm_ratio(taps, e_idx, l_idx)
    if method == 'delta':
        return sqm_delta(taps, e_idx, l_idx)
    if method == 'wscm':
        return sqm_wscm_EL(taps)
    raise ValueError(f'未知方法 {method}')


# ---------------------------------------------------------------------------
# Stand-alone run per metric
# ---------------------------------------------------------------------------
def run_method_v2(method, beta, m_ms, n_boot, seg_s, dyn_mode, seed, npz_dir, alphas,
                  e_idx, l_idx, stars_filter=None, exclude_ds5=False, exclude_ds2=False,
                  verbose=False, calib_raw=CALIB_RAW_S, out_prefix=None):


    m = m_ms
    seg_len = int(seg_s * 1000)
    alphas = list(alphas)
    a_ref = alphas[-1]
    files = select_files_v2(npz_dir, dyn_mode)

    if stars_filter:
        files = [f for f in files if any(s in os.path.basename(f) for s in stars_filter)]
    else:
        if exclude_ds5:
            files = [f for f in files if 'ds5_r2' not in os.path.basename(f)]
        if exclude_ds2:
            files = [f for f in files if 'ds2_' not in os.path.basename(f)]

    if not files:
        print(f'[ERROR] {npz_dir} 下没找到 pertap_*.npz(含 taps_raw)。')
        return []

    delta_idx = int(round(DELTA_GUARD_S * 1000))
    c_lo_s, c_hi_s = calib_raw
    print(f'\n{"=" * 100}')
    print(f'[{method.upper()} v2] beta={beta:.0e} m={m}ms nboot={n_boot} seg={seg_s}s '
          f'dyn={dyn_mode} seed={seed}  ({len(files)} stars)  δguard={DELTA_GUARD_S}s')
    print(f'[{method.upper()} v2] α 档位({len(alphas)}): '
          f'{", ".join(alpha_tag(a) for a in alphas)}')
    print(f'[{method.upper()} v2] rng: 每星独立(seed + crc32(星名)) '
          f'—— 与 LSTM 批同一实现, 可干净并排 [E4]')
    if method in ('ratio', 'delta'):
        print(f'[{method.upper()} v2] E/L tap = idx{e_idx}/{l_idx} '
              f'(Δτ={(e_idx - PROMPT_IDX) * 0.2:+.1f}/{(l_idx - PROMPT_IDX) * 0.2:+.1f} chip)')
    print(f'{"star":16s} {"L":>4s} {"h":>8s} | {"cls":4s} {"evLat":>8s} | {"#FA":>4s} | '
          f'{"MaoAlm@" + alpha_tag(a_ref):>13s} {"pDR":>6s} {"pDR5":>6s}')
    print('-' * 100)

    rows, detrate_rows = [], []
    for f in files:
        taps, cn0, lab, scen, prn, ss = load_star_v2(f)
        star = star_name_from_path(f)
        asr_ms = ALIGN_START_MS.get(scen, 0)
        onset = AUTH[scen]
        oi = int(round(onset * 1000 - 100 - asr_ms))


        rng = star_rng(seed, star)

        metric = get_metric(method, taps, cn0, e_idx, l_idx)


        if scen in ABS_CALIB_SCENARIOS:
            lo_idx = max(0, int(round(c_lo_s * 1000 - 100 - asr_ms)))
            hi_idx = min(len(metric), int(round(c_hi_s * 1000 - 100 - asr_ms)))
            cmask = np.zeros(len(metric), dtype=bool)
            if hi_idx > lo_idx:
                cmask[lo_idx:hi_idx] = True
            if cmask.sum() < 10:
                cmask = (lab == 0)
        else:
            cmask = (lab == 0)


        med = float(np.median(metric[cmask]))
        S = np.abs(metric - med)

        Sc = S[cmask]
        mu, sd = Sc.mean(), Sc.std()
        if sd < 1e-12:
            sd = 1e-12
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
        p = f'{out_prefix}_{method}_detrate.csv'
        with open(p, 'w', newline='', encoding='utf-8-sig') as fh:
            w = csv.writer(fh)
            w.writerow(detrate_csv_header())
            w.writerows(detrate_rows)
        print(f'[saved] {p}  ({len(detrate_rows)} 行)')
    return rows


def summarize_method(method, rows, beta, alphas, out_path=None):
    """Summary with the columns of the learned scores (far_mc_beta_neweval_v2.summarize_v2)."""
    summarize_v2(rows, beta, alphas, out_path, tag=f'{method.upper()} v2')


def load_lstm_csv(path):
    out = {}
    with open(path, encoding='utf-8-sig') as fh:
        for row in csv.DictReader(fh):
            out[row['star']] = row
    return out


def _fmt_cell(cls, lat):
    if cls in (None, '', 'MISS'):
        return 'MISS' if cls == 'MISS' else 'NA'
    if cls == 'H_NOT_FOUND':
        return 'NF'
    if lat in (None, ''):
        return str(cls)
    try:
        return f'{float(lat):+.2f}'
    except (TypeError, ValueError):
        return str(cls)


def report_compare(method_rows, lstm_map, beta, out_path=None):
    methods = [m for m in ('ratio', 'delta', 'wscm', 'cn0') if m in method_rows]
    base = sorted(method_rows[methods[0]], key=lambda x: (x['scen'], x['prn']))
    by_star = {m: {r['star']: r for r in method_rows[m]} for m in methods}

    if out_path:
        hdr = ['star', 'scenario', 'prn']
        if lstm_map is not None:
            hdr += ['LSTM_class', 'LSTM_latency_s', 'LSTM_nFA']
        for m in methods:
            hdr += [f'{m}_class', f'{m}_latency_s', f'{m}_nFA']
        with open(out_path, 'w', newline='', encoding='utf-8-sig') as fh:
            w = csv.writer(fh); w.writerow(hdr)
            for b in base:
                star = b['star']; row = [star, b['scen'], b['prn']]
                if lstm_map is not None:
                    lr = lstm_map.get(star)
                    row += ([lr.get('class', ''), lr.get('latency_s', ''),
                             lr.get('n_pre_onset_FA', '')] if lr else ['', '', ''])
                for m in methods:
                    r = by_star[m].get(star)
                    row += ([r['cls'], f'{r["lat"]:.4f}' if r['lat'] is not None else '',
                             r['n_fa'] if r['n_fa'] is not None else ''] if r else ['', '', ''])
                w.writerow(row)
        print(f'[saved] {out_path}')

    print('\n' + '=' * 92)
    print(f'对照 @ β={beta:.0e}  事件级延迟(s, MISS=漏检, NF=h未命中)')
    print('=' * 92)
    cols = (['LSTM'] if lstm_map is not None else []) + [m.capitalize() for m in methods]
    print(f'{"star":16s} ' + ' '.join(f'{c:>10s}' for c in cols))
    print('-' * 92)
    for b in base:
        star = b['star']; cells = []
        if lstm_map is not None:
            lr = lstm_map.get(star)
            cells.append(_fmt_cell(lr.get('class'), lr.get('latency_s')) if lr else 'NA')
        for m in methods:
            r = by_star[m].get(star)
            cells.append(_fmt_cell(r['cls'], r['lat']) if r else 'NA')
        print(f'{star:16s} ' + ' '.join(f'{c:>10s}' for c in cells))


# ---------------------------------------------------------------------------
# Reference values for a side-by-side table (not used in the paper)
# ---------------------------------------------------------------------------
MAO_PUBLISHED = {
    'ds2': {'Ratio': (23592, 31.81), 'Delta': (42136, 30.13), 'ELP': (43656, 30.28),
            'Q-energy': (19457, 33.70), 'WSCM': (1, 99.99), 'LSTM': (5, 100.00)},
    'ds3': {'Ratio': (4032, 39.25), 'Delta': (5028, 37.69), 'ELP': (5183, 40.54),
            'Q-energy': (4325, 42.91), 'WSCM': (4, 50.23), 'LSTM': (5, 99.96)},
    'ds4': {'Ratio': (2450, 40.52), 'Delta': (142805, 50.70), 'ELP': (130257, 68.68),
            'Q-energy': (129836, 68.81), 'WSCM': (2568, 65.42), 'LSTM': (2, 98.80)},
    'ds5': {'Ratio': (2450, 27.00), 'Delta': (142805, 25.50), 'ELP': (130257, 25.52),
            'Q-energy': (129836, 26.04), 'WSCM': (2, 99.99), 'LSTM': (5, 99.87)},
    'ds6': {'Ratio': (82375, 30.92), 'Delta': (66752, 31.28), 'ELP': (59258, 33.79),
            'Q-energy': (60523, 33.54), 'WSCM': (3, 66.16), 'LSTM': (5, 99.62)},
}


REP_PRN = {'ds2': 3, 'ds3': 23, 'ds4_r2': 13, 'ds5_r2': 15, 'ds6_r2': 18,
           'ds7': 3, 'ds8': 3}


GS_PUBLISHED_DET = {
    'ds3': {'QR-LSTM': (100.0, 99.7), 'Ratio': (49.2, 23.8), 'WSCM': (67.6, 27.5)},
    'ds4': {'QR-LSTM': (97.1, 83.7),  'Ratio': (80.5, 67.9), 'WSCM': (37.3, 23.3)},
    'ds5': {'QR-LSTM': (100.0, 99.8), 'Ratio': (4.7, 0.0),   'WSCM': (100.0, 100.0)},
    'ds6': {'QR-LSTM': (99.8, 97.3),  'Ratio': (59.9, 20.9), 'WSCM': (96.2, 84.2)},
    'ds7': {'QR-LSTM': (87.8, 73.0),  'Ratio': (72.1, 56.2), 'WSCM': (3.9, 0.0)},
    'ds8': {'QR-LSTM': (91.8, 74.0),  'Ratio': (72.9, 56.4), 'WSCM': (6.3, 2.1)},
}

GS_PUBLISHED_LAT_S = {'ds3': 0.089, 'ds4': 0.179, 'ds5': 0.077,
                      'ds6': 0.004, 'ds7': 45.180, 'ds8': 19.386}

GS_ALPHA_LOOSE, GS_ALPHA_STRICT = 0.10, 1e-4


def _rep_row(rows, scen):
    """Row of the representative satellite of a scenario, or None."""
    rp = REP_PRN.get(scen)
    for r in rows:
        if r['scen'] == scen and r['prn'] == rp:
            return r
    return None


def _pick_alpha(alphas, target):
    """Alpha closest to target on a log scale."""
    if not alphas:
        return None
    return min(alphas, key=lambda a: abs(np.log10(a) - np.log10(target)))


def report_parity(method_rows, lstm_rows_by_star, beta, alphas, out_path=None):
    """Side-by-side table on representative satellites; different definitions, so differences are expected.
    """
    scen_order = ['ds2', 'ds3', 'ds4_r2', 'ds5_r2', 'ds6_r2', 'ds7', 'ds8']
    our_methods = [m for m in ('ratio', 'delta', 'wscm', 'cn0') if m in method_rows]
    a_loose  = _pick_alpha(alphas, GS_ALPHA_LOOSE)
    a_strict = _pick_alpha(alphas, 1e-3)

    def our_cell(scen, method, al):
        r = _rep_row(method_rows[method], scen)
        if r is None or al not in r.get('per_alpha', {}):
            return ('', '')
        d = r['per_alpha'][al]
        alarm_ms = f'{d["mao_alarm"] * 1000:.0f}' if d['mao_alarm'] is not None else 'none'
        dr = f'{d["post_dr"] * 100:.2f}' if d['post_dr'] == d['post_dr'] else ''
        return (alarm_ms, dr)

    def lstm_cell(scen, al):
        """Value for a learned score from its csv."""
        rp = REP_PRN.get(scen)
        t = alpha_tag(al)
        for star, row in lstm_rows_by_star.items():
            if row.get('scenario') == scen and str(row.get('prn')) == str(rp):
                ma  = row.get(f'mao_alarm_s@{t}', '')
                pdr = row.get(f'post_dr@{t}', '')
                alarm_ms = f'{float(ma) * 1000:.0f}' if ma not in ('', None) else 'none'
                drp = f'{float(pdr) * 100:.2f}' if pdr not in ('', None) else ''
                return (alarm_ms, drp)
        return ('', '')

    def ev_lat(scen):
        """Event-level delay of the representative satellite."""
        for star, row in lstm_rows_by_star.items():
            if row.get('scenario') == scen and str(row.get('prn')) == str(REP_PRN.get(scen)):
                v = row.get('latency_s', '')
                if v in ('', None):
                    return row.get('class', '')
                try:
                    return f'{float(v):+.3f}'
                except ValueError:
                    return str(v)
        return ''

    print('\n' + '=' * 104)
    print('三方 parity — 代表星(GS 名单: ds3=23 ds4=13 ds5=15 ds6=18 ds7=3 ds8=3, ds2=3)')
    print('  ⛔ 三套口径不同, 数字本就不该一致 —— 本表用于【主动摆明差异】:')
    print('     SP(ours) = event-level β-CUSUM(h 由 block-bootstrap 标到 β), 逐样本列用同 α 的 γ')
    print('     GS       = per-sample γ(clean p95 族), K=1 算 Det / K=5 算 alarm, CAL:TEST 50:50')
    print('     Mao      = 监督 LSTM, 连续 5 次报警, 【无标定 FAR】, 且不测 ds7/ds8')
    print('=' * 104)


    for al, label, gs_slot in ((a_loose, '宽松档', 0), (a_strict, '严格档', 1)):
        if al is None:
            continue
        gs_note = ('可直接对照 GS αnom=10%' if gs_slot == 0
                   else f'⚠ GS 那档是 0.01%, 我们是 {alpha_tag(al)} '
                        f'—— 不是同一个工作点, 只作趋势参照')
        print(f'\n--- 检出率 % @ {label} α={alpha_tag(al)}  ({gs_note}) ---')
        cols = ['LSTM(SP)'] + [m.capitalize() + '(SP)' for m in our_methods] + \
               ['| QR-LSTM(GS)', 'Ratio(GS)', 'WSCM(GS)']
        print(f'{"scen":8s} ' + ' '.join(f'{c:>14s}' for c in cols))
        print('-' * 104)
        for scen in scen_order:
            cells = [lstm_cell(scen, al)[1] or '—']
            for m in our_methods:
                cells.append(our_cell(scen, m, al)[1] or '—')
            gs = GS_PUBLISHED_DET.get(scen.replace('_r2', ''))
            if gs:
                cells += [f'| {gs["QR-LSTM"][gs_slot]:.1f}',
                          f'{gs["Ratio"][gs_slot]:.1f}', f'{gs["WSCM"][gs_slot]:.1f}']
            else:
                cells += ['| —', '—', '—']
            print(f'{scen:8s} ' + ' '.join(f'{c:>14s}' for c in cells))


    print(f'\n--- 报警延迟 (s) — ★ 同编码器同分数序列, 只换判决层 ---')
    print(f'{"scen":8s} {"SP事件级":>12s} {"GS(K=5)":>12s} {"Mao(LSTM)":>12s} {"SP−GS":>12s}')
    print('-' * 104)
    for scen in scen_order:
        sp = ev_lat(scen)
        gs = GS_PUBLISHED_LAT_S.get(scen.replace('_r2', ''))
        mao = MAO_PUBLISHED.get(scen.replace('_r2', ''))
        mao_s = f'{mao["LSTM"][0] / 1000.0:+.3f}' if mao else '—'
        gs_s = f'{gs:+.3f}' if gs is not None else '—'
        diff = ''
        try:
            if gs is not None and sp not in ('', 'MISS', None):
                diff = f'{float(sp) - gs:+.3f}'
        except ValueError:
            diff = ''
        print(f'{scen:8s} {str(sp) or "—":>12s} {gs_s:>12s} {mao_s:>12s} {diff:>12s}')
    print('  ★ 负的 SP−GS = SP 判决层更快。ds8 尤其值得看: 同一份 S(t), CUSUM 会累积')
    print('    那些凑不齐 K=5 的孤立越界点, 而 K=5 会把它们全部丢掉。')

    # ---- (C) CSV ----
    if out_path:
        recs = []
        for scen in scen_order:
            rec = {'scen': scen, 'rep_prn': REP_PRN.get(scen, ''),
                   'SP_event_latency_s': ev_lat(scen)}
            for al, tagname in ((a_loose, 'loose'), (a_strict, 'strict')):
                if al is None:
                    continue
                rec[f'alpha_{tagname}'] = alpha_tag(al)
                lc = lstm_cell(scen, al)
                rec[f'SP_LSTM_dr%_{tagname}'] = lc[1]
                rec[f'SP_LSTM_maoalarm_ms_{tagname}'] = lc[0]
                for m in our_methods:
                    c = our_cell(scen, m, al)
                    rec[f'SP_{m}_dr%_{tagname}'] = c[1]
                    rec[f'SP_{m}_maoalarm_ms_{tagname}'] = c[0]
            gs = GS_PUBLISHED_DET.get(scen.replace('_r2', ''))
            if gs:
                rec['GS_QRLSTM_dr%_10pct']  = gs['QR-LSTM'][0]
                rec['GS_Ratio_dr%_10pct']   = gs['Ratio'][0]
                rec['GS_WSCM_dr%_10pct']    = gs['WSCM'][0]
                rec['GS_QRLSTM_dr%_001pct'] = gs['QR-LSTM'][1]
                rec['GS_Ratio_dr%_001pct']  = gs['Ratio'][1]
                rec['GS_WSCM_dr%_001pct']   = gs['WSCM'][1]
            gl = GS_PUBLISHED_LAT_S.get(scen.replace('_r2', ''))
            if gl is not None:
                rec['GS_latency_s_K5'] = gl
            mao = MAO_PUBLISHED.get(scen.replace('_r2', ''))
            if mao:
                rec['Mao_LSTM_alarm_ms'] = mao['LSTM'][0]
                rec['Mao_LSTM_dr%']      = mao['LSTM'][1]
                rec['Mao_WSCM_alarm_ms'] = mao['WSCM'][0]
                rec['Mao_WSCM_dr%']      = mao['WSCM'][1]
            recs.append(rec)
        keys = []
        for r in recs:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(out_path, 'w', newline='', encoding='utf-8-sig') as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for rec in recs:
                w.writerow({k: rec.get(k, '') for k in keys})
        print(f'\n[saved] {out_path}')


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--beta', type=float, default=1e-5)
    ap.add_argument('--m', type=int, default=1000)
    ap.add_argument('--nboot', type=int, default=1000)
    ap.add_argument('--seg', type=float, default=80.0)
    ap.add_argument('--ds6', dest='dyn', choices=['static', 'dynamic'], default='dynamic',
                    help='动态场景(ds5/ds6)取 _dyn 还是静态')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--dir', dest='npz_dir', default='.',
                    help=r'npz 目录(递归); 本机指 J:\Database\TEXBAT\processed_Mao')
    ap.add_argument('--stars', nargs='+', default=None)
    ap.add_argument('--exclude-ds5', action='store_true', help='排除 ds5_r2(默认纳入)')
    ap.add_argument('--exclude-ds2', action='store_true',
                    help='[P4] 排除 ds2(默认【不】排除, ds2 照跑)')
    ap.add_argument('--methods', nargs='+', default=['ratio', 'delta', 'wscm', 'cn0'],
                    choices=['ratio', 'delta', 'wscm', 'cn0'])
    ap.add_argument('--e-idx', type=int, default=DEFAULT_E_IDX)
    ap.add_argument('--l-idx', type=int, default=DEFAULT_L_IDX)
    ap.add_argument('--alphas', type=float, nargs='+', default=DEFAULT_ALPHAS,
                    help='[P2] 逐样本检出率虚警水平, 可多档 '
                         '(默认 0.1 0.05 0.01 1e-3 1e-4)')
    ap.add_argument('--out', default=None, help='输出前缀')
    ap.add_argument('--lstm-csv', default=None,
                    help='far_mc_v2 的 LSTM 事件级 CSV(给了就并对照表 + parity)')
    ap.add_argument('--verbose', action='store_true')
    a = ap.parse_args()

    alphas = sorted(set(a.alphas), reverse=True)
    method_rows = {}
    for method in a.methods:
        rows = run_method_v2(method, a.beta, a.m, a.nboot, a.seg, a.dyn, a.seed,
                             a.npz_dir, alphas, a.e_idx, a.l_idx, a.stars,
                             a.exclude_ds5, a.exclude_ds2, a.verbose, out_prefix=a.out)
        if not rows:
            continue
        summarize_method(method, rows, a.beta, alphas,
                         (f'{a.out}_{method}.csv' if a.out else None))
        method_rows[method] = rows

    if method_rows:
        lstm_map = None
        if a.lstm_csv and os.path.exists(a.lstm_csv):
            lstm_map = load_lstm_csv(a.lstm_csv)
        elif a.lstm_csv:
            print(f'\n[warn] --lstm-csv 不存在: {a.lstm_csv}; 只出 SQM/C-N0 对照')
        report_compare(method_rows, lstm_map, a.beta,
                       (f'{a.out}_compare.csv' if a.out else None))

        report_parity(method_rows, lstm_map if lstm_map else {}, a.beta, alphas,
                      (f'{a.out}_parity.csv' if a.out else None))
