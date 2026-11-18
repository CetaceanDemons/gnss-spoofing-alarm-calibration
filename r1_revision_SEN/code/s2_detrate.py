# -*- coding: utf-8 -*-
"""Per-instant detection rate after onset for the five scores (Table 8, Figure 11).

gamma is the (1-alpha) quantile of the score on the calibration segment, alpha in
{0.1, 0.05, 0.01}; the rate is the share of epochs from onset to the end of the
recording with S > gamma. Also writes the pairs of detection rate and CUSUM delay.
Run from the repository root (the working folder that also holds the data folders); see --help.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEF_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))   # J:\Database\TEXBAT\processed_Mao

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
ALPHAS = (0.10, 0.05, 0.01)
SCORES = (('qr', 'QR-LSTM'), ('ae', 'LSTM-AE'), ('tcn', 'TCN'), ('cn0', 'C/N$_0$ proxy'), ('wscm', 'WSCM'))
SC_CLASS = {'ds3': 'static', 'ds4_r2': 'static', 'ds5_r2': 'dynamic', 'ds6_r2': 'dynamic',
            'ds7': 'complex', 'ds8': 'complex'}
EXPECT_AUTH = {'ds3': 119.6, 'ds4_r2': 113.8, 'ds5_r2': 102.1, 'ds6_r2': 104.4, 'ds7': 110.0, 'ds8': 110.0}
# ds3 onset after the correction; the other onsets are those of Table 1.
ONSET_NEW = dict(EXPECT_AUTH, ds3=118.9)
OLD = {'qr': ('results_v2b_1e-5', 'far_mc_v2_1e-5_lstm.csv'),
       'ae': ('results_prerun_nb1000', 'pre_ae_nb1000_lstm.csv'),
       'cn0': ('results_v2b_1e-5', 'sqm_v2_1e-5_cn0.csv'),
       'wscm': ('results_v2b_1e-5', 'sqm_v2_1e-5_wscm.csv')}
# Table 8 of the first submission, for comparison.
OLD_T8 = {'qr': (91.89, 90.16, 87.01), 'ae': (91.15, 89.05, 84.84), 'cn0': (67.03, 61.62, 52.42),
          'wscm': (49.18, 42.99, 33.73)}
TOL = 1e-4


class Tee:
    def __init__(self, path):
        self.f = open(path, 'w', encoding='utf-8')

    def __call__(self, *a):
        s = ' '.join(str(x) for x in a)
        print(s, flush=True)
        self.f.write(s + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def setup_paths(root):
    code = os.path.join(root, 're_ii', 'code')
    for p in (code, root):
        if p not in sys.path:
            sys.path.insert(0, p)


def ns(s):
    s = str(s).strip()
    if s.startswith('pertap_'):
        s = s[7:]
    return s[:-4] if s.endswith('_dyn') else s


def fl(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


def read_csv(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8-sig', newline='') as fh:
        return list(csv.DictReader(fh))


def write_csv(path, rows, fields=None):
    if not rows:
        return
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: (f'{v:.6g}' if isinstance(v, float) else v) for k, v in r.items()})


def rankavg(x):
    x = np.asarray(x, float)
    o = np.argsort(x, kind='mergesort')
    r = np.empty(len(x))
    xs = x[o]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return r


def spearman(x, y):
    """Spearman correlation (Pearson correlation of average ranks)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 4 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float('nan'), len(x)
    return float(np.corrcoef(rankavg(x), rankavg(y))[0, 1]), len(x)


# ---------------------------------------------------------------------------
# Scores, read with the loaders of the sequential rules
# ---------------------------------------------------------------------------
def load_score(k, path, gidx, fc, v2, sq, rs, e_idx, l_idx):
    """Returns S, S_old (None for the TCN), labels, scenario and PRN."""
    if k == 'qr':
        S, lab, scen, prn = fc.load_star(path)
        S = np.asarray(S, dtype=np.float64)
        return S, S, np.asarray(lab), scen, prn
    if k in ('ae', 'tcn'):
        S, lab, scen, prn = v2.load_S_scores(path)
        S = np.asarray(S, dtype=np.float64)
        return S, (S if k == 'ae' else None), np.asarray(lab), scen, prn

    taps, cn0, lab, scen, prn, _ = sq.load_star_v2(path)
    metric = np.asarray(sq.get_metric(k, taps, cn0, e_idx, l_idx), dtype=np.float64).ravel()
    lab = np.asarray(lab)
    if len(lab) != len(metric):
        raise ValueError(f'{os.path.basename(path)}: labels {len(lab)} 与 {k} 长度 {len(metric)} 不一致')
    bad = ~np.isfinite(metric)
    cm = fc.segments(metric, lab, scen, gidx)['cmask']
    med = float(np.median(metric[cm & ~bad]))
    if bad.any():
        metric = np.where(bad, med, metric)
    S = np.abs(metric - med)
    m1 = rs.v1_window(lab, scen, len(metric))
    S_old = np.abs(metric - float(np.median(metric[m1])))
    return S, S_old, lab, scen, prn


# ---------------------------------------------------------------------------
# Per-instant detection rate after onset (Table 8)
# ---------------------------------------------------------------------------
def detrates(S, Sc, oi, v2):
    out = {}
    for al in ALPHAS:
        g = v2.calib_gamma(Sc, al)
        dec = (S > g).astype(np.float64)
        out[al] = dict(gamma=g, cal_exc=float(np.mean(Sc > g)), post_dr=v2.post_onset_detrate(dec, oi))
    return out


# ---------------------------------------------------------------------------
# Table 8 and the pairs of Figure 11
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=DEF_ROOT, help=r'代码根目录（含 far_mc_beta_neweval.py 与 re_ii\code）')
    ap.add_argument('--dir', dest='data_dir', default=None, help='数据根目录，默认与 --root 相同')
    ap.add_argument('--s1-dir', default=None, help=r'S1 输出目录，默认 <root>\r1_revision_SEN\s1')
    ap.add_argument('--out-dir', default=None, help=r'默认 <root>\r1_revision_SEN\s2')
    ap.add_argument('--scores', nargs='+', default=[k for k, _ in SCORES], choices=[k for k, _ in SCORES])
    ap.add_argument('--stars', nargs='+', default=None)
    ap.add_argument('--guard-s', dest='guard_s', type=float, default=0.1)
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    D = os.path.abspath(a.data_dir or root)
    s1dir = os.path.abspath(a.s1_dir or os.path.join(root, 'r1_revision_SEN', 's1'))
    out = os.path.abspath(a.out_dir or os.path.join(root, 'r1_revision_SEN', 's2'))
    os.makedirs(out, exist_ok=True)
    log = Tee(os.path.join(out, 's2_log.txt'))
    t0 = time.time()

    setup_paths(root)
    import far_calib_v32 as fc
    import far_mc_beta_neweval_v2 as v2
    import sqm_beta_neweval_v2 as sq
    import rev_main as rm
    import rev_sqm as rs

    log(f'[version] s2_detrate.py v1.0（2026-11-17）  root={root}  data={D}')
    for name, dct in (('far_calib_v32.AUTH', fc.AUTH), ('v2.AUTH', v2.AUTH), ('sqm.AUTH', sq.AUTH)):
        for sc, v in EXPECT_AUTH.items():
            if abs(dct[sc] - v) > 1e-9:
                raise SystemExit(f'[STOP] {name}[{sc}] = {dct[sc]}，预期 {v}（本脚本不改 AUTH，旧口径要用原值）')
    log(f'[cfg] 新 onset = {ONSET_NEW}；ACTIVATION_RAW = {fc.ACTIVATION_RAW}；guard = {a.guard_s} s；α = {ALPHAS}')
    gidx = int(round(a.guard_s * 1000))
    e_idx, l_idx = sq.DEFAULT_E_IDX, sq.DEFAULT_L_IDX

    rows, by = [], {}
    for k in a.scores:
        if k in ('qr', 'cn0', 'wscm'):
            items = rm.qr_items(D, a.stars)
        else:
            sub, s1, s2 = rm.DEFAULT_DUAL[k]
            items = rm.dual_items(os.path.join(D, sub, s1), os.path.join(D, sub, s2), a.stars, log)
        if not a.stars and len(items) != 21:
            log(f'[WARN] {k}: 找到 {len(items)} 颗星，预期 21')
        oldp = os.path.join(D, *OLD[k]) if k in OLD else None
        old = read_csv(oldp) if oldp else None
        oldmap = {ns(r['star']): r for r in old} if old else {}
        log(f'\n== {k}: {len(items)} 颗星；旧口径参照 ' +
            (oldp if old is not None else ('无（TCN 没有旧结果）' if k == 'tcn' else f'找不到 {oldp}')))
        for base, path, _ in items:
            S, S_old, lab, scen, prn = load_score(k, path, gidx, fc, v2, sq, rs, e_idx, l_idx)
            seg = fc.segments(S, lab, scen, gidx)
            cm, asr, n = seg['cmask'], seg['asr'], len(S)
            oi = min(fc.idx_of(ONSET_NEW[scen], asr), n)
            new = detrates(S, S[cm], oi, v2)
            r = dict(score=k, star=base, scenario=scen, group=SC_CLASS[scen], prn=prn, n=n,
                     cal_raw_start_s=seg['raw'](seg['c0']), cal_raw_end_s=seg['raw'](seg['c1']),
                     n_cal=int(cm.sum()), onset_new_s=ONSET_NEW[scen], post_s=(n - oi) / 1000.0)
            for al in ALPHAS:
                t = v2.alpha_tag(al)
                r[f'gamma@{t}'] = new[al]['gamma']
                r[f'cal_exc@{t}'] = new[al]['cal_exc']
                r[f'post_dr@{t}'] = new[al]['post_dr']

            r['parity'] = 'no_ref'
            if S_old is not None and k in OLD:
                cm_old = rs.v1_window(lab, scen, n)
                oi_old = int(round(EXPECT_AUTH[scen] * 1000 - 100 - asr))
                oldv = detrates(S_old, S_old[cm_old], oi_old, v2)
                ref = oldmap.get(base)
                ok = ref is not None
                for al in ALPHAS:
                    t = v2.alpha_tag(al)
                    r[f'old_recalc@{t}'] = oldv[al]['post_dr']
                    rv = fl(ref.get(f'post_dr@{t}')) if ref else float('nan')
                    r[f'old_csv@{t}'] = rv
                    ok = ok and np.isfinite(rv) and abs(rv - oldv[al]['post_dr']) <= TOL
                r['parity'] = 'match' if ok else ('no_ref' if ref is None else 'DIFF')
            rows.append(r)
            by.setdefault(k, []).append(r)
            log(f'  {base:16s} cal {r["cal_raw_start_s"]:6.1f}–{r["cal_raw_end_s"]:6.1f} s  post {r["post_s"]:6.1f} s  '
                + '  '.join(f'DR@{v2.alpha_tag(al)}={100 * r[f"post_dr@{v2.alpha_tag(al)}"]:6.2f}%' for al in ALPHAS)
                + f'  核对={r["parity"]}')
        tag5 = [v2.alpha_tag(al) for al in ALPHAS]
        write_csv(os.path.join(out, f's2_{k}_postdr.csv'),
                  [dict(star=r['star'], scen=r['scenario'], prn=r['prn'],
                        **{f'post_dr@{t}': r[f'post_dr@{t}'] for t in tag5}) for r in by.get(k, [])])
    write_csv(os.path.join(out, 's2_star.csv'), rows)


    B = []
    tags = [v2.alpha_tag(al) for al in ALPHAS]
    B.append(f'[version] s2_detrate.py v1.0  星数 = ' + ', '.join(f'{k}:{len(by.get(k, []))}' for k in a.scores))
    B.append('[口径] γ = 标定段（表 1）的 (1−α) 分位；检测率 = onset→录音结束；ds3 onset = 118.9 s')
    pk = []
    for k in a.scores:
        rr = by.get(k, [])
        if k == 'tcn':
            continue
        nm = sum(r['parity'] == 'match' for r in rr)
        bad = [r['star'] for r in rr if r['parity'] != 'match']
        pk.append(f'{k} {nm}/{len(rr)}' + (f'(不一致: {",".join(bad[:4])})' if bad else ''))
    B.append('[取数核对] 旧口径复算 vs 旧 CSV（容差 1e-4）: ' + '；'.join(pk))
    exc = max((abs(r[f'cal_exc@{t}'] - al) / al for r in rows for al, t in zip(ALPHAS, tags)), default=float('nan'))
    B.append(f'[自检] 标定段超阈率与 α 的最大相对偏差 = {exc:.2e}（应 ≈ 0）')
    t1o = []
    B.append('== 新表 8：逐时刻检测率（%，21 颗星均值 [最小]）；括号外"旧"为稿件现值 ==')
    B.append(f'{"score":5s} | ' + ' | '.join(f'α={al:<5g} 新 均值[最小]   旧' for al in ALPHAS))
    tex_rows = []
    for k, lab in SCORES:
        rr = by.get(k, [])
        if not rr:
            continue
        cells, tcells = [], []
        rec = dict(score=k, n=len(rr))
        for i, (al, t) in enumerate(zip(ALPHAS, tags)):
            v = np.array([100 * r[f'post_dr@{t}'] for r in rr])
            mean, mn = float(np.mean(v)), float(np.min(v))
            rec[f'mean@{t}'], rec[f'min@{t}'] = mean, mn
            for g in ('static', 'dynamic', 'complex'):
                vg = [100 * r[f'post_dr@{t}'] for r in rr if r['group'] == g]
                rec[f'{g}_mean@{t}'] = float(np.mean(vg)) if vg else float('nan')
                rec[f'{g}_median@{t}'] = float(np.median(vg)) if vg else float('nan')
            o = OLD_T8.get(k, (float('nan'),) * 3)[i]
            cells.append(f'{mean:6.2f} [{mn:6.2f}] {o:6.2f}')
            tcells.append(f'{mean:.2f} [{mn:.2f}]')
        t1o.append(rec)
        tex_rows.append(f'{lab} & ' + ' & '.join(tcells) + '\\\\')
        B.append(f'{k:5s} | ' + ' | '.join(cells))
    write_csv(os.path.join(out, 's2_T1o.csv'), t1o)
    with open(os.path.join(out, 's2_table8.tex'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join([
            '\\begin{table}[!htbp]',
            '\\caption{Per-instant detection rate after onset at a nominal per-epoch false-alarm rate $\\alpha$ '
            '(\\%; mean over 21 satellites, minimum in brackets).\\label{tab:T1o}}',
            '\\centering', '\\small', '\\begin{tabular}{@{}lccc@{}}', '\\toprule',
            'Score & $\\alpha=10^{-1}$ & $5\\times10^{-2}$ & $10^{-2}$\\\\', '\\midrule'] + tex_rows + [
            '\\bottomrule', '\\end{tabular}', '',
            '\\noindent{\\footnotesize Threshold: $(1-\\alpha)$ quantile of the score on the calibration segment of '
            'Table~\\ref{tab:datasets}; rate counted from the onset to the end of the record.}',
            '\\end{table}', '']))

    t05 = v2.alpha_tag(0.05)
    dif = [(abs(r[f'post_dr@{t05}'] - r.get(f'old_csv@{t05}', float('nan'))), r) for r in rows
           if np.isfinite(r.get(f'old_csv@{t05}', float('nan')))]
    dif.sort(key=lambda x: -x[0])
    if dif:
        B.append('[新−旧] α=0.05 逐星差最大的 5 个: ' + '；'.join(
            f'{r["score"]}:{r["star"]} {100 * (r[f"post_dr@{t05}"] - r[f"old_csv@{t05}"]):+.2f}pp' for _, r in dif[:5]))
        for g in ('static', 'dynamic', 'complex'):
            dg = [100 * abs(r[f'post_dr@{t05}'] - r[f'old_csv@{t05}']) for _, r in dif if r['group'] == g]
            B.append(f'  {g}: |新−旧| 中位 {np.median(dg):.2f} pp，最大 {np.max(dg):.2f} pp（n={len(dg)}）' if dg else f'  {g}: —')


    B.append('== 图 11：逐时刻检测率（α=0.05，新）vs CUSUM 时延（β=0.02，S1）==')
    pairs, s1rows = [], {}
    for k in a.scores:
        p = os.path.join(s1dir, f'rev_{k}_star.csv')
        s1rows[k] = read_csv(p)
        if s1rows[k] is None:
            B.append(f'[WARN] 找不到 S1 结果 {p}')
    for g in ('static', 'dynamic'):
        dcol = 'delay_onset_s' if g == 'static' else 'delay_act_s'
        per = []
        for k in a.scores:
            if not s1rows.get(k):
                continue
            dr = {r['star']: 100 * r[f'post_dr@{t05}'] for r in by.get(k, [])}
            drold = {r['star']: 100 * r[f'old_csv@{t05}'] for r in by.get(k, []) if f'old_csv@{t05}' in r}
            xs, ys, xo = [], [], []
            for r in s1rows[k]:
                if r['rule'] != 'cusum' or abs(fl(r['beta']) - 0.02) > 1e-9 or r['group'] != g:
                    continue
                st, d = ns(r['star']), fl(r[dcol])
                if st in dr and np.isfinite(d) and d > 0:
                    xs.append(dr[st]); ys.append(d); xo.append(drold.get(st, float('nan')))
                    pairs.append(dict(group=g, score=k, star=st, dr05_new=dr[st], dr05_old=drold.get(st, float('nan')),
                                      cusum_delay_s=d))
            if xs:
                per.append((k, len(xs), float(np.median(xs)), float(np.median(ys)), float(np.max(ys))))
        if not per:
            continue
        B.append(f'[{g}] ' + '  '.join(f'{k}: n={n} DR中位 {m:.1f}% 时延中位 {d:.3f}[{mx:.3f}]s' for k, n, m, d, mx in per))
        mdr, mdl = [p[2] for p in per], [p[3] for p in per]
        B.append(f'[{g}] 跨分数：DR 中位 {min(mdr):.1f}–{max(mdr):.1f}%（差 {max(mdr) - min(mdr):.1f} pp）；'
                 f'CUSUM 时延中位 {min(mdl):.3f}–{max(mdl):.3f} s（差 {max(mdl) - min(mdl):.3f} s）')
        pg = [p for p in pairs if p['group'] == g]
        r5 = spearman([p['dr05_new'] for p in pg], [p['cusum_delay_s'] for p in pg])
        p4 = [p for p in pg if p['score'] != 'tcn']
        r4 = spearman([p['dr05_new'] for p in p4], [p['cusum_delay_s'] for p in p4])
        r4o = spearman([p['dr05_old'] for p in p4], [p['cusum_delay_s'] for p in p4])
        old_rho = '−0.18 (n=28)' if g == 'static' else '−0.05 (n=40)'
        B.append(f'[{g}] Spearman ρ_S：五种 {r5[0]:+.3f} (n={r5[1]})；四种(无TCN) {r4[0]:+.3f} (n={r4[1]})；'
                 f'四种用旧DR {r4o[0]:+.3f} (n={r4o[1]})；稿件 {old_rho}')
    write_csv(os.path.join(out, 's2_fig11_pairs.csv'), pairs)


    cmp = []
    for k in a.scores:
        rr = s1rows.get(k) or []

        def med(rule):
            v = [fl(r['delay_onset_s']) for r in rr if r['rule'] == rule and abs(fl(r['beta']) - 0.02) < 1e-9
                 and r['group'] == 'static' and np.isfinite(fl(r['delay_onset_s']))]
            return (float(np.median(v)), float(np.max(v))) if v else (float('nan'), float('nan'))
        c, kb = med('cusum'), med('k_blk')
        if np.isfinite(c[0]):
            cmp.append(f'{k} {c[0]:.3f}[{c[1]:.3f}] vs {kb[0]:.3f}[{kb[1]:.3f}] Δ={c[0] - kb[0]:+.3f}')
    if cmp:
        B.append('[check] 静态类 β=0.02 时延中位[最大] CUSUM vs k_blk：' + '；'.join(cmp))
    B.append(f'[done] 用时 {(time.time() - t0) / 60:.1f} 分钟；输出目录 {out}')
    with open(os.path.join(out, 's2_brief.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(B) + '\n')
    log('\n' + '\n'.join(B))
    log.close()


if __name__ == '__main__':
    main()
