"""Baseline shift between calibration and held-out segments (Section 4.4, Figure 9).

Median shift in robust units, block-mean shift, within-segment trend and exceedance
ratios of the score series, and their rank correlation with the held-out false
alarms.
Run from the repository root (the working folder that also holds the data folders); see --help.
"""
import numpy as np, os, sys, argparse, csv, time, re

_here = os.path.dirname(os.path.abspath(__file__))
for _d in (_here, os.path.dirname(_here), os.path.dirname(os.path.dirname(_here))):
    if os.path.exists(os.path.join(_d, 'sqm_beta_neweval_v2.py')):
        if _d not in sys.path:
            sys.path.insert(0, _d)
        break
if _here not in sys.path:
    sys.path.insert(0, _here)

import c1p_blockcn0 as cp

fc, rm, sq = cp.fc, cp.rm, cp.sq

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
SERIES = ('cn0_lin', 'cn0_db', 'qr', 'wscm')
SCORES = ('qr', 'cn0_fold', 'wscm_fold')
QS = (0.90, 0.99)
GRPS = ('static', 'dynamic', 'complex')
TBLK = 1000
NPERM = 20000


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def mad_scale(x):
    med = float(np.median(x))
    return med, 1.4826 * float(np.median(np.abs(x - med)))


def linfit(t, y):
    t, y = np.asarray(t, float), np.asarray(y, float)
    if len(t) < 3:
        return float('nan'), float('nan'), float('nan')
    tm, ym = t.mean(), y.mean()
    sxx = float(((t - tm) ** 2).sum())
    b = float(((t - tm) * (y - ym)).sum() / sxx)
    a = ym - b * tm
    sst = float(((y - ym) ** 2).sum())
    r2 = 1.0 - float(((y - a - b * t) ** 2).sum()) / sst if sst > 0 else float('nan')
    return b, a, r2


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


def spearman(x, y, rng, nperm=NPERM):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 4 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float('nan'), float('nan'), n
    rx, ry = rankavg(x), rankavg(y)
    rx = (rx - rx.mean()) / rx.std()
    ry = (ry - ry.mean()) / ry.std()
    rho = float((rx * ry).mean())
    P = np.argsort(rng.random((nperm, n)), axis=1)
    rp = (ry[P] * rx[None, :]).mean(1)
    return rho, float((1 + (rp >= rho - 1e-12).sum()) / (nperm + 1)), n


def norm_star(s):
    s = os.path.basename(str(s).strip())
    if s.startswith('pertap_'):
        s = s[len('pertap_'):]
    if s.endswith('.npz'):
        s = s[:-4]
    if s.endswith('_dyn'):
        s = s[:-4]
    return s


def _nrm(v):
    return re.sub(r'[^0-9a-z]', '', str(v).lower())


def _flt(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


# ---------------------------------------------------------------------------
# Earlier results
# ---------------------------------------------------------------------------
def read_rows(path):
    with open(path, encoding='utf-8-sig', newline='') as fh:
        return list(csv.DictReader(fh))


def pick_col(hdr, cands):
    low = {h.strip().lower(): h for h in hdr}
    for c in cands:
        if c in low:
            return low[c]
    return None


def med(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else float('nan')


def fa_from_rev(path, rule, beta, log):
    if not os.path.exists(path):
        return {}, f'找不到 {path}'
    rows = read_rows(path)
    if not rows:
        return {}, '空文件'
    hdr = list(rows[0].keys())
    cs = pick_col(hdr, ('star', 'base', 'key', 'name'))
    cr = pick_col(hdr, ('rule', 'method', 'decision', 'calib'))
    cb = pick_col(hdr, ('beta', 'level', 'target'))
    cf = pick_col(hdr, ('ho_frac',))
    if None in (cs, cr, cb, cf):
        log(f'[WARN] {os.path.basename(path)} 列名识别失败（星={cs} 规则={cr} β={cb} 误报={cf}）；表头前 40 列：{hdr[:40]}')
        return {}, '列名识别失败'
    out, ndup = {}, 0
    for r in rows:
        if _nrm(r[cr]) != _nrm(rule) or abs(_flt(r[cb]) - beta) > 1e-9:
            continue
        k = norm_star(r[cs])
        if k in out:
            ndup += 1
            continue
        out[k] = _flt(r[cf])
    if not out:
        log(f'[WARN] {os.path.basename(path)}：规则={rule} β={beta} 没匹配到行；{cr} 取值 '
            f'{sorted({r[cr] for r in rows})[:12]}，{cb} 取值 {sorted({r[cb] for r in rows})[:12]}')
    return out, f'列=({cs},{cr},{cb},{cf})' + (f'，同星重复行 {ndup} 已忽略' if ndup else '')


def fa_from_c1p(path):
    if not os.path.exists(path):
        return {}, f'找不到 {path}'
    out = {}
    for r in read_rows(path):
        if (int(_flt(r['T_ms'])) == 20 and r['mu1'] == 'p1sd' and r['arm'] == 'blk'
                and abs(_flt(r['beta']) - 0.05) < 1e-9):
            out[norm_star(r['star'])] = _flt(r.get('ho_frac', ''))
    return out, 'T=20, p1sd, blk, β=0.05'


def acf_ref(path):
    if not os.path.exists(path):
        return None
    return {norm_star(r['star']): _flt(r['acf1']) for r in read_rows(path)
            if r['score'] == 'cn0_db' and int(_flt(r['T_ms'])) == 1}


# ---------------------------------------------------------------------------
# Shift between calibration and held-out segments (Section 4.4, Figure 9)
# ---------------------------------------------------------------------------
def star_stats(base, key, path, gidx, log):
    series, cn0, lab, scen, prn, seg = cp.load_all(path, gidx, log)
    taps, cn0r, *_ = sq.load_star_v2(path)
    wscm = np.asarray(sq.get_metric('wscm', taps, cn0r, sq.DEFAULT_E_IDX, sq.DEFAULT_L_IDX), float).ravel()
    cm = seg['cmask']
    bad = ~np.isfinite(wscm)
    if bad.any():
        wscm = np.where(bad, float(np.median(wscm[cm & ~bad])), wscm)
    sers = {'cn0_lin': cn0, 'cn0_db': series['cn0_db'], 'qr': series['qr'], 'wscm': wscm}
    scs = {'qr': series['qr'], 'cn0_fold': series['cn0_fold'], 'wscm_fold': series['wscm_fold']}
    c0, c1, ha, hb, raw = seg['c0'], seg['c1'], seg['ho_a'], seg['hb_act'], seg['raw']
    nh = hb - ha
    grp = fc.group_of(scen)
    r = dict(star=base, key=key, scenario=scen, group=grp, prn=prn, pooled_excluded=(scen in rm.POOL_EXCLUDE),
             clean_contiguous=seg['contiguous'], clean_s=int(cm.sum()) / 1000.0, ho_s=nh / 1000.0,
             t_clean0=raw(c0), t_clean1=raw(c1), t_ho1=raw(hb), has_ho=(nh >= TBLK),
             acf1_cn0db=cp.acf_nan(series['cn0_db'][c0:c1], 1))
    a, b = cp.blk_range(c0, c1, TBLK)
    tb = raw(np.arange(a, b) * TBLK + (TBLK - 1) / 2.0)
    for nm, x in sers.items():
        xc = x[cm]
        m_c, s_c = mad_scale(xc)
        xb = cp.block_mean(x, TBLK)[a:b]
        slope, icpt, r2 = linfit(tb, xb)
        mean_c, sd_blk = float(xb.mean()), float(xb.std())
        r.update({f'{nm}_m_c': m_c, f'{nm}_s_c': s_c, f'{nm}_sd_blk_c': sd_blk, f'{nm}_trend_r2': r2,
                  f'{nm}_drift_clean_sc': slope * (tb[-1] - tb[0]) / s_c if s_c > 0 else float('nan')})
        if nh >= TBLK and s_c > 0:
            xh = x[ha:hb]
            m_h, s_h = mad_scale(xh)
            mean_h = float(xh.mean())
            t_mid = raw((ha + hb - 1) / 2.0)
            r.update({f'{nm}_delta': (m_h - m_c) / s_c, f'{nm}_scale_ratio': s_h / s_c,
                      f'{nm}_delta_mean': (mean_h - mean_c) / s_c,
                      f'{nm}_delta_pred': (icpt + slope * t_mid - mean_c) / s_c,
                      f'{nm}_delta_blk': (mean_h - mean_c) / sd_blk if sd_blk > 0 else float('nan')})
    for nm, x in scs.items():
        xc = x[cm]
        for q in QS:
            g = float(np.quantile(xc, q))
            p_c = float(np.mean(xc > g))
            p_h = float(np.mean(x[ha:hb] > g)) if nh >= TBLK else float('nan')
            r[f'{nm}_er{int(round(q * 100))}'] = p_h / p_c if p_c > 0 else float('nan')
    return r


def group_rows(rows):
    out = []
    use = [r for r in rows if r['has_ho'] and not r['pooled_excluded']]
    for g in GRPS + ('all',):
        sel = [r for r in use if g == 'all' or r['group'] == g]
        if not sel:
            continue
        for nm in SERIES:
            out.append(dict(kind='series', name=nm, group=g, n=len(sel),
                            abs_delta_med=med([abs(r.get(f'{nm}_delta', np.nan)) for r in sel]),
                            abs_delta_blk_med=med([abs(r.get(f'{nm}_delta_blk', np.nan)) for r in sel]),
                            scale_ratio_med=med([r.get(f'{nm}_scale_ratio', np.nan) for r in sel]),
                            abs_delta_pred_med=med([abs(r.get(f'{nm}_delta_pred', np.nan)) for r in sel]),
                            abs_drift_clean_med=med([abs(r.get(f'{nm}_drift_clean_sc', np.nan)) for r in sel]),
                            trend_r2_med=med([r.get(f'{nm}_trend_r2', np.nan) for r in sel])))
        for nm in SCORES:
            e90 = [r.get(f'{nm}_er90', np.nan) for r in sel]
            out.append(dict(kind='score', name=nm, group=g, n=len(sel), er90_med=med(e90),
                            er90_max=max((v for v in e90 if np.isfinite(v)), default=float('nan')),
                            er99_med=med([r.get(f'{nm}_er99', np.nan) for r in sel])))
    return out


# ---------------------------------------------------------------------------
# Rank correlation with the held-out false alarms
# ---------------------------------------------------------------------------
CORR_X = (('cn0_db', 'delta_blk', True), ('cn0_db', 'delta', True), ('cn0_fold', 'er90', False),
          ('cn0_fold', 'er99', False), ('qr', 'delta_blk', True), ('wscm', 'delta_blk', True))


def corr_rows(rows, rng):
    out = []
    use = [r for r in rows if r['has_ho'] and not r['pooled_excluded']]
    for sub in ('all', 'static'):
        sel = [r for r in use if sub == 'all' or r['group'] == sub]
        for nm, met, ab in CORR_X:
            x = [abs(r.get(f'{nm}_{met}', np.nan)) if ab else r.get(f'{nm}_{met}', np.nan) for r in sel]
            for fa in ('fa_kblk', 'fa_c1p'):
                rho, p, n = spearman(x, [r.get(fa, np.nan) for r in sel], rng)
                out.append(dict(subset=sub, x=f'{"|" if ab else ""}{nm}_{met}{"|" if ab else ""}', fa=fa,
                                rho=rho, p_one_sided=p, n=n))
    return out


def f2(v, w=6, d=2):
    return f'{v:{w}.{d}f}' if v is not None and np.isfinite(v) else ' ' * (w - 1) + '-'


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', dest='npz_dir', default='.')
    ap.add_argument('--stars', nargs='+', default=None)
    ap.add_argument('--guard-s', dest='guard_s', type=float, default=0.1)
    ap.add_argument('--rev-cn0', default=None, help=r'默认 <dir>\re_ii\results_c\rev_cn0_star.csv')
    ap.add_argument('--c1p-star', default=None, help=r'默认 <dir>\re_ii\results_c\c1p_c1p_star.csv')
    ap.add_argument('--c1b-star', default=None, help=r'默认 <dir>\re_ii\results_c\c1p_c1b_star.csv')
    ap.add_argument('--rule', default='k_blk')
    ap.add_argument('--beta', type=float, default=0.05)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    D = os.path.abspath(a.npz_dir)
    rc = os.path.join(D, 're_ii', 'results_c')
    prefix = os.path.splitext(a.out)[0]
    os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
    log = fc.Tee(prefix + '_log.txt')
    brief = []

    def B(s=''):
        log(s)
        brief.append(s)

    t0 = time.time()
    B('[version] c1d_drift.py v1.0（2026-10-25）  净段 vs 留出段 ho（终点=开启−0.1 s）；ds8 不进汇总')
    items = rm.qr_items(D, a.stars)
    rows = []
    for i, (base, path, key) in enumerate(items, 1):
        try:
            r = star_stats(base, key, path, int(round(a.guard_s * 1000)), log)
        except Exception as e:
            log(f'[skip] {base}: {e}')
            continue
        rows.append(r)
        log(f'[{i}/{len(items)}] {base} ({r["group"]}) ho {r["ho_s"]:.1f} s  cn0dB Δ={f2(r.get("cn0_db_delta"))} '
            f'Δblk={f2(r.get("cn0_db_delta_blk"))}  ER90 cn0f={f2(r.get("cn0_fold_er90"))}')

    fa_k, tag_k = fa_from_rev(a.rev_cn0 or os.path.join(rc, 'rev_cn0_star.csv'), a.rule, a.beta, log)
    fa_c, tag_c = fa_from_c1p(a.c1p_star or os.path.join(rc, 'c1p_c1p_star.csv'))
    for r in rows:
        k = norm_star(r['star'])
        r['fa_kblk'] = fa_k.get(k, fa_k.get(norm_star(r['key']), float('nan')))
        r['fa_c1p'] = fa_c.get(k, float('nan'))

    nho = sum(r['has_ho'] for r in rows)
    B(f'== 自检① 星数 {len(rows)}（应 21），有留出段 {nho}（应 16），进汇总 '
      f'{sum(r["has_ho"] and not r["pooled_excluded"] for r in rows)}（应 14）')
    ref = acf_ref(a.c1b_star or os.path.join(rc, 'c1p_c1b_star.csv'))
    if ref is None:
        B('== 自检② 找不到 c1p_c1b_star.csv，跳过')
    else:
        dif = [r['star'] for r in rows if not (norm_star(r['star']) in ref
                                               and abs(ref[norm_star(r['star'])] - r['acf1_cn0db']) <= 1e-5)]
        B(f'== 自检② cn0dB 1 ms ACF(1) 对 c1p_c1b_star：{len(rows) - len(dif)} 一致，{len(dif)} 不一致'
          + (f'  {dif}' if dif else ''))
    B(f'== 自检③ fa_kblk 匹配 {sum(np.isfinite(r["fa_kblk"]) for r in rows if r["has_ho"])}/{nho} [{tag_k}]；'
      f'fa_c1p 匹配 {sum(np.isfinite(r["fa_c1p"]) for r in rows if r["has_ho"])}/{nho} [{tag_c}]')

    gr = group_rows(rows)
    cr = corr_rows(rows, np.random.default_rng(a.seed))
    fields = list(dict.fromkeys(k for r in rows for k in r))
    fc.save_csv(prefix + '_star.csv', rows, fields)
    fc.save_csv(prefix + '_group.csv', gr, list(dict.fromkeys(k for r in gr for k in r)))
    fc.save_csv(prefix + '_corr.csv', cr, ['subset', 'x', 'fa', 'rho', 'p_one_sided', 'n'])

    B('\n== A. 原始序列：净段 → 留出段（跨星中位数；单位：Δ、Δpred、净段内漂移 = s_c；Δblk = 净段 1 s 块均值的 sd）==')
    B('序列     组       n | |Δ|   |Δblk| s_h/s_c |Δpred| |净段漂移| 趋势R²')
    for g in GRPS + ('all',):
        for d in (x for x in gr if x['kind'] == 'series' and x['group'] == g):
            B(f'{d["name"]:8s} {g:8s} {d["n"]:2d} |{f2(d["abs_delta_med"])}{f2(d["abs_delta_blk_med"], 7)}'
              f'{f2(d["scale_ratio_med"], 8)}{f2(d["abs_delta_pred_med"], 8)}{f2(d["abs_drift_clean_med"], 9)}'
              f'{f2(d["trend_r2_med"], 8)}')
    B('\n== B. 打分序列超限率比 ER = 留出段超限率 / 净段超限率（γ = 净段分位；1 = 无变化）==')
    B('打分       组       | ER90中位 ER90最大 ER99中位')
    for g in GRPS + ('all',):
        for d in (x for x in gr if x['kind'] == 'score' and x['group'] == g):
            B(f'{d["name"]:10s} {g:8s} |{f2(d["er90_med"], 8)}{f2(d["er90_max"], 9)}{f2(d["er99_med"], 9)}')
    B('\n== C. 与误报的 Spearman 秩相关：ρ (单侧 p, n) ==')
    B('x                  | fa_kblk 全部(14)   fa_kblk 静态(7)   | fa_c1p 全部(14)    fa_c1p 静态(7)')
    ci = {(c['subset'], c['x'], c['fa']): c for c in cr}
    for nm, met, ab in CORR_X:
        xn = f'{"|" if ab else ""}{nm}_{met}{"|" if ab else ""}'
        cells = []
        for fa in ('fa_kblk', 'fa_c1p'):
            for sub in ('all', 'static'):
                c = ci[(sub, xn, fa)]
                cells.append(f'{f2(c["rho"], 5)} ({f2(c["p_one_sided"], 4, 3)},{c["n"]:2d})')
        B(f'{xn:18s} | {cells[0]}  {cells[1]}  | {cells[2]}  {cells[3]}')
    B('\n== D. 逐星（有留出段；* = ds8，不进汇总）。Δ 有符号；ER90 为打分序列；fa 为 β=0.05 的 ho 误报 ==')
    B('星              组   cn0dB:Δ  Δblk  Δpred | ER90: cn0f   qr  wscmf | fa_kblk fa_c1p')
    for r in rows:
        if not r['has_ho']:
            continue
        B(f'{r["star"][:15]:15s}{"*" if r["pooled_excluded"] else " "}{r["group"][:4]:4s}'
          f'{f2(r.get("cn0_db_delta"), 7)}{f2(r.get("cn0_db_delta_blk"), 6)}{f2(r.get("cn0_db_delta_pred"), 7)} |'
          f'{f2(r.get("cn0_fold_er90"), 9)}{f2(r.get("qr_er90"), 6)}{f2(r.get("wscm_fold_er90"), 6)} |'
          f'{f2(r.get("fa_kblk"), 8, 4)}{f2(r.get("fa_c1p"), 8, 4)}')
    B('注：判据见文件头。n=14/7，秩相关统计力弱，只作方向参考；判断主要看 A、B 两表静态行 cn0 与 QR/WSCM 的差距。')
    B(f'[saved] {prefix}_star.csv / _group.csv / _corr.csv / _log.txt / _brief.txt   [done] {time.time() - t0:.0f} s')
    with open(prefix + '_brief.txt', 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(brief) + '\n')
    log.close()
