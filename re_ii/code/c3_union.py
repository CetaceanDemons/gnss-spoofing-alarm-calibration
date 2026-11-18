"""Receiver-level false alarms (Section 4.7, Table A2).

Takes the per-satellite limits from the _star.csv files, forms the union of the alarms
of all satellites of one recording over their common held-out stretch, and compares
the union rate with beta and with 1-(1-beta)^n.
Run from the repository root (the working folder that also holds the data folders); see --help.
"""
import numpy as np, os, sys, argparse, time
from types import SimpleNamespace

_here = os.path.dirname(os.path.abspath(__file__))
for _d in (_here, os.path.dirname(_here), os.path.dirname(os.path.dirname(_here))):
    if os.path.exists(os.path.join(_d, 'sqm_beta_neweval_v2.py')):
        if _d not in sys.path:
            sys.path.insert(0, _d)
        break
if _here not in sys.path:
    sys.path.insert(0, _here)

import rev_main as rm
import far_calib_v32 as fc
import krule_clean_far_v12 as kr
import rev_sqm as rsq
import sqm_beta_neweval_v2 as sq

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
RULES = [('cusum', b) for b in rm.BETAS_MAIN] + [('k_blk', b) for b in rm.BETAS_MAIN] + \
        [('k_iid', b) for b in rm.BETAS_MAIN] + [('k_p95', float('nan'))]
CSV_SUB = {'qr': 'results_rev', 'ae': 'results_rev', 'tcn': 'results_rev', 'cn0': 'results_c', 'wscm': 'results_c'}


def bkey(b):
    return -1.0 if not np.isfinite(b) else round(float(b), 12)


def load_items(enc, D, a, log):
    if enc in ('qr', 'cn0', 'wscm'):
        return rm.qr_items(D, None)
    sub, s1, s2 = rm.DEFAULT_DUAL[enc]
    return rm.dual_items(os.path.join(D, sub, s1), os.path.join(D, sub, s2), None, log)


def make_loader(enc, a, log):
    if enc == 'qr':
        return lambda p: fc.load_star(p)
    if enc in ('ae', 'tcn'):
        return lambda p: rm.v2.load_S_scores(p)
    ns = SimpleNamespace(guard_s=a.guard_s, e_idx=sq.DEFAULT_E_IDX, l_idx=sq.DEFAULT_L_IDX)
    ld = rsq.make_loader(enc, ns, log, {})
    return lambda p: ld(p, enc)


# ---------------------------------------------------------------------------
# Per-satellite limits read back from the _star.csv files
# ---------------------------------------------------------------------------
def star_rules(S, lab, scen, ref_rows, a):
    """Alarm functions, compared series and thresholds of each rule for one satellite."""
    gidx = int(round(a.guard_s * 1000))
    S = np.ascontiguousarray(np.asarray(S, dtype=np.float64))
    seg = fc.segments(S, lab, scen, gidx)
    cm = seg['cmask']
    Sc = np.ascontiguousarray(S[cm])
    mu, sd = float(Sc.mean()), float(Sc.std())
    Z = np.ascontiguousarray((S - mu) / sd)
    Z32 = np.ascontiguousarray(Z.astype(np.float32).astype(np.float64))
    cand = np.unique(Z[cm].astype(np.float32)).astype(np.float64)
    K, m = a.K, a.m
    out = {}
    for rule, b in RULES:
        rr = ref_rows.get((rule, bkey(b)))
        if rr is None or rr.get('thr', '') == '':
            continue
        if rule == 'cusum':
            h = rm._f(rr['thr'])
            out[(rule, bkey(b))] = (lambda X, _h=h: fc.alarm_idx(X, rm.K0, _h), Z, h, rr)
        elif rule == 'k_blk':
            gz0 = rm._f(rr['thr_z'])
            gz = float(cand[int(np.argmin(np.abs(cand - gz0)))])
            out[(rule, bkey(b))] = (lambda X, _g=gz: kr.krun_idx(X, _g, K), Z32, gz, rr)
        else:
            p_t = kr.p_for_target(b, K, m) if rule == 'k_iid' else 0.05
            g = float(np.quantile(Sc, 1.0 - p_t))
            out[(rule, bkey(b))] = (lambda X, _g=g: kr.krun_idx(X, _g, K), S, g, rr)
    return out, seg


# ---------------------------------------------------------------------------
# Union of the alarms over the common held-out stretch (Section 4.7, Table A2)
# ---------------------------------------------------------------------------
# Compared with beta and with 1-(1-beta)^n; the union bound n*beta holds for any
# dependence between satellites (Section 3.5).
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', dest='npz_dir', default='.')
    ap.add_argument('--encoders', nargs='+', default=['qr', 'ae', 'cn0', 'wscm'], choices=list(CSV_SUB))
    ap.add_argument('--stars', nargs='+', default=None, help='只用星名含这些子串的星（冒烟用）')
    ap.add_argument('--csv-dir', default=None, help=r'覆盖 _star.csv 所在目录（默认按编码器：re_ii\results_rev 或 re_ii\results_c）')
    ap.add_argument('--K', type=int, default=5)
    ap.add_argument('--m', type=int, default=1000)
    ap.add_argument('--guard-s', dest='guard_s', type=float, default=0.1)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    D = os.path.abspath(a.npz_dir)
    prefix = os.path.splitext(a.out)[0]
    os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
    log = fc.Tee(prefix + '_log.txt')
    brief = []

    def B(s=''):
        log(s)
        brief.append(s)

    t0 = time.time()
    B('[version] c3_union.py v1.0（2026-10-24）')
    B(f'[cfg] encoders={a.encoders}  K={a.K}  m={a.m}  guard={a.guard_s}  只评 ho（主口径）；合并不含 ds8')
    scen_rows, reg_rows, regsum = [], [], []
    for enc in a.encoders:
        cp = os.path.join(a.csv_dir or os.path.join(D, 're_ii', CSV_SUB[enc]), f'rev_{enc}_star.csv')
        if not os.path.exists(cp):
            B(f'[{enc}] 找不到 {cp}，跳过')
            continue
        ref = {}
        for r in rm._read_csv(cp):
            ref.setdefault(r['star'], {})[(r['rule'], bkey(rm._f(r['beta'])))] = r
        items = [it for it in load_items(enc, D, a, log) if not a.stars or any(s in it[0] for s in a.stars)]
        loader = make_loader(enc, a, log)
        log(f'\n==== [{enc}] {len(items)} 颗星，门限来自 {cp}')
        per_scen = {}
        n_ok = n_bad = 0
        for base, path, key in items:
            if base not in ref:
                log(f'  [skip] {base}: csv 里没有这颗星')
                continue
            S, lab, scen, prn = loader(path)
            rules, seg = star_rules(S, np.asarray(lab), scen, ref[base], a)
            per_scen.setdefault(scen, []).append((base, rules, seg))
            for (rule, bk), (fn, X, thr, rr) in rules.items():
                res = fc.ho_eval(fn, X, seg['ho_a'], seg['hb_act'], [], a.m)
                ev0 = int(rm._f(rr.get('ho_n_event', 'nan'))) if rr.get('ho_n_event', '') != '' else -1
                same = (res['n_event'] == ev0)
                n_ok += same
                n_bad += not same
                thr_csv = rm._f(rr['thr_z'] if rule == 'k_blk' else rr['thr'])
                reg_rows.append(dict(encoder=enc, star=base, rule=rule, beta=(bk if bk > 0 else float('nan')),
                                     thr_used=thr, thr_csv=thr_csv, ho_event_now=res['n_event'], ho_event_csv=ev0,
                                     match=('match' if same else 'DIFF')))
        regsum.append(f'  {enc}: 逐星 ho 事件数重算 {n_ok} 一致 / {n_bad} 不一致' +
                      ('' if n_bad == 0 else '  [WARN] 看 _regress.csv 的 DIFF 行'))
        for scen, lst in sorted(per_scen.items()):
            A = max(s['ho_a'] for _, _, s in lst)
            Bnd = min(s['hb_act'] for _, _, s in lst)
            for rule, b in RULES:
                bk = bkey(b)
                have = [(base, rules[(rule, bk)]) for base, rules, _ in lst if (rule, bk) in rules]
                if not have:
                    continue
                n = len(have)
                U, single_frac, single_ev, decl = [], [], [], []
                for base, (fn, X, thr, rr) in have:
                    if Bnd <= A:
                        continue
                    full = np.asarray(fn(X[:Bnd]), dtype=np.int64)
                    idx = full[(full >= A) & (full < Bnd)]
                    U.append(idx)
                    c, w = fc.window_cov(idx, A, Bnd, a.m)
                    single_frac.append(c / w if w > 0 else float('nan'))
                    single_ev.append(fc.n_events(idx, a.m))
                    decl.append(b if rule != 'k_p95' else rm._f(rr.get('ref', 'nan')))
                if Bnd - A < a.m or not U:
                    scen_rows.append(dict(encoder=enc, scenario=scen, group=fc.group_of(scen), rule=rule,
                                          beta=b, n_stars=n, verdict_single='too_short', verdict_union='too_short'))
                    continue
                u = np.unique(np.concatenate(U))
                cov, nwin = fc.window_cov(u, A, Bnd, a.m)
                nev = fc.n_events(u, a.m)
                lo, up = fc.win_rate_bounds(nev, cov, nwin, a.m)
                d1 = float(np.nanmean(decl))
                dn = 1.0 - (1.0 - d1) ** n
                raw = lst[0][2]['raw']
                scen_rows.append(dict(encoder=enc, scenario=scen, group=fc.group_of(scen), rule=rule, beta=b,
                                      n_stars=n, A_raw_s=raw(A), B_raw_s=raw(Bnd), len_s=(Bnd - A) / 1000.0,
                                      n_event=nev, cov=cov, nwin=nwin, frac=cov / nwin, lo95=lo, up95=up,
                                      decl_single=d1, ref_union=dn,
                                      single_frac_mean=float(np.nanmean(single_frac)),
                                      single_event_sum=int(sum(single_ev)),
                                      verdict_single=fc.verdict(lo, up, d1, nwin),
                                      verdict_union=fc.verdict(lo, up, dn, nwin),
                                      pooled_excluded=(scen in rm.POOL_EXCLUDE)))

    pool = []
    for enc in a.encoders:
        for rule, b in RULES:
            for g in ('all', 'static', 'dynamic', 'complex'):
                sel = [r for r in scen_rows if r['encoder'] == enc and r['rule'] == rule and bkey(r['beta']) == bkey(b)
                       and r.get('nwin', 0) > 0 and not r['pooled_excluded'] and (g == 'all' or r['group'] == g)]
                if not sel:
                    continue
                nw = sum(r['nwin'] for r in sel)
                cov = sum(r['cov'] for r in sel)
                nev = sum(r['n_event'] for r in sel)
                d1 = sum(r['decl_single'] * r['nwin'] for r in sel) / nw
                dn = sum(r['ref_union'] * r['nwin'] for r in sel) / nw
                sf = sum(r['single_frac_mean'] * r['nwin'] for r in sel) / nw
                lo, up = fc.win_rate_bounds(nev, cov, nw, a.m)
                pool.append(dict(encoder=enc, rule=rule, beta=b, group=g, n_scen=len(sel),
                                 n_stars=sum(r['n_stars'] for r in sel), total_s=sum(r['len_s'] for r in sel),
                                 n_event=nev, frac=cov / nw, lo95=lo, up95=up, decl_single=d1, ref_union=dn,
                                 single_frac_mean=sf, ratio_union_single=(cov / nw / sf if sf > 0 else float('nan')),
                                 verdict_single=fc.verdict(lo, up, d1, nw), verdict_union=fc.verdict(lo, up, dn, nw)))
    SC_F = ['encoder', 'scenario', 'group', 'rule', 'beta', 'n_stars', 'A_raw_s', 'B_raw_s', 'len_s', 'n_event', 'cov',
            'nwin', 'frac', 'lo95', 'up95', 'decl_single', 'ref_union', 'single_frac_mean', 'single_event_sum',
            'verdict_single', 'verdict_union', 'pooled_excluded']
    PO_F = ['encoder', 'rule', 'beta', 'group', 'n_scen', 'n_stars', 'total_s', 'n_event', 'frac', 'lo95', 'up95',
            'decl_single', 'ref_union', 'single_frac_mean', 'ratio_union_single', 'verdict_single', 'verdict_union']
    RG_F = ['encoder', 'star', 'rule', 'beta', 'thr_used', 'thr_csv', 'ho_event_now', 'ho_event_csv', 'match']
    fc.save_csv(prefix + '_scen.csv', scen_rows, SC_F)
    fc.save_csv(prefix + '_pooled.csv', pool, PO_F)
    fc.save_csv(prefix + '_regress.csv', reg_rows, RG_F)

    vs = {'verified': 'W', 'inconclusive': 'C', 'violated': 'E', 'too_short': 'S', 'none': '-'}
    B('== 回归（逐星 ho 事件数 vs 原 _star.csv）==')
    for s in regsum:
        B(s)
    B('\n== 接收机级 ho 误报（各场景并集后合并，不含 ds8）==')
    B('enc   rule   β     | 星数 场景 | 并集率    [95%界]             事件 | 单星均值   并集/单星 | 对β判 | 对1−(1−β)^n 参照  判')
    for p in pool:
        if p['group'] != 'all':
            continue
        bt = 'p95  ' if not np.isfinite(p['beta']) else f'{p["beta"]:.0e}'
        B(f'{p["encoder"]:5s} {p["rule"]:6s} {bt} | {p["n_stars"]:3d}  {p["n_scen"]:3d}  | {p["frac"]:8.2e} '
          f'[{p["lo95"]:.1e}, {p["up95"]:.1e}] {p["n_event"]:4d} | {p["single_frac_mean"]:8.2e} {p["ratio_union_single"]:6.2f}   | '
          f'  {vs.get(p["verdict_single"], "?")}   | {p["ref_union"]:8.2e}         {vs.get(p["verdict_union"], "?")}')
    B('注：对β判 = 拿接收机级并集率去比单星声明值（E 即"单星 β 不等于接收机 β"）；对 1−(1−β)^n 判 = 比独立星假设下的并集参照。')
    B('注：并集/单星 ≈ n 说明各星误报互不重叠；明显 < n 说明误报在同一时刻共模出现（同场景共享前端/干扰）。')
    B(f'[saved] {prefix}_scen.csv / _pooled.csv / _regress.csv / _log.txt / _brief.txt')
    B(f'[done] 用时 {(time.time() - t0) / 60:.1f} 分钟')
    with open(prefix + '_brief.txt', 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(brief) + '\n')
    log.close()


if __name__ == '__main__':
    main()
