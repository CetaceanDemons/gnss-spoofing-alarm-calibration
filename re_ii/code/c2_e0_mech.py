"""Mechanism of the excess false alarms of count rules (Table 4, Figure 7).

Estimates the exceedance rate p and the persistence r = P(X_t > gamma | X_{t-1} > gamma)
on the calibration segments and predicts the window false-alarm rate with a two-state
Markov chain (Eq. (13)); repeats the comparison on simulated AR(1) series.
Run from the repository root (the working folder that also holds the data folders); see --help.
"""
import numpy as np, os, sys, argparse, time

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
from c1p_blockcn0 import load_all, med

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
SCORES = ('qr', 'cn0_fold', 'wscm_fold')
KS = (3, 5, 10)
LEVELS = [('beta', b) for b in (0.1, 0.05, 0.03, 0.02)] + [('alpha', a) for a in (0.05, 0.01)]
RHOS = (0.0, 0.1, 0.3, 0.5, 0.8)
M = 1000
GROUPS = ('all', 'static', 'dynamic', 'complex')


# ---------------------------------------------------------------------------
# Two-state Markov chain for the count rule (Eq. (13))
# ---------------------------------------------------------------------------
def markov_window(p, r, K, m):
    """Probability of at least one K-run alarm in m steps for a two-state Markov chain with persistence r and rate p.
    """
    if not (np.isfinite(p) and np.isfinite(r)) or p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    r = min(max(r, 0.0), 1.0)
    q = min(1.0, p * (1.0 - r) / (1.0 - p))
    n = K + 1
    Q = np.zeros((n, n))
    A = np.zeros(n)
    for st in range(n):
        pe = q if st == 0 else r
        run = 0 if st in (0, K) else st
        if run + 1 >= K:
            A[st] += pe
        else:
            Q[st, run + 1] += pe
        Q[st, 0] += 1.0 - pe
    P = Q.copy()
    P[:, K] += A
    Mx = np.vstack([P.T - np.eye(n), np.ones(n)])
    rhs = np.zeros(n + 1)
    rhs[-1] = 1.0
    pi = np.linalg.lstsq(Mx, rhs, rcond=None)[0]
    pi = np.clip(pi, 0.0, None)
    pi /= pi.sum()
    v = pi @ np.linalg.matrix_power(Q, m)
    return float(min(1.0, max(0.0, 1.0 - v.sum())))


def persist(e):
    d = int(e[:-1].sum())
    return float((e[1:] & e[:-1]).sum() / d) if d > 0 else float('nan')


_PT = {}


def p_target(beta, K):
    if (beta, K) not in _PT:
        _PT[(beta, K)] = kr.p_for_target(beta, K, M)
    return _PT[(beta, K)]


def eval_series(Sc, p_t, K):
    """Count rule on a calibration segment as in krule_clean_far_v12, with r and the two predictions.
    """
    N = len(Sc)
    g = float(np.quantile(Sc, 1.0 - p_t))
    e = Sc > g
    p_hat = float(e.mean())
    r_hat = persist(e)
    nom = kr.nominal_iid(p_hat, K, M)
    idx = kr.krun_idx(np.ascontiguousarray(Sc), g, K)
    cov, nwin = kr.window_cov(idx, 0, N, M)
    nev = kr.n_events(idx, M)
    lo, up = kr.win_rate_bounds(nev, cov, nwin, M)
    dlv = cov / nwin if nwin > 0 else float('nan')
    w_mk = markov_window(p_hat, r_hat, K, M)
    w_ii = markov_window(p_hat, p_hat, K, M)
    return dict(gamma=g, p_hat=p_hat, r_hat=r_hat, r_over_p=(r_hat / p_hat if p_hat > 0 else float('nan')),
                pred_approx=((r_hat / p_hat) ** (K - 1) if p_hat > 0 and np.isfinite(r_hat) else float('nan')),
                w_markov=w_mk, w_iid_mk=w_ii, pred_exact=(w_mk / w_ii if w_ii > 0 else float('nan')),
                nominal_iid=nom, dlv=dlv, n_event=nev, cov=cov, nwin=nwin, lo95=lo, up95=up,
                D=(dlv / nom if nom > 0 else float('nan')), resolvable_3q=(nom >= 3.0 * M / N),
                verdict=kr.verdict(lo, up, nom, nwin))


# ---------------------------------------------------------------------------
# Part A: calibration segments of the real satellites (Table 4, Figure 7)
# ---------------------------------------------------------------------------
def part_a(items, a, log):
    rows = []
    gidx = int(round(a.guard_s * 1000))
    for i, (base, path, key) in enumerate(items, 1):
        try:
            series, _, _, scen, prn, seg = load_all(path, gidx, log)
        except Exception as ex:
            log(f'[skip] {base}: {ex}')
            continue
        grp = fc.group_of(scen)
        cm = seg['cmask']
        t0 = time.time()
        for sc in SCORES:
            Sc = np.ascontiguousarray(series[sc][cm])
            for K in KS:
                for mode, lev in LEVELS:
                    p_t = p_target(lev, K) if mode == 'beta' else lev
                    r = dict(star=base, key=key, scenario=scen, group=grp, score=sc, K=K, mode=mode, level=lev,
                             p_target=p_t, N_clean=len(Sc), pooled_excluded=(scen in rm.POOL_EXCLUDE))
                    r.update(eval_series(Sc, p_t, K))
                    rows.append(r)
        q = {(r['score'], r['K']): r for r in rows if r['star'] == base and r['mode'] == 'beta' and r['level'] == 0.05}
        log(f'[{i}/{len(items)}] {base} ({grp})  β=0.05：' + '  '.join(
            f'{sc} K5 r/p={q[(sc, 5)]["r_over_p"]:.2f} D={q[(sc, 5)]["D"]:.2f} 预测={q[(sc, 5)]["pred_exact"]:.2f}'
            for sc in SCORES) + f'  ({time.time() - t0:.0f} s)')
    return rows


def regress(rows, D, a, log):
    out = []
    for name, Ks in (('krule_v12.csv', (5,)), ('krule_K3510.csv', (3, 5, 10))):
        p = os.path.join(a.krule_dir or os.path.join(D, 're_ii', 'results_v32'), name)
        if not os.path.exists(p):
            out.append(f'  {name}：找不到，跳过')
            continue
        ref = {(r['star'], r['mode'], int(rm._f(r['K'])), round(rm._f(r['level']), 12)): r for r in rm._read_csv(p)}
        ok = bad = miss = 0
        for r in rows:
            if r['score'] != 'qr' or r['K'] not in Ks:
                continue
            rr = ref.get((r['key'], r['mode'], r['K'], round(r['level'], 12)))
            if rr is None:
                miss += 1
                continue
            d0 = rm._f(rr['D_clean'])
            same = rm.close(r['D'], d0) or (r['D'] == 0 and d0 == 0) or (np.isnan(r['D']) and np.isnan(d0))
            ok += same
            bad += not same
            if not same and bad <= 3:
                log(f'  [回归 DIFF] {name} {r["star"]} K={r["K"]} {r["mode"]} {r["level"]}: {r["D"]:.6g} vs {d0:.6g}')
        out.append(f'  {name}：qr 的 D_clean {ok} 一致 / {bad} 不一致 / {miss} 无对照')
    return out


def pooled(rows):
    out = []
    for sc in SCORES:
        for K in KS:
            for mode, lev in LEVELS:
                for g in GROUPS:
                    sel = [r for r in rows if r['score'] == sc and r['K'] == K and r['mode'] == mode
                           and r['level'] == lev and not r['pooled_excluded'] and (g == 'all' or r['group'] == g)]
                    if not sel:
                        continue
                    nw = sum(r['nwin'] for r in sel)
                    cov = sum(r['cov'] for r in sel)
                    nev = sum(r['n_event'] for r in sel)
                    nom = sum(r['nominal_iid'] * r['nwin'] for r in sel) / nw
                    wmk = sum(r['w_markov'] * r['nwin'] for r in sel) / nw
                    wii = sum(r['w_iid_mk'] * r['nwin'] for r in sel) / nw
                    lo, up = kr.win_rate_bounds(nev, cov, nw, M)
                    out.append(dict(score=sc, K=K, mode=mode, level=lev, group=g, n_stars=len(sel),
                                    p_hat_med=med([r['p_hat'] for r in sel]), r_hat_med=med([r['r_hat'] for r in sel]),
                                    r_over_p_med=med([r['r_over_p'] for r in sel]),
                                    pred_approx_med=med([r['pred_approx'] for r in sel]),
                                    pred_exact=(wmk / wii if wii > 0 else float('nan')),
                                    nominal_iid=nom, dlv=cov / nw, n_event=nev, lo95=lo, up95=up,
                                    D=(cov / nw / nom if nom > 0 else float('nan')),
                                    D_lo=(lo / nom if nom > 0 else float('nan')), D_up=(up / nom if nom > 0 else float('nan')),
                                    resolvable_3q=(nom >= 3.0 * M / sum(r['N_clean'] for r in sel)),
                                    verdict=kr.verdict(lo, up, nom, nw)))
    return out


# ---------------------------------------------------------------------------
# Part B: Gaussian AR(1) series (Figure 7)
# ---------------------------------------------------------------------------
def ar1(n, rho, rng):
    e = rng.standard_normal(n)
    if rho == 0.0:
        return e
    try:
        from scipy.signal import lfilter
        x = lfilter([np.sqrt(1.0 - rho * rho)], [1.0, -rho], e)
        x[0] = e[0]
        return x
    except Exception:
        x = np.empty(n)
        x[0] = e[0]
        c = np.sqrt(1.0 - rho * rho)
        for t in range(1, n):
            x[t] = rho * x[t - 1] + c * e[t]
        return x


def part_b(a, log):
    rows = []
    rng = np.random.default_rng([a.seed, 20261024])
    for rho in RHOS:
        for K in KS:
            for md, p_t in (('b05', p_target(0.05, K)), ('p95', 0.05)):
                Ds, Ps, rs, nev = [], [], [], 0
                for _ in range(a.reps):
                    x = ar1(a.n_ar, rho, rng)
                    res = eval_series(x, p_t, K)
                    Ds.append(res['D'])
                    Ps.append(res['pred_exact'])
                    rs.append(res['r_over_p'])
                    nev += res['n_event']
                Ds = np.array(Ds)
                rows.append(dict(rho=rho, K=K, mode=md, p_target=p_t, reps=a.reps, N=a.n_ar,
                                 D_mean=float(np.nanmean(Ds)), D_q025=float(np.nanquantile(Ds, 0.025)),
                                 D_q975=float(np.nanquantile(Ds, 0.975)), pred_exact_mean=float(np.nanmean(Ps)),
                                 r_over_p_mean=float(np.nanmean(rs)), n_event_total=nev,
                                 resolvable_3q=(kr.nominal_iid(p_t, K, M) >= 3.0 * M / a.n_ar)))
        log(f'  AR(1) ρ={rho}: ' + '  '.join(f'K={r["K"]} D={r["D_mean"]:.2f}/预测 {r["pred_exact_mean"]:.2f}'
                                            for r in rows if r['rho'] == rho and r['mode'] == 'b05'))
    return rows


# ---------------------------------------------------------------------------
# Diagnostic plot
# ---------------------------------------------------------------------------
def plot(rows_b, rows_a, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for K, c in zip(KS, ('#1f77b4', '#d62728', '#2ca02c')):
        sel = sorted([r for r in rows_b if r['K'] == K and r['mode'] == 'b05'], key=lambda r: r['rho'])
        x = [r['rho'] for r in sel]
        y = np.array([r['D_mean'] for r in sel])
        lo = np.clip(y - np.array([r['D_q025'] for r in sel]), 0.0, None)
        hi = np.clip(np.array([r['D_q975'] for r in sel]) - y, 0.0, None)
        ax[0].errorbar(x, y, yerr=[lo, hi], fmt='o-', color=c, capsize=3, label=f'K={K} simulated')
        ax[0].plot(x, [r['pred_exact_mean'] for r in sel], '--', color=c, alpha=0.7, label=f'K={K} Markov pred.')
    ax[0].axhline(1.0, color='k', lw=0.8)
    ax[0].axhline(20.0, color='gray', ls=':', lw=1, label='cap = 1/β')
    ax[0].set_yscale('log')
    ax[0].set_xlabel('AR(1) coefficient ρ')
    ax[0].set_ylabel('D = actual / nominal (iid)')
    ax[0].set_title('Gaussian AR(1), γ set for β=0.05 under iid')
    ax[0].legend(fontsize=8)
    ax[0].grid(True, ls=':', alpha=0.5)
    mk = {'qr': 'o', 'cn0_fold': 's', 'wscm_fold': '^'}
    for sc in SCORES:
        sel = [r for r in rows_a if r['score'] == sc and r['K'] == 5 and r['mode'] == 'beta' and r['level'] == 0.05
               and not r['pooled_excluded'] and r['D'] > 0 and r['pred_exact'] > 0]
        ax[1].scatter([r['pred_exact'] for r in sel], [r['D'] for r in sel], marker=mk[sc], alpha=0.7, label=sc)
    lim = [0.3, 300]
    ax[1].plot(lim, lim, 'k--', lw=0.8)
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlabel('Markov-predicted D (from r̂, p̂)')
    ax[1].set_ylabel('actual D on clean segment')
    ax[1].set_title('Real stars, K=5, β=0.05 (ds8 excluded; D=0 not shown)')
    ax[1].legend(fontsize=8)
    ax[1].grid(True, ls=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
A_FIELDS = ['star', 'scenario', 'group', 'score', 'K', 'mode', 'level', 'p_target', 'N_clean', 'pooled_excluded',
            'gamma', 'p_hat', 'r_hat', 'r_over_p', 'pred_approx', 'w_markov', 'w_iid_mk', 'pred_exact', 'nominal_iid',
            'dlv', 'n_event', 'cov', 'nwin', 'lo95', 'up95', 'D', 'resolvable_3q', 'verdict']
P_FIELDS = ['score', 'K', 'mode', 'level', 'group', 'n_stars', 'p_hat_med', 'r_hat_med', 'r_over_p_med',
            'pred_approx_med', 'pred_exact', 'nominal_iid', 'dlv', 'n_event', 'lo95', 'up95', 'D', 'D_lo', 'D_up',
            'resolvable_3q', 'verdict']
B_FIELDS = ['rho', 'K', 'mode', 'p_target', 'reps', 'N', 'D_mean', 'D_q025', 'D_q975', 'pred_exact_mean',
            'r_over_p_mean', 'n_event_total', 'resolvable_3q']


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', dest='npz_dir', default='.')
    ap.add_argument('--stars', nargs='+', default=None)
    ap.add_argument('--reps', type=int, default=20)
    ap.add_argument('--n-ar', dest='n_ar', type=int, default=80000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--guard-s', dest='guard_s', type=float, default=0.1)
    ap.add_argument('--krule-dir', default=None, help=r'默认 <dir>\re_ii\results_v32')
    ap.add_argument('--skip-real', action='store_true', help='只跑 AR(1)')
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
    B('[version] c2_e0_mech.py v1.0（2026-10-24）')
    B(f'[cfg] m={M}  K={list(KS)}  分数={list(SCORES)}  AR(1): ρ={list(RHOS)} N={a.n_ar} reps={a.reps} seed={a.seed}'
      f'  numba={"ON" if rm.HAVE_NUMBA else "OFF"}')
    rows_a, pool = [], []
    if not a.skip_real:
        items = rm.qr_items(D, a.stars)
        B(f'[cfg] {len(items)} 颗星' + ('' if a.stars or len(items) == 21 else '  [WARN] ≠ 21'))
        rows_a = part_a(items, a, log)
        B('== 回归（qr 的 D_clean 对 krule 旧结果）==')
        for s in regress(rows_a, D, a, log):
            B(s)
        pool = pooled(rows_a)
        fc.save_csv(prefix + '_star.csv', rows_a, A_FIELDS)
        fc.save_csv(prefix + '_pooled.csv', pool, P_FIELDS)
    log('\n---- AR(1) 仿真 ----')
    rows_b = part_b(a, log)
    fc.save_csv(prefix + '_ar1.csv', rows_b, B_FIELDS)
    try:
        plot(rows_b, rows_a, prefix + '_fig.png')
        figmsg = f'{prefix}_fig.png'
    except Exception as ex:
        figmsg = f'出图失败：{ex}'

    if pool:
        B('\n== A 真实数据（合并不含 ds8）：D 实测 vs 马尔可夫精确预测；r/p 与 (r/p)^(K−1) 为跨星中位数 ==')
        B('score     K  档位     | p̂中位  r̂中位  r/p  | 近似(r/p)^(K-1) | 精确预测 | D实测 [95%界]          事件 判 可分辨')
        pi = {(p['score'], p['K'], p['mode'], p['level'], p['group']): p for p in pool}
        for sc in SCORES:
            for K in KS:
                for mode, lev in (('beta', 0.05), ('alpha', 0.05)):
                    p = pi.get((sc, K, mode, lev, 'all'))
                    if not p:
                        continue
                    tag = 'β=0.05' if mode == 'beta' else 'p95   '
                    B(f'{sc:9s} {K:2d} {tag} | {p["p_hat_med"]:.4f} {p["r_hat_med"]:.3f} {p["r_over_p_med"]:5.2f} | '
                      f'{p["pred_approx_med"]:15.3g} | {p["pred_exact"]:8.3g} | {p["D"]:7.3g} '
                      f'[{p["D_lo"]:.2g}, {p["D_up"]:.2g}] {p["n_event"]:5d} {p["verdict"][:4]} '
                      f'{"是" if p["resolvable_3q"] else "否"}')
        B('  分组（K=5 β=0.05）：' + '  '.join(
            f'{sc}:' + '/'.join(f'{pi[(sc, 5, "beta", 0.05, g)]["D"]:.2g}' if (sc, 5, 'beta', 0.05, g) in pi else '-'
                                for g in ('static', 'dynamic', 'complex')) for sc in SCORES) + '（静/动/复 的 D 实测）')
    B('\n== B AR(1)（γ 按独立假设反推到 β=0.05；D 为 reps 次均值 [2.5%, 97.5%]）==')
    B('ρ     | K=3: D实测 [范围] / 预测         | K=5                              | K=10')
    for rho in RHOS:
        cells = []
        for K in KS:
            r = next(x for x in rows_b if x['rho'] == rho and x['K'] == K and x['mode'] == 'b05')
            cells.append(f'{r["D_mean"]:6.3g} [{r["D_q025"]:.2g},{r["D_q975"]:.2g}] / {r["pred_exact_mean"]:6.3g}')
        B(f'{rho:4.1f}  | ' + ' | '.join(cells))
    B('注：ρ=0 时 D 应≈1（算法自检）。D 有上限 = 1/参照值（β=0.05 → 20，窗口误报不会超过 1）：顶到 20 的格子是封顶，不是"恰好 20 倍"。')
    B('注：真实星 r/p 落在 AR(1) 的哪个 ρ 区间，就对应哪一档超支；实测 D ≫ 精确预测 → 一阶持续性解释不完，还有长程记忆。')
    B(f'[saved] {prefix}_star.csv / _pooled.csv / _ar1.csv / _log.txt / _brief.txt；图 {figmsg}')
    B(f'[done] 用时 {(time.time() - t0) / 60:.1f} 分钟')
    with open(prefix + '_brief.txt', 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(brief) + '\n')
    log.close()
