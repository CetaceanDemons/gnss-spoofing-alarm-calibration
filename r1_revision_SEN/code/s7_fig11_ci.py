# -*- coding: utf-8 -*-
"""Bootstrap confidence intervals for the Spearman correlations of Figure 11 (file Figure10.png).

Figure 11: CUSUM delay at beta = 0.02 from the onset against the per-instant detection rate
at alpha = 0.05, per satellite and score, static (a) and dynamic (b) scenarios.
Paper: Section 4.6, Figure 11 and Table 9.

The pairs are taken exactly as in s6_figs.fig11(), and the point estimates must reproduce
rho_S = 0.102 (n = 35) and 0.397 (n = 48) before anything else is computed.
Two resampling schemes, percentile 95% intervals:
  cluster - satellites drawn with replacement, each with all of its score pairs (primary:
            the five scores of one satellite see the same change and are not independent);
  pair    - satellite-score pairs drawn with replacement (for comparison only).
Also reports the pairs that fig11() leaves out (delay <= 0 or missing) and the correlation
with the zero-delay pairs put back, and redraws Figure 11 with the interval in each panel.

Run from the repository root (J:\\Database\\TEXBAT\\processed_Mao):
    python r1_revision_SEN/code/s7_fig11_ci.py
Outputs: r1_revision_SEN/s7/ (csv + brief) and r1_revision_SEN/figs_r2/Figure10.{png,pdf}.
"""
import argparse
import csv
import os
import sys

import numpy as np

VERSION = 's7_fig11_ci.py v1.1 (2026-11-23)'
HERE = os.path.dirname(os.path.abspath(__file__))
DEF_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))

SCORES = ('qr', 'ae', 'tcn', 'cn0', 'wscm')
GROUPS = ('static', 'dynamic')
# Published values (s6_figs log, S2 brief, s2b brief): rho_S and n per panel.
EXPECT = {'static': (0.102, 35), 'dynamic': (0.397, 48)}
TOL_RHO = 5e-4


# ---------------------------------------------------------------------------
# Helpers (same definitions as figs_journal.py)
# ---------------------------------------------------------------------------
def ns(s):
    """Satellite key: strips the 'pertap_' prefix and the '_dyn' suffix."""
    s = str(s).strip()
    if s.startswith('pertap_'):
        s = s[7:]
    return s[:-4] if s.endswith('_dyn') else s


def fl(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


def rankavg(x):
    """Average ranks (ties share the mean rank), as in figs_journal.rankavg."""
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
    """Spearman correlation (Pearson correlation of average ranks); nan if degenerate."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 4 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float('nan')
    return float(np.corrcoef(rankavg(x), rankavg(y))[0, 1])


def read(path):
    if not os.path.exists(path):
        raise SystemExit(f'[STOP] 找不到 {path}')
    with open(path, encoding='utf-8-sig', newline='') as fh:
        return list(csv.DictReader(fh))


def write(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Pairs of Figure 11, selected as in s6_figs.fig11()
# ---------------------------------------------------------------------------
def collect(root):
    s1 = os.path.join(root, 'r1_revision_SEN', 's1')
    s2 = os.path.join(root, 'r1_revision_SEN', 's2')
    kept, left = [], []
    for k in SCORES:
        dr = {ns(r['star']): fl(r['post_dr@5e-02']) for r in read(os.path.join(s2, f's2_{k}_postdr.csv'))}
        for r in read(os.path.join(s1, f'rev_{k}_star.csv')):
            if r['rule'] != 'cusum' or abs(fl(r['beta']) - 0.02) > 1e-9 or r['group'] not in GROUPS:
                continue
            sat = ns(r['star'])
            d, o = fl(r['delay_onset_s']), dr.get(sat, float('nan'))
            rec = dict(group=r['group'], score=k, sat=sat, dr05_pct=100 * o, delay_s=d)
            if np.isfinite(o) and np.isfinite(d) and d > 0:
                kept.append(rec)
            else:
                left.append(rec)
    return kept, left


# ---------------------------------------------------------------------------
# Bootstrap of the Spearman correlation
# ---------------------------------------------------------------------------
def boot(pairs, scheme, nboot, rng):
    x = np.array([p['dr05_pct'] for p in pairs])
    y = np.array([p['delay_s'] for p in pairs])
    sat = np.array([p['sat'] for p in pairs])
    sats = np.unique(sat)
    members = [np.flatnonzero(sat == s) for s in sats]
    out = np.full(nboot, np.nan)
    for b in range(nboot):
        if scheme == 'cluster':
            pick = rng.integers(0, len(sats), len(sats))
            ii = np.concatenate([members[j] for j in pick])
        else:
            ii = rng.integers(0, len(x), len(x))
        out[b] = spearman(x[ii], y[ii])
    ok = out[np.isfinite(out)]
    return ok, int(nboot - len(ok)), len(sats)


def summarise(group, scheme, pairs, rho, nboot, rng):
    v, ndeg, nsat = boot(pairs, scheme, nboot, rng)
    return dict(group=group, scheme=scheme, rho_S=rho, n_pairs=len(pairs), n_sat=nsat, nboot=nboot,
                n_degenerate=ndeg, ci95_lo=float(np.percentile(v, 2.5)), ci95_hi=float(np.percentile(v, 97.5)),
                ci90_lo=float(np.percentile(v, 5.0)), ci90_hi=float(np.percentile(v, 95.0)),
                boot_median=float(np.median(v)), share_le_0=float(np.mean(v <= 0)),
                share_le_m05=float(np.mean(v <= -0.5)), share_ge_05=float(np.mean(v >= 0.5)))


# ---------------------------------------------------------------------------
# Figure 11 with the interval (layout of s6_figs.fig11)
# ---------------------------------------------------------------------------
def mtxt(v):
    s = f'{v:.2f}'
    return s.replace('-', '\u2212')


def redraw(root, pairs_by_group, ci, out_dir, top_factor):
    sys.path.insert(0, os.path.join(root, 're_ii', 'figs_journal', 'code'))
    import fig_style as S              # noqa: E402
    import matplotlib.ticker as mtick  # noqa: E402

    def nice_log(axis, lo, hi):
        span = np.log10(hi / lo)
        mult = (1, 2, 3, 5) if span <= 1.3 else (1, 2, 5) if span <= 2.3 else (1, 3) if span <= 3.5 else (1,)
        c = [round(m * 10.0 ** e, 10) for e in range(-5, 4) for m in mult]
        c = [v for v in c if lo <= v <= hi]
        while len(c) > 6:
            c = c[::2]
        axis.set_major_locator(mtick.FixedLocator(c))
        axis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f'{v:g}'))
        axis.set_minor_formatter(mtick.NullFormatter())

    fig, axes = S.new_fig('text', 0.52, 1, 2, sharey=False)
    for ax, g in zip(axes, GROUPS):
        allx, ally = [], []
        for k in SCORES:
            pp = [p for p in pairs_by_group[g] if p['score'] == k]
            xs, ys = [p['dr05_pct'] for p in pp], [p['delay_s'] for p in pp]
            sp = S.SCORE[k]
            ax.plot(xs, ys, ls='none', marker=sp['marker'], color=sp['color'], label=sp['label'], alpha=0.9)
            allx += xs
            ally += ys
        c = ci[g]
        ax.text(0.03, 0.97, f'{S.CLASS_LABEL[g]}: $\\rho_S$ = {mtxt(c["rho_S"])} (n = {c["n_pairs"]})\n'
                f'95% CI [{mtxt(c["ci95_lo"])}, {mtxt(c["ci95_hi"])}]',
                transform=ax.transAxes, ha='left', va='top', fontsize=8, linespacing=1.25)
        ax.set_yscale('log')
        ax.set_ylim(min(ally) / 1.6, max(ally) * top_factor)
        nice_log(ax.yaxis, *ax.get_ylim())
        ax.set_xlim(0, 105)
        ax.set_xlabel('Per-instant detection rate\nat $\\alpha$ = 0.05 (%)')
        ax.set_ylabel('CUSUM delay at $\\beta$ = 0.02,\nfrom onset (s)')
    hh, ll = axes[0].get_legend_handles_labels()
    try:
        fig.legend(hh, ll, loc='outside upper center', ncol=5, fontsize=7)
    except Exception:
        axes[0].legend(hh, ll, loc='lower left', fontsize=7)
    S.panel_labels(axes)
    S.save(fig, out_dir, 'Figure10')


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--root', default=DEF_ROOT, help='processed_Mao 根目录（默认为脚本上两级）')
    ap.add_argument('--nboot', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=20261123)
    ap.add_argument('--top-factor', type=float, default=5.0, help='y 轴上限 = 最大时延 × 此值（R1 原图为 4）')
    ap.add_argument('--skip-check', action='store_true', help='不核对已发表的 ρ_S（仅冒烟测试用）')
    ap.add_argument('--no-fig', action='store_true')
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    out = os.path.join(root, 'r1_revision_SEN', 's7')
    figdir = os.path.join(root, 'r1_revision_SEN', 'figs_r2')
    os.makedirs(out, exist_ok=True)
    B = [f'[{VERSION}]  root = {root}  nboot = {a.nboot}  seed = {a.seed}']

    kept, left = collect(root)
    by = {g: [p for p in kept if p['group'] == g] for g in GROUPS}

    # 1. Point estimates must reproduce the published values.
    B.append('== 1. 点估计核对（应与稿件一致）==')
    rho = {}
    bad = []
    for g in GROUPS:
        x = [p['dr05_pct'] for p in by[g]]
        y = [p['delay_s'] for p in by[g]]
        rho[g] = spearman(x, y)
        er, en = EXPECT[g]
        ok = abs(rho[g] - er) <= TOL_RHO and len(x) == en
        B.append(f'  {g:8s} ρ_S = {rho[g]:+.4f} (n = {len(x)}, 卫星 {len(set(p["sat"] for p in by[g]))})  '
                 f'稿件 {er:+.3f} (n = {en})  {"一致" if ok else "不一致"}')
        if not ok:
            bad.append(g)
    if bad and not a.skip_check:
        B.append('[STOP] 点估计与稿件不一致，未继续。请把本文件发回。')
        open(os.path.join(out, 's7_brief.txt'), 'w', encoding='utf-8').write('\n'.join(B) + '\n')
        print('\n'.join(B))
        sys.exit(1)

    # 2. Bootstrap intervals.
    B.append(f'== 2. 自助法 95% 区间（百分位；{a.nboot} 次）==')
    rng = np.random.default_rng(a.seed)
    rows, ci = [], {}
    for g in GROUPS:
        for scheme in ('cluster', 'pair'):
            r = summarise(g, scheme, by[g], rho[g], a.nboot, rng)
            rows.append(r)
            if scheme == 'cluster':
                ci[g] = r
            B.append(f'  {g:8s} {scheme:7s} ρ_S = {r["rho_S"]:+.3f}  95% [{r["ci95_lo"]:+.3f}, {r["ci95_hi"]:+.3f}]  '
                     f'90% [{r["ci90_lo"]:+.3f}, {r["ci90_hi"]:+.3f}]  P(ρ*≤0) = {r["share_le_0"]:.3f}  '
                     f'P(ρ*≤−0.5) = {r["share_le_m05"]:.3f}  P(ρ*≥0.5) = {r["share_ge_05"]:.3f}  '
                     f'退化 {r["n_degenerate"]}  (卫星 {r["n_sat"]})')
    write(os.path.join(out, 's7_fig11_ci.csv'), rows)
    write(os.path.join(out, 's7_fig11_pairs.csv'), kept)

    # 3. Pairs left out by fig11() and the correlation with zero delays put back.
    B.append('== 3. fig11() 未纳入的对（时延 ≤ 0 或缺失）==')
    for p in left:
        B.append(f'  {p["group"]:8s} {p["score"]:5s} {p["sat"]:14s} DR = {p["dr05_pct"]:.2f}%  delay = {p["delay_s"]}')
    write(os.path.join(out, 's7_fig11_left_out.csv'), left)
    for g in GROUPS:
        z = [p for p in left if p['group'] == g and np.isfinite(p['delay_s']) and p['delay_s'] == 0
             and np.isfinite(p['dr05_pct'])]
        if z:
            allp = by[g] + z
            rz = spearman([p['dr05_pct'] for p in allp], [p['delay_s'] for p in allp])
            rr = summarise(g, 'cluster+zero', allp, rz, a.nboot, rng)
            B.append(f'  {g:8s} 纳入 {len(z)} 个零时延对：ρ_S = {rz:+.3f} (n = {len(allp)})  '
                     f'整簇 95% [{rr["ci95_lo"]:+.3f}, {rr["ci95_hi"]:+.3f}]')

    if not a.no_fig:
        redraw(root, by, ci, figdir, a.top_factor)
        B.append(f'[fig] {os.path.join(figdir, "Figure10.png")}（及 .pdf）')
    B.append(f'[saved] {out}')
    open(os.path.join(out, 's7_brief.txt'), 'w', encoding='utf-8').write('\n'.join(B) + '\n')
    print('\n'.join(B))


if __name__ == '__main__':
    main()
