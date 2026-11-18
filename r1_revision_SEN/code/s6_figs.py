# -*- coding: utf-8 -*-
"""Redraws Figures 2, 10 and 11 (files Figure2, Figure9, Figure10) and Figure A3.

Uses fig_style.py and figs_journal.py with the revised onsets and delays.
Run: python r1_revision_SEN/code/s6_figs.py (from the repository root).
"""
import os
import sys
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 're_ii', 'figs_journal', 'code'))
import fig_style as S            # noqa: E402
import figs_journal as FJ        # noqa: E402
from matplotlib.lines import Line2D   # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import matplotlib.ticker as mtick     # noqa: E402

FJ.D = ROOT
OUT = os.path.join(ROOT, 'r1_revision_SEN', 'figs')
S1 = ('r1_revision_SEN', 's1')
S2 = ('r1_revision_SEN', 's2')
SCORES = ('qr', 'ae', 'tcn', 'cn0', 'wscm')
A3NAME = {'ae': 'FigureA3a', 'tcn': 'FigureA3b', 'cn0': 'FigureA3c', 'wscm': 'FigureA3d'}
SEG = tuple((n, c, cal, ho, act, (118.9 if n == 'ds3' else on)) for n, c, cal, ho, act, on in FJ.SEG)
log = FJ.log

# ---------------------------------------------------------------------------
# Axis and label helpers
# ---------------------------------------------------------------------------
def nice_log(axis, lo, hi):
    """Log-axis ticks from a 1-2-3-5, 1-2-5, 1-3 or 1 sequence (at most six)."""
    span = np.log10(hi / lo)
    mult = (1, 2, 3, 5) if span <= 1.3 else (1, 2, 5) if span <= 2.3 else (1, 3) if span <= 3.5 else (1,)
    c = [round(m * 10.0 ** e, 10) for e in range(-5, 4) for m in mult]
    c = [v for v in c if lo <= v <= hi]
    while len(c) > 6:
        c = c[::2]
    axis.set_major_locator(mtick.FixedLocator(c))
    axis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f'{v:g}'))
    axis.set_minor_formatter(mtick.NullFormatter())


def place_labels(fig, ax, lines, labels):
    """Places each beta label at the corner that overlaps least with lines, markers and other labels.
    """
    fig.canvas.draw()
    T, bb, px = ax.transData, ax.bbox, fig.dpi / 72.0
    obs = []
    for xs, ys in lines:
        d = T.transform(np.column_stack([xs, ys]))
        for a, b in zip(d[:-1], d[1:]):
            obs += list(a + (b - a) * t for t in np.linspace(0, 1, 40))
        obs += list(d)
    obs += [(bb.x1 - u * px, bb.y1 - v * px) for u in np.linspace(2, 40, 12) for v in np.linspace(2, 11, 4)]
    for x, y, txt, pref in labels:
        X, Y = T.transform((x, y))
        w, h = (4.2 * len(txt) - 6) * px, 7.5 * px
        best = None
        for dx in (pref, -pref):
            for dy in (1, -1):
                x0 = X + 4 * px if dx > 0 else X - 4 * px - w
                y0 = Y + 4 * px if dy > 0 else Y - 4 * px - h
                sc = sum(1 for ox, oy in obs if x0 - 2 * px <= ox <= x0 + w + 2 * px and y0 - 2 * px <= oy <= y0 + h + 2 * px)
                if x0 < bb.x0 or x0 + w > bb.x1 or y0 < bb.y0 or y0 + h > bb.y1:
                    sc += 1000
                if best is None or sc < best[0]:
                    best = (sc, dx, dy, x0, y0)
        sc, dx, dy, x0, y0 = best
        ax.annotate(txt, (x, y), fontsize=7, xytext=(4 * dx, 4 * dy), textcoords='offset points',
                    path_effects=S.STROKE, ha='left' if dx > 0 else 'right', va='bottom' if dy > 0 else 'top')
        obs += [(x0 + u * w, y0 + v * h) for u in np.linspace(0, 1, 8) for v in np.linspace(0, 1, 3)]


# ---------------------------------------------------------------------------
# Figure 2: segments and reference instants
# ---------------------------------------------------------------------------
def fig2():
    fig, ax = S.new_fig('text', 0.40)
    n = len(SEG)
    for i, (name, cl, cal, ho, act, on) in enumerate(SEG):
        y = n - 1 - i
        dyn = cl == 'dynamic'
        ax.barh(y, cal[1] - cal[0], left=cal[0], height=0.5, color='#CCCCCC', lw=0)
        if ho:
            ax.barh(y, ho[1] - ho[0], left=ho[0], height=0.5, color=S.CLASS[cl], lw=0,
                    hatch='////' if name == 'ds8' else None, edgecolor='white')
        if on > act:
            ax.barh(y, on - act, left=act, height=0.5, facecolor='none', edgecolor='0.35', hatch='xxxx', lw=0)
        ax.plot([act, act], [y - 0.32, y + 0.32], color='k', lw=1.1, ls=(0, (2, 1.2)) if dyn else '-')
        ax.plot(on, y + 0.42, marker='v', color=S.OI['verm'], ms=4, ls='none')
        if dyn:
            ax.plot(58.0, y, marker='|', color='0.15', ms=8, mew=1.0)
    ax.set_yticks(range(n))
    ax.set_yticklabels([s[0] for s in SEG][::-1])
    ax.set_xlim(0, 125)
    ax.set_ylim(-0.6, n - 0.2)
    ax.set_xlabel('Time in recording (s)')
    h = [Patch(color='#CCCCCC', label='Calibration')] + \
        [Patch(color=S.CLASS[c], label=f'Held-out ({c})') for c in FJ.GROUPS] + \
        [Patch(facecolor='none', edgecolor='0.35', hatch='xxxx', label='Activation or event to onset'),
         Line2D([], [], color='k', lw=1.1, label='Activation signature'),
         Line2D([], [], color='k', lw=1.1, ls=(0, (2, 1.2)), label='97 s common-mode event'),
         Line2D([], [], color=S.OI['verm'], marker='v', ls='none', label='Onset'),
         Line2D([], [], color='0.15', marker='|', ls='none', ms=8, label='58 s common-mode event')]
    ax.legend(handles=h, loc='center left', bbox_to_anchor=(1.01, 0.5))
    log('图2 onset: ' + ', '.join(f'{s[0]}={s[5]}' for s in SEG))
    S.save(fig, OUT, 'Figure2')


# ---------------------------------------------------------------------------
# Figures 10 and A3: median delay against delivered rate
# ---------------------------------------------------------------------------
def fig_delay(k):
    P = FJ.rd(*S1, f'rev_{k}_pooled.csv')
    fig, axes = S.new_fig('full', 0.32, 1, 3)
    for ax, g in zip(axes, FJ.GROUPS):
        ymax, lines, labs = [], [], []
        for rule in ('cusum', 'k_blk', 'k_iid'):
            rr = sorted([r for r in P if r['rule'] == rule and r['group'] == g and r['beta'] != ''
                         and FJ.fl(r['beta']) >= 0.02 - 1e-9 and FJ.fl(r['cal_frac']) > 0],
                        key=lambda r: FJ.fl(r['beta']))
            x, y = [FJ.fl(r['cal_frac']) for r in rr], [FJ.fl(r['delay_onset_med']) for r in rr]
            ymax += y
            lines.append((x, y))
            ax.plot(x, y, color=S.RULE[rule]['color'], marker=S.RULE[rule]['marker'], lw=1.0, ms=3.5,
                    label=S.RULE[rule]['label'])
            for r in rr:
                if rule == 'cusum' and FJ.fl(r['beta']) in (0.1, 0.02):
                    labs.append((FJ.fl(r['cal_frac']), FJ.fl(r['delay_onset_med']), f'$\\beta$={FJ.fl(r["beta"]):g}',
                                 1 if FJ.fl(r['beta']) == 0.1 else -1))
            log(f'图10/A3 {k} {g} {rule}: ' + ', '.join(f'{FJ.fl(r["beta"]):g}:{FJ.fl(r["delay_onset_med"]):.3f}'
                                                         for r in rr))
        p = [r for r in P if r['rule'] == 'k_p95' and r['group'] == g and FJ.fl(r['cal_frac']) > 0]
        if p:
            ax.plot(FJ.fl(p[0]['cal_frac']), FJ.fl(p[0]['delay_onset_med']), color=S.RULE['k_p95']['color'],
                    marker='*', ms=7, ls='none', label=S.RULE['k_p95']['label'])
            ymax.append(FJ.fl(p[0]['delay_onset_med']))
            lines.append(([FJ.fl(p[0]['cal_frac'])], [FJ.fl(p[0]['delay_onset_med'])]))
        ax.set_xscale('log')
        ok = [v for v in ymax if np.isfinite(v) and v > 0]
        if g == 'static':
            ax.set_ylim(0, max(ok) * 1.2 if ok else 1)
        else:
            ax.set_yscale('log')
            lo_, hi_ = ax.get_ylim()
            ax.set_ylim(lo_ / 1.5, hi_ * 1.7)
            nice_log(ax.yaxis, *ax.get_ylim())
        nice_log(ax.xaxis, *ax.get_xlim())
        ax.text(0.97, 0.97, S.CLASS_LABEL[g], transform=ax.transAxes, ha='right', va='top', fontsize=8)
        ax.set_xlabel('Delivered rate per 1-s window\n(calibration segments)')
        ax.set_ylabel('Median delay from onset (s)')
        place_labels(fig, ax, lines, labs)
    hh, ll = axes[0].get_legend_handles_labels()
    try:
        fig.legend(hh, ll, loc='outside upper center', ncol=4)
    except Exception:
        axes[0].legend(hh, ll, loc='lower left', fontsize=7)
    S.panel_labels(axes)
    S.save(fig, OUT, 'Figure9' if k == 'qr' else A3NAME[k])


# ---------------------------------------------------------------------------
# Figure 11: per-instant detection rate against CUSUM delay
# ---------------------------------------------------------------------------
def fig11():
    fig, axes = S.new_fig('text', 0.52, 1, 2, sharey=False)
    for ax, g in zip(axes, ('static', 'dynamic')):
        allx, ally = [], []
        for k in SCORES:
            dr = {FJ.ns(r['star']): FJ.fl(r['post_dr@5e-02']) for r in FJ.rd(*S2, f's2_{k}_postdr.csv')}
            xs, ys = [], []
            for r in FJ.rd(*S1, f'rev_{k}_star.csv'):
                if r['rule'] != 'cusum' or abs(FJ.fl(r['beta']) - 0.02) > 1e-9 or r['group'] != g:
                    continue
                d, o = FJ.fl(r['delay_onset_s']), dr.get(FJ.ns(r['star']), np.nan)
                if np.isfinite(o) and np.isfinite(d) and d > 0:
                    xs.append(100 * o)
                    ys.append(d)
            sp = S.SCORE[k]
            ax.plot(xs, ys, ls='none', marker=sp['marker'], color=sp['color'], label=sp['label'], alpha=0.9)
            allx += xs
            ally += ys
            log(f'图11 {g} {k}: n={len(xs)} DR中位 {np.median(xs) if xs else np.nan:.1f}% '
                f'时延中位 {np.median(ys) if ys else np.nan:.3f} s')
        rho, n = FJ.spearman(allx, ally)
        ax.text(0.03, 0.97, f'{S.CLASS_LABEL[g]}: Spearman $\\rho_s$ = {rho:.2f} (n = {n})', transform=ax.transAxes,
                ha='left', va='top', fontsize=8)
        log(f'图11 {g}: Spearman {rho:.3f} n={n}（S2 brief：静态 +0.102 n=35；s2b：动态 +0.397 n=48）')
        ax.set_yscale('log')
        ax.set_ylim(min(ally) / 1.6, max(ally) * 4.0)
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
    S.save(fig, OUT, 'Figure10')


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    log('[version] s6_figs.py v1.1（2026-11-18）')
    jobs = [('Figure2', fig2), ('Figure10', fig11)] + [(f'delay_{k}', (lambda k=k: fig_delay(k))) for k in SCORES]
    for name, fn in jobs:
        try:
            fn()
            log(f'[ok] {name}')
        except Exception:
            log(f'[FAIL] {name}\n' + traceback.format_exc())
    with open(os.path.join(OUT, 's6_figs_log.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(FJ.LOG) + '\n')
    print(f'[saved] {OUT}')
