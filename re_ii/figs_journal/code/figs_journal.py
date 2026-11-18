"""Figures 1, 3-8, 9 (file Figure11.png) and A2 of the paper, from the result files.

Figures 2, 10 and 11 and Figure A3 are redrawn by r1_revision_SEN/code/s6_figs.py.
Run from the repository root (the working folder that also holds the data folders); see --help.
"""
import os, sys, csv, argparse, traceback
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig_style as S
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.ticker as mtick

D = OUT = '.'
LOG = []
GROUPS = ('static', 'dynamic', 'complex')
SC_CLASS = {'ds3': 'static', 'ds4_r2': 'static', 'ds5_r2': 'dynamic', 'ds6_r2': 'dynamic', 'ds7': 'complex',
            'ds8': 'complex'}
CL_ORDER = {'static': 0, 'dynamic': 1, 'complex': 2}
SC_PATHS = (('qr', ('re_ii', 'results_rev', 'rev_qr_pooled.csv')), ('ae', ('re_ii', 'results_rev', 'rev_ae_pooled.csv')),
            ('tcn', ('re_ii', 'results_rev', 'rev_tcn_pooled.csv')), ('cn0', ('re_ii', 'results_c', 'rev_cn0_pooled.csv')),
            ('wscm', ('re_ii', 'results_c', 'rev_wscm_pooled.csv')))
REVSTAR = {'qr': ('re_ii', 'results_rev', 'rev_qr_star.csv'), 'ae': ('re_ii', 'results_rev', 'rev_ae_star.csv'),
           'cn0': ('re_ii', 'results_c', 'rev_cn0_star.csv'), 'wscm': ('re_ii', 'results_c', 'rev_wscm_star.csv')}
OLD = {'qr': ('results_v2b_1e-5', 'far_mc_v2_1e-5_lstm.csv'), 'ae': ('results_prerun_nb1000', 'pre_ae_nb1000_lstm.csv'),
       'cn0': ('results_v2b_1e-5', 'sqm_v2_1e-5_cn0.csv'), 'wscm': ('results_v2b_1e-5', 'sqm_v2_1e-5_wscm.csv')}
C1B = {'qr': ('QR-LSTM score', '#000000', 'o'), 'cn0_db': ('C/N$_0$ proxy (dB)', '#A6761D', 'P'),
       'cn0_fold': ('C/N$_0$ proxy, folded', '#D9B26F', 'p'), 'wscm_fold': ('WSCM, folded', '#7570B3', 'X')}
SEG = (('ds3', 'static', (0.1, 80.0), (80.0, 114.0), 114.1, 119.6),
       ('ds4', 'static', (0.1, 80.0), (80.0, 109.8), 109.9, 113.8),
       ('ds5', 'dynamic', (21.2, 96.7), None, 96.8, 102.1),
       ('ds6', 'dynamic', (0.1, 80.0), (80.0, 96.4), 96.5, 104.4),
       ('ds7', 'complex', (0.1, 80.0), (80.0, 109.9), 110.0, 110.0),
       ('ds8', 'complex', (16.1, 96.0), (96.0, 109.9), 110.0, 110.0))


# ---------------------------------------------------------------------------
# Reading results and helpers
# ---------------------------------------------------------------------------
def rd(*p):
    with open(os.path.join(D, *p), encoding='utf-8-sig', newline='') as fh:
        return list(csv.DictReader(fh))


def fl(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


def ns(s):
    s = str(s).strip()
    if s.startswith('pertap_'):
        s = s[7:]
    return s[:-4] if s.endswith('_dyn') else s


def parts(st):
    scen, prn = ns(st).rsplit('_prn', 1)
    return scen, int(prn)


def skey(st):
    scen, prn = parts(st)
    return CL_ORDER[SC_CLASS[scen]], scen, prn


def short(st):
    scen, prn = parts(st)
    return f"{scen.replace('_r2', '')}·{prn}"


def log(s):
    LOG.append(s)
    print(s)


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
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 4 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float('nan'), len(x)
    return float(np.corrcoef(rankavg(x), rankavg(y))[0, 1]), len(x)


def arrow_down(ax, x, y, color, alpha=0.6):
    ax.annotate('', xy=(x, y * 0.45), xytext=(x, y),
                arrowprops=dict(arrowstyle='-|>', lw=0.7, color=color, alpha=alpha, mutation_scale=6))


def err(y, lo, hi):
    return [[max(0.0, y - lo)], [max(0.0, hi - y)]]


def nice_log_ticks(ax):
    lo, hi = ax.get_ylim()
    c = [v for v in (0.1, 0.2, 0.3, 0.5, 0.7, 1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 200) if lo <= v <= hi]
    while len(c) > 6:
        c = c[::2]
    ax.yaxis.set_major_locator(mtick.FixedLocator(c))
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.yaxis.set_minor_formatter(mtick.NullFormatter())


# ---------------------------------------------------------------------------
# Figure 1: correlation peak at three stages of an attack (Eq. (2))
# ---------------------------------------------------------------------------
def F02():
    taps = np.round(np.arange(-2.0, 2.0001, 0.2), 1)
    fine = np.linspace(-2.0, 2.0, 801)
    tri = lambda x: np.maximum(0.0, 1.0 - np.abs(x))
    Aa, As = 1.0, 1.4
    cases = (('Code delays aligned', 0.0, 0.0), ('Code delays separating', 0.0, 0.6),
             ('Authentic peak outside the window', -3.0, 0.0))
    fig, axes = S.new_fig('full', 0.30, 1, 3, sharey=True)
    for ax, (ttl, ta, ts) in zip(axes, cases):
        ax.plot(fine, Aa * tri(fine - ta), color='#4D4D4D', lw=1.0, ls=(0, (4, 2)), label='Authentic component')
        ax.plot(fine, As * tri(fine - ts), color='#999999', lw=1.0, ls=(0, (6, 2, 1, 2)), label='Counterfeit component')
        ax.plot(taps, Aa * tri(taps - ta) + As * tri(taps - ts), color='k', lw=0.8, marker='o', ms=3.5,
                label='Sampled peak (21 taps)')
        ax.text(0.03, 0.97, ttl, transform=ax.transAxes, ha='left', va='top', fontsize=8)
        ax.set_xlim(-2.1, 2.1)
        ax.set_xticks([-2, -1, 0, 1, 2])
        ax.set_xlabel('Correlator offset $\\Delta\\tau$ (chips)')
    axes[0].set_ylim(0, 2.75)
    axes[0].set_ylabel('Correlation (normalised)')
    hh, ll = axes[0].get_legend_handles_labels()
    try:
        fig.legend(hh, ll, loc='outside lower center', ncol=3)
    except Exception:
        axes[2].legend(hh, ll, loc='upper right', fontsize=7)
    S.panel_labels(axes)
    S.save(fig, OUT, 'F02_peak_stages')


# ---------------------------------------------------------------------------
# Segment timeline (first submission; Figure 2 is redrawn by s6_figs.py)
# ---------------------------------------------------------------------------
def F03():
    fig, ax = S.new_fig('text', 0.40)
    n = len(SEG)
    for i, (name, cl, cal, ho, act, on) in enumerate(SEG):
        y = n - 1 - i
        ax.barh(y, cal[1] - cal[0], left=cal[0], height=0.5, color='#CCCCCC', lw=0)
        if ho:
            ax.barh(y, ho[1] - ho[0], left=ho[0], height=0.5, color=S.CLASS[cl], lw=0,
                    hatch='////' if name == 'ds8' else None, edgecolor='white')
        if on > act:
            ax.barh(y, on - act, left=act, height=0.5, facecolor='none', edgecolor='0.35', hatch='xxxx', lw=0)
        ax.plot([act, act], [y - 0.32, y + 0.32], color='k', lw=1.1)
        ax.plot(on, y + 0.42, marker='v', color=S.OI['verm'], ms=4, ls='none')
        if name in ('ds5', 'ds6'):
            ax.plot(58.0, y, marker='|', color='0.15', ms=8, mew=1.0)
    ax.set_yticks(range(n))
    ax.set_yticklabels([s[0] for s in SEG][::-1])
    ax.set_xlim(0, 125)
    ax.set_ylim(-0.6, n - 0.2)
    ax.set_xlabel('Time in recording (s)')
    h = [Patch(color='#CCCCCC', label='Calibration')] + \
        [Patch(color=S.CLASS[c], label=f'Held-out ({c})') for c in GROUPS] + \
        [Patch(facecolor='none', edgecolor='0.35', hatch='xxxx', label='Activation to onset'),
         Line2D([], [], color='k', lw=1.1, label='Activation signature'),
         Line2D([], [], color=S.OI['verm'], marker='v', ls='none', label='Published onset'),
         Line2D([], [], color='0.15', marker='|', ls='none', ms=8, label='58 s common-mode event')]
    ax.legend(handles=h, loc='center left', bbox_to_anchor=(1.01, 0.5))
    S.save(fig, OUT, 'F03_timeline')


# ---------------------------------------------------------------------------
# Figure 3: lag-one autocorrelation per satellite
# ---------------------------------------------------------------------------
def F04():
    rows = [r for r in rd('re_ii', 'results_c', 'c1p_c1b_star.csv') if int(fl(r['T_ms'])) == 1]
    val = {(ns(r['star']), r['score']): fl(r['acf1']) for r in rows}
    cls = {ns(r['star']): r['group'] for r in rows}
    stars = sorted(cls, key=skey)
    e1b = {(ns(r['star']), r['variant']): fl(r['acf_1']) for r in rd('re_ii', 'results_e1b', 'e1b_acf.csv')}
    x = {s: i for i, s in enumerate(stars)}
    fig, ax = S.new_fig('text', 0.50)
    for cl in GROUPS:
        idx = [x[s] for s in stars if cls[s] == cl]
        if idx:
            ax.axvspan(min(idx) - 0.5, max(idx) + 0.5, color=S.CLASS[cl], alpha=0.10, lw=0)
            ax.text((min(idx) + max(idx)) / 2, 0.97, S.CLASS_LABEL[cl], transform=ax.get_xaxis_transform(),
                    ha='center', va='top', fontsize=8)
    for k, (lab, col, mk) in C1B.items():
        xs = [x[s] for s in stars if np.isfinite(val.get((s, k), np.nan))]
        ys = [val[(s, k)] for s in stars if np.isfinite(val.get((s, k), np.nan))]
        ax.plot(xs, ys, ls='none', marker=mk, color=col, label=lab, zorder=3)
    xm = [x[s] for s in stars if np.isfinite(e1b.get((s, 'mask58'), np.nan))]
    ym = [e1b[(s, 'mask58')] for s in stars if np.isfinite(e1b.get((s, 'mask58'), np.nan))]
    ax.plot(xm, ym, ls='none', marker='o', ms=3, color='0.6', label='QR-LSTM, 57–60 s masked', zorder=2)
    ax.axhline(0, color='0.5', lw=0.5)
    ax.set_xticks(range(len(stars)))
    ax.set_xticklabels([short(s) for s in stars], rotation=90)
    ax.set_xlim(-0.6, len(stars) - 0.4)
    ax.set_ylim(-0.1, 1.0)
    ax.set_ylabel('Lag-one autocorrelation $\\rho_1$\n(calibration segment, 1 ms)')
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.01), ncol=3)
    log('F04 ρ1 QR 动态范围: ' + ', '.join(f'{short(s)}={val[(s, "qr")]:.3f}' for s in stars if cls[s] == 'dynamic'))
    S.save(fig, OUT, 'F04_rho1_per_star')


# ---------------------------------------------------------------------------
# Figure 4: autocorrelation of representative satellites
# ---------------------------------------------------------------------------
def F05():
    z = np.load(os.path.join(D, 're_ii', 'results_e1b', 'e1b_acf_acf.npz'))
    raw = {ns(k.split('|')[0]): k.split('|')[0] for k in z.files}
    e1b = {(ns(r['star']), r['variant']): r for r in rd('re_ii', 'results_e1b', 'e1b_acf.csv')}
    reps = (('ds4_r2_prn13', 'static'), ('ds6_r2_prn9', 'dynamic'), ('ds7_prn23', 'complex'))
    fig, axes = S.new_fig('full', 0.30, 1, 3, sharey=True)
    for ax, (st, cl) in zip(axes, reps):
        a = np.asarray(z[f'{raw[st]}|full'], float)
        lags = np.arange(len(a))
        r = e1b[(st, 'full')]
        b = 1.96 / np.sqrt(fl(r['N_clean_s']) * 1000.0)
        ax.axhspan(-b, b, color='0.85', lw=0, zorder=0)
        ax.plot(lags[1:], a[1:], color=S.CLASS[cl], lw=0.9, zorder=2)
        ax.axhline(0.05, color='k', lw=0.6, ls=(0, (4, 2)))
        L = int(fl(r['L_pick_farmc']))
        ax.axvline(L, color='k', lw=0.6, ls=':')
        ax.text(L + 8 if L < 300 else L - 8, 0.22, f'$L$ = {L}', path_effects=S.STROKE, fontsize=8,
                ha='left' if L < 300 else 'right')
        ax.text(0.97, 0.97, f'{S.CLASS_LABEL[cl]}: {short(st)}, $\\rho_1$ = {fl(r["rho1_farmc"]):.3f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=8)
        ax.set_xlim(0, len(a) - 1)
        ax.set_ylim(-0.03, 0.33)
        ax.set_xlabel('Lag (epochs of 1 ms)')
        log(f'F05 {st}: L={L} ρ1={fl(r["rho1_farmc"]):.3f} 带宽 ±{b:.4f}')
    axes[0].set_ylabel('ACF of the clean QR-LSTM score')
    S.panel_labels(axes)
    S.save(fig, OUT, 'F05_acf_representatives')


# ---------------------------------------------------------------------------
# Figure 5: serial correlation of block means
# ---------------------------------------------------------------------------
def F06():
    P = {(r['score'], int(fl(r['T_ms'])), r['group']): r for r in rd('re_ii', 'results_c', 'c1p_c1b_pooled.csv')}
    Ts = np.array([1, 20, 100, 1000], float)
    srows = [r for r in rd('re_ii', 'results_c', 'c1p_c1b_star.csv') if r['pooled_excluded'] != 'yes']
    fig, (a, b) = S.new_fig('text', 0.48, 1, 2)
    for i, (k, (lab, col, mk)) in enumerate(C1B.items()):
        xx = Ts * 10 ** (0.035 * (i - 1.5))
        rr = [P[(k, int(T), 'all')] for T in Ts]
        y = np.array([fl(r['acf1_med']) for r in rr])
        q = [np.nanpercentile([fl(r['acf1']) for r in srows if r['score'] == k and int(fl(r['T_ms'])) == int(T)],
                              [25, 75]) for T in Ts]
        lo, hi = np.array([v[0] for v in q]), np.array([v[1] for v in q])
        a.errorbar(xx, y, yerr=[np.maximum(0, y - lo), np.maximum(0, hi - y)], color=col, marker=mk, lw=1.1,
                   elinewidth=0.6, label=lab)
        Dv = np.array([fl(r['b05_D']) for r in rr])
        nom = np.array([fl(r['b05_nom']) for r in rr])
        dlo = np.array([fl(r['b05_lo95']) for r in rr]) / nom
        dhi = np.array([fl(r['b05_up95']) for r in rr]) / nom
        b.errorbar(xx, Dv, yerr=[np.maximum(0, Dv - dlo), np.maximum(0, dhi - Dv)], color=col, marker=mk, lw=1.1,
                   elinewidth=0.6)
        log(f'F06 {k}: ρ1 中位 {np.round(y, 2).tolist()}  D_b05 {np.round(Dv, 2).tolist()}')
    for ax in (a, b):
        ax.set_xscale('log')
        ax.set_xticks(Ts)
        ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f'{v:g}'))
        ax.xaxis.set_minor_locator(mtick.NullLocator())
        ax.set_xlabel('Block length $T$ (ms)')
    a.set_ylim(-0.1, 1.0)
    a.set_ylabel('Lag-one autocorrelation\nof block means')
    hh, ll = a.get_legend_handles_labels()
    try:
        fig.legend(hh, ll, loc='outside upper center', ncol=4, fontsize=7)
    except Exception:
        b.legend(hh, ll, loc='lower left', fontsize=7)
    b.set_yscale('log')
    b.set_yticks([1, 2, 3, 4, 6, 8])
    b.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f'{v:g}'))
    b.yaxis.set_minor_formatter(mtick.NullFormatter())
    b.axhline(1, color='k', lw=0.6, ls=(0, (4, 2)))
    b.set_ylabel('$K$=5 rule: delivered /\nindependence reference')
    S.panel_labels((a, b))
    S.save(fig, OUT, 'F06_block_means')


# ---------------------------------------------------------------------------
# Figure 6: count rule
# ---------------------------------------------------------------------------
def F07():
    kv = [r for r in rd('re_ii', 'results_v32', 'krule_v12.csv') if r['mode'] == 'beta' and r['pooled_excluded'] != 'yes']
    kk = rd('re_ii', 'results_rev', 'krule_K3510_pooled.csv')
    t1p = {r['score']: r for r in rd('re_ii', 'results_c', 'tables', 'c_T1p.csv') if r['group'] == 'dynamic'}
    fig, (a, b, c) = S.new_fig('full', 0.32, 1, 3)
    lv = sorted({fl(r['level']) for r in kv})
    sh = [[fl(r['D_shuffle']) for r in kv if fl(r['level']) == l and fl(r['D_shuffle']) > 0] for l in lv]
    a.fill_between(lv, [min(s) if s else np.nan for s in sh], [max(s) if s else np.nan for s in sh], color='0.85',
                   lw=0, label='Shuffled in time')
    for st in sorted({r['star'] for r in kv}, key=skey):
        rr = sorted([r for r in kv if r['star'] == st and fl(r['D_clean']) > 0], key=lambda r: fl(r['level']))
        if rr:
            a.plot([fl(r['level']) for r in rr], [fl(r['D_clean']) for r in rr], color=S.CLASS[rr[0]['group']],
                   lw=0.5, alpha=0.45)
    for cl in GROUPS:
        med = [np.median([fl(r['D_clean']) for r in kv if r['group'] == cl and fl(r['level']) == l]) for l in lv]
        a.plot(lv, med, color=S.CLASS[cl], lw=1.8, marker='o', ms=3.5)
        if med[-1] > 0:
            a.text(lv[-1] * 1.08, med[-1] * {'static': 0.90, 'dynamic': 1.0, 'complex': 1.35}[cl], S.CLASS_LABEL[cl], color=S.CLASS[cl], va='center', fontsize=8,
                   path_effects=S.STROKE)
        log(f'F07a {cl} 中位 D: ' + ', '.join(f'{l:g}:{m:.2f}' for l, m in zip(lv, med)))
    a.axhline(1, color='k', lw=0.6, ls=(0, (4, 2)))
    a.set_xscale('log')
    a.set_yscale('log')
    a.set_xlim(lv[0] / 1.3, lv[-1] * 2.2)
    a.set_xticks(lv)
    a.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f'{v:g}'))
    a.xaxis.set_minor_formatter(mtick.NullFormatter())
    a.set_xlabel('Target $\\beta$ per 1-s window')
    a.set_ylabel('Delivered / independence reference\n(calibration segments)')
    a.legend(loc='upper left')
    Ks = (3, 5, 10)
    nomv = []
    for j, cl in enumerate(GROUPS):
        for K in Ks:
            r = next(r for r in kk if r['mode'] == 'alpha' and int(fl(r['K'])) == K and abs(fl(r['level']) - 0.05) < 1e-9
                     and r['group'] == cl)
            x = K * 10 ** (0.03 * (j - 1))
            if int(fl(r['clean_n_event'])) > 0:
                y = fl(r['dlv_clean'])
                b.errorbar([x], [y], yerr=err(y, fl(r['clean_lo95']), fl(r['clean_up95'])), color=S.CLASS[cl],
                           marker='o', elinewidth=0.6)
            else:
                b.plot(x, fl(r['clean_up95']), marker='o', color=S.CLASS[cl], alpha=0.5, ls='none')
                arrow_down(b, x, fl(r['clean_up95']), S.CLASS[cl])
            if j == 0:
                nomv.append(fl(r['nominal_iid']))
    b.plot(Ks, nomv, color='k', lw=0.8, ls=(0, (4, 2)))
    xr = 7.3
    yr = 10 ** np.interp(xr, Ks, np.log10(nomv))
    b.annotate('Independence\nreference', xy=(xr, yr), xytext=(6.5, yr * 60), fontsize=8, ha='left', va='bottom',
               arrowprops=dict(arrowstyle='-', lw=0.6, color='k'), path_effects=S.STROKE)
    for cl in GROUPS:
        b.plot([], [], color=S.CLASS[cl], marker='o', ls='none', label=S.CLASS_LABEL[cl])
    b.set_yscale('log')
    b.set_xticks(Ks)
    b.set_xlabel('Consecutive exceedances $K$ (95th-percentile $\\gamma$)')
    b.set_ylabel('False-alarm rate per 1-s window')
    b.legend(loc='lower left')
    log('F07b 参照值 K=3/5/10: ' + ', '.join(f'{v:.2e}' for v in nomv))
    for i, (k, sp) in enumerate(S.SCORE.items()):
        r = t1p.get(k)
        if r is None:
            continue
        Dv = fl(r['D'])
        c.errorbar([Dv], [i], xerr=[[max(0, Dv - fl(r['D_lo']))], [max(0, fl(r['D_up']) - Dv)]], color=sp['color'],
                   marker=sp['marker'], elinewidth=0.7, ls='none')
        c.text(fl(r['D_up']) * 1.15, i, f'{Dv:.0f}', va='center', fontsize=8)
        log(f'F07c {k} 动态 D={Dv:.1f} [{fl(r["D_lo"]):.1f}, {fl(r["D_up"]):.1f}]')
    c.set_yticks(range(len(S.SCORE)))
    c.set_yticklabels([sp['label'] for sp in S.SCORE.values()])
    c.set_ylim(len(S.SCORE) - 0.4, -0.6)
    c.set_xscale('log')
    c.set_xlim(10, 3000)
    c.set_xlabel('Delivered / reference\n(95th pct., $K$=5, dynamic)')
    S.panel_labels((a, b, c))
    S.save(fig, OUT, 'F07_krule')


# ---------------------------------------------------------------------------
# Figure 7: Markov-chain prediction
# ---------------------------------------------------------------------------
def F08():
    ar = [r for r in rd('re_ii', 'results_c', 'c2_k3510_ar1.csv') if r['mode'] == 'b05']
    st = [r for r in rd('re_ii', 'results_c', 'c2_k3510_star.csv')
          if r['mode'] == 'beta' and abs(fl(r['level']) - 0.05) < 1e-9 and int(fl(r['K'])) == 5
          and r['pooled_excluded'] != 'yes' and r['resolvable_3q'] == 'yes']
    fig, (a, b) = S.new_fig('text', 0.46, 1, 2)
    for K, col, mk in ((3, '#A0A0A0', '^'), (5, '#555555', 'o'), (10, '#000000', 's')):
        rr = sorted([r for r in ar if int(fl(r['K'])) == K], key=lambda r: fl(r['rho']))
        rho = [fl(r['rho']) for r in rr]
        a.fill_between(rho, [fl(r['D_q025']) for r in rr], [fl(r['D_q975']) for r in rr], color=col, alpha=0.18, lw=0)
        a.plot(rho, [fl(r['D_mean']) for r in rr], color=col, marker=mk, label=f'$K$ = {K}')
        a.plot(rho, [fl(r['pred_exact_mean']) for r in rr], color=col, lw=0.9, ls=(0, (3, 2)))
        log(f'F08a K={K}: D_mean ' + ', '.join(f'{fl(r["rho"]):g}:{fl(r["D_mean"]):.2f}' for r in rr))
    a.plot([], [], color='0.4', lw=0.9, ls=(0, (3, 2)), label='Markov prediction')
    a.axhline(1, color='k', lw=0.5)
    a.axhline(20, color='k', lw=0.6, ls=':')
    a.text(0.02, 21.5, 'upper limit $1/\\beta$ = 20', fontsize=7, va='bottom')
    a.set_yscale('log')
    a.set_ylim(0.2, 40)
    a.set_xlabel('AR(1) coefficient $\\rho$')
    a.set_ylabel('Delivered / independence reference')
    a.legend(loc='lower right', ncol=2, handlelength=1.4, columnspacing=1.0)
    smk = {'qr': 'o', 'cn0_fold': 'P', 'wscm_fold': 'X'}
    xs, ys = [], []
    for r in st:
        x, y, nom = fl(r['pred_exact']), fl(r['D']), fl(r['nominal_iid'])
        if not (x > 0 and y > 0):
            continue
        b.errorbar([x], [y], yerr=err(y, fl(r['lo95']) / nom, fl(r['up95']) / nom), color=S.CLASS[r['group']],
                   marker=smk.get(r['score'], 'o'), elinewidth=0.5, ls='none', alpha=0.85)
        xs.append(x)
        ys.append(y)
    lim = [min(xs + ys) / 1.5, max(xs + ys) * 3.0]
    b.plot(lim, lim, color='k', lw=0.6, ls=(0, (4, 2)))
    b.set_xlim(lim)
    b.set_ylim(lim)
    b.set_xscale('log')
    b.set_yscale('log')
    b.set_xlabel('Markov-predicted ratio')
    b.set_ylabel('Measured ratio (calibration segments)')
    h = [Patch(color=S.CLASS[c], label=S.CLASS_LABEL[c]) for c in GROUPS] + \
        [Line2D([], [], color='0.3', marker=m, ls='none', label=l) for l, m in
         (('QR-LSTM', 'o'), ('C/N$_0$, folded', 'P'), ('WSCM, folded', 'X'))]
    b.legend(handles=h, loc='upper left', fontsize=7, ncol=2, columnspacing=0.5, handlelength=1.2, handletextpad=0.5)
    rho_s, n = spearman(xs, ys)
    log(f'F08b 逐星点数 {n}，预测与实测 Spearman {rho_s:.2f}')
    S.panel_labels((a, b))
    S.save(fig, OUT, 'F08_markov')


# ---------------------------------------------------------------------------
# Figure 8: held-out verification
# ---------------------------------------------------------------------------
def F09():
    fig, axes = S.new_fig('full', 0.34, 1, 3, sharey=True, sharex=True)
    for ax, rule in zip(axes, ('cusum', 'k_blk', 'k_iid')):
        col = S.RULE[rule]['color']
        tot = None
        for i, (k, path) in enumerate(SC_PATHS):
            sp = S.SCORE[k]
            rr = sorted([r for r in rd(*path) if r['rule'] == rule and r['group'] == 'all' and r['beta'] != ''],
                        key=lambda r: fl(r['ref']))
            for r in rr:
                x = fl(r['ref']) * 10 ** (0.03 * (i - 2))
                nev, fr, lo, up = int(fl(r['ho_n_event'])), fl(r['ho_frac']), fl(r['ho_lo95']), fl(r['ho_up95'])
                tot = fl(r['ho_total_s'])
                if nev > 0:
                    ax.errorbar([x], [fr], yerr=err(fr, lo, up), color=col, marker=sp['marker'], elinewidth=0.6,
                                ls='none', ms=4)
                else:
                    ax.plot(x, up, marker=sp['marker'], color=col, alpha=0.45, ls='none', ms=4)
                    arrow_down(ax, x, up, col, alpha=0.45)
            if rule == 'k_iid':
                log(f'F09 {k} k_iid ho: ' + ', '.join(f'{fl(r["ref"]):g}:{fl(r["ho_frac"]):.3f}' for r in rr))
        xb = 2.996 / tot if tot else 8e-3
        ax.axvspan(5e-4, xb, color='0.92', lw=0, zorder=0)
        ax.plot([5e-4, 0.3], [5e-4, 0.3], color='k', lw=0.6, ls=(0, (4, 2)))
        ax.text(0.04, 0.96, S.RULE[rule]['label'], transform=ax.transAxes, ha='left', va='top', fontsize=8,
                color=col, path_effects=S.STROKE)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(7e-4, 0.15)
        ax.set_ylim(7e-4, 0.6)
        ax.set_xlabel('Target $\\beta$ per 1-s window')
    axes[0].set_ylabel('Delivered rate on held-out data')
    h = [Line2D([], [], color='0.25', marker=sp['marker'], ls='none', label=sp['label']) for sp in S.SCORE.values()]
    axes[2].legend(handles=h, loc='lower right', fontsize=7)
    S.panel_labels(axes)
    S.save(fig, OUT, 'F09_heldout')


# ---------------------------------------------------------------------------
# Delay figures of the first submission (redrawn by s6_figs.py: Figures 10, A3)
# ---------------------------------------------------------------------------
def F10():
    for k, path in SC_PATHS:
        P = rd(*path)
        fig, axes = S.new_fig('full', 0.32, 1, 3)
        for ax, g in zip(axes, GROUPS):
            dk = 'delay_act_med' if g == 'dynamic' else 'delay_onset_med'
            for rule in ('cusum', 'k_blk', 'k_iid'):
                rr = sorted([r for r in P if r['rule'] == rule and r['group'] == g and r['beta'] != ''
                             and fl(r['beta']) >= 0.02 - 1e-9 and fl(r['cal_frac']) > 0], key=lambda r: fl(r['beta']))
                x, y = [fl(r['cal_frac']) for r in rr], [fl(r[dk]) for r in rr]
                ax.plot(x, y, color=S.RULE[rule]['color'], marker=S.RULE[rule]['marker'], lw=1.0, ms=3.5,
                        label=S.RULE[rule]['label'])
                for r in rr:
                    if fl(r['beta']) in (0.1, 0.02) and rule == 'cusum':
                        lft = fl(r['beta']) == 0.1
                        ax.annotate(f'$\\beta$={fl(r["beta"]):g}', (fl(r['cal_frac']), fl(r[dk])), fontsize=7,
                                    xytext=(-4 if lft else 4, 5), textcoords='offset points', path_effects=S.STROKE,
                                    ha='right' if lft else 'left')
            p = [r for r in P if r['rule'] == 'k_p95' and r['group'] == g and fl(r['cal_frac']) > 0]
            if p:
                ax.plot(fl(p[0]['cal_frac']), fl(p[0][dk]), color=S.RULE['k_p95']['color'], marker='*', ms=7,
                        ls='none', label=S.RULE['k_p95']['label'])
            ax.set_xscale('log')
            if g == 'static':
                ax.set_ylim(0, 0.3)
            else:
                ax.set_yscale('log')
                lo_, hi_ = ax.get_ylim()
                ax.set_ylim(lo_ / 1.25, hi_ * 1.7)
                nice_log_ticks(ax)
            ax.text(0.97, 0.97, S.CLASS_LABEL[g], transform=ax.transAxes, ha='right', va='top', fontsize=8)
            ax.set_xlabel('Delivered rate per 1-s window\n(calibration segments)')
            ax.set_ylabel(f'Median delay from {"activation" if g == "dynamic" else "onset"} (s)')
        hh, ll = axes[0].get_legend_handles_labels()
        try:
            fig.legend(hh, ll, loc='outside upper center', ncol=4)
        except Exception:
            axes[0].legend(hh, ll, loc='lower left', fontsize=7)
        S.panel_labels(axes)
        S.save(fig, OUT, f'F10_delay_{k}')
        log(f'F10 {k}: 已出图')


# ---------------------------------------------------------------------------
# Evaluation criteria, first submission (redrawn by s6_figs.py: Figure 11)
# ---------------------------------------------------------------------------
def F11():
    fig, axes = S.new_fig('text', 0.52, 1, 2, sharey=False)
    for ax, g in zip(axes, ('static', 'dynamic')):
        allx, ally = [], []
        for k in ('qr', 'ae', 'cn0', 'wscm'):
            old = {ns(r['star']): fl(r['post_dr@5e-02']) for r in rd(*OLD[k])}
            xs, ys = [], []
            for r in rd(*REVSTAR[k]):
                if r['rule'] != 'cusum' or abs(fl(r['beta']) - 0.02) > 1e-9 or r['group'] != g:
                    continue
                d = fl(r['delay_act_s'] if g == 'dynamic' else r['delay_onset_s'])
                o = old.get(ns(r['star']), np.nan)
                if np.isfinite(o) and np.isfinite(d) and d > 0:
                    xs.append(100 * o)
                    ys.append(d)
            sp = S.SCORE[k]
            ax.plot(xs, ys, ls='none', marker=sp['marker'], color=sp['color'], label=sp['label'], alpha=0.9)
            allx += xs
            ally += ys
            log(f'F11 {g} {k}: n={len(xs)} DR 中位 {np.median(xs) if xs else np.nan:.1f}% 时延中位 '
                f'{np.median(ys) if ys else np.nan:.3f} s')
        rho, n = spearman(allx, ally)
        ax.text(0.03, 0.97, f'{S.CLASS_LABEL[g]}: Spearman $\\rho_s$ = {rho:.2f} (n = {n})', transform=ax.transAxes,
                ha='left', va='top', fontsize=8)
        log(f'F11 {g}: Spearman {rho:.3f} n={n}')
        ax.set_yscale('log')
        ax.set_ylim(min(ally) / 1.6, max(ally) * 4.0)
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f'{v:g}'))
        ax.set_xlim(0, 105)
        ax.set_xlabel('Per-instant detection rate at $\\alpha$ = 0.05 (%)')
        ax.set_ylabel(f'CUSUM delay at $\\beta$ = 0.02,\nfrom {"activation" if g == "dynamic" else "onset"} (s)')
    hh, ll = axes[0].get_legend_handles_labels()
    try:
        fig.legend(hh, ll, loc='outside upper center', ncol=4, fontsize=7)
    except Exception:
        axes[0].legend(hh, ll, loc='lower left', fontsize=7)
    S.panel_labels(axes)
    S.save(fig, OUT, 'F11_two_standards')


# ---------------------------------------------------------------------------
# Figure A2: self-consistency of the calibration
# ---------------------------------------------------------------------------
def A2():
    rows = rd('re_ii', 'results_v32', 'calib_v32_dyn.csv')
    blk = {(r['star'], fl(r['beta'])): r for r in rows if r['calib'] == 'blk'}
    iid = [r for r in rows if r['calib'] == 'iid']
    fig, (a, b, c) = S.new_fig('full', 0.32, 1, 3)
    for st in sorted({r['star'] for r in iid}, key=skey):
        rr = sorted([r for r in iid if r['star'] == st and r['h_ge_gm'] != 'yes'
                     and blk.get((st, fl(r['beta'])), {}).get('h_ge_gm') != 'yes' and fl(r['D_boot']) > 0],
                    key=lambda r: fl(r['beta']))
        if rr:
            a.plot([fl(r['beta']) for r in rr], [fl(r['D_boot']) for r in rr], color=S.CLASS[rr[0]['group']],
                   lw=0.7, marker='o', ms=2, alpha=0.4 if rr[0]['pooled_excluded'] == 'yes' else 0.8)
    a.axhline(1, color='k', lw=0.6, ls=(0, (4, 2)))
    a.axvspan(5e-4, 0.02, color='0.92', lw=0, zorder=0)
    a.set_xscale('log')
    a.set_yscale('log')
    a.set_xlim(7e-4, 0.13)
    a.set_xlabel('Target $\\beta$ per 1-s window')
    a.set_ylabel('i.i.d.-calibrated limit:\ndelivered / target (block resamples)')
    for g in GROUPS:
        a.plot([], [], color=S.CLASS[g], label=S.CLASS_LABEL[g])
    a.legend(loc='upper right')
    pts = [(r, blk.get((r['star'], fl(r['beta'])))) for r in iid if abs(fl(r['beta']) - 0.02) < 1e-9]
    for r, q in pts:
        if q is None:
            continue
        al = 0.4 if r['pooled_excluded'] == 'yes' else 0.9
        b.plot(fl(r['rho1']), fl(q['h']) / fl(r['h']), marker='o', color=S.CLASS[r['group']], alpha=al, ls='none')
        c.plot(fl(r['rho1']), fl(r['D_boot']), marker='o', color=S.CLASS[r['group']], alpha=al, ls='none')
    for ax in (b, c):
        ax.axhline(1, color='k', lw=0.6, ls=(0, (4, 2)))
        ax.set_xlabel('Lag-one autocorrelation $\\rho_1$')
    b.set_ylabel('$h_{\\mathrm{block}}/h_{\\mathrm{iid}}$ at $\\beta$ = 0.02')
    c.set_ylabel('i.i.d. limit: delivered / target\nat $\\beta$ = 0.02')
    S.panel_labels((a, b, c))
    S.save(fig, OUT, 'A2_selfconsistency')


# ---------------------------------------------------------------------------
# Figure 9: baseline shift of the carrier-to-noise proxy (file Figure11.png)
# ---------------------------------------------------------------------------
def A3():
    rows = [r for r in rd('re_ii', 'results_c', 'c1d_star.csv') if r['has_ho'] == 'yes']
    key = {'ds3_prn3', 'ds4_r2_prn3', 'ds7_prn3', 'ds6_r2_prn9', 'ds4_r2_prn7'}
    fig, (a, b) = S.new_fig('text', 0.46, 1, 2)
    xa, ya, xb, yb = [], [], [], []
    for r in rows:
        ex = r['pooled_excluded'] == 'yes'
        kw = dict(marker='o', color=S.CLASS[r['group']], alpha=0.35 if ex else 0.9, ls='none')
        x1, y1, x2, y2 = fl(r['cn0_fold_er90']), fl(r['fa_kblk']), fl(r['cn0_db_delta']), fl(r['fa_c1p'])
        a.plot(x1, y1, **kw)
        b.plot(x2, y2, **kw)
        if not ex:
            xa.append(x1)
            ya.append(y1)
            xb.append(x2)
            yb.append(y2)
        if ns(r['star']) in key:
            for ax, x, y in ((a, x1, y1), (b, x2, y2)):
                low = ax is b and ns(r['star']) == 'ds3_prn3'
                ax.annotate(short(r['star']), (x, y), xytext=(-4, -9) if low else (3, 3), textcoords='offset points',
                            fontsize=7, path_effects=S.STROKE, ha='right' if low else 'left')
    ra, na = spearman(xa, ya)
    rb, nb = spearman(xb, yb)
    a.text(0.03, 0.97, f'Spearman $\\rho_s$ = {ra:.2f} (n = {na})', transform=a.transAxes, ha='left', va='top')
    b.text(0.03, 0.97, f'Spearman $\\rho_s$ = {rb:.2f} (n = {nb})', transform=b.transAxes, ha='left', va='top')
    log(f'A3 ER90 对 fa_kblk Spearman {ra:.2f} (n={na})；有符号 Δ 对 fa_c1p Spearman {rb:.2f} (n={nb})')
    a.axvline(1, color='k', lw=0.6, ls=(0, (4, 2)))
    b.axvline(0, color='k', lw=0.6, ls=(0, (4, 2)))
    a.set_xlabel('Held-out / calibration exceedance ratio\n(folded C/N$_0$ proxy, 90th pct.)')
    a.set_ylabel('Held-out rate, $K$=5 block $\\gamma$,\n$\\beta$ = 0.05')
    b.set_xlabel('Held-out shift of C/N$_0$ proxy (dB)\nin robust calibration units')
    b.set_ylabel('Held-out rate, LLR CUSUM\n(block $h$, $\\mu_0+\\sigma_0$, $\\beta$ = 0.05)')
    h = [Line2D([], [], color=S.CLASS[g], marker='o', ls='none', label=S.CLASS_LABEL[g]) for g in GROUPS]
    a.legend(handles=h, loc='center right', fontsize=7)
    S.panel_labels((a, b))
    S.save(fig, OUT, 'A3_cn0_shift')


FIGS = {'F02': F02, 'F03': F03, 'F04': F04, 'F05': F05, 'F06': F06, 'F07': F07, 'F08': F08, 'F09': F09, 'F10': F10,
        'F11': F11, 'A2': A2, 'A3': A3}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='.')
    ap.add_argument('--only', nargs='+', default=None)
    a = ap.parse_args()
    D = os.path.abspath(a.dir)
    OUT = os.path.join(D, 're_ii', 'figs_journal', 'out')
    os.makedirs(OUT, exist_ok=True)
    log('[version] figs_journal.py v1.5（2026-10-28）')
    for name in (a.only or list(FIGS)):
        try:
            FIGS[name]()
            log(f'[ok] {name}')
        except Exception:
            log(f'[FAIL] {name}\n' + traceback.format_exc())
    with open(os.path.join(OUT, 'figs_log.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(LOG) + '\n')
    print(f'[saved] {OUT}')
