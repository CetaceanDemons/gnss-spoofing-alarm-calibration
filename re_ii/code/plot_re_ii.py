"""Figures of the first manuscript version; this paper uses only the centring comparison (Figure A1).

Run from the repository root (the working folder that also holds the data folders); see --help.
"""
import argparse, glob, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9.5,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.4,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.8,
    'figure.dpi': 120, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'legend.frameon': True, 'legend.framealpha': 1.0, 'legend.edgecolor': '0.6',
    'legend.fancybox': False, 'legend.fontsize': 7.5,
})
C_BLK, C_IID, C_K, C_OLD, C_REF = '#1f3a5f', '#8b1a1a', '#222222', '#8c8c8c', '#333333'
GROUP_COLOR = {'static': '#1b5e3f', 'dynamic': '#5b2a86', 'complex': '#8a5a1a'}
GROUP_MARK = {'static': 'o', 'dynamic': 's', 'complex': '^'}
GROUPS = ['static', 'dynamic', 'complex']
ASR_MS = {'ds5_r2': 21100, 'ds8': 16000}       # raw_t = (i + 100 + asr) / 1000
K0, M_WIN = 4.0, 1000
Z95 = -np.log(0.05)
REP = [('fig4', 'ds4_r2_prn13'), ('fig5', 'ds6_r2_prn9_dyn'), ('fig6', 'ds7_prn23')]


def short(star):
    return star.replace('_r2', '').replace('_dyn', '')


def savefig(fig, out, name):
    p = os.path.join(out, name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    fig.savefig(p)
    plt.close(fig)
    print(f'  [saved] {p}')


def beta_axis(ax, lo=6e-4, hi=0.16):
    ax.set_xscale('log')
    ax.set_xlim(lo, hi)
    ax.set_xlabel(r'nominal target $\beta$')


def read_csv(p):
    return pd.read_csv(p, encoding='utf-8-sig')


def star_table(cal):
    s = cal.drop_duplicates('star').set_index('star')
    s['yes_excl'] = s['pooled_excluded'].astype(str).str.lower().eq('yes')
    return s


def npz_index(root):
    idx = {}
    for f in sorted(glob.glob(os.path.join(root, '**', 'pertap_*.npz'), recursive=True)):
        name = os.path.basename(f)[len('pertap_'):-len('.npz')]
        idx.setdefault(name, f)
    return idx


def load_series(f, scen):
    d = np.load(f, allow_pickle=True)
    S = d['per'].astype(np.float64).sum(1)
    lab = np.asarray(d['labels'])
    raw = (np.arange(len(S)) + 100 + ASR_MS.get(scen, 0)) / 1000.0
    model = str(d['model']) if 'model' in d.files else '(no model field)'
    return S, lab, raw, model


def acf(x, maxlag):
    x = x - x.mean()
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    a = np.fft.irfft(f * np.conj(f))[:maxlag + 1]
    return a / a[0]


def cusum_reset(Z, k0, h):
    g = np.zeros(len(Z)); al = np.zeros(len(Z), bool); cur = 0.0
    for t, z in enumerate(Z):
        cur = max(0.0, cur + z - k0)
        if cur >= h:
            al[t] = True; cur = 0.0
        g[t] = cur
    return g, al


def envelope_max(t, y, w):
    nb = len(y) // w
    if w <= 1 or nb < 2:
        return t, y
    return t[:nb * w].reshape(nb, w).mean(1), y[:nb * w].reshape(nb, w).max(1)


# ---------------- fig2: rho1 ----------------
def fig_rho1(st, out):
    order = sorted(st.index, key=lambda s: (GROUPS.index(st.loc[s, 'group']), s))
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    seen = set()
    for x, s in enumerate(order):
        g = st.loc[s, 'group']; ex = st.loc[s, 'yes_excl']
        ax.scatter(x, st.loc[s, 'rho1'], s=46, marker=GROUP_MARK[g],
                   facecolor='white' if ex else GROUP_COLOR[g], edgecolor=GROUP_COLOR[g],
                   linewidth=1.1, zorder=3, label=None if g in seen else g)
        seen.add(g)
    ax.axhline(0, color='k', lw=0.7)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([short(s) for s in order], rotation=50, ha='right', fontsize=7.5)
    ax.set_ylabel(r'lag-1 autocorrelation $\rho_1$')
    ax.legend(title='scenario', loc='upper left')
    ax.margins(x=0.03)
    savefig(fig, out, 'fig2.png')


def exp1_numbers(st, idx, maxlag, out, lines):
    acfs, cums = {}, {}
    lines.append('== Exp1: clean-segment ACF per star (lags 1..max(L,10)) ==')
    lines.append('star, group, N, rho1_csv, rho1_recomputed, L, min_acf, n_sig_negative, first_lag_in_band, encoder_model, cumsumS_at_clean_end')
    for s in st.index:
        if s not in idx:
            lines.append(f'{s}: npz not found'); continue
        r = st.loc[s]
        S, lab, raw, model = load_series(idx[s], r['scenario'])
        m = (lab == 0) & (raw >= r['clean_raw_start_s'] - 5e-4) & (raw <= r['clean_raw_end_s'] + 5e-4)
        x = S[m]; N = len(x)
        a = acf(x, max(maxlag, int(r['L']) + 10))
        band = 1.96 / np.sqrt(N)
        Lr = max(int(r['L']), 10)
        seg = a[1:Lr + 1]
        inb = np.where(np.abs(a[1:]) < band)[0]
        first_in = int(inb[0] + 1) if len(inb) else -1
        acfs[s] = (a, band, N)
        cums[s] = float(np.cumsum(x)[-1])
        lines.append(f'{s}, {r["group"]}, {N}, {r["rho1"]:.4f}, {a[1]:.4f}, {int(r["L"])}, '
                     f'{seg.min():.4f}, {int((seg < -band).sum())}, {first_in}, {model}, {cums[s]:.0f}')
    lines.append('')
    return acfs


def fig_acf_panel(s, st, acfs, maxlag, out, name):
    a, band, N = acfs[s]
    g = st.loc[s, 'group']
    lags = np.arange(1, maxlag + 1)
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    ax.vlines(lags, 0, a[1:maxlag + 1], color=GROUP_COLOR[g], lw=0.8)
    ax.axhline(0, color='k', lw=0.7)
    for y in (band, -band):
        ax.axhline(y, color=C_BLK, ls='--', lw=0.8)
    ax.set_xlim(0, maxlag)
    ax.set_xlabel('lag (ms)'); ax.set_ylabel('ACF')
    ax.set_title(f'{short(s)}  ($\\rho_1$={st.loc[s, "rho1"]:.3f}, $L$={int(st.loc[s, "L"])})', fontsize=9)
    savefig(fig, out, name)


# ---------------- fig4b/5b/6b: h vs beta ----------------
def fig_limits_panel(s, cal, st, out, name):
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    for calib, col, mk, lab in [('iid', C_IID, 'o', 'i.i.d. reshuffle'), ('blk', C_BLK, 's', 'block bootstrap')]:
        d = cal[(cal.star == s) & (cal.calib == calib)].sort_values('beta')
        sat = d['h_ge_gm'].astype(str).str.lower().eq('yes').values
        ax.plot(d['beta'], d['h'], color=col, lw=1.3, zorder=2)
        ax.scatter(d['beta'][~sat], d['h'][~sat], marker=mk, s=28, color=col, edgecolor='k', lw=0.4, zorder=3, label=lab)
        if sat.any():
            ax.scatter(d['beta'][sat], d['h'][sat], marker=mk, s=28, facecolor='white', edgecolor=col, lw=1.0, zorder=3)
    gm = st.loc[s, 'gm']
    ax.axhline(gm, color=C_REF, ls=':', lw=0.9)
    ax.text(0.99, gm, r' clean max of $g$', transform=ax.get_yaxis_transform(), ha='right', va='bottom', fontsize=7, color='0.3')
    beta_axis(ax)
    ax.set_ylabel(r'control limit $h$')
    ax.set_title(f'{short(s)}  ($\\rho_1$={st.loc[s, "rho1"]:.3f})', fontsize=9)
    ax.legend(loc='lower left', title='open = at or above clean max', title_fontsize=7)
    savefig(fig, out, name)


def fig_selfcons(cal, st, beta_cmp, out, lines):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), constrained_layout=True)
    ax = axes[0]
    for s in sorted(st.index, key=lambda s: (GROUPS.index(st.loc[s, 'group']), s)):
        g = st.loc[s, 'group']
        d = cal[(cal.star == s) & (cal.calib == 'iid')].sort_values('beta')
        sat = d['h_ge_gm'].astype(str).str.lower().eq('yes').values
        D = d['D_boot'].values; b = d['beta'].values
        ok = ~sat & (D > 0)
        ax.plot(b[ok], D[ok], color=GROUP_COLOR[g], lw=0.9, alpha=0.9)
        ax.scatter(b[ok], D[ok], marker=GROUP_MARK[g], s=14, color=GROUP_COLOR[g], zorder=3)
    ax.axhline(1, color='k', lw=0.9)
    beta_axis(ax); ax.set_yscale('log')
    ax.set_ylabel(r'$D_{\mathrm{iid}}$ = delivered / nominal (bootstrap)')
    ax.set_title('(a) i.i.d.-calibrated limit', fontsize=9, loc='left')
    for g in GROUPS:
        ax.scatter([], [], marker=GROUP_MARK[g], color=GROUP_COLOR[g], label=g)
    ax.legend(loc='upper right')

    sub = cal[np.isclose(cal.beta, beta_cmp)]
    hb = sub[sub.calib == 'blk'].set_index('star')['h']; hi = sub[sub.calib == 'iid'].set_index('star')['h']
    Di = sub[sub.calib == 'iid'].set_index('star')['D_boot']
    satb = sub[sub.calib == 'blk'].set_index('star')['h_ge_gm'].astype(str).str.lower().eq('yes')
    lines.append(f'== fig7: beta_cmp={beta_cmp} per star: rho1, h_blk, h_iid, ratio, D_iid(boot), blk_saturated ==')
    for k, (yv, yl, ttl) in enumerate([(hb / hi, r'$h_{\mathrm{blk}}/h_{\mathrm{iid}}$', '(b) limit ratio'),
                                        (Di, r'$D_{\mathrm{iid}}$ (bootstrap)', '(c) i.i.d. mis-delivery')]):
        ax = axes[k + 1]
        for s in st.index:
            g = st.loc[s, 'group']; ex = st.loc[s, 'yes_excl'] or bool(satb.get(s, False))
            ax.scatter(st.loc[s, 'rho1'], yv[s], marker=GROUP_MARK[g], s=34,
                       facecolor='white' if ex else GROUP_COLOR[g], edgecolor=GROUP_COLOR[g], lw=1.0, zorder=3)
        ax.axhline(1, color='k', lw=0.9)
        ax.set_xlabel(r'lag-1 autocorrelation $\rho_1$'); ax.set_ylabel(yl)
        ax.set_title(f'{ttl}, $\\beta$={beta_cmp:g}', fontsize=9, loc='left')
    for s in st.index:
        lines.append(f'{s}, {st.loc[s, "rho1"]:.4f}, {hb[s]:.3f}, {hi[s]:.3f}, {hb[s] / hi[s]:.3f}, {Di[s]:.3f}, {bool(satb.get(s, False))}')
    lines.append('')
    savefig(fig, out, 'fig7.png')


def fig_timeline(st, out):
    scen = st.drop_duplicates('scenario').set_index('scenario')
    order = [s for s in ['ds3', 'ds4_r2', 'ds5_r2', 'ds6_r2', 'ds7', 'ds8'] if s in scen.index]
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    used = set()

    def L(name):
        if name in used:
            return None
        used.add(name); return name
    for y, sc in enumerate(order[::-1]):
        r = scen.loc[sc]
        ax.barh(y, r['clean_raw_end_s'] - r['clean_raw_start_s'], left=r['clean_raw_start_s'], height=0.5,
                color=C_BLK, alpha=0.85, label=L('calibration (clean)'))
        if r['ho_raw_end_s'] > r['ho_raw_start_s']:
            ax.barh(y, r['ho_raw_end_s'] - r['ho_raw_start_s'], left=r['ho_raw_start_s'], height=0.5,
                    color='#7fa7c9', label=L('held-out'))
        if r['ho0_raw_end_s'] > r['act_raw_s']:
            ax.barh(y, r['ho0_raw_end_s'] - r['act_raw_s'], left=r['act_raw_s'], height=0.5, color='white',
                    edgecolor=C_OLD, hatch='////', lw=0.6, label=L('activation to onset (excluded)'))
        ax.plot([r['act_raw_s']] * 2, [y - 0.36, y + 0.36], color=C_IID, lw=1.6, label=L('activation'))
        ax.plot([r['ho0_raw_end_s']] * 2, [y - 0.36, y + 0.36], color='k', lw=1.6, ls=':', label=L('published onset'))
        if bool(r['yes_excl']):
            ax.text(r['ho0_raw_end_s'] + 1.5, y, 'not pooled', va='center', fontsize=7, color='0.35')
    ax.set_yticks(range(len(order))); ax.set_yticklabels([short(s) for s in order[::-1]])
    ax.set_xlabel('recording time (s)'); ax.set_xlim(0, 128)
    ax.grid(axis='y', visible=False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=5, fontsize=7)
    savefig(fig, out, 'fig8.png')


def floor_by_group(cal):
    one = cal[~cal['pooled_excluded'].astype(str).str.lower().eq('yes')].drop_duplicates('star')
    res = {}
    for g in GROUPS:
        for v in ['ho', 'ho0']:
            n = one[(one.group == g) & (one[f'{v}_len_s'] > 0)][f'{v}_nwin'].sum()
            res[(g, v)] = Z95 * M_WIN / n if n > 0 else np.nan
    return res


def plot_series(ax, df, xcol, fcol, lo, up, ncol, x_mult, marker, color, filled, label):
    df = df.sort_values(xcol)
    x = df[xcol].values * x_mult
    first = True
    for xi, f, l, u, n in zip(x, df[fcol], df[lo], df[up], df[ncol]):
        lab = label if first else None; first = False
        if n > 0:
            ax.errorbar(xi, f, yerr=[[max(f - max(l, 1e-6), 0)], [max(u - f, 0)]], fmt=marker, ms=5, color=color,
                        mfc=color if filled else 'white', mec=color, elinewidth=0.9, capsize=2, zorder=3, label=lab)
        else:
            ax.scatter(xi, u, marker='v', s=30, facecolor=color if filled else 'white', edgecolor=color, zorder=3, label=lab)


def fig_heldout(pool, kpool, cal, out, lines):
    fl = floor_by_group(cal)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=True, constrained_layout=True)
    lines.append('== fig9: verifiability floor (zero-event 95% upper bound) per group ==')
    for ax, g in zip(axes, GROUPS):
        p = pool[pool.group == g]
        plot_series(ax, p[(p.variant == 'ho') & (p.calib == 'blk')], 'beta', 'frac', 'lo95', 'up95', 'n_event',
                    1.0, 's', C_BLK, True, 'CUSUM, block')
        plot_series(ax, p[(p.variant == 'ho') & (p.calib == 'iid')], 'beta', 'frac', 'lo95', 'up95', 'n_event',
                    1.15, 'o', C_IID, False, 'CUSUM, i.i.d.')
        k = kpool[(kpool['mode'] == 'beta') & (kpool.group == g) & (kpool['ho_total_s'] > 0)]
        plot_series(ax, k, 'level', 'ho_frac', 'ho_lo95', 'ho_up95', 'ho_n_event', 0.87, 'D', C_K, True,
                    'K=5 rule (i.i.d. design)')
        if g == 'dynamic':
            plot_series(ax, p[(p.variant == 'ho0') & (p.calib == 'blk')], 'beta', 'frac', 'lo95', 'up95', 'n_event',
                        1.3, 'X', C_OLD, True, 'CUSUM, block, to onset')
        xx = np.array([6e-4, 0.16])
        ax.plot(xx, xx, color=C_REF, ls='--', lw=0.9, label='delivered = nominal')
        f = fl[(g, 'ho')]
        if np.isfinite(f):
            ax.axvspan(6e-4, f, color='0.9', zorder=0)
        lines.append(f'{g}: ho floor = {f:.4g}, ho0 floor = {fl[(g, "ho0")]:.4g}')
        beta_axis(ax); ax.set_yscale('log'); ax.set_ylim(1e-3, 0.8)
        ph = p[p.variant == 'ho']
        n_st = int(ph['n_stars'].iloc[0]) if len(ph) else 0
        T = ph['total_ho_s'].iloc[0] if len(ph) else 0
        ax.set_title(f'{g} ({n_st} sat., {T:.0f} sat.-s held out)', fontsize=9, loc='left')
    axes[0].set_ylabel('delivered windowed false-alarm rate (held-out)')
    hs, ls = [], []
    for ax in axes:
        for h_, l_ in zip(*ax.get_legend_handles_labels()):
            if l_ not in ls:
                hs.append(h_); ls.append(l_)
    fig.legend(hs, ls, loc='outside lower center', ncol=5, fontsize=7.5)
    lines.append('')
    savefig(fig, out, 'fig9.png')


def fig_krule(kr, st, out, lines):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), constrained_layout=True, gridspec_kw={'width_ratios': [1, 1.25]})
    ax = axes[0]
    b = kr[kr['mode'] == 'beta']
    sh = b['D_shuffle'].replace([np.inf], np.nan).dropna()
    sh = sh[sh > 0]
    if len(sh):
        ax.axhspan(sh.min(), sh.max(), color='0.88', zorder=0, label='time-shuffled clean')
    for s in sorted(b.star.unique(), key=lambda s: (GROUPS.index(st.loc[s, 'group']), s)):
        g = st.loc[s, 'group']
        d = b[b.star == s].sort_values('level')
        ax.plot(d['level'], d['D_clean'], color=GROUP_COLOR[g], lw=0.9)
        ax.scatter(d['level'], d['D_clean'], marker=GROUP_MARK[g], s=14, color=GROUP_COLOR[g], zorder=3)
    for g in GROUPS:
        ax.scatter([], [], marker=GROUP_MARK[g], color=GROUP_COLOR[g], label=g)
    ax.axhline(1, color='k', lw=0.9)
    beta_axis(ax, 1.4e-2, 0.13); ax.set_yscale('log')
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
    ax.xaxis.set_major_locator(FixedLocator([0.02, 0.03, 0.05, 0.1]))
    ax.xaxis.set_major_formatter(FixedFormatter(['0.02', '0.03', '0.05', '0.1']))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel(r'$D_{\mathrm{clean}}$ = delivered / nominal')
    ax.set_title('(a) K=5 rule, threshold set for nominal $\\beta$', fontsize=9, loc='left')
    ax.legend(loc='upper right')
    lines.append(f'== fig10a: D_shuffle range over beta-mode rows: {sh.min():.3f} .. {sh.max():.3f} ==')

    ax = axes[1]
    a = kr[(kr['mode'] == 'alpha') & np.isclose(kr['level'], 0.05)].copy()
    a['g'] = a.star.map(st['group'])
    a = a.sort_values(['g', 'star'], key=lambda c: c.map(lambda v: GROUPS.index(v) if v in GROUPS else v) if c.name == 'g' else c)
    nominal = a['nominal_iid'].iloc[0]
    for x, (_, r) in enumerate(a.iterrows()):
        g = r['g']; ex = bool(st.loc[r['star'], 'yes_excl'])
        col = GROUP_COLOR[g]
        if r['n_event_clean'] > 0:
            ax.errorbar(x, r['dlv_clean'], yerr=[[max(r['dlv_clean'] - max(r['clean_lo95'], 1e-7), 0)], [max(r['clean_up95'] - r['dlv_clean'], 0)]],
                        fmt=GROUP_MARK[g], ms=5, color=col, mfc='white' if ex else col, elinewidth=0.8, capsize=2, zorder=3)
        else:
            ax.scatter(x, r['clean_up95'], marker='v', s=26, facecolor='white', edgecolor=col, zorder=3)
    ax.axhline(nominal, color=C_IID, ls='--', lw=1.0)
    ax.text(0.01, nominal, f' nominal under independence = {nominal:.3g}', transform=ax.get_yaxis_transform(),
            va='bottom', fontsize=7, color=C_IID)
    ax.set_yscale('log'); ax.set_ylim(nominal / 3, 0.5)
    ax.set_xticks(range(len(a))); ax.set_xticklabels([short(s) for s in a.star], rotation=50, ha='right', fontsize=7)
    ax.set_ylabel('delivered windowed false-alarm rate (clean)')
    ax.set_title('(b) 95th-percentile threshold, K=5', fontsize=9, loc='left')
    lines.append('== fig10b: prevailing p95+K=5 per star: star, dlv_clean, D_clean, n_event_clean ==')
    for _, r in a.iterrows():
        lines.append(f'{r["star"]}, {r["dlv_clean"]:.4g}, {r["D_clean"]:.1f}, {int(r["n_event_clean"])}')
    lines.append('')
    savefig(fig, out, 'fig10.png')


# ---------------------------------------------------------------------------
# Figure A1: uncentred and centred CUSUM
# ---------------------------------------------------------------------------
def fig_centring(s, cal, st, idx, beta_h, win_ms, out, lines):
    r = st.loc[s]
    S, lab, raw, model = load_series(idx[s], r['scenario'])
    t_end = r['ho_raw_end_s'] if r['ho_raw_end_s'] > r['ho_raw_start_s'] else r['clean_raw_end_s']
    clean = (lab == 0) & (raw >= r['clean_raw_start_s'] - 5e-4) & (raw <= r['clean_raw_end_s'] + 5e-4)
    w = (raw >= r['clean_raw_start_s'] - 5e-4) & (raw < t_end)
    mu, sd = S[clean].mean(), S[clean].std()
    Z = (S[w] - mu) / sd; t = raw[w]
    h = float(cal[(cal.star == s) & (cal.calib == 'blk') & np.isclose(cal.beta, beta_h)]['h'].iloc[0])
    g, al = cusum_reset(Z, K0, h)
    cum = np.cumsum(S[w])
    ce = r['clean_raw_end_s']
    inc, inh = t <= ce + 5e-4, t > ce + 5e-4
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.6, 4.4), sharex=True, constrained_layout=True)
    te, ce_env = envelope_max(t, np.maximum(cum, 1.0), win_ms)
    a1.plot(te, ce_env, color=C_IID, lw=1.3)
    a1.set_yscale('log'); a1.set_ylabel(r'uncentred $\sum_t S(t)$')
    te2, ge2 = envelope_max(t, g, win_ms)
    a2.fill_between(te2, 0, ge2, color=C_BLK, alpha=0.25, lw=0)
    a2.plot(te2, ge2, color=C_BLK, lw=0.7)
    a2.axhline(h, color=C_REF, ls='--', lw=1.0)
    a2.text(0.005, h, f' $h$ ($\\beta$={beta_h:g}, block)', transform=a2.get_yaxis_transform(), va='bottom', fontsize=7.5)
    a2.set_ylim(-0.02 * h * 1.3, h * 1.3)
    a2.set_ylabel(r'centred CUSUM $g_t$'); a2.set_xlabel('recording time (s)')
    for a in (a1, a2):
        a.axvline(ce, color='0.4', ls=':', lw=1.1)
    a1.text(ce, 0.05, ' calibration | held-out', transform=a1.get_xaxis_transform(), fontsize=7.5, color='0.3')
    a1.set_xlim(t[0], t[-1])
    savefig(fig, out, 'fig3.png')
    lines.append(f'== fig3 (Exp2) star={s}, encoder={model} ==')
    lines.append(f'window {t[0]:.3f}..{t[-1]:.3f} s; clean mean S = {mu:.4f}, sd = {sd:.4f}; h(beta={beta_h}) = {h:.3f}; gm = {r["gm"]:.3f}')
    lines.append(f'cumsum S at clean end = {cum[inc][-1]:.0f}; at window end = {cum[-1]:.0f}')
    lines.append(f'clean: frac(g==0) = {(g[inc] == 0).mean():.4f}, median g = {np.median(g[inc]):.3f}, '
                 f'p99.9 g = {np.quantile(g[inc], 0.999):.2f}, max g = {g[inc].max():.2f}, alarms = {int(al[inc].sum())}')
    if inh.any():
        lines.append(f'held-out: frac(g==0) = {(g[inh] == 0).mean():.4f}, max g = {g[inh].max():.2f}, alarms = {int(al[inh].sum())}')
    lines.append('')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--res', required=True)
    ap.add_argument('--dir', default=None, help='npz 根目录 (图 3/4a/5a/6a 需要; 不给则跳过)')
    ap.add_argument('--out', default='figs_v32')
    ap.add_argument('--maxlag', type=int, default=300)
    ap.add_argument('--beta-cmp', dest='beta_cmp', type=float, default=0.02)
    ap.add_argument('--centring-star', dest='cstar', default='ds6_r2_prn9_dyn')
    ap.add_argument('--centring-beta', dest='cbeta', type=float, default=0.01)
    ap.add_argument('--win-ms', dest='win_ms', type=int, default=100)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cal = read_csv(os.path.join(a.res, 'calib_v32_dyn.csv'))
    pool = read_csv(os.path.join(a.res, 'calib_v32_dyn_pooled.csv'))
    kr = read_csv(os.path.join(a.res, 'krule_v12.csv'))
    kpool = read_csv(os.path.join(a.res, 'krule_v12_pooled.csv'))
    st = star_table(cal)
    print(f'[in] {len(st)} stars, {len(cal)} calibration rows')
    lines = []
    fig_rho1(st, a.out)
    for fig, s in REP:
        if s in st.index:
            fig_limits_panel(s, cal, st, a.out, f'{fig}/b.png')
    fig_selfcons(cal, st, a.beta_cmp, a.out, lines)
    fig_timeline(st, a.out)
    fig_heldout(pool, kpool, cal, a.out, lines)
    fig_krule(kr, st, a.out, lines)
    if a.dir:
        idx = npz_index(a.dir)
        print(f'[in] {len(idx)} pertap npz under {a.dir}')
        acfs = exp1_numbers(st, idx, a.maxlag, a.out, lines)
        for fig, s in REP:
            if s in acfs:
                fig_acf_panel(s, st, acfs, a.maxlag, a.out, f'{fig}/a.png')
        if a.cstar in idx:
            fig_centring(a.cstar, cal, st, idx, a.cbeta, a.win_ms, a.out, lines)
        else:
            print(f'[warn] centring star {a.cstar} npz not found, fig3 skipped')
    else:
        print('[warn] --dir not given: fig3 / fig4a / fig5a / fig6a / Exp1 numbers skipped')
    p = os.path.join(a.out, 'numbers_for_tex.txt')
    with open(p, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))
    print(f'  [saved] {p}\n[done]')


if __name__ == '__main__':
    main()
