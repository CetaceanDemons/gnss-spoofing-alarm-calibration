"""Plot style for the journal figures (MDPI page widths, Palatino text, PDF and 600-dpi PNG).
"""
import os
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

CM = 1 / 2.54
TW, FW = 13.86 * CM, 18.46 * CM
OI = dict(black='#000000', orange='#E69F00', sky='#56B4E9', green='#009E73', yellow='#F0E442',
          blue='#0072B2', verm='#D55E00', pink='#CC79A7')
RULE = {'cusum': dict(color=OI['black'], marker='o', label='CUSUM, block $h$'),
        'k_blk': dict(color=OI['blue'], marker='s', label='$K$=5, block $\\gamma$'),
        'k_iid': dict(color=OI['verm'], marker='^', label='$K$=5, indep. $\\gamma$'),
        'k_p95': dict(color=OI['pink'], marker='*', label='$K$=5, 95th pct.')}
CLASS = {'static': OI['green'], 'dynamic': OI['orange'], 'complex': OI['sky']}
CLASS_LABEL = {'static': 'Static', 'dynamic': 'Dynamic', 'complex': 'Complex'}
SCORE = {'qr': dict(label='QR-LSTM', color='#000000', marker='o'),
         'ae': dict(label='LSTM-AE', color='#555555', marker='s'),
         'tcn': dict(label='TCN', color='#999999', marker='D'),
         'cn0': dict(label='C/N$_0$ proxy', color='#A6761D', marker='P'),
         'wscm': dict(label='WSCM', color='#7570B3', marker='X')}
STROKE = [pe.withStroke(linewidth=2.2, foreground='white')]

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Palatino Linotype', 'Palatino', 'TeX Gyre Pagella', 'URW Palladio L', 'STIXGeneral',
                   'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 8, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'axes.linewidth': 0.6, 'lines.linewidth': 1.2, 'lines.markersize': 4.5,
    'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4,
    'xtick.major.size': 3, 'ytick.major.size': 3, 'xtick.minor.size': 1.5, 'ytick.minor.size': 1.5,
    'axes.grid': False, 'legend.frameon': False, 'legend.handlelength': 1.8, 'errorbar.capsize': 0,
    'axes.formatter.use_mathtext': True,
    'figure.dpi': 100, 'figure.constrained_layout.use': True,
    'savefig.bbox': 'standard', 'pdf.fonttype': 42, 'ps.fonttype': 42,
})


def new_fig(width='text', aspect=0.62, nrows=1, ncols=1, **kw):
    """New figure of text width (13.86 cm) or full width (18.46 cm); aspect = height / width."""
    w = TW if width == 'text' else FW
    return plt.subplots(nrows, ncols, figsize=(w, w * aspect), **kw)


def panel_labels(axes):
    for i, ax in enumerate(axes):
        ax.text(-0.02, 1.035, f'({chr(97 + i)})', transform=ax.transAxes, ha='right', va='bottom', fontsize=9)


def save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, f'{name}.pdf'))
    fig.savefig(os.path.join(outdir, f'{name}.png'), dpi=600)
    plt.close(fig)
