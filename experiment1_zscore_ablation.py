"""Alignment and standardisation of the TEXBAT tracking outputs.

The scoring scripts import align_scenario, compute_local_mu_sigma and the scenario
table from this module. Run on its own, it compares per-scenario and global
standardisation of the QR score, a check from the companion study [22] that is not
part of this paper.
"""

import numpy as np
import os, gc, math, time
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
TEXBAT_ROOT  = r"."   # run from J:\Database\TEXBAT\processed_Mao
OAKBAT_NPY   = r"J:\Database\OAKBAT\processed_03_26\Mao_base\oakbat_single_star_v2"
OAKBAT_MODEL = r"J:\Database\OAKBAT\processed_03_26\Mao_base\qr_lstm_results_v2\qr_lstm_v2.keras"
OUT_DIR      = os.path.join(TEXBAT_ROOT, "experiment1_results")
os.makedirs(OUT_DIR, exist_ok=True)

W            = 100
TAU_L, TAU_U = 0.0228, 0.9772
N_TAPS       = 21
PROMPT_IDX   = 10
K_ALARM      = 5
BATCH        = 2048

# OAKBAT cleanStatic training prefixes (for global mu/sigma)
OAKBAT_CLEAN_FILES = [
    "cleanStatic_th_0.7_ch0_prn8_raw_taps.npy",
    "cleanStatic_th_0.7_ch1_prn32_raw_taps.npy",
    "cleanStatic_th_0.7_ch2_prn31_raw_taps.npy",
    "cleanStatic_th_0.7_ch4_prn25_raw_taps.npy",
    "cleanStatic_th_0.7_ch7_prn10_raw_taps.npy",
]

record_dtype = np.dtype([
    ('taps',       np.float32, (21,)),
    ('prompt_iq',  np.float32, (2,)),
    ('timestamp',  np.uint64),
    ('states',     np.float32, (12,)),
    ('next_epoch', np.uint64),
    ('prn',        np.int32)
])

FS = 25_000_000  # TEXBAT 25 MHz
THRESHOLD_GAP_SAMPLES = 1.5 * (FS / 1000)


# clean_end / spoof_start in ms (relative to aligned start).
# onset_sec is for plot annotation only.
SCENARIO_CONFIG = {
    'ds3': {
        'dir': 'ds3',
        'channels': [0, 1],          # PRN 3, 23
        'clean_end': 80_000,         # 80s, well before Lemmenes activation 114.1s
        'spoof_start': 125_000,      # Lemmenes onset 118.9s + 6.1s margin (matched to ds4 logic)
        'onset_sec': 118.9,
    },
    'ds4_r2': {
        'dir': 'ds4_r2',
        'channels': [0, 1, 2, 3, 4], # PRN 3, 23, 6, 7, 13
        'clean_end': 80_000,
        'spoof_start': 120_000,      # Lemmenes onset 113.8s + 6.2s
        'onset_sec': 113.8,
    },
    'ds5_r2': {
        'dir': 'ds5_r2',
        'channels': [0, 1, 2, 3, 4], # PRN 6, 9, 15, 18, 21
        'clean_end': 80_000,
        'spoof_start': 130_000,      # NO precise literature; conservative ~100s onset
        'onset_sec': 100,
    },
    'ds6_r2': {
        'dir': 'ds6_r2',
        'channels': [0, 1, 2, 3, 4], # PRN 6, 9, 15, 18, 22
        'clean_end': 80_000,
        'spoof_start': 130_000,      # Same caveat as ds5_r2
        'onset_sec': 100,
    },
    'ds8': {
        'dir': 'ds8',
        'channels': [0, 1],          # PRN 3, 23
        'clean_end': 80_000,
        'spoof_start': 140_000,      # Humphreys ds7 paper: pull-off begins 136s
        'onset_sec': 110,
    },
    'ds7': {
        'dir': 'ds7',
        'channels': [0, 1],          # PRN 3, 23
        'clean_end': 80_000,
        'spoof_start': 140_000,
        'onset_sec': 110,
    },
}


SCENARIOS_TO_RUN = ['ds7']

plt.rcParams.update({'font.family': 'serif', 'font.size': 11,
                     'figure.dpi': 120, 'savefig.dpi': 200})


def compute_global_mu_sigma():
    print("--- Computing GLOBAL µ/σ from OAKBAT cleanStatic ---")
    chunks = []
    for fn in OAKBAT_CLEAN_FILES:
        path = os.path.join(OAKBAT_NPY, fn)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing OAKBAT clean file: {path}")
        t = np.load(path).astype(np.float32)
        chunks.append(t)
        print(f"  loaded {fn}: shape={t.shape}")
    cat = np.concatenate(chunks, axis=0)
    mu  = np.mean(cat, axis=0, keepdims=True).astype(np.float32)
    sig = (np.std(cat, axis=0, keepdims=True) + 1e-8).astype(np.float32)
    print(f"  total samples: {len(cat)}")
    print(f"  µ_prompt    = {mu[0, PROMPT_IDX]:.0f}")
    print(f"  σ_prompt    = {sig[0, PROMPT_IDX]:.0f}")
    print(f"  µ all 21 taps:\n    {np.round(mu[0], 1)}")
    print(f"  σ all 21 taps:\n    {np.round(sig[0], 1)}")
    del cat, chunks; gc.collect()
    return mu, sig


def align_scenario(sc_name, cfg):
    sc_path = os.path.join(TEXBAT_ROOT, cfg['dir'])
    ranges = []
    prn_map = {}
    for ch in cfg['channels']:
        fpath = os.path.join(sc_path, f"trk_ch_{ch}.dat")
        if not os.path.exists(fpath):
            print(f"  [WARN] {fpath} missing")
            continue
        data = np.fromfile(fpath, dtype=record_dtype)
        if len(data) < 200:
            continue
        dt = np.diff(data['timestamp'])
        gaps = np.where(dt > THRESHOLD_GAP_SAMPLES)[0] + 1
        bp = np.unique(np.concatenate(([0], gaps, [len(data)])))
        best, mlen = (0, 0), 0
        for i in range(len(bp) - 1):
            s, e = bp[i], bp[i+1]
            if (e - s) > mlen:
                mlen = e - s
                best = (int(s), int(e))
        prn_val = int(data['prn'][best[0]])
        ranges.append(best)
        prn_map[ch] = prn_val
        print(f"  CH{ch} (PRN{prn_val}): {best[0]}->{best[1]} ({(best[1]-best[0])/1000:.1f}s)")
        del data
    if not ranges:
        return None, None
    start = int(math.ceil(max(r[0] for r in ranges) / 100.0) * 100)
    end   = min(r[1] for r in ranges)
    print(f"  Aligned: {start} -> {end} ({(end-start)/1000:.1f}s)")
    return {'start': start, 'end': end}, prn_map


def compute_local_mu_sigma(sc_name, cfg, alignment):
    sc_path = os.path.join(TEXBAT_ROOT, cfg['dir'])
    clean_end = min(alignment['start'] + cfg['clean_end'], alignment['end'])
    chunks = []
    for ch in cfg['channels']:
        fpath = os.path.join(sc_path, f"trk_ch_{ch}.dat")
        if not os.path.exists(fpath):
            continue
        data = np.fromfile(fpath, dtype=record_dtype)
        taps = data['taps'][alignment['start']:clean_end].astype(np.float32)
        chunks.append(taps)
        del data
    cat = np.concatenate(chunks, axis=0)
    mu  = np.mean(cat, axis=0, keepdims=True).astype(np.float32)
    sig = (np.std(cat, axis=0, keepdims=True) + 1e-8).astype(np.float32)
    print(f"  LOCAL µ/σ: {len(cat)} clean samples, µ_prompt={mu[0,PROMPT_IDX]:.0f}, σ_prompt={sig[0,PROMPT_IDX]:.0f}")
    del cat, chunks; gc.collect()
    return mu, sig


def score_taps(model, taps_raw, mu, sig):
    taps_n = (taps_raw - mu) / sig
    T = len(taps_n)
    N = T - W
    if N <= 0:
        return None, None, None, None

    S_tot = np.zeros(N, dtype=np.float32)
    S_e   = np.zeros(N, dtype=np.float32)
    S_l   = np.zeros(N, dtype=np.float32)

    for start in range(0, N, BATCH):
        end = min(start + BATCH, N)
        bs  = end - start
        X = np.zeros((bs, W, N_TAPS), dtype=np.float32)
        for j in range(bs):
            X[j] = taps_n[start + j : start + j + W]
        q_lo, q_hi = model.predict(X, batch_size=bs, verbose=0)
        y = taps_n[start + W : end + W]
        per = np.maximum(0, y - q_hi) + np.maximum(0, q_lo - y)
        S_tot[start:end] = per.sum(axis=1)
        S_e[start:end]   = per[:, :PROMPT_IDX].sum(axis=1)
        S_l[start:end]   = per[:, PROMPT_IDX+1:].sum(axis=1)
        del X, q_lo, q_hi
    times = np.arange(W, W + N)
    return times, S_tot, S_e, S_l

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def alarm_time_K5(S, lab, gamma, K, times, onset_sec):
    """First run of K consecutive exceedances whose last sample is at or after the onset."""
    if K <= 0 or len(S) < K:
        return None, None
    binary = (S > gamma).astype(np.int8)
    csum = np.cumsum(np.concatenate(([0], binary)))
    rolling = csum[K:] - csum[:-K]
    onset_ms = int(round(onset_sec * 1000))
    for i in range(len(rolling)):
        if rolling[i] == K:
            t_ms = int(times[i + K - 1])
            if t_ms >= onset_ms:
                t_s = t_ms / 1000.0
                return t_s, t_s - onset_sec
    return None, None

def metrics(S_tot, S_e, S_l, lab, gamma, times, onset_sec):
    c = (lab == 0); s = (lab == 1)
    out = {'gamma': float(gamma)}
    if c.any():
        out['clean_mean'] = float(S_tot[c].mean())
        out['clean_p95']  = float(np.percentile(S_tot[c], 95))
        out['FAR_K1']     = float((S_tot[c] > gamma).mean() * 100)
    if s.any():
        out['spoof_mean'] = float(S_tot[s].mean())
        out['Det_K1']     = float((S_tot[s] > gamma).mean() * 100)
        out['SE_mean']    = float(S_e[s].mean())
        out['SL_mean']    = float(S_l[s].mean())
        out['LE']         = float(out['SL_mean'] / (out['SE_mean'] + 1e-8))
    if c.any() and s.any():
        t_abs, t_rel = alarm_time_K5(S_tot, lab, gamma, K_ALARM, times, onset_sec)
        out['alarm_K5_abs_s'] = t_abs
        out['alarm_K5_rel_s'] = t_rel
    return out

# ---------------------------------------------------------------------------
# Per-star plot: old vs new overlay
# ---------------------------------------------------------------------------
def plot_overlay(times, S_old, S_new, lab, name, onset_s, spoof_start_ms,
                 g_old, out_path):
    t_sec = times / 1000.0
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    for ax, S, title, color in [(axes[0], S_old, 'OLD: per-scenario µ/σ (v3.4 baseline)', '#1f77b4'),
                                 (axes[1], S_new, 'NEW: OAKBAT cleanStatic global µ/σ',  '#d62728')]:
        ax.plot(t_sec, S, color=color, lw=0.4, alpha=0.75)
        ax.axhline(g_old, color='black', ls='--', lw=1, alpha=0.7,
                   label=f'γ = {g_old:.3f} (locked from OLD clean p95)')
        ax.axvline(onset_s, color='gray', ls=':', lw=1, alpha=0.7,
                   label=f'onset~{onset_s}s')
        ax.axvline(spoof_start_ms / 1000.0, color='red', ls='--', lw=1, alpha=0.7,
                   label=f'spoof_start={spoof_start_ms/1000:.0f}s')
        sp_idx = np.where(lab == 1)[0]
        if len(sp_idx):
            ax.axvspan(t_sec[sp_idx[0]], t_sec[min(sp_idx[-1], len(t_sec)-1)],
                       alpha=0.06, color='red')
        ax.set_ylabel('S(t)'); ax.set_title(title); ax.grid(True, ls=':', alpha=0.4)
        ax.legend(fontsize=8, loc='upper right')
    axes[-1].set_xlabel('Time (s)')
    plt.suptitle(f'Experiment 1 — {name}', y=1.005)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------------------------
# Process one scenario
# ---------------------------------------------------------------------------
def process_scenario(sc_name, cfg, model, mu_global, sig_global, rows):
    print(f"\n{'='*70}\n  Scenario: {sc_name}\n{'='*70}")
    alignment, prn_map = align_scenario(sc_name, cfg)
    if alignment is None:
        print(f"  [SKIP] no alignment"); return
    mu_local, sig_local = compute_local_mu_sigma(sc_name, cfg, alignment)

    sc_path = os.path.join(TEXBAT_ROOT, cfg['dir'])
    for ch in cfg['channels']:
        fpath = os.path.join(sc_path, f"trk_ch_{ch}.dat")
        if not os.path.exists(fpath):
            continue
        data = np.fromfile(fpath, dtype=record_dtype)
        prn_val = int(data['prn'][alignment['start']])
        taps = data['taps'][alignment['start']:alignment['end']].astype(np.float32)
        del data; gc.collect()

        n_samples = len(taps)
        labels = np.full(n_samples, -1, dtype=np.int8)
        labels[:cfg['clean_end']] = 0
        if cfg['spoof_start'] < n_samples:
            labels[cfg['spoof_start']:] = 1

        name = f"{sc_name}_prn{prn_val}"
        print(f"\n  >>> {name} (CH{ch})  samples={n_samples}, "
              f"clean={(labels==0).sum()}, guard={(labels==-1).sum()}, spoof={(labels==1).sum()}")

        # OLD
        print(f"      [OLD] running inference with local µ/σ ...")
        t0 = time.time()
        times, S_o_tot, S_o_e, S_o_l = score_taps(model, taps, mu_local, sig_local)
        t_old = time.time() - t0
        # NEW
        print(f"      [NEW] running inference with global µ/σ ...")
        t0 = time.time()
        _, S_n_tot, S_n_e, S_n_l = score_taps(model, taps, mu_global, sig_global)
        t_new = time.time() - t0
        print(f"      inference: OLD {t_old:.1f}s, NEW {t_new:.1f}s")

        # align labels to S length
        N   = len(S_o_tot)
        lab = labels[W:W+N]


        c = (lab == 0); s = (lab == 1)
        if not (c.any() and s.any()):
            print(f"      [SKIP] no clean or spoof samples"); continue
        gamma = np.percentile(S_o_tot[c], 95)

        m_old = metrics(S_o_tot, S_o_e, S_o_l, lab, gamma, times, cfg['onset_sec'])
        m_new = metrics(S_n_tot, S_n_e, S_n_l, lab, gamma, times, cfg['onset_sec'])

        print(f"      γ = {gamma:.4f}  (locked from OLD clean p95)")
        print(f"      OLD  Det%(K=1)={m_old['Det_K1']:6.2f}  FAR={m_old['FAR_K1']:5.2f}  "
              f"L/E={m_old['LE']:5.2f}  alarm_K5_rel={m_old.get('alarm_K5_rel_s')}")
        print(f"      NEW  Det%(K=1)={m_new['Det_K1']:6.2f}  FAR={m_new['FAR_K1']:5.2f}  "
              f"L/E={m_new['LE']:5.2f}  alarm_K5_rel={m_new.get('alarm_K5_rel_s')}")
        delta = m_new['Det_K1'] - m_old['Det_K1']
        print(f"      Δ Det% = {delta:+6.2f} pp")

        # save dual npz
        np.savez_compressed(
            os.path.join(OUT_DIR, f"{name}_scores_dual.npz"),
            times=times,
            S_total_old=S_o_tot, S_early_old=S_o_e, S_late_old=S_o_l,
            S_total_new=S_n_tot, S_early_new=S_n_e, S_late_new=S_n_l,
            labels=labels, gamma=gamma,
            mu_local=mu_local, sigma_local=sig_local,
            mu_global=mu_global, sigma_global=sig_global,
        )
        # plot overlay
        plot_overlay(times, S_o_tot, S_n_tot, lab, name,
                     cfg['onset_sec'], cfg['spoof_start'], gamma,
                     os.path.join(OUT_DIR, f"{name}_overlay.png"))

        # row for csv
        rows.append({
            'scenario':   sc_name,
            'prn':        prn_val,
            'gamma_locked': gamma,
            'Det_K1_old': m_old['Det_K1'],
            'Det_K1_new': m_new['Det_K1'],
            'Delta_pp':   delta,
            'FAR_K1_old': m_old['FAR_K1'],
            'FAR_K1_new': m_new['FAR_K1'],
            'LE_old':     m_old['LE'],
            'LE_new':     m_new['LE'],
            'SE_old':     m_old['SE_mean'],
            'SL_old':     m_old['SL_mean'],
            'SE_new':     m_new['SE_mean'],
            'SL_new':     m_new['SL_mean'],
            'alarmK5_rel_old': m_old.get('alarm_K5_rel_s'),
            'alarmK5_rel_new': m_new.get('alarm_K5_rel_s'),
            'clean_mean_old': m_old['clean_mean'],
            'clean_mean_new': m_new['clean_mean'],
            'spoof_mean_old': m_old['spoof_mean'],
            'spoof_mean_new': m_new['spoof_mean'],
        })

        del taps, S_o_tot, S_o_e, S_o_l, S_n_tot, S_n_e, S_n_l, labels
        gc.collect()

# ---------------------------------------------------------------------------
# Summary writers
# ---------------------------------------------------------------------------
def write_csv(rows, path):
    if not rows: return
    keys = list(rows[0].keys())
    with open(path, 'w', encoding='utf-8') as f:
        f.write(','.join(keys) + '\n')
        for r in rows:
            f.write(','.join(
                f"{r[k]:.4f}" if isinstance(r[k], float) and r[k] is not None
                else ('' if r[k] is None else str(r[k]))
                for k in keys
            ) + '\n')

def write_md(rows, path):
    with open(path, 'w', encoding='utf-8') as f:
        f.write("# Experiment 1 — Z-score Ablation Summary\n\n")
        f.write("γ locked from OLD clean p95.  K=1 Det% reported (matches v3.4 baseline metric).\n\n")
        f.write("| Scenario | PRN | Det% OLD | Det% NEW | Δ pp | FAR_new | L/E OLD | L/E NEW | alarm_K5 OLD | alarm_K5 NEW |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            def fmt(v, dec=2):
                return f"{v:.{dec}f}" if v is not None else "—"
            f.write(f"| {r['scenario']} | {r['prn']} | "
                    f"{fmt(r['Det_K1_old'])} | {fmt(r['Det_K1_new'])} | "
                    f"{fmt(r['Delta_pp'],2)} | {fmt(r['FAR_K1_new'])} | "
                    f"{fmt(r['LE_old'])} | {fmt(r['LE_new'])} | "
                    f"{fmt(r['alarmK5_rel_old'],3)} | {fmt(r['alarmK5_rel_new'],3)} |\n")
        # aggregate stats
        old_dets = [r['Det_K1_old'] for r in rows]
        new_dets = [r['Det_K1_new'] for r in rows]
        new_fars = [r['FAR_K1_new'] for r in rows]
        deltas   = [r['Delta_pp']   for r in rows]
        f.write("\n## Aggregate\n\n")
        f.write(f"- Mean Det% OLD: {np.mean(old_dets):.2f}\n")
        f.write(f"- Mean Det% NEW: {np.mean(new_dets):.2f}\n")
        f.write(f"- Mean Δ pp    : {np.mean(deltas):+.2f}\n")
        f.write(f"- Worst Δ pp   : {min(deltas):+.2f} (star = "
                f"{rows[int(np.argmin(deltas))]['scenario']}_prn{rows[int(np.argmin(deltas))]['prn']})\n")
        f.write(f"- Mean FAR new : {np.mean(new_fars):.2f}\n")
        f.write(f"- Max  FAR new : {max(new_fars):.2f}\n\n")
        f.write("## Decision rule\n\n")
        f.write("- mean |Δ| ≤ 5 pp & FAR_new ≤ 10%   → method robust, add as supplementary table\n")
        f.write("- 5 < mean |Δ| ≤ 15 pp              → acceptable, requires extra explanation\n")
        f.write("- mean |Δ| > 15 pp OR FAR_new > 30% → METHOD-LEVEL ALARM, redesign normalization\n")

# ---------------------------------------------------------------------------
# Pinball loss (for compile after load)
# ---------------------------------------------------------------------------
def pinball_loss(tau):
    def loss(y_true, y_pred):
        err = y_true - y_pred
        return tf.reduce_mean(tf.maximum(tau * err, (tau - 1.0) * err))
    return loss

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("="*70)
    print("Experiment 1 — Z-score Ablation")
    print(f"  Scenarios: {SCENARIOS_TO_RUN}")
    print(f"  Model:     {OAKBAT_MODEL}")
    print(f"  OAKBAT:    {OAKBAT_NPY}")
    print(f"  Output:    {OUT_DIR}")
    print("="*70)


    mu_global, sig_global = compute_global_mu_sigma()

    # load model
    if not os.path.exists(OAKBAT_MODEL):
        raise FileNotFoundError(f"Model not found: {OAKBAT_MODEL}")
    model = load_model(OAKBAT_MODEL, compile=False)
    model.compile(optimizer=Adam(1e-3),
                  loss={'q_lower': pinball_loss(TAU_L),
                        'q_upper': pinball_loss(TAU_U)})
    print(f"\nModel loaded: {model.count_params()} params")

    # process each scenario, accumulating rows
    rows = []
    t_start = time.time()
    for sc in SCENARIOS_TO_RUN:
        if sc not in SCENARIO_CONFIG:
            print(f"[WARN] {sc} not in SCENARIO_CONFIG, skipping"); continue
        process_scenario(sc, SCENARIO_CONFIG[sc], model,
                         mu_global, sig_global, rows)

    print(f"\n\nTotal time: {time.time()-t_start:.1f}s")

    # write CSV + MD
    csv_path = os.path.join(OUT_DIR, "exp1_summary.csv")
    md_path  = os.path.join(OUT_DIR, "exp1_summary.md")
    write_csv(rows, csv_path)
    write_md(rows, md_path)

    # console summary
    print("\n" + "="*78)
    print("SUMMARY (Det% K=1):")
    print("="*78)
    print(f"{'Scenario':<10} {'PRN':<5} {'OLD':>7} {'NEW':>7} {'Δ pp':>7} {'FAR_new':>8} {'L/E_old':>8} {'L/E_new':>8}")
    print("-"*78)
    for r in rows:
        print(f"{r['scenario']:<10} {r['prn']:<5} "
              f"{r['Det_K1_old']:7.2f} {r['Det_K1_new']:7.2f} "
              f"{r['Delta_pp']:+7.2f} {r['FAR_K1_new']:8.2f} "
              f"{r['LE_old']:8.2f} {r['LE_new']:8.2f}")
    print("-"*78)
    if rows:
        print(f"{'MEAN':<16} "
              f"{np.mean([r['Det_K1_old'] for r in rows]):7.2f} "
              f"{np.mean([r['Det_K1_new'] for r in rows]):7.2f} "
              f"{np.mean([r['Delta_pp']   for r in rows]):+7.2f} "
              f"{np.mean([r['FAR_K1_new'] for r in rows]):8.2f}")
    print("="*78)
    print(f"\nWritten: {csv_path}")
    print(f"Written: {md_path}")
    print(f"Per-star npz + overlay PNGs in: {OUT_DIR}\n")

    print("=== SANITY CHECK ===")
    print("Det% OLD should match the per-scenario baseline.")
    print("If they differ by >0.5 pp, alignment / labels are off — investigate before trusting NEW numbers.")
