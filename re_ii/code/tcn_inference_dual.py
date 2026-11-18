"""Scores the TEXBAT scenarios with a trained TCN.

Same procedure and output fields as lstm_ae_inference_dual.py; the paper uses
S_total_old. --dynamic selects the dynamic model for ds5 and ds6. Model weights are
not included.
Run from the repository root (the working folder that also holds the data folders); see --help.
"""
import argparse
import os
import sys
import gc
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_here = os.path.dirname(os.path.abspath(__file__))
for _d in (_here, os.path.dirname(_here), os.path.dirname(os.path.dirname(_here))):
    if os.path.exists(os.path.join(_d, "experiment1_zscore_ablation.py")):
        if _d not in sys.path:
            sys.path.insert(0, _d)
        break

import tensorflow as tf
from tensorflow.keras.models import load_model

import experiment1_zscore_ablation as exp1


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42,
                    help='Which trained LSTM-AE seed to load (default 42).')
parser.add_argument('--scenarios', type=str, default='all',
                    help='Comma-separated subset (e.g. "ds3,ds7") or "all".')


parser.add_argument('--dynamic', action='store_true',
                    help='[DYN] 用 cleanDynamic 训练的动态 AE 模型, '
                         '只跑 ds5_r2/ds6_r2(动态场景)。')
args = parser.parse_args()
SEED = args.seed
DYNAMIC = args.dynamic


_MODEL_TAG   = 'dynamic_' if DYNAMIC else ''          # tcn_dynamic_seed42.keras
_OUT_SUFFIX  = '_dual_dyn' if DYNAMIC else '_dual'    # seed42_dual_dyn\
NPZ_SUFFIX   = '_dyn' if DYNAMIC else ''              # ..._scores_dual_dyn.npz

LSTM_AE_MODEL = (
    r"J:\Database\OAKBAT\processed_03_26\Mao_base\tcn_results"
    rf"\tcn_{_MODEL_TAG}seed{SEED}.keras"
)
OUT_DIR = os.path.join(
    r"J:\Database\TEXBAT\processed_Mao\tcn_results",
    f"seed{SEED}{_OUT_SUFFIX}"
)
os.makedirs(OUT_DIR, exist_ok=True)

EXPECTED_PARAMS = 23_265

# inherit from exp1
W           = exp1.W
N_TAPS      = exp1.N_TAPS
PROMPT_IDX  = exp1.PROMPT_IDX
K_ALARM     = exp1.K_ALARM
BATCH       = exp1.BATCH


DYN_SCENARIOS = ['ds5_r2', 'ds6_r2']
ALL_SCENARIOS = ['ds3', 'ds4_r2', 'ds5_r2', 'ds6_r2', 'ds7', 'ds8']

if args.scenarios.lower() == 'all':
    SCENARIOS_TO_RUN = list(DYN_SCENARIOS) if DYNAMIC else list(ALL_SCENARIOS)
else:
    SCENARIOS_TO_RUN = [s.strip() for s in args.scenarios.split(',')]
    if DYNAMIC:
        bad = [s for s in SCENARIOS_TO_RUN if s not in DYN_SCENARIOS]
        if bad:
            print(f"[WARN] --dynamic 下指定了非动态场景 {bad} —— "
                  f"这是拿动态模型跑静态场景, 违反'静态场景静态模型'原则。"
                  f"确认你是在调试。")
    else:
        bad = [s for s in SCENARIOS_TO_RUN if s in DYN_SCENARIOS]
        if bad:
            print(f"[WARN] 静态模式下指定了动态场景 {bad} —— "
                  f"正式实验里 ds5/ds6 应走 --dynamic。确认你是在调试。")

plt.rcParams.update({'font.family': 'serif', 'font.size': 11,
                     'figure.dpi': 120, 'savefig.dpi': 200})


# ---------------------------------------------------------------------------
# LSTM-AE-specific scoring (squared reconstruction error)
# ---------------------------------------------------------------------------
def score_taps_lstm_ae(model, taps_raw, mu, sig):
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
        recon = model.predict(X, batch_size=bs, verbose=0)
        if isinstance(recon, list):
            raise RuntimeError("Multi-head output detected; expected Arch-3.")
        y = taps_n[start + W : end + W]
        per = (y - recon) ** 2
        S_tot[start:end] = per.sum(axis=1)
        S_e[start:end]   = per[:, :PROMPT_IDX].sum(axis=1)
        S_l[start:end]   = per[:, PROMPT_IDX+1:].sum(axis=1)
        del X, recon
    times = np.arange(W, W + N)
    return times, S_tot, S_e, S_l


def process_scenario_dual(sc_name, cfg, model, mu_global, sig_global, rows):
    print(f"\n{'='*70}\n  Scenario: {sc_name}\n{'='*70}")
    alignment, prn_map = exp1.align_scenario(sc_name, cfg)
    if alignment is None:
        print(f"  [SKIP] no alignment"); return
    mu_local, sig_local = exp1.compute_local_mu_sigma(sc_name, cfg, alignment)

    sc_path = os.path.join(exp1.TEXBAT_ROOT, cfg['dir'])
    for ch in cfg['channels']:
        fpath = os.path.join(sc_path, f"trk_ch_{ch}.dat")
        if not os.path.exists(fpath):
            continue
        data = np.fromfile(fpath, dtype=exp1.record_dtype)
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
              f"clean={(labels==0).sum()}, "
              f"guard={(labels==-1).sum()}, "
              f"spoof={(labels==1).sum()}")

        # OLD
        print(f"      [OLD] running with local mu/sigma ...")
        t0 = time.time()
        times, S_o_tot, S_o_e, S_o_l = score_taps_lstm_ae(
            model, taps, mu_local, sig_local)
        t_old = time.time() - t0

        # NEW
        print(f"      [NEW] running with OAKBAT global mu/sigma ...")
        t0 = time.time()
        _, S_n_tot, S_n_e, S_n_l = score_taps_lstm_ae(
            model, taps, mu_global, sig_global)
        t_new = time.time() - t0
        print(f"      inference: OLD {t_old:.1f}s, NEW {t_new:.1f}s")

        # align labels
        N   = len(S_o_tot)
        lab = labels[W:W+N]
        c = (lab == 0); s = (lab == 1)
        if not (c.any() and s.any()):
            print(f"      [SKIP] no clean or spoof samples"); continue


        gamma_old = float(np.percentile(S_o_tot[c], 95))
        gamma_new = float(np.percentile(S_n_tot[c], 95))

        m_old = exp1.metrics(S_o_tot, S_o_e, S_o_l, lab, gamma_old, times,
                             cfg['onset_sec'])
        m_new = exp1.metrics(S_n_tot, S_n_e, S_n_l, lab, gamma_new, times,
                             cfg['onset_sec'])

        print(f"      gamma_old = {gamma_old:.4f}, gamma_new = {gamma_new:.4f}")
        print(f"      OLD  Det%={m_old['Det_K1']:6.2f}  FAR={m_old['FAR_K1']:5.2f}  "
              f"L/E={m_old['LE']:5.2f}")
        print(f"      NEW  Det%={m_new['Det_K1']:6.2f}  FAR={m_new['FAR_K1']:5.2f}  "
              f"L/E={m_new['LE']:5.2f}")
        delta_b = m_new['Det_K1'] - m_old['Det_K1']
        print(f"      Delta_b = {delta_b:+6.2f} pp")


        npz_path = os.path.join(
            OUT_DIR, f"{name}_tcn_scores_dual{NPZ_SUFFIX}.npz")
        np.savez_compressed(
            npz_path,
            times=times, labels=labels,
            S_total_old=S_o_tot, S_early_old=S_o_e, S_late_old=S_o_l,
            S_total_new=S_n_tot, S_early_new=S_n_e, S_late_new=S_n_l,
            gamma_old=gamma_old, gamma_new=gamma_new,
            mu_local=mu_local, sigma_local=sig_local,
            mu_global=mu_global, sigma_global=sig_global,
            scenario=sc_name, prn=prn_val, seed=SEED,


            arch=("TCN k2 d1-64 F39 (dynamic, cleanDynamic-trained)"
                  if DYNAMIC else "TCN k2 d1-64 F39"),
        )

        # overlay 2-row plot (OLD on top, NEW on bottom)
        png_path = os.path.join(
            OUT_DIR, f"{name}_tcn_overlay_dual{NPZ_SUFFIX}.png")
        plot_overlay_dual(times, S_o_tot, S_n_tot, lab, name,
                          cfg['onset_sec'], cfg['spoof_start'],
                          gamma_old, gamma_new, png_path)

        rows.append({
            'scenario':   sc_name,
            'prn':        prn_val,
            'gamma_old':  gamma_old,
            'gamma_new':  gamma_new,
            'Det_old':    m_old['Det_K1'],
            'Det_new':    m_new['Det_K1'],
            'Delta_b':    delta_b,
            'FAR_old':    m_old['FAR_K1'],
            'FAR_new':    m_new['FAR_K1'],
            'LE_old':     m_old['LE'],
            'LE_new':     m_new['LE'],
            'SE_old':     m_old['SE_mean'],
            'SL_old':     m_old['SL_mean'],
            'SE_new':     m_new['SE_mean'],
            'SL_new':     m_new['SL_mean'],
            'alarm_K5_old': m_old.get('alarm_K5_rel_s'),
            'alarm_K5_new': m_new.get('alarm_K5_rel_s'),
            'clean_mean_old': m_old['clean_mean'],
            'clean_mean_new': m_new['clean_mean'],
            'spoof_mean_old': m_old['spoof_mean'],
            'spoof_mean_new': m_new['spoof_mean'],
        })

        del taps, S_o_tot, S_o_e, S_o_l, S_n_tot, S_n_e, S_n_l, labels
        gc.collect()


def plot_overlay_dual(times, S_old, S_new, lab, name, onset_s, spoof_start_ms,
                      g_old, g_new, out_path):
    t_sec = times / 1000.0
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    pairs = [(axes[0], S_old, g_old, 'OLD: per-scenario mu/sigma', '#2ca02c'),
             (axes[1], S_new, g_new, 'NEW: OAKBAT global mu/sigma', '#d62728')]
    for ax, S, g, title, color in pairs:
        ax.plot(t_sec, S, color=color, lw=0.4, alpha=0.8)
        ax.axhline(g, color='black', ls='--', lw=1, alpha=0.7,
                   label=f'gamma = {g:.3f}')
        ax.axvline(onset_s, color='gray', ls=':', lw=1, alpha=0.7,
                   label=f'onset~{onset_s}s')
        ax.axvline(spoof_start_ms / 1000.0, color='red', ls='--', lw=1,
                   alpha=0.7,
                   label=f'spoof_start={spoof_start_ms/1000:.0f}s')
        sp_idx = np.where(lab == 1)[0]
        if len(sp_idx):
            ax.axvspan(t_sec[sp_idx[0]], t_sec[min(sp_idx[-1], len(t_sec)-1)],
                       alpha=0.06, color='red')
        ax.set_ylabel('S(t)'); ax.set_title(title)
        ax.grid(True, ls=':', alpha=0.4)
        ax.legend(fontsize=9, loc='upper right')
    axes[-1].set_xlabel('Time (s)')
    plt.suptitle(f'TCN dual — {name}', y=1.005)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Summary writers
# ---------------------------------------------------------------------------
def write_summary_csv(rows, path):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, 'w', encoding='utf-8') as f:
        f.write(','.join(keys) + '\n')
        for r in rows:
            f.write(','.join(
                f"{r[k]:.4f}" if isinstance(r[k], float) and r[k] is not None
                else ('' if r[k] is None else str(r[k]))
                for k in keys
            ) + '\n')


def write_summary_md(rows, path, total_time_s):
    qr_mod1_mean_delta_b = -5.95
    qr_mod1_ds78_delta_b = -34.23

    dets_old = np.array([r['Det_old'] for r in rows])
    dets_new = np.array([r['Det_new'] for r in rows])
    deltas_b = np.array([r['Delta_b'] for r in rows])

    advanced = [r for r in rows if r['scenario'] in ('ds7', 'ds8')]
    others   = [r for r in rows if r['scenario'] not in ('ds7', 'ds8')]
    delta_advanced = np.array([r['Delta_b'] for r in advanced]) if advanced else np.array([])
    delta_others   = np.array([r['Delta_b'] for r in others])   if others   else np.array([])

    with open(path, 'w', encoding='utf-8') as f:
        f.write(f"# LSTM-AE Dual-Mode Inference Summary (seed={SEED})\n\n")
        f.write(f"**Mode**: dual (OLD per-scenario mu/sigma + NEW OAKBAT global mu/sigma)  \n")
        f.write(f"**Stars**: {len(rows)}  \n")
        f.write(f"**Total time**: {total_time_s:.0f}s "
                f"({total_time_s/60:.1f} min)  \n")
        f.write(f"**gamma rule**: p95 of S(t) on each mode's own clean "
                f"segment (FAR=5% by design, matches Mod-1 NEW-(b))\n\n")

        f.write("## Per-star results\n\n")
        f.write("| Sc | PRN | Det OLD | Det NEW | Delta_b | "
                "FAR OLD | FAR NEW | L/E OLD | L/E NEW | gamma OLD | gamma NEW |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['scenario']} | {r['prn']} | "
                    f"{r['Det_old']:.2f} | {r['Det_new']:.2f} | "
                    f"{r['Delta_b']:+.2f} | "
                    f"{r['FAR_old']:.2f} | {r['FAR_new']:.2f} | "
                    f"{r['LE_old']:.2f} | {r['LE_new']:.2f} | "
                    f"{r['gamma_old']:.2f} | {r['gamma_new']:.2f} |\n")

        f.write("\n## Aggregate (LSTM-AE)\n\n")
        f.write(f"- Mean Det% OLD: {dets_old.mean():.2f}\n")
        f.write(f"- Mean Det% NEW: {dets_new.mean():.2f}\n")
        f.write(f"- **Mean Delta_b** (NEW - OLD): "
                f"**{deltas_b.mean():+.2f} pp** "
                f"(median {np.median(deltas_b):+.2f}, "
                f"std {deltas_b.std():.2f})\n")
        f.write(f"- Worst Delta_b: {deltas_b.min():+.2f} pp ({rows[int(np.argmin(deltas_b))]['scenario']}_prn{rows[int(np.argmin(deltas_b))]['prn']})\n")
        f.write(f"- Best  Delta_b: {deltas_b.max():+.2f} pp ({rows[int(np.argmax(deltas_b))]['scenario']}_prn{rows[int(np.argmax(deltas_b))]['prn']})\n\n")

        f.write("## Attack-type stratification\n\n")
        if delta_advanced.size:
            f.write(f"- ds7+ds8 (advanced) {len(delta_advanced)} stars: "
                    f"mean Delta_b = **{delta_advanced.mean():+.2f}** pp "
                    f"(std {delta_advanced.std():.2f})\n")
        if delta_others.size:
            f.write(f"- Other 17 stars: mean Delta_b = "
                    f"**{delta_others.mean():+.2f}** pp "
                    f"(std {delta_others.std():.2f})\n")

        f.write("\n## QR-LSTM Mod-1 reference (for comparison)\n\n")
        f.write(f"- QR mean Delta_b (21 stars):  **{qr_mod1_mean_delta_b:+.2f}** pp "
                f"(from Mod-1 aggregate)\n")
        f.write(f"- QR ds7+ds8 mean Delta_b:     **{qr_mod1_ds78_delta_b:+.2f}** pp\n\n")

        f.write("## β route bet decision\n\n")
        ae_mean = deltas_b.mean()
        delta_vs_qr = ae_mean - qr_mod1_mean_delta_b
        f.write(f"AE mean Delta_b vs QR mean Delta_b: "
                f"{ae_mean:+.2f} - ({qr_mod1_mean_delta_b:+.2f}) = "
                f"**{delta_vs_qr:+.2f} pp**\n\n")
        f.write("Interpretation:\n")
        f.write("- delta < -10 pp -> AE degrades MUCH MORE than QR; QR has "
                "deployment-realistic robustness advantage. **β bet WIN**.\n")
        f.write("- -10 <= delta <= +5 pp -> Both methods degrade similarly; "
                "QR has no robustness differentiation. **β bet LOSE -> α route**.\n")
        f.write("- delta > +5 pp -> AE degrades LESS than QR (AE more robust); "
                "**bad outcome**, paper Section 1 must be heavily rewritten.\n\n")

        if delta_vs_qr < -10:
            f.write("**Auto-verdict: β BET WIN.** LSTM-AE degrades "
                    f"{-delta_vs_qr:.1f} pp more than QR-LSTM under deployment-"
                    "realistic normalization. Paper Section 1 contribution 1 "
                    "(unsupervised paradigm) AND Section 6 (deployment "
                    "robustness) both have data backing.\n")
        elif delta_vs_qr <= 5:
            f.write("**Auto-verdict: β BET LOSE.** LSTM-AE degrades "
                    "comparably to QR-LSTM. Section 5.7 cannot use NEW-mode "
                    "comparison as a differentiator. Pivot to α route: "
                    "rewrite Section 1 around (a) ds4_r2/ds7/ds8 local "
                    "advantages, (b) statistical interpretability of QR.\n")
        else:
            f.write("**Auto-verdict: REVERSAL.** LSTM-AE is MORE robust than "
                    "QR-LSTM under global normalization. Investigate "
                    "immediately — this contradicts the expected physics and "
                    "may indicate a normalization or scoring bug.\n")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("="*70)
    print(f"TCN Dual-Mode Inference (C 组 TCN-b，复制自 lstm_ae_inference_dual.py)")
    print(f"  MODE:      {'DYNAMIC (cleanDynamic model)' if DYNAMIC else 'STATIC (cleanStatic model)'}")
    print(f"  Seed:      {SEED}")
    print(f"  Scenarios: {SCENARIOS_TO_RUN}")
    print(f"  Model:     {LSTM_AE_MODEL}")
    print(f"  Output:    {OUT_DIR}")
    if NPZ_SUFFIX:
        print(f"  npz 后缀:  *_scores_dual{NPZ_SUFFIX}.npz")
    print("="*70)

    # ---- Sanity ----
    n_stars_expected = 0
    for sc in SCENARIOS_TO_RUN:
        if sc not in exp1.SCENARIO_CONFIG:
            raise SystemExit(f"[FATAL] '{sc}' not in SCENARIO_CONFIG")
        n_stars_expected += len(exp1.SCENARIO_CONFIG[sc]['channels'])
    print(f"\n[SANITY] Will attempt {n_stars_expected} stars across "
          f"{len(SCENARIOS_TO_RUN)} scenarios.\n")

    # ---- GLOBAL mu/sigma (NEW source) ----
    mu_global, sig_global = exp1.compute_global_mu_sigma()

    # ---- Load LSTM-AE model ----
    if not os.path.exists(LSTM_AE_MODEL):
        raise SystemExit(f"[FATAL] Model not found: {LSTM_AE_MODEL}")
    print(f"\n--- Loading TCN model ---")
    model = load_model(LSTM_AE_MODEL, compile=False)
    actual_params = model.count_params()
    print(f"  Total params: {actual_params:,}")
    if actual_params != EXPECTED_PARAMS:
        raise SystemExit(
            f"[FATAL] Param count mismatch: {actual_params} vs "
            f"{EXPECTED_PARAMS}.")
    out_shape = model.output_shape
    print(f"  Output shape: {out_shape}")
    if not (isinstance(out_shape, tuple) and len(out_shape) == 2
            and out_shape[1] == N_TAPS):
        raise SystemExit(
            f"[FATAL] Unexpected output shape {out_shape}.")
    print(f"  [SANITY OK] TCN params / output shape OK.\n")

    # ---- Run dual inference ----
    rows = []
    t_start = time.time()
    for sc in SCENARIOS_TO_RUN:
        process_scenario_dual(sc, exp1.SCENARIO_CONFIG[sc], model,
                              mu_global, sig_global, rows)

    total_time = time.time() - t_start
    print(f"\n\nTotal inference time: {total_time:.0f}s "
          f"({total_time/60:.1f} min) over {len(rows)} stars")

    # ---- Write summary ----
    csv_path = os.path.join(OUT_DIR, "tcn_dual_summary.csv")
    md_path  = os.path.join(OUT_DIR, "tcn_dual_summary.md")
    write_summary_csv(rows, csv_path)
    write_summary_md(rows, md_path, total_time)


    _b = ['[version] tcn_inference_dual.py v1.0（2026-10-24）',
          f'[cfg] mode={"DYNAMIC" if DYNAMIC else "STATIC"}  seed={SEED}  model={LSTM_AE_MODEL}',
          f'[cfg] out={OUT_DIR}  共 {len(rows)} 星（预期 {n_stars_expected}）  用时 {total_time / 60:.1f} min',
          '星 | Det_old% | FAR_old% | alarmK5_old(s) | Det_new% | Delta_b',
          '--- 以下 Det/FAR 为 p95、K=1 的旧口径，只用来肉眼看"分数有没有正常分开"，不进论文 ---']
    for r in rows:
        _a = r.get('alarm_K5_old')
        _b.append(f"{r['scenario']}_prn{r['prn']} | {r['Det_old']:.2f} | {r['FAR_old']:.2f} | "
                  f"{(f'{_a:+.3f}' if isinstance(_a, (int, float)) else '—')} | {r['Det_new']:.2f} | {r['Delta_b']:+.2f}")
    if rows:
        _b.append(f"MEAN Det_old {np.mean([r['Det_old'] for r in rows]):.2f}  Det_new {np.mean([r['Det_new'] for r in rows]):.2f}")
    if len(rows) != n_stars_expected:
        _b.append(f'[WARN] 星数 {len(rows)} ≠ {n_stars_expected}，先查日志里的 [SKIP]')
    _b.append(f'[saved] npz 文件名形如 <场景>_prn<N>_tcn_scores_dual{NPZ_SUFFIX}.npz')
    with open(os.path.join(OUT_DIR, 'tcn_dual_brief.txt'), 'w', encoding='utf-8') as _fh:
        _fh.write('\n'.join(_b) + '\n')
    print(f"Written: {os.path.join(OUT_DIR, 'tcn_dual_brief.txt')}  ← 只回传这个")

    # ---- Console summary ----
    print("\n" + "="*78)
    print(f"TCN DUAL SUMMARY (seed={SEED}, {len(rows)} stars)")
    print("="*78)
    print(f"{'Sc':<10} {'PRN':<5} {'Det OLD':>8} {'Det NEW':>8} "
          f"{'Delta_b':>8} {'L/E OLD':>8} {'L/E NEW':>8}")
    print("-"*78)
    for r in rows:
        print(f"{r['scenario']:<10} {r['prn']:<5} "
              f"{r['Det_old']:8.2f} {r['Det_new']:8.2f} "
              f"{r['Delta_b']:+8.2f} "
              f"{r['LE_old']:8.2f} {r['LE_new']:8.2f}")
    print("-"*78)
    if rows:
        dets_old = np.array([r['Det_old'] for r in rows])
        dets_new = np.array([r['Det_new'] for r in rows])
        deltas_b = np.array([r['Delta_b'] for r in rows])
        print(f"{'MEAN':<16} {dets_old.mean():8.2f} {dets_new.mean():8.2f} "
              f"{deltas_b.mean():+8.2f}")
    print("="*78)
    print(f"\nWritten: {csv_path}")
    print(f"Written: {md_path}\n")


    if rows:
        ae_delta_b = deltas_b.mean()
        qr_delta_b = -5.95
        diff = ae_delta_b - qr_delta_b
        print("=== β ROUTE BET DECISION ===")
        print(f"  AE mean Delta_b:  {ae_delta_b:+.2f} pp")
        print(f"  QR mean Delta_b:  {qr_delta_b:+.2f} pp (Mod-1 ref)")
        print(f"  AE vs QR diff:    {diff:+.2f} pp")
        if diff < -10:
            print(f"  -> WIN: AE degrades much more, QR robustness advantage.")
        elif diff <= 5:
            print(f"  -> LOSE: comparable robustness, switch to α route.")
        else:
            print(f"  -> REVERSAL: AE more robust — investigate.\n")
