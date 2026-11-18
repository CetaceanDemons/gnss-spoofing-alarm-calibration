"""As dump_pertap_residual_v2.py for ds7 and ds8.

ds7 and ds8 are digital overlays on the cleanStatic recording; their onsets follow
Humphreys et al. (2016). Usage: python dump_pertap_residual_v3.py <seed>.
"""
import sys
import os
import importlib.util

# ---------------------------------------------------------------------------
# args
# ---------------------------------------------------------------------------
if len(sys.argv) != 2:
    print("Usage: python dump_pertap_residual_v3.py <seed>   (e.g. 42)")
    sys.exit(1)
SEED = int(sys.argv[1])

# ---------------------------------------------------------------------------
# paths (edit if your tree differs)
# ---------------------------------------------------------------------------
EXP1_PATH       = r"J:\Database\TEXBAT\processed_Mao\experiment1_zscore_ablation.py"
TEXBAT_ROOT_ABS = r"J:\Database\TEXBAT\processed_Mao"
MODEL_PATH      = (
    rf"J:\Database\OAKBAT\processed_03_26\Mao_base\qr_lstm_results_v2_mod1"
    rf"\qr_lstm_v1_seed{SEED}.keras"
)
OUT_DIR = os.path.join(TEXBAT_ROOT_ABS, f"pertap_residual_seed{SEED}_ds78")


SCENARIOS_TO_DUMP = ['ds7', 'ds8']

USE_GLOBAL = False


PROMPT_IDX_CN0 = 10
NOISE_IDX_CN0  = (0, 20)


AUTH_ONSET = {'ds7': 110.0, 'ds8': 110.0}

# ---------------------------------------------------------------------------
# load exp1 as a module
# ---------------------------------------------------------------------------
if not os.path.exists(EXP1_PATH):
    print(f"ERROR: experiment1_zscore_ablation.py not found at {EXP1_PATH}")
    sys.exit(1)

os.chdir(TEXBAT_ROOT_ABS)
spec = importlib.util.spec_from_file_location("experiment1_zscore_ablation", EXP1_PATH)
exp1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp1)
exp1.TEXBAT_ROOT = TEXBAT_ROOT_ABS

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

W      = exp1.W          # 100
N_TAPS = exp1.N_TAPS     # 21
BATCH  = exp1.BATCH      # 2048

os.makedirs(OUT_DIR, exist_ok=True)


for sc in SCENARIOS_TO_DUMP:
    if sc not in exp1.SCENARIO_CONFIG:
        print(f"[v3][ERROR] {sc} 不在 exp1.SCENARIO_CONFIG! 确认 experiment1_zscore_ablation.py "
              f"里有 {sc} 的配置(dir/channels/clean_end/spoof_start)。")
        sys.exit(1)


def score_pertap_cn0(model, taps_raw, mu, sig, clean_end_idx):
    taps_n = (taps_raw - mu) / sig
    T = len(taps_n)
    N = T - W
    if N <= 0:
        return None, None, None, None, None, None
    per_all = np.zeros((N, N_TAPS), dtype=np.float32)
    for start in range(0, N, BATCH):
        end = min(start + BATCH, N)
        bs = end - start
        X = np.zeros((bs, W, N_TAPS), dtype=np.float32)
        for j in range(bs):
            X[j] = taps_n[start + j: start + j + W]
        q_lo, q_hi = model.predict(X, batch_size=bs, verbose=0)
        y = taps_n[start + W: end + W]
        per_all[start:end] = np.maximum(0, y - q_hi) + np.maximum(0, q_lo - y)
        del X, q_lo, q_hi
    times = np.arange(W, W + N)

    raw_win = taps_raw[W:W + N]
    prompt_pow = (raw_win[:, PROMPT_IDX_CN0].astype(np.float64) ** 2)
    ce = min(clean_end_idx, T)
    clean_hi = max(0, ce - W)
    if clean_hi < 10:
        noise_block = np.concatenate([raw_win[:, NOISE_IDX_CN0[0]],
                                      raw_win[:, NOISE_IDX_CN0[1]]]).astype(np.float64)
        print(f"      [CN0][WARN] clean window <10 samples; "
              f"noise floor estimated from FULL segment edges")
    else:
        noise_block = np.concatenate(
            [raw_win[:clean_hi, NOISE_IDX_CN0[0]],
             raw_win[:clean_hi, NOISE_IDX_CN0[1]]]).astype(np.float64)
    noise_var = float(np.var(noise_block)) + 1e-12
    cn0 = (prompt_pow / noise_var).astype(np.float32)


    return (times, per_all, cn0, prompt_pow.astype(np.float32),
            np.float32(noise_var), raw_win.astype(np.float32))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: model not found: {MODEL_PATH}")
    sys.exit(1)

print(f"[v3] seed={SEED}  model(STATIC)={MODEL_PATH}")
print(f"[v3] normalization = {'GLOBAL (Cal-T)' if USE_GLOBAL else 'LOCAL (Cal-S)'}")
print(f"[v3] scenarios = {SCENARIOS_TO_DUMP}  (ds7/ds8 frequency-locked code-push)")
print(f"[v3] C/N0: prompt idx={PROMPT_IDX_CN0}, noise idx={NOISE_IDX_CN0} (double-sided var)")
print(f"[v3] 预期: ds7/ds8 功率不变 -> C/N0 跳变很小(类 ds4/ds6 功率匹配)")

model = load_model(MODEL_PATH, compile=False)
model.compile(optimizer=Adam(1e-3),
              loss={'q_lower': exp1.pinball_loss(exp1.TAU_L),
                    'q_upper': exp1.pinball_loss(exp1.TAU_U)})
print(f"[v3] model loaded: {model.count_params()} params")

mu_global = sig_global = None
if USE_GLOBAL:
    mu_global, sig_global = exp1.compute_global_mu_sigma()

for sc in SCENARIOS_TO_DUMP:
    cfg = exp1.SCENARIO_CONFIG[sc]
    print(f"\n{'='*60}\n  {sc}  (onset={AUTH_ONSET.get(sc, cfg['onset_sec'])}s, "
          f"spoof_start={cfg['spoof_start']})\n{'='*60}")
    alignment, prn_map = exp1.align_scenario(sc, cfg)
    if alignment is None:
        print(f"  [SKIP] no alignment")
        continue

    if USE_GLOBAL:
        mu, sig = mu_global, sig_global
    else:
        mu, sig = exp1.compute_local_mu_sigma(sc, cfg, alignment)

    onset_auth = AUTH_ONSET.get(sc, cfg['onset_sec'])

    sc_path = os.path.join(TEXBAT_ROOT_ABS, cfg['dir'])
    for ch in cfg['channels']:
        fpath = os.path.join(sc_path, f"trk_ch_{ch}.dat")
        if not os.path.exists(fpath):
            print(f"  [WARN] missing {fpath}")
            continue
        data = np.fromfile(fpath, dtype=exp1.record_dtype)
        prn_val = int(data['prn'][alignment['start']])
        taps = data['taps'][alignment['start']:alignment['end']].astype(np.float32)

        piq = data['prompt_iq'][alignment['start']:alignment['end']].astype(np.float32)
        del data

        n_samples = len(taps)
        labels = np.full(n_samples, -1, dtype=np.int8)
        labels[:cfg['clean_end']] = 0
        if cfg['spoof_start'] < n_samples:
            labels[cfg['spoof_start']:] = 1

        times, per, cn0, prompt_pow, noise_var, taps_raw = score_pertap_cn0(
            model, taps, mu, sig, cfg['clean_end'])
        if per is None:
            print(f"  [SKIP] {sc} ch{ch}: too short")
            continue
        N = len(per)
        lab = labels[W:W + N]
        prompt_iq = piq[W:W + N]

        out = os.path.join(OUT_DIR, f"pertap_{sc}_prn{prn_val}.npz")
        np.savez_compressed(out, per=per, labels=lab, times=times,
                            prn=prn_val, scenario=sc, seed=SEED,
                            onset_sec=onset_auth,
                            spoof_start=cfg['spoof_start'],
                            cn0=cn0, prompt_pow=prompt_pow,
                            noise_var=noise_var,
                            cn0_idx=np.array([PROMPT_IDX_CN0,
                                              NOISE_IDX_CN0[0],
                                              NOISE_IDX_CN0[1]], dtype=np.int32),

                            taps_raw=taps_raw, prompt_iq=prompt_iq)

        import numpy as _np
        cdb = 10*_np.log10(cn0[lab==0].astype(_np.float64).mean()+1e-9)
        sdb = (10*_np.log10(cn0[lab==1].astype(_np.float64).mean()+1e-9)
               if (lab==1).any() else cdb)
        mb = os.path.getsize(out) / 1e6
        print(f"  saved {out}  per={per.shape} taps_raw={taps_raw.shape} "
              f"iq={prompt_iq.shape} cn0={cn0.shape} "
              f"C/N0_jump={sdb-cdb:+.1f}dB  noise_var={float(noise_var):.3g}  "
              f"clean={(lab==0).sum()} spoof={(lab==1).sum()}  ({mb:.1f} MB)")

print(f"\n[v3] DONE. Upload the .npz files in: {OUT_DIR}")
print(f"[v3] 共 {len(SCENARIOS_TO_DUMP)*2} 个文件(ds7/ds8 各 2 颗 PRN3/23)")
print(f"[v3] 若 C/N0_jump 都很小(<3dB) = 证实 ds7/ds8 功率不变、必须看形状")
