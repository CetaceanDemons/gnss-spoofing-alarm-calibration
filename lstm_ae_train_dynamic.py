"""As lstm_ae_train.py on OAKBAT cleanDynamic (model for ds5 and ds6).

Training satellites, standardisation and output names differ from the static
version; architecture and hyperparameters are the same.
Usage: python lstm_ae_train_dynamic.py --seed 42.
"""

import argparse
import os
import sys
import gc
import time
import random


# CRITICAL: Lock seeds BEFORE importing tf/keras/numpy ops

def lock_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    return np


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed (default 42, matches Mod-1 baseline)')
args = parser.parse_args()
SEED = args.seed

print(f"[LSTM-AE] Locking seed = {SEED}")
np = lock_seeds(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

print(f"[LSTM-AE] TF {tf.__version__}, "
      f"GPU visible: {tf.config.list_physical_devices('GPU')}")


DATA_DIR = r"J:\Database\OAKBAT\processed_03_26\Mao_base\oakbat_single_star_v2"
OUT_DIR  = r"J:\Database\OAKBAT\processed_03_26\Mao_base\lstm_ae_results"
os.makedirs(OUT_DIR, exist_ok=True)


TAG = "dynamic_"

W = 100
N_TAPS = 21
PROMPT_IDX = 10
EPOCHS = 10
BATCH = 512
LR = 0.001

TRAIN_PREFIX = "cleanDynamic_th_0.7"


CLEAN_DYNAMIC_STARS = [
    "cleanDynamic_th_0.7_r2_ch1_prn20",
    "cleanDynamic_th_0.7_r2_ch4_prn15",
    "cleanDynamic_th_0.7_r2_ch5_prn32",
    "cleanDynamic_th_0.7_r2_ch6_prn12",
    "cleanDynamic_th_0.7_r2_ch7_prn8",
]

EXPECTED_PARAMS = 23_381  # 22,016 (LSTM 64) + 1,365 (Dense 21)
                          # vs QR-LSTM v1: 24,746  (= 22,016 + 1,365 * 2)

plt.rcParams.update({'font.family': 'serif', 'font.size': 11,
                      'figure.dpi': 150, 'savefig.dpi': 200})


def discover_train_files(data_dir):
    train_files = []
    missing = []
    for base in CLEAN_DYNAMIC_STARS:
        taps_path = os.path.join(data_dir, base + "_raw_taps.npy")
        y_path    = os.path.join(data_dir, base + "_y_seq.npy")
        if not os.path.exists(taps_path):
            missing.append(taps_path)
            continue
        if not os.path.exists(y_path):
            missing.append(y_path)
            continue
        train_files.append({
            'taps_path': taps_path,
            'y_path': y_path,
            'scenario': TRAIN_PREFIX, 'ch': '-', 'prn': '-',
            'name': base
        })
    if missing:
        print(f"[FATAL] 缺 {len(missing)} 个 cleanDynamic 数据文件:")
        for p in missing:
            print(f"    {p}")
        print(f"  确认 DATA_DIR 指对了: {data_dir}")
        sys.exit(1)
    return train_files


def make_windows_strided(arr, W):
    T, F = arr.shape
    if T <= W:
        return np.empty((0, W, F), dtype=arr.dtype), \
               np.empty((0, F), dtype=arr.dtype)
    shape = (T - W, W, F)
    strides = (arr.strides[0], arr.strides[0], arr.strides[1])
    X = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
    Y = arr[W:]
    return X, Y


def sample_training_data(train_files, mu, sigma, W, max_per_star=50000):
    X_list, Y_list = [], []
    for info in train_files:
        taps = np.load(info['taps_path']).astype(np.float32)
        taps_normed = (taps - mu) / sigma
        X_view, Y_view = make_windows_strided(taps_normed, W)
        n = len(X_view)
        if n == 0:
            continue
        if n > max_per_star:
            idx = np.random.choice(n, max_per_star, replace=False)
            idx.sort()
        else:
            idx = np.arange(n)
        X_list.append(X_view[idx].copy())
        Y_list.append(Y_view[idx].copy())
        print(f"  {info['name']}: {len(idx)} windows (from {n} available)")
        del taps, taps_normed, X_view, Y_view
        gc.collect()
    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)
    del X_list, Y_list
    gc.collect()
    return X, Y


# ---------------------------------------------------------------------------
# MODEL  (Architecture 3: QR-LSTM-analog with single MSE head)
# ---------------------------------------------------------------------------
def build_lstm_ae(W=100, n_taps=21):
    """LSTM-AE Arch 3:
        Input(W, 21) -> LSTM(64) -> Dropout(0.1) -> Dense(21) [recon]
        Loss: MSE  on  (X[t] - X_hat[t])
        Total params: 23,381
    """
    inp = Input(shape=(W, n_taps))
    x = LSTM(64, return_sequences=False)(inp)
    x = Dropout(0.1)(x)
    out = Dense(n_taps, name='recon')(x)
    m = Model(inputs=inp, outputs=out)
    m.compile(optimizer=Adam(LR), loss='mse')
    return m


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 65)
    print(f"LSTM-AE DYNAMIC training (Arch 3: QR-LSTM-analog), seed = {SEED}")
    print(f"Train corpus: cleanDynamic x5  (静态版是 cleanStatic x5)")
    print(f"Architecture: Input({W},{N_TAPS}) -> LSTM(64) -> Dropout -> "
          f"Dense({N_TAPS})")
    print(f"Expected params: {EXPECTED_PARAMS}  "
          f"(QR-LSTM: 24,746; we share LSTM, drop one Dense head)")
    print(f"Loss: MSE on next-step 21-tap reconstruction")
    print(f"W={W}, Epochs={EPOCHS}, Batch={BATCH}, LR={LR}   "
          f"<- ★ 与静态 AE 完全相同(不是 dynamic_experiment 的 30/256)")
    print("=" * 65)

    train_files = discover_train_files(DATA_DIR)
    print(f"\nTraining stars ({len(train_files)}):")
    for f in train_files:
        print(f"  {f['name']}")
    if len(train_files) != 5:
        print(f"[FATAL] Expected 5 cleanDynamic stars, found "
              f"{len(train_files)}. Aborting.")
        sys.exit(1)


    print("\n--- Computing normalization from 5 cleanDynamic stars ---")
    clean_chunks = []
    for info in train_files:
        t = np.load(info['taps_path']).astype(np.float32)
        clean_chunks.append(t)
        print(f"  {info['name']}: {len(t)} samples")
    all_clean = np.concatenate(clean_chunks, axis=0)
    mu = np.mean(all_clean, axis=0, keepdims=True)
    sigma = np.std(all_clean, axis=0, keepdims=True) + 1e-8
    print(f"  mu range=[{mu.min():.3e}, {mu.max():.3e}]")
    print(f"  sigma range=[{sigma.min():.3e}, {sigma.max():.3e}]")
    del all_clean, clean_chunks
    gc.collect()

    musigma_path = os.path.join(OUT_DIR, f"mu_sigma_{TAG}seed{SEED}.npz")
    np.savez_compressed(musigma_path, mu=mu, sigma=sigma, seed=SEED)
    print(f"  saved: {musigma_path}")

    print("\n--- Building training set ---")
    X_train, Y_train = sample_training_data(train_files, mu, sigma, W,
                                             max_per_star=50000)
    print(f"\nTotal training: {len(X_train)}")

    perm = np.random.permutation(len(X_train))
    X_train, Y_train = X_train[perm], Y_train[perm]

    n_val = int(len(X_train) * 0.1)
    X_val, Y_val = X_train[:n_val], Y_train[:n_val]
    X_tr, Y_tr = X_train[n_val:], Y_train[n_val:]
    del X_train, Y_train
    gc.collect()
    print(f"Train: {len(X_tr)}, Val: {len(X_val)}")

    print("\n--- Training LSTM-AE ---")
    model = build_lstm_ae(W, N_TAPS)
    model.summary()

    actual_params = model.count_params()
    print(f"\n[SANITY] Actual trainable params: {actual_params}")
    print(f"[SANITY] Expected: {EXPECTED_PARAMS}")
    if actual_params != EXPECTED_PARAMS:
        print(f"[ERROR] Param count mismatch ({actual_params} vs "
              f"{EXPECTED_PARAMS}). Aborting.")
        sys.exit(1)
    print(f"[SANITY OK] Architecture matches Arch-3 spec.")

    t0 = time.time()
    hist = model.fit(
        X_tr, Y_tr,
        validation_data=(X_val, Y_val),
        epochs=EPOCHS, batch_size=BATCH,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)
        ],
        verbose=1
    )
    train_time = time.time() - t0
    print(f"\nTraining done in {train_time:.0f}s ({train_time/60:.1f} min)")

    model_path = os.path.join(OUT_DIR, f"lstm_ae_{TAG}seed{SEED}.keras")
    model.save(model_path)
    print(f"saved: {model_path}")

    history_path = os.path.join(OUT_DIR, f"train_history_{TAG}seed{SEED}.npz")
    np.savez_compressed(
        history_path,
        loss=np.array(hist.history['loss']),
        val_loss=np.array(hist.history['val_loss']),
        seed=SEED,
        train_time_s=train_time,
        actual_params=actual_params,
        epochs_run=len(hist.history['loss']),
    )
    print(f"saved: {history_path}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hist.history['loss'], label='Train', marker='o', markersize=4)
    ax.plot(hist.history['val_loss'], label='Val', marker='s', markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.legend()
    ax.set_title(f'LSTM-AE Arch-3 DYNAMIC (cleanDynamic x5), seed={SEED}, '
                 f'final val_loss={hist.history["val_loss"][-1]:.5f}')
    ax.grid(True, ls=':', alpha=0.4)
    plt.tight_layout()
    curves_path = os.path.join(OUT_DIR, f"training_curves_{TAG}seed{SEED}.png")
    plt.savefig(curves_path)
    plt.close()
    print(f"saved: {curves_path}")

    print("\n" + "=" * 65)
    print(f"DONE dynamic seed={SEED}")
    print(f"  final train_loss = {hist.history['loss'][-1]:.5f}")
    print(f"  final val_loss   = {hist.history['val_loss'][-1]:.5f}")
    print(f"  epochs run       = {len(hist.history['loss'])}")
    print(f"  params           = {actual_params}")
    print(f"  output dir       = {OUT_DIR}")
    print("=" * 65)
