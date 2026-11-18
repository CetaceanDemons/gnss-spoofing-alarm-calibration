"""Trains the TCN next-epoch predictor (Section 4.1).

Dilated causal convolutions (seven residual blocks, 39 filters, kernel 2, dilations 1 to
64), with the data, windowing and standardisation of the LSTM autoencoder; static model
on OAKBAT cleanStatic, --dynamic on OAKBAT cleanDynamic.
Run from the repository root (the working folder that also holds the data folders); see --help.
"""

import argparse
import os
import sys
import gc
import time
import random


def lock_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    return np


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--dynamic', action='store_true',
                    help='用 cleanDynamic×5 训练动态模型（与 lstm_ae_train_dynamic.py 同一批星）')
parser.add_argument('--smoke', action='store_true',
                    help='管线检查：每星 4000 窗、1 epoch，产物带 _SMOKE 后缀，并估计正式训练每 epoch 用时')
parser.add_argument('--data-dir', default=r"J:\Database\OAKBAT\processed_03_26\Mao_base\oakbat_single_star_v2")
parser.add_argument('--out-dir', default=r"J:\Database\OAKBAT\processed_03_26\Mao_base\tcn_results")
args = parser.parse_args()
SEED = args.seed
DYNAMIC = args.dynamic
SMOKE = args.smoke

print(f"[TCN] Locking seed = {SEED}")
np = lock_seeds(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv1D, Add, Activation, Cropping1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

print(f"[TCN] TF {tf.__version__}, GPU visible: {tf.config.list_physical_devices('GPU')}")


DATA_DIR = args.data_dir
OUT_DIR = args.out_dir
os.makedirs(OUT_DIR, exist_ok=True)

TAG = "dynamic_" if DYNAMIC else ""
SUF = "_SMOKE" if SMOKE else ""

W = 100
N_TAPS = 21
PROMPT_IDX = 10
EPOCHS = 1 if SMOKE else 10
BATCH = 512
LR = 0.001
MAX_PER_STAR = 4000 if SMOKE else 50000
MAX_PER_STAR_FULL = 50000

FILTERS = 39
KERNEL = 2
DILATIONS = (1, 2, 4, 8, 16, 32, 64)
RECEPTIVE_FIELD = 1 + (KERNEL - 1) * sum(DILATIONS)          # 128
EXPECTED_PARAMS = 23_265


TRAIN_PREFIX = "cleanDynamic_th_0.7" if DYNAMIC else "cleanStatic_th_0.7"

CLEAN_DYNAMIC_STARS = [
    "cleanDynamic_th_0.7_r2_ch1_prn20",
    "cleanDynamic_th_0.7_r2_ch4_prn15",
    "cleanDynamic_th_0.7_r2_ch5_prn32",
    "cleanDynamic_th_0.7_r2_ch6_prn12",
    "cleanDynamic_th_0.7_r2_ch7_prn8",
]

plt.rcParams.update({'font.family': 'serif', 'font.size': 11, 'figure.dpi': 150, 'savefig.dpi': 200})


def discover_static(data_dir):
    train_files = []
    for fn in sorted(os.listdir(data_dir)):
        if not fn.endswith("_raw_taps.npy"):
            continue
        base = fn.replace("_raw_taps.npy", "")
        y_path = os.path.join(data_dir, base + "_y_seq.npy")
        if not os.path.exists(y_path):
            continue
        parts = base.split("_")
        if parts[0] != "cleanStatic":
            continue
        sc = "cleanStatic_th_" + parts[2]
        ch, prn = parts[3], parts[4]
        train_files.append({'taps_path': os.path.join(data_dir, fn), 'y_path': y_path,
                            'scenario': sc, 'ch': ch, 'prn': prn, 'name': f"{sc}_{ch}_{prn}"})
    return train_files


def discover_dynamic(data_dir):
    train_files, missing = [], []
    for base in CLEAN_DYNAMIC_STARS:
        taps_path = os.path.join(data_dir, base + "_raw_taps.npy")
        y_path = os.path.join(data_dir, base + "_y_seq.npy")
        if not os.path.exists(taps_path):
            missing.append(taps_path)
            continue
        if not os.path.exists(y_path):
            missing.append(y_path)
            continue
        train_files.append({'taps_path': taps_path, 'y_path': y_path,
                            'scenario': TRAIN_PREFIX, 'ch': '-', 'prn': '-', 'name': base})
    if missing:
        print(f"[FATAL] 缺 {len(missing)} 个 cleanDynamic 数据文件:")
        for p in missing:
            print(f"    {p}")
        sys.exit(1)
    return train_files


def make_windows_strided(arr, W):
    T, F = arr.shape
    if T <= W:
        return np.empty((0, W, F), dtype=arr.dtype), np.empty((0, F), dtype=arr.dtype)
    shape = (T - W, W, F)
    strides = (arr.strides[0], arr.strides[0], arr.strides[1])
    X = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
    Y = arr[W:]
    return X, Y


def sample_training_data(train_files, mu, sigma, W, max_per_star=50000):
    X_list, Y_list, n_avail = [], [], []
    for info in train_files:
        taps = np.load(info['taps_path']).astype(np.float32)
        taps_normed = (taps - mu) / sigma
        X_view, Y_view = make_windows_strided(taps_normed, W)
        n = len(X_view)
        n_avail.append(n)
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
    return X, Y, n_avail


def build_tcn(W=100, n_taps=21):
    inp = Input(shape=(W, n_taps))
    x = Conv1D(FILTERS, 1, name='proj')(inp)
    for d in DILATIONS:
        h = Activation('relu', name=f'relu_d{d}')(x)
        h = Conv1D(FILTERS, KERNEL, padding='causal', dilation_rate=d, name=f'tcn_d{d}')(h)
        x = Add(name=f'res_d{d}')([x, h])
    x = Activation('relu', name='relu_out')(x)
    x = Cropping1D((W - 1, 0), name='last_step')(x)
    x = Flatten(name='flat')(x)
    x = Dropout(0.1)(x)
    out = Dense(n_taps, name='recon')(x)
    m = Model(inputs=inp, outputs=out)
    m.compile(optimizer=Adam(LR), loss='mse')
    return m


class EpochTimer(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.times = []

    def on_epoch_begin(self, epoch, logs=None):
        self._t = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.time() - self._t)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    t_all = time.time()
    brief = []

    def B(s=''):
        print(s)
        brief.append(s)

    B('[version] tcn_train.py v1.0（2026-10-24）')
    B(f'[cfg] {"DYNAMIC (cleanDynamic×5)" if DYNAMIC else "STATIC (cleanStatic×5)"}  seed={SEED}'
      + ('  [SMOKE：每星 4000 窗、1 epoch，不是正式模型]' if SMOKE else ''))
    B(f'[cfg] TF {tf.__version__}  W={W}  epochs={EPOCHS}  batch={BATCH}  lr={LR}  max/星={MAX_PER_STAR}')
    B(f'[cfg] TCN: proj 1×1 → {len(DILATIONS)} 层预激活残差 Conv1D({FILTERS}, k={KERNEL}, causal, d={list(DILATIONS)})'
      f' → 末时刻 → Dropout(0.1) → Dense({N_TAPS})；感受野 {RECEPTIVE_FIELD}')
    if RECEPTIVE_FIELD < W:
        B(f'[FATAL] 感受野 {RECEPTIVE_FIELD} < W={W}')
        sys.exit(1)

    train_files = discover_dynamic(DATA_DIR) if DYNAMIC else discover_static(DATA_DIR)
    print(f"\nTraining stars ({len(train_files)}):")
    for f in train_files:
        print(f"  {f['name']}")
    if len(train_files) != 5:
        msg = f"Expected 5 {'cleanDynamic' if DYNAMIC else 'cleanStatic'} stars, found {len(train_files)}"
        if DYNAMIC:
            B(f'[FATAL] {msg}')
            sys.exit(1)
        B(f'[WARN] {msg}')
    B(f'[data] {len(train_files)} 颗训练星：' + ', '.join(f['name'] for f in train_files))


    clean_chunks = [np.load(info['taps_path']).astype(np.float32) for info in train_files]
    all_clean = np.concatenate(clean_chunks, axis=0)
    mu = np.mean(all_clean, axis=0, keepdims=True)
    sigma = np.std(all_clean, axis=0, keepdims=True) + 1e-8
    print(f"  mu range=[{mu.min():.3e}, {mu.max():.3e}]  sigma range=[{sigma.min():.3e}, {sigma.max():.3e}]")
    del all_clean, clean_chunks
    gc.collect()
    musigma_path = os.path.join(OUT_DIR, f"mu_sigma_{TAG}seed{SEED}{SUF}.npz")
    np.savez_compressed(musigma_path, mu=mu, sigma=sigma, seed=SEED)

    print("\n--- Building training set ---")
    X_train, Y_train, n_avail = sample_training_data(train_files, mu, sigma, W, max_per_star=MAX_PER_STAR)
    perm = np.random.permutation(len(X_train))
    X_train, Y_train = X_train[perm], Y_train[perm]
    n_val = int(len(X_train) * 0.1)
    X_val, Y_val = X_train[:n_val], Y_train[:n_val]
    X_tr, Y_tr = X_train[n_val:], Y_train[n_val:]
    del X_train, Y_train
    gc.collect()
    B(f'[data] Train {len(X_tr)}  Val {len(X_val)}  （首个训练窗索引指纹 perm[:3]={perm[:3].tolist()}）')

    model = build_tcn(W, N_TAPS)
    model.summary()
    actual_params = model.count_params()
    B(f'[SANITY] params {actual_params}（预期 {EXPECTED_PARAMS}；LSTM-AE 23,381）')
    if actual_params != EXPECTED_PARAMS:
        B('[ERROR] 参数数不符，中止')
        sys.exit(1)

    timer = EpochTimer()
    t0 = time.time()
    hist = model.fit(
        X_tr, Y_tr, validation_data=(X_val, Y_val), epochs=EPOCHS, batch_size=BATCH,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                   tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5),
                   timer],
        verbose=1)
    train_time = time.time() - t0

    model_path = os.path.join(OUT_DIR, f"tcn_{TAG}seed{SEED}{SUF}.keras")
    model.save(model_path)


    m2 = load_model(model_path, compile=False)
    p1 = model.predict(X_val[:1024], verbose=0)
    p2 = m2.predict(X_val[:1024], verbose=0)
    maxdiff = float(np.max(np.abs(p1 - p2)))
    out_shape = m2.output_shape
    rt_ok = (m2.count_params() == EXPECTED_PARAMS and isinstance(out_shape, tuple) and len(out_shape) == 2
             and out_shape[1] == N_TAPS and maxdiff < 1e-5)
    del m2

    history_path = os.path.join(OUT_DIR, f"train_history_{TAG}seed{SEED}{SUF}.npz")
    np.savez_compressed(history_path, loss=np.array(hist.history['loss']),
                        val_loss=np.array(hist.history['val_loss']), seed=SEED, train_time_s=train_time,
                        actual_params=actual_params, epochs_run=len(hist.history['loss']),
                        epoch_times_s=np.array(timer.times))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hist.history['loss'], label='Train', marker='o', markersize=4)
    ax.plot(hist.history['val_loss'], label='Val', marker='s', markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.legend()
    ax.set_title(f'TCN {"DYNAMIC" if DYNAMIC else "STATIC"}, seed={SEED}, '
                 f'final val_loss={hist.history["val_loss"][-1]:.5f}')
    ax.grid(True, ls=':', alpha=0.4)
    plt.tight_layout()
    curves_path = os.path.join(OUT_DIR, f"training_curves_{TAG}seed{SEED}{SUF}.png")
    plt.savefig(curves_path)
    plt.close()

    vl = hist.history['val_loss']
    B('')
    B(f'[train] epochs run {len(vl)}  用时 {train_time:.0f} s（{train_time / 60:.1f} min）')
    B('[train] 每 epoch 用时 s：' + ' '.join(f'{t:.0f}' for t in timer.times))
    B('[train] loss    ：' + ' '.join(f'{v:.5f}' for v in hist.history['loss']))
    B('[train] val_loss：' + ' '.join(f'{v:.5f}' for v in vl))
    B(f'[train] best val_loss {min(vl):.5f}（第 {int(np.argmin(vl)) + 1} epoch；restore_best_weights 已恢复到这一轮）')
    if SMOKE:
        n_full = int(sum(min(n, MAX_PER_STAR_FULL) for n in n_avail) * 0.9)
        est = timer.times[0] * n_full / max(1, len(X_tr))
        B(f'[estimate] 正式训练集约 {n_full} 窗 → 每 epoch 约 {est / 60:.1f} min，10 epoch 约 {est * 10 / 60:.0f} min'
          '（冒烟第 1 个 epoch 含图编译开销，略偏高）')
    B(f'[roundtrip] load_model 回读：params {EXPECTED_PARAMS}，output_shape {out_shape}，'
      f'预测最大差 {maxdiff:.2e} → {"OK" if rt_ok else "FAIL（先别跑 TCN-b，把本文件发回）"}')
    B(f'[saved] {model_path}')
    B(f'[saved] {musigma_path}')
    B(f'[saved] {history_path}')
    B(f'[saved] {curves_path}')
    B(f'[done] 总用时 {(time.time() - t_all) / 60:.1f} min')
    brief_path = os.path.join(OUT_DIR, f"train_brief_{TAG}seed{SEED}{SUF}.txt")
    with open(brief_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(brief) + '\n')
    print(f'[saved] {brief_path}  ← 只回传这个文件')
