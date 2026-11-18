# -*- coding: utf-8 -*-
"""Per-channel check of the GNSS-SDR output for TEXBAT cleanDynamic against ds5 and ds6.

Reports tracked PRN, duration, C/N0 and lock indicators, and losses of lock.
Run from the repository root (the working folder that also holds the data folders); see --help.
"""
import os, sys, argparse
import numpy as np

DT = np.dtype([('taps', '<f4', 21), ('iq', '<f4', 2), ('ts', '<u8'),
               ('st', '<f4', 12), ('ne', '<f8'), ('prn', '<u4')])
FS = 25e6
ROOT = r'J:\Database\TEXBAT\processed_Mao'


def summarize(d, log):
    if not os.path.isdir(d):
        log(f'  [缺] {d}')
        return
    files = sorted(f for f in os.listdir(d) if f.startswith('trk_ch_') and f.endswith('.dat'))
    for f in files:
        p = os.path.join(d, f)
        sz = os.path.getsize(p)
        if sz % DT.itemsize:
            log(f'  {f}: 大小 {sz} 不是 160 的整数倍 → 格式异常')
            continue
        r = np.fromfile(p, DT)
        if r.size == 0:
            log(f'  {f}: 空文件')
            continue
        for prn in np.unique(r['prn']):
            s = r[r['prn'] == prn]
            t = s['ts'] / FS
            dtm = np.median(np.diff(t)) * 1e3 if s.size > 1 else np.nan
            cn0 = s['st'][:, 9]
            lock = s['st'][:, 10]
            log(f'  {f[4:-4]:>5} PRN{int(prn):>3} n={s.size:>7} t=[{t[0]:7.2f},{t[-1]:7.2f}]s '
                f'dt={dtm:5.1f}ms CN0中位={np.median(cn0):5.1f} CN0<35占比={np.mean(cn0 < 35):5.3f} '
                f'锁定量中位={np.median(lock):5.2f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--sets', nargs='+', default=['cleanDynamic_r2', 'ds5_r2', 'ds6_r2'])
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(a.root, 'r1_revision_SEN', 's3', 'cd_r2_check_brief.txt')
    lines = []
    log = lambda s: (print(s), lines.append(s))
    log('cd_r2_check v1.0 | 每行 = 一个通道里的一颗星；t 为录音内绝对时间（timestamp/25e6）')
    for s in a.sets:
        log(f'== {s}')
        summarize(os.path.join(a.root, s), log)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print(f'[写出] {out}')


if __name__ == '__main__':
    main()
