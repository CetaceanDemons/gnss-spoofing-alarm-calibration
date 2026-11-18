# -*- coding: utf-8 -*-
"""Lists the dynamic-class delays longer than 1 s with their limits (Section 4.6).
Run from the repository root (the working folder that also holds the data folders); see --help.
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
S1 = os.path.join(ROOT, 'r1_revision_SEN', 's1')
OUT = os.path.join(ROOT, 'r1_revision_SEN', 's2', 's2c_brief.txt')


def fl(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


B = ['[s2c] v1.0  动态类 onset 时延 > 1 s 的行（β = 0.02；p95 无 β）']
B.append(f'{"score":5s} {"star":14s} {"rule":6s} {"delay_s":>8s} {"thr_z":>8s} {"gm":>8s} {"L":>4s} '
         f'{"首报raw_s":>9s} {"act→onset报警":>12s}')
for k in ('qr', 'ae', 'tcn', 'cn0', 'wscm'):
    with open(os.path.join(S1, f'rev_{k}_star.csv'), encoding='utf-8-sig', newline='') as fh:
        for r in csv.DictReader(fh):
            if r['group'] != 'dynamic':
                continue
            if not (r['rule'] == 'k_p95' or abs(fl(r['beta']) - 0.02) < 1e-9):
                continue
            d = fl(r['delay_onset_s'])
            if d == d and d > 1.0:
                B.append(f'{k:5s} {r["star"]:14s} {r["rule"]:6s} {d:8.2f} {fl(r["thr_z"]):8.1f} {fl(r["gm"]):8.1f} '
                         f'{r["L"]:>4s} {fl(r["det_onset_raw_s"]):9.2f} {r["n_alarm_act_onset"]:>12s}')
with open(OUT, 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(B) + '\n')
print('\n'.join(B))
