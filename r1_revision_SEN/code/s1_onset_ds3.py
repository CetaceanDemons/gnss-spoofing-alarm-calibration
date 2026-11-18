# -*- coding: utf-8 -*-
"""Reruns the pipeline with the ds3 onset at 118.9 s (Lemmenes et al., 2016).

Patches the onset tables in memory and runs rev_main.py (qr, ae, tcn) and rev_sqm.py
(cn0, wscm) unchanged; only the onset-related columns of ds3 should change.
Run from the repository root (the working folder that also holds the data folders); see --help.
"""
import argparse
import os
import runpy
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEF_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))   # J:\Database\TEXBAT\processed_Mao
# ds3 onset before and after the correction (Lemmenes et al., 2016).
OLD = {'ds3': 119.6}
NEW = {'ds3': 118.9}
JOBS = ('qr', 'ae', 'tcn', 'sqm')
NOREF = '__s1_no_regress__.csv'


# ---------------------------------------------------------------------------
# Onset tables patched in memory; the modules on disk stay unchanged
# ---------------------------------------------------------------------------
def patch_auth(root, data_dir, out_dir):
    """Sets the ds3 onset in the three onset tables and records the check."""
    code = os.path.join(root, 're_ii', 'code')
    for p in (root, code):
        if p not in sys.path:
            sys.path.insert(0, p)
    import far_mc_beta_neweval as v1
    import far_calib_v32 as fc
    import krule_clean_far_v12 as kr
    import far_mc_beta_neweval_v2 as v2
    import sqm_beta_neweval_v2 as sq
    copies = {'v1 本体（far_calib_v32 / krule_v12 / rev_main 共用）': v1.AUTH,
              'far_mc_beta_neweval_v2 私载的 v1 副本': v2.AUTH,
              'sqm_beta_neweval_v2 私载的 v1 副本': sq.AUTH}
    lines = [f'[s1] {time.strftime("%Y-%m-%d %H:%M:%S")}  root={root}  data={data_dir}']
    for name, d in copies.items():
        for k, v in NEW.items():
            if abs(d[k] - OLD[k]) > 1e-9 and abs(d[k] - v) > 1e-9:
                raise SystemExit(f'[STOP] {name} 里 {k} = {d[k]}，既不是旧值 {OLD[k]} 也不是新值 {v}')
            d[k] = v
        lines.append(f'  {name}: id={id(d)}  ds3={d["ds3"]}  其余={ {k: x for k, x in d.items() if k != "ds3"} }')
    assert fc.AUTH is v1.AUTH and kr.AUTH is v1.AUTH, 'far_calib_v32 / krule_v12 的 AUTH 不是 v1 本体'
    lines.append(f'  far_calib_v32.AUTH is v1 本体: {fc.AUTH is v1.AUTH}；krule_v12.AUTH is v1 本体: {kr.AUTH is v1.AUTH}')
    lines.append(f'  ACTIVATION_RAW（不改）: {fc.ACTIVATION_RAW}')
    with open(os.path.join(out_dir, 's1_patch_check.txt'), 'a', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return copies


# ---------------------------------------------------------------------------
# rev_main.py / rev_sqm.py run as __main__ with the patched tables
# ---------------------------------------------------------------------------
def run_one(job, root, data_dir, out_dir, extra):
    patch_auth(root, data_dir, out_dir)
    code = os.path.join(root, 're_ii', 'code')
    if job in ('qr', 'ae', 'tcn'):
        script = os.path.join(code, 'rev_main.py')
        argv = [script, '--dir', data_dir, '--encoder', job,
                '--regress-cal', NOREF, '--regress-k', NOREF,
                '--out', os.path.join(out_dir, f'rev_{job}.csv')] + extra
    else:
        script = os.path.join(code, 'rev_sqm.py')
        argv = [script, '--dir', data_dir, '--methods', 'cn0', 'wscm',
                '--out', os.path.join(out_dir, 'rev')] + extra
    print(f'[s1] 运行: {" ".join(argv)}')
    sys.argv = argv
    runpy.run_path(script, run_name='__main__')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--job', required=True, choices=JOBS + ('all',))
    ap.add_argument('--root', default=DEF_ROOT, help=r'代码根目录（含 far_mc_beta_neweval.py 与 re_ii\code）')
    ap.add_argument('--dir', dest='data_dir', default=None, help='数据根目录，默认与 --root 相同')
    ap.add_argument('--out-dir', default=None, help=r'默认 <root>\r1_revision_SEN\s1')
    a, extra = ap.parse_known_args()
    root = os.path.abspath(a.root)
    data_dir = os.path.abspath(a.data_dir or root)
    out_dir = os.path.abspath(a.out_dir or os.path.join(root, 'r1_revision_SEN', 's1'))
    os.makedirs(out_dir, exist_ok=True)

    if a.job != 'all':
        run_one(a.job, root, data_dir, out_dir, extra)
        return
    t0 = time.time()
    for j in JOBS:
        t1 = time.time()
        cmd = [sys.executable, os.path.abspath(__file__), '--job', j, '--root', root,
               '--dir', data_dir, '--out-dir', out_dir] + extra
        r = subprocess.run(cmd)
        print(f'[s1] {j} 结束，返回码 {r.returncode}，用时 {(time.time() - t1) / 60:.1f} min')
        if r.returncode != 0:
            raise SystemExit(f'[STOP] {j} 失败，先把 {out_dir} 里对应的 _log.txt 发回来')
    print(f'[s1] 全部完成，用时 {(time.time() - t0) / 60:.1f} min。下一步：python r1_revision_SEN\\code\\s1_compare.py')


if __name__ == '__main__':
    main()
