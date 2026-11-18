"""Pipeline of rev_main.py for the two hand-crafted scores (carrier-to-noise proxy, WSCM).

--block-len fixes the bootstrap block length (used to test whether the block-length
rule explains the excess of the block-calibrated count rule on the proxy).
Run from the repository root (the working folder that also holds the data folders); see --help.
"""
import numpy as np, os, sys, argparse, time

_here = os.path.dirname(os.path.abspath(__file__))
for _d in (_here, os.path.dirname(_here), os.path.dirname(os.path.dirname(_here))):
    if os.path.exists(os.path.join(_d, 'sqm_beta_neweval_v2.py')):
        if _d not in sys.path:
            sys.path.insert(0, _d)
        break
if _here not in sys.path:
    sys.path.insert(0, _here)

import rev_main as rm
import far_calib_v32 as fc
import sqm_beta_neweval_v2 as sq

PARITY_COL = 'mao_alarm_s@5e-02'


# ---------------------------------------------------------------------------
# Loaders for the hand-crafted scores; the rules are those of rev_main.run_star
# ---------------------------------------------------------------------------
def v1_window(lab, scen, n):
    """Calibration window of the earlier hand-crafted pipeline, used only for a consistency check.
    """
    asr = sq.ALIGN_START_MS.get(scen, 0)
    if scen in sq.ABS_CALIB_SCENARIOS:
        lo = max(0, int(round(sq.CALIB_RAW_S[0] * 1000 - 100 - asr)))
        hi = min(n, int(round(sq.CALIB_RAW_S[1] * 1000 - 100 - asr)))
        m = np.zeros(n, dtype=bool)
        if hi > lo:
            m[lo:hi] = True
        if m.sum() >= 10:
            return m
    return lab == 0


def make_loader(method, a, log, parity):
    gidx = int(round(a.guard_s * 1000))

    def load(path, enc):
        taps, cn0, lab, scen, prn, _ = sq.load_star_v2(path)
        metric = np.asarray(sq.get_metric(method, taps, cn0, a.e_idx, a.l_idx), dtype=np.float64).ravel()
        lab = np.asarray(lab)
        if len(lab) != len(metric):
            raise ValueError(f'{os.path.basename(path)}: labels {len(lab)} 与 {method} 长度 {len(metric)} 不一致')
        bad = ~np.isfinite(metric)
        cm = fc.segments(metric, lab, scen, gidx)['cmask']
        med = float(np.median(metric[cm & ~bad]))
        if bad.any():
            log(f'  [WARN] {os.path.basename(path)}: {method} 有 {int(bad.sum())} 个非有限值，已填为净段中位数')
            metric = np.where(bad, med, metric)
        S = np.abs(metric - med)

        m1 = v1_window(lab, scen, len(metric))
        S2 = np.abs(metric - float(np.median(metric[m1])))
        a5 = sq.consecutive_alarm_idx(S2, float(np.quantile(S2[m1], 0.95)), 5)
        asr = sq.ALIGN_START_MS.get(scen, 0)
        parity[sq.star_name_from_path(path)] = ((a5 + 100 + asr) / 1000.0 - sq.AUTH[scen]) if a5 >= 0 else float('nan')
        return S, lab, scen, prn
    return load


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_pool(log, pool):
    log('\n==== 合并（误报不含 ds8；时延含 ds8）====')
    log(f'{"rule":6s} {"β":>7s} {"group":8s} {"参照":>9s} | {"净段":>9s} {"判定":12s} | {"ho事件":>5s} {"ho率":>9s} '
        f'{"判定":12s} | {"检出":>5s} {"act中位":>8s} | {"onset中位":>9s} {"onset最大":>9s}')
    for p in pool:
        if p['rule'] == 'cusum' and p['beta'] not in rm.BETAS_MAIN:
            continue
        b = f'{p["beta"]:.0e}' if np.isfinite(p['beta']) else '  p95'
        log(f'{p["rule"]:6s} {b:>7s} {p["group"]:8s} {p["ref"]:9.2e} | {p.get("cal_frac", float("nan")):9.2e} '
            f'{p.get("cal_verdict", ""):12s} | {p.get("ho_n_event", 0):5d} {p.get("ho_frac", float("nan")):9.2e} '
            f'{p.get("ho_verdict", ""):12s} | {p["n_det"]:2d}/{p["n_stars"]:<2d} '
            f'{p.get("delay_act_med", float("nan")):+8.3f} | {p.get("delay_onset_med", float("nan")):+9.3f} '
            f'{p.get("delay_onset_max", float("nan")):+9.3f}')


def _vs(v):
    return {'verified': 'W', 'inconclusive': 'C', 'violated': 'E', 'none': '-', 'too_short': 'S'}.get(v, '?')


def write_brief(base_out, method, a, pool, rows, chks, nc, nr, secs):
    """Short summary of the pooled held-out results and delays."""
    b = [f'[version] rev_sqm.py v1.1  method={method}  nboot={a.nboot}  seed={a.seed}  '
         f'块长={"规则" if a.block_len is None else f"固定 {a.block_len}"}  星数={len({r["star"] for r in rows})}',
         '== ho 留出段误报（合并不含 ds8）：率 判定(事件数)；W/C/E = verified/inconclusive/violated ==',
         'rule   β      | all                 | static              | dynamic             | complex             '
         '| 检出 | onset中位 | onset最大']
    pi = {(p['rule'], p['beta'] if np.isfinite(p['beta']) else -1.0, p['group']): p for p in pool}
    for rule, betas in (('cusum', rm.BETAS_MAIN), ('k_blk', rm.BETAS_MAIN), ('k_iid', rm.BETAS_MAIN), ('k_p95', [-1.0])):
        for bt in betas:
            cells = []
            for g in rm.GROUPS:
                p = pi.get((rule, bt, g), {})
                cells.append(f'{p.get("ho_frac", float("nan")):8.2e} {_vs(p.get("ho_verdict", "none"))}'
                             f'({p.get("ho_n_event", 0):3d})  ')
            pa = pi.get((rule, bt, 'all'), {})
            b.append(f'{rule:6s} {("p95" if bt < 0 else f"{bt:.0e}"):6s} | ' + '| '.join(cells) +
                     f'| {pa.get("n_det", 0):2d}/{pa.get("n_stars", 0):<2d}| {pa.get("delay_onset_med", float("nan")):+9.3f} '
                     f'| {pa.get("delay_onset_max", float("nan")):+9.3f}')
    b.append('== 参照值：cusum / k_blk = β；k_iid / k_p95 = 各星独立假设参照值的加权平均 ==')
    for rule in ('k_iid', 'k_p95'):
        bts = rm.BETAS_MAIN if rule == 'k_iid' else [-1.0]
        b.append(f'  {rule}: ' + '  '.join(f'{("p95" if bt < 0 else f"{bt:.0e}")}→{pi.get((rule, bt, "all"), {}).get("ref", float("nan")):.2e}'
                                          for bt in bts))
    bad = [r['star'] for r in rows if r['rule'] == 'cusum' and r['beta'] == rm.BETAS_ALL[0] and r['status'] != 'ok']
    nm = sum(1 for c in chks if c.get('v2_match') == 'match')
    capped = f'{nc}/{nr}' if a.block_len is None else '（块长已固定，此项无意义）'
    b.append(f'[诊断] 块长顶到 500：{capped}；cusum β=0.1 非 ok：{bad if bad else "无"}；取数核对 {nm}/{len(chks)} 一致')
    kb = [r for r in rows if r['rule'] == 'k_blk' and r['status'] != 'ok']
    if kb:
        b.append(f'[诊断] k_blk 非 ok（saturated/degenerate）{len(kb)} 行：' +
                 ', '.join(f'{r["star"]}@{r["beta"]:g}:{r["status"]}' for r in kb[:8]) + (' …' if len(kb) > 8 else ''))
    b.append(f'[done] 用时 {secs / 60:.1f} 分钟')
    with open(base_out + '_brief.txt', 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(b) + '\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', dest='npz_dir', default='.', help=r'processed_Mao 根目录')
    ap.add_argument('--methods', nargs='+', default=['cn0', 'wscm'], choices=['cn0', 'wscm', 'ratio', 'delta'])
    ap.add_argument('--stars', nargs='+', default=None)
    ap.add_argument('--K', type=int, default=5)
    ap.add_argument('--nboot', type=int, default=2000)
    ap.add_argument('--seg', type=float, default=80.0)
    ap.add_argument('--m', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--guard-s', dest='guard_s', type=float, default=0.1)
    ap.add_argument('--e-idx', type=int, default=sq.DEFAULT_E_IDX)
    ap.add_argument('--l-idx', type=int, default=sq.DEFAULT_L_IDX)
    ap.add_argument('--parity-dir', default=None, help=r'取数核对参照目录，默认 <dir>\results_v2b_1e-5')
    ap.add_argument('--block-len', dest='block_len', type=int, default=None,
                    help='[v1.1] 固定块自助块长（C1c 诊断用）；不给 = 按规则（与 v1.0 相同）')
    ap.add_argument('--out', required=True, help=r'输出前缀，例 re_ii\results_c\rev')
    a = ap.parse_args()
    a.pertap_map = {}
    D = os.path.abspath(a.npz_dir)
    pdir = a.parity_dir or os.path.join(D, 'results_v2b_1e-5')
    prefix = os.path.splitext(a.out)[0]
    os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
    items = rm.qr_items(D, a.stars)
    if a.block_len is not None:
        _L_fixed = int(a.block_len)
        fc.pick_block_len = lambda Z, _L=_L_fixed: _L

    for method in a.methods:
        base_out = f'{prefix}_{method}'
        log = fc.Tee(base_out + '_log.txt')
        log('[version] rev_sqm.py v1.1（2026-10-24）')
        if a.block_len is not None:
            log(f'[cfg] ★ 块自助块长固定为 L={a.block_len}（C1c 诊断；cusum 与 k_blk 受影响，k_iid / k_p95 与 v1.0 相同）')
        log(f'[numba] {"ON" if rm.HAVE_NUMBA else "OFF（会很慢）"}')
        log(f'[cfg] method={method}  K={a.K}  K0={rm.K0}  m={a.m}  nboot={a.nboot}  seg={a.seg} s  seed={a.seed}  '
            f'guard={a.guard_s} s' + (f'  E/L tap={a.e_idx}/{a.l_idx}' if method in ('ratio', 'delta') else ''))
        log(f'[cfg] {len(items)} 颗星：' + ', '.join(b for b, _, _ in items))
        if not a.stars and len(items) != 21:
            log(f'[WARN] 星数 {len(items)} ≠ 21')
        parity = {}
        rm.load_any = make_loader(method, a, log, parity)
        reg, mism, rows, chks = {'cal': {}, 'k': {}}, [], [], []
        t0 = time.time()
        for i, it in enumerate(items, 1):
            log(f'[{i}/{len(items)}]')
            rr, chk, _ = rm.run_star(it, a, log, reg, mism, method)
            rows += rr
            if chk:
                chks.append(chk)

        ref_p = os.path.join(pdir, f'sqm_v2_1e-5_{method}.csv')
        if os.path.exists(ref_p):
            ref = {r['star']: r for r in rm._read_csv(ref_p)}
            nd = 0
            for c in chks:
                p = parity.get(c['star'], float('nan'))
                c['parity_mao_alarm_s'] = p
                if c['star'] not in ref:
                    c['v2_match'] = 'no_ref'
                    continue
                v = rm._f(ref[c['star']].get(PARITY_COL, ''))
                c['v2_csv_mao_alarm_s'] = v
                ok = (np.isnan(v) and np.isnan(p)) or (np.isfinite(v) and np.isfinite(p) and abs(v - p) < 1.5e-3)
                c['v2_match'] = 'match' if ok else 'DIFF'
                nd += (not ok)
            log(f'\n[取数核对] 对照 {ref_p} 的 {PARITY_COL}：{len(chks) - nd}/{len(chks)} 颗星一致（容差 1.5 ms）'
                + ('' if nd == 0 else '  [WARN] 有不一致，先把日志发回来再用结果'))
        else:
            log(f'\n[取数核对] 找不到 {ref_p}，跳过')

        rm.save_csv(base_out + '_star.csv', rows, rm.STAR_FIELDS)
        pool = rm.pooled(rows, a.m, rm.GROUPS)
        rm.save_csv(base_out + '_pooled.csv', pool, rm.POOL_FIELDS)
        rm.save_csv(base_out + '_pooled_scen.csv', rm.pooled(rows, a.m, sorted({r['scenario'] for r in rows})),
                    rm.POOL_FIELDS)
        rm.save_csv(base_out + '_check.csv', chks, rm.CHK_FIELDS + ['parity_mao_alarm_s'])
        print_pool(log, pool)
        nc = sum(1 for r in rows if r['rule'] == 'cusum' and r['beta'] == rm.BETAS_ALL[0] and r['L_capped'])
        nr = sum(1 for r in rows if r['rule'] == 'cusum' and r['beta'] == rm.BETAS_ALL[0])
        log(f'\n[诊断] 块长顶到 500 的星：{nc}/{nr}（若占多数，先停下来讨论块长上限）')
        log(f'[saved] {base_out}_star.csv / _pooled.csv / _pooled_scen.csv / _check.csv / _log.txt')
        log(f'[done] {method} 用时 {(time.time() - t0) / 60:.1f} 分钟')
        write_brief(base_out, method, a, pool, rows, chks, nc, nr, time.time() - t0)
        log.close()
