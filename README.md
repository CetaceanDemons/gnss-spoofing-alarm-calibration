# False-alarm control for sequential GNSS spoofing alarms

Code for

> Z. Wang, X. Liao, J. Zhang, S. He, Q. He, Y. Liang, W. Cao, "False-Alarm Control for Sequential GNSS Spoofing Alarms under Temporal Dependence", submitted to *Sensors* (MDPI), 2026.

The article extends the conference paper [3] (ICECAI 2026, in press). The QR-LSTM score comes from the companion study [22] (submitted); its model and training code belong to that study and will be released with it. Bracketed numbers refer to the reference list of the article.

## Pipeline

1. **Receiver** (`receiver/`): the TEXBAT recordings are processed with GNSS-SDR v0.0.20, modified to write the 21 correlator outputs and the loop states of every channel once per millisecond.
2. **Scores** (repository root, `re_ii/code/tcn_*.py`): QR-LSTM per-tap residuals (model of [22]), LSTM autoencoder and TCN prediction errors; the carrier-to-noise proxy and the WSCM are computed from the stored taps. Networks are trained on OAKBAT.
3. **Alarm layer and experiments** (`re_ii/code/`): order-preserving block-bootstrap calibration of the CUSUM and of the count rule, held-out verification with Poisson bounds, rules designed under independence, block means, Markov-chain prediction, receiver-level false alarms, baseline shift, sensitivity.
4. **Revision experiments** (`r1_revision_SEN/code/`): ds3 onset, per-instant detection rates, long clean recordings (TEXBAT cleanStatic, OAKBAT cleanDynamic, TEXBAT cleanDynamic), common-mode events of the clean drive.
5. **Tables and figures** (`re_ii/code/c_tables.py`, `re_ii/figs_journal/code/`, `r1_revision_SEN/code/s6_figs.py`).

## Where each result comes from

| Paper item | Scripts |
|---|---|
| Table 1 (segments, onsets) | `re_ii/code/far_calib_v32.py`, `far_mc_beta_neweval.py`, `r1_revision_SEN/code/s1_onset_ds3.py` |
| Table 2, Figure 5 (block means) | `re_ii/code/c1p_blockcn0.py` |
| Table 3, Figure 6 (count rule) | `re_ii/code/krule_clean_far_v12.py`, `rev_main.py`, `rev_sqm.py` |
| Table 4, Figure 7 (Markov chain) | `re_ii/code/c2_e0_mech.py` |
| Table 5, Figure 8 (held-out verification) | `re_ii/code/rev_main.py`, `rev_sqm.py` |
| Table 6 (long recordings) | `r1_revision_SEN/code/s3a_infer_long.py`, `s3b_verify_long.py`, `s3c_verify_cd.py` |
| Table 7, Figures 10 and A3 (delays) | `r1_revision_SEN/code/s1_onset_ds3.py`, `s1_compare.py`, `s2b_dyn_onset.py`, `s2c_maxdelay.py` |
| Table 8, Figure 11 (per-instant detection rate; bootstrap intervals of ρ_S) | `r1_revision_SEN/code/s2_detrate.py`, `s2b_dyn_onset.py`, `s7_fig11_ci.py` |
| Table A1 (QR score, eight targets) | `re_ii/code/far_calib_v32.py`, `rev_main.py`, `r1_revision_SEN/code/s2b_dyn_onset.py` |
| Table A2 (receiver level) | `re_ii/code/c3_union.py` |
| Table A3 (by scenario class) | `re_ii/code/c_tables.py` |
| Table A4 (alternative activation instants) | `r1_revision_SEN/code/s8_dyn_activation_scan.py` (with `re_ii/code/rev_main.py`, `rev_sqm.py`) |
| Table A5 (sensitivity) | `re_ii/code/rev_factors.py` |
| Table A6 (log-likelihood-ratio CUSUM) | `re_ii/code/c1p_blockcn0.py` |
| Figures 3 and 4 (autocorrelation) | `re_ii/code/e1b_acf_transient.py`, `c1p_blockcn0.py` |
| Figure 9 (baseline shift) | `re_ii/code/c1d_drift.py` |
| Appendix D (event near 97 s, alternative activation instants) | `r1_revision_SEN/code/s3a_infer_long.py`, `s5_cd_events.py`, `s2b_dyn_onset.py`, `s8_dyn_activation_scan.py` |
| Tables 2–5, A2, A3, A6 (assembly) | `re_ii/code/c_tables.py` |
| Figures 1, 3–9, A2 (drawing) | `re_ii/figs_journal/code/figs_journal.py` |
| Figures 2, 10, A3 (drawing) | `r1_revision_SEN/code/s6_figs.py` |
| Figure 11 (drawing, with the intervals) | `r1_revision_SEN/code/s7_fig11_ci.py` |
| Figure A1 (drawing) | `re_ii/code/plot_re_ii.py` |

Networks: `lstm_ae_train.py`, `lstm_ae_train_dynamic.py`, `re_ii/code/tcn_train.py`. Scoring: `dump_pertap_residual_v2.py` (static scenarios), `dump_pertap_residual_v3.py` (ds7, ds8), `dump_pertap_residual_dynamic_v2.py` (ds6), `dump_pertap_residual_dynamic_ds5.py` (ds5), `lstm_ae_inference_dual.py`, `re_ii/code/tcn_inference_dual.py`. Shared modules: `far_mc_beta_neweval.py`, `far_mc_beta_neweval_v2.py`, `sqm_beta_neweval_v2.py`, `experiment1_zscore_ablation.py`.

## Order of execution

1. `receiver/`: process the recordings (see below).
2. Train the networks and score the scenarios.
3. `re_ii/code/`: `far_calib_v32.py` → `krule_clean_far_v12.py` → `rev_main.py --encoder qr|ae|tcn` → `rev_sqm.py` → `c1p_blockcn0.py`, `c2_e0_mech.py`, `c3_union.py`, `c1d_drift.py`, `e1b_acf_transient.py`, `rev_factors.py` → `c_tables.py`.
4. `r1_revision_SEN/code/`: `s1_onset_ds3.py` → `s1_compare.py` → `s2_detrate.py` → `s2b_dyn_onset.py`, `s2c_maxdelay.py`; `cd_r2_check.py` → `s3a_infer_long.py` → `s3b_verify_long.py`, `s3c_verify_cd.py` → `s5_cd_events.py`; then `s7_fig11_ci.py` and `s8_dyn_activation_scan.py` (second revision round).
5. Figures: `re_ii/figs_journal/code/figs_journal.py`, `re_ii/code/plot_re_ii.py`, `r1_revision_SEN/code/s6_figs.py`, `r1_revision_SEN/code/s7_fig11_ci.py`.

Every script is run from the repository root, which is also the folder that holds the processed recordings (`ds3/`, `ds4_r2/`, …, `cleanDynamic_r2/`), the score folders (`pertap_residual_*/`, `lstm_ae_results/`, `tcn_results/`) and the result folders the scripts create. The default paths point to our machine (`J:\Database\...`); most scripts take `--dir` or `--root`, and the scoring and training scripts define their paths at the top. `python <script> --help` lists the options.

## Data and related software

Neither the data nor trained network weights are redistributed here.

- **TEXBAT**, the Texas Spoofing Test Battery [6]: https://radionavlab.ae.utexas.edu/texbat/ (recordings at https://rnl-data.ae.utexas.edu/datastore/texbat/). T. E. Humphreys, J. A. Bhatti, D. P. Shepard and K. D. Wesson, "The Texas spoofing test battery: Toward a standard for evaluating GPS signal authentication techniques," ION GNSS 2012, pp. 3569–3583.
- **OAKBAT**, the Oak Ridge Spoofing and Interference Test Battery [49]: https://doi.org/10.13139/ORNLNCCS/1664429 (metadata at https://github.com/oakbat). A. Albright, S. Powers, J. Bonior and F. Combs, "A tool for furthering GNSS security research: The Oak Ridge spoofing and interference test battery (OAKBAT)," ION GNSS+ 2020, pp. 3697–3712, doi:10.33012/2020.17712.
- **GNSS-SDR**, version 0.0.20: https://gnss-sdr.org, https://github.com/gnss-sdr/gnss-sdr. C. Fernández-Prades, J. Arribas, P. Closas, C. Avilés and L. Esteve, "GNSS-SDR: an open source tool for researchers and developers," in Proc. 24th International Technical Meeting of the Satellite Division of the Institute of Navigation (ION GNSS 2011), Portland, Oregon, 19–23 September 2011, pp. 780–794.
- Onsets and activation instants of the static scenarios follow [50], A. Lemmenes, P. Corbell and S. Gunawardena, "Detailed analysis of the TEXBAT datasets using a high fidelity software GPS receiver," ION GNSS+ 2016, pp. 3027–3032, doi:10.33012/2016.14668; those of ds7 and ds8 follow [51], T. E. Humphreys, "TEXBAT data sets 7 and 8," The University of Texas, 2016.

## Receiver processing (`receiver/`)

GNSS-SDR v0.0.20 with `gnss_sdr_v0020_local_mods.patch` applied: 21 correlators spaced 0.2 chips, and one 160-byte record per channel and millisecond (21 tap magnitudes, prompt I/Q, sample stamp, twelve loop states including C/N0 and the carrier-lock indicator, PRN). Fixed satellites whose acquisition fails are retried in place. The patch applies to the `v0.0.20` tag.

| Recordings | Configuration | Script |
|---|---|---|
| ds3, ds7, ds8 | `current_ds3.conf`, `current_ds7.conf`, `current_ds8.conf`: two channels, PRN 3 and 23 | `run_all.sh` |
| ds4, ds5, ds6 (folders `*_r2`) | `texbat_base.conf` with five fixed satellites per scenario, other channels searching freely | `run_r2.sh` |
| TEXBAT cleanDynamic | `cleanDynamic_r2_used.conf` (fixed satellites 6 9 15 18 21 22); checksums in `cleanDynamic_r2_provenance.txt` | `run_cd_r2.sh` |
| TEXBAT cleanStatic (Section 4.5) | two channels, PRN 3 and 23; the configuration file of this run (1 April 2026) was not kept | — |

`texbat_base.conf` is the later form of the base configuration (eight channels, no fixed satellites). `run_all.sh` wrote each `current_<scenario>.conf` from the earlier form, which differed only in the channel settings, as a comparison of the files shows. `move_texbat_data.py` moves the `.dat` output to the shared folder. The configuration files are byte-identical to those used (MD5 of `texbat_base.conf`: `b72beb042400dc3b52d7d13e78cb6fee`; of `cleanDynamic_r2_used.conf`: `66416757e69323fa2960b77408553b9a`), so their comments remain in Chinese.

## Requirements

Python 3.10 with the versions in `requirements.txt`. TensorFlow is needed only for training and scoring; numba is optional and speeds up the bootstrap. The figures use the Palatino Linotype font.

## Notes on this release

The scripts are the versions that produced the results of the paper. For this release, comments and docstrings were rewritten in English and nine console messages were edited; the code is otherwise unchanged, as a comparison of the syntax trees of the original and released files confirms. Some console messages and argument help texts are still in Chinese. In `receiver/`, the comments of the patch and of the scripts were translated and the home directory of the virtual machine is written as `$HOME`; the code lines of the patch are unchanged. The second revision round (November 2026) added `r1_revision_SEN/code/s7_fig11_ci.py` and `s8_dyn_activation_scan.py`; the appendix table numbers above follow the revised paper.

## License

MIT (see `LICENSE`) for the code in this repository. `receiver/gnss_sdr_v0020_local_mods.patch` modifies GNSS-SDR and is distributed under the GNU General Public License v3.0, as GNSS-SDR is.
