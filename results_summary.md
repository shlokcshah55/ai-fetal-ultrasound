# Results Summary

Current date: 2026-06-11. All metric values are rounded to 3 decimal places. Values reported as `mean +/- std` use the saved folds named in the relevant table unless explicitly marked as single-fold only. No accuracy values are reported.

## 1. RUN INVENTORY

Run date/time below is the saved result/checkpoint file modification time, because the result CSVs do not contain an explicit run-start timestamp.

### Cross-validation result sets

| Method/result set | Folds available | Run date/time | Checkpoint/config provenance |
|---|---:|---|---|
| LR baseline | 3/3 | 2026-06-02 09:38:55 +0100 | `results/heldout_disease_logreg_cv_summary.csv`; checkpoints `checkpoints/heldout_disease_logreg_fold{0,1,2}of3_mean_LogisticRegression.joblib`; config `config.yaml`; split metadata `results/heldout_disease_logreg_fold{0,1,2}of3_split_info.json` |
| MLP head | 3/3 | 2026-06-02 09:38:55 +0100 | `results/heldout_disease_baseline_cv_summary.csv`; checkpoints `checkpoints/heldout_disease_baseline_fold{0,1,2}of3_mean_MLP.pt`; config `config.yaml`; split metadata `results/heldout_disease_baseline_fold{0,1,2}of3_split_info.json` |
| MLP + MC Dropout, p=0.3, T=10 | 3/3 | 2026-06-02 09:38:55 +0100 | `results/heldout_disease_mc_dropout_mean_p0.3_T10_cv_summary.csv`; checkpoints `checkpoints/heldout_disease_mc_dropout_fold{0,1,2}of3_mean_p0.3_mlp.pt`; config `config.yaml` |
| MLP + MC Dropout, p=0.3, T=50 | 3/3 | 2026-06-02 09:38:55 +0100 | `results/heldout_disease_mc_dropout_mean_p0.3_T50_cv_summary.csv`; checkpoints `checkpoints/heldout_disease_mc_dropout_fold{0,1,2}of3_mean_p0.3_mlp.pt`; config `config.yaml` |
| MLP + MC Dropout, p=0.3, T=100 | 3/3 | 2026-06-02 09:38:55 +0100 | `results/heldout_disease_mc_dropout_mean_p0.3_T100_cv_summary.csv`; checkpoints `checkpoints/heldout_disease_mc_dropout_fold{0,1,2}of3_mean_p0.3_mlp.pt`; config `config.yaml` |
| MLP + MC Dropout, p=0.3, T=1000 | 3/3 | 2026-06-02 09:38:55 +0100 | `results/heldout_disease_mc_dropout_mean_p0.3_T1000_cv_summary.csv`; checkpoints `checkpoints/heldout_disease_mc_dropout_fold{0,1,2}of3_mean_p0.3_mlp.pt`; config `config.yaml` |
| Energy MLP | 3/3 | 2026-06-02 09:38:55 +0100 | `results/heldout_disease_energy_cv_summary.csv`; checkpoints `checkpoints/heldout_disease_energy_fold{0,1,2}of3_mean_t1_energy_mlp.pt`; config `config.yaml` |
| VOS MLP | 3/3 | 2026-06-02 09:38:55 +0100 | `results/heldout_disease_vos_cv_summary.csv`; checkpoints `checkpoints/heldout_disease_vos_fold{0,1,2}of3_mean_vos_mlp.pt` and Gaussian stats `checkpoints/heldout_disease_vos_fold{0,1,2}of3_mean_vos_gaussian_stats.npz`; config `config.yaml` |
| DINOv2 representation probe | 3/3 | latest 2026-06-10 11:18:31 +0100 | `results/heldout_disease_representation_probe_fold{0,1,2}of3_mean_summary.csv`; checkpoint/config id `facebookresearch_dinov2_vitb14` |
| EDL softplus ann25 kl0.10 | 3/3 | 2026-06-02 09:38:55 +0100 | `results/heldout_disease_edl_mean_softplus_ann25_kl0p10_cv_summary.csv`; checkpoints `checkpoints/heldout_disease_edl_fold{0,1,2}of3_mean_softplus_ann25_kl0p10_edl_mlp.pt`; config `config.yaml` |
| EDL softplus ann25 kl0.10, T=10 | 3/3 | 2026-06-02 09:38:55 +0100 | `results/heldout_disease_edl_mean_softplus_ann25_kl0p10_T10_cv_summary.csv`; same EDL fold checkpoints as above |
| EDL softplus ann25 kl0.10, T=100 | 3/3 | 2026-06-02 09:38:55 +0100 | `results/heldout_disease_edl_mean_softplus_ann25_kl0p10_T100_cv_summary.csv`; same EDL fold checkpoints as above |
| EDL softplus ann25 kl0.10, T=1000 | 3/3 | 2026-06-02 09:38:55 +0100 | `results/heldout_disease_edl_mean_softplus_ann25_kl0p10_T1000_cv_summary.csv`; same EDL fold checkpoints as above |
| Fetal-clip LR/probe baseline | 3/3 | latest 2026-06-10 22:22:59 +0100 | `results/heldout_disease_representation_probe_fetal_clip_mean_cv_summary.csv` and fold files `results/heldout_disease_representation_probe_fetal_clip_fold{0,1,2}of3_mean_summary.csv`; checkpoints `checkpoints/heldout_disease_baseline_fetal_clip_fold{0,1,2}of3_mean_LogisticRegression.joblib`; checkpoint/config id `fetalclip_weights_c3ea7e11_fetalclip_config_e5fc56d1` |
| Fetal-clip MLP head | 3/3 | 2026-06-10 22:22:59 +0100 | `results/heldout_disease_baseline_fetal_clip_cv_summary.csv` and fold files `results/heldout_disease_baseline_fetal_clip_fold{0,1,2}of3_summary.csv`; checkpoints `checkpoints/heldout_disease_baseline_fetal_clip_fold{0,1,2}of3_mean_MLP.pt`; same fetal-clip checkpoint/config id |
| Fetal-clip representation probe | 3/3 | latest 2026-06-10 22:22:59 +0100 | `results/heldout_disease_representation_probe_fetal_clip_mean_cv_summary.csv` and fold files `results/heldout_disease_representation_probe_fetal_clip_fold{0,1,2}of3_mean_summary.csv`; same fetal-clip checkpoint/config id |
| Fetal-clip MC Dropout, p=0.3, T=100 | 3/3 | latest 2026-06-10 22:22:59 +0100 | `results/heldout_disease_mc_dropout_fetal_clip_mean_p0.3_T100_cv_summary.csv` and fold files `results/heldout_disease_mc_dropout_fetal_clip_fold{0,1,2}of3_mean_p0.3_T100_summary.csv`; checkpoints `checkpoints/heldout_disease_mc_dropout_fetal_clip_fold{0,1,2}of3_mean_p0.3_mlp.pt`; same fetal-clip checkpoint/config id |
| Fetal-clip Energy | 3/3 | latest 2026-06-10 22:22:59 +0100 | `results/heldout_disease_energy_fetal_clip_cv_summary.csv` and fold files `results/heldout_disease_energy_fetal_clip_fold{0,1,2}of3_summary.csv`; checkpoints `checkpoints/heldout_disease_energy_fetal_clip_fold{0,1,2}of3_mean_t1_energy_mlp.pt`; checkpoint/config id `fetalclip_weights_550f38e5_fetalclip_config_580f6925` |
| Fetal-clip VOS | 3/3 | latest 2026-06-10 22:22:59 +0100 | `results/heldout_disease_vos_fetal_clip_cv_summary.csv` and fold files `results/heldout_disease_vos_fetal_clip_fold{0,1,2}of3_summary.csv`; checkpoints `checkpoints/heldout_disease_vos_fetal_clip_fold{0,1,2}of3_mean_vos_mlp.pt`; Gaussian stats `checkpoints/heldout_disease_vos_fetal_clip_fold{0,1,2}of3_mean_vos_gaussian_stats.npz`; same fetal-clip checkpoint/config id |
| Fetal-clip EDL softplus ann25 kl0.10, T=100 | 3/3 | latest 2026-06-10 22:22:59 +0100 | `results/heldout_disease_edl_fetal_clip_mean_softplus_ann25_kl0p10_T100_cv_summary.csv` and fold files `results/heldout_disease_edl_fetal_clip_fold{0,1,2}of3_mean_softplus_ann25_kl0p10_T100_summary.csv`; checkpoints `checkpoints/heldout_disease_edl_fetal_clip_fold{0,1,2}of3_mean_softplus_ann25_kl0p10_edl_mlp.pt`; same fetal-clip checkpoint/config id |

### Single-split saved result sets

These are saved results but are not treated as 3-fold CV evidence in the main comparison tables.

| Method/result set | Folds available | Run date/time | Checkpoint/config provenance |
|---|---:|---|---|
| MLP head single split | 1 | 2026-05-15 12:36:28 +0100 | `results/heldout_disease_baseline_summary.csv`; checkpoint `checkpoints/heldout_disease_baseline_mean_MLP.pt`; config `config.yaml` |
| MC Dropout p=0.3 T=50 single split | 1 | 2026-05-15 12:36:28 +0100 | `results/heldout_disease_mc_dropout_mean_p0.3_T50_summary.csv`; checkpoint `checkpoints/heldout_disease_mc_dropout_mean_p0.3_mlp.pt`; config `config.yaml` |
| Energy MLP single split | 1 | 2026-06-01 22:44:22 +0100 | `results/heldout_disease_energy_summary.csv`; checkpoint `checkpoints/heldout_disease_energy_mean_t1_energy_mlp.pt`; config `config.yaml` |
| VOS MLP single split | 1 | 2026-06-01 22:44:22 +0100 | `results/heldout_disease_vos_summary.csv`; checkpoint `checkpoints/heldout_disease_vos_mean_vos_mlp.pt`; Gaussian stats `checkpoints/heldout_disease_vos_mean_vos_gaussian_stats.npz`; config `config.yaml` |
| EDL relu ann10 single split | 1 | 2026-05-29 10:44:46 +0100 | `results/heldout_disease_edl_mean_relu_ann10_summary.csv`; checkpoint `checkpoints/heldout_disease_edl_mean_relu_ann10_edl_mlp.pt`; config `config.yaml` |
| EDL softplus ann25 kl0.00 single split | 1 | 2026-06-01 22:37:37 +0100 | `results/heldout_disease_edl_mean_softplus_ann25_kl0p00_summary.csv`; checkpoint `checkpoints/heldout_disease_edl_mean_softplus_ann25_kl0p00_edl_mlp.pt`; config `config.yaml` |
| EDL softplus balanced ann25 single split | 1 | 2026-06-01 19:37:02 +0100 | `results/heldout_disease_edl_softplus_balanced_mean_softplus_ann25_summary.csv`; checkpoint `checkpoints/heldout_disease_edl_softplus_balanced_mean_softplus_ann25_edl_mlp.pt`; config `config.yaml` |
| Groupwise per-disease analysis | 1 | 2026-06-02 00:57:44 +0100 | `results/groupwise_per_disease.csv` |

### Planned/no-result items

| Planned method/stage | Status |
|---|---|
| Final model | NOT YET AVAILABLE: no `results/*final*` or final-model checkpoint/result file found |
| Reliability-curve data | NOT YET AVAILABLE: no saved reliability-curve bins/plots found |
| Temperature-scaled calibration for LR, MLP, MC Dropout, Energy, VOS | NOT YET AVAILABLE: calibrated probabilities/ECE are saved only for EDL. The saved non-EDL prediction CSVs contain only `id_test` and `heldout_disease`, so validation predictions/logits needed to fit temperature are not currently saved. |

EDL is not missing: EDL has both 3-fold CV summaries and older single-split variants.

## 2. IN-DISTRIBUTION CLASSIFICATION TABLE

| Stage/method | AUPRC | AUROC | F1 | Sensitivity | Specificity | Delta AUPRC vs previous |
|---|---:|---:|---:|---:|---:|---:|
| LR baseline | 0.237 +/- 0.017 | 0.681 +/- 0.003 | 0.536 +/- 0.007 | 0.616 +/- 0.029 | 0.652 +/- 0.024 | N/A |
| MLP head | 0.247 +/- 0.017 | 0.685 +/- 0.007 | 0.510 +/- 0.024 | 0.678 +/- 0.055 | 0.587 +/- 0.064 | +0.010 |
| MLP + MC Dropout, p=0.3, T=100 | 0.305 +/- 0.011 | 0.776 +/- 0.017 | 0.561 +/- 0.057 | 0.716 +/- 0.129 | 0.680 +/- 0.127 | +0.058 |
| Energy MLP | 0.297 +/- 0.016 | 0.769 +/- 0.017 | 0.531 +/- 0.041 | 0.779 +/- 0.077 | 0.616 +/- 0.085 | -0.008 |
| VOS MLP | 0.310 +/- 0.021 | 0.771 +/- 0.037 | 0.580 +/- 0.009 | 0.681 +/- 0.077 | 0.722 +/- 0.029 | +0.013 |
| EDL softplus ann25 kl0.10 | 0.311 +/- 0.033 | 0.770 +/- 0.028 | 0.609 +/- 0.015 | 0.659 +/- 0.079 | 0.770 +/- 0.036 | +0.001 |

Source note: values were read from `results/heldout_disease_logreg_cv_summary.csv`, `results/heldout_disease_baseline_cv_summary.csv`, `results/heldout_disease_mc_dropout_mean_p0.3_T100_cv_summary.csv`, `results/heldout_disease_energy_cv_summary.csv`, `results/heldout_disease_vos_cv_summary.csv`, and `results/heldout_disease_edl_mean_softplus_ann25_kl0p10_cv_summary.csv`.

## 3. OOD DETECTION TABLE

OOD positive class was defined by the saved split files as the three held-out conditions `tetralogy`, `avsd`, and `a_stenosis`. The OOD negative class is the saved `id_test` split: healthy plus known/seen disease conditions (`hlhs`, `tga`, `raa`, `coa`, `p_atresia`, `p_stenosis`).

| UQ/OOD method | OOD AUROC | OOD AUPRC | FPR@95TPR | ECE |
|---|---:|---:|---:|---:|
| MC Dropout entropy, p=0.3, T=100 | 0.605 +/- 0.222 | 0.231 +/- 0.125 | 0.809 +/- 0.133 | 0.185 +/- 0.183 |
| Energy score | 0.630 +/- 0.074 | 0.205 +/- 0.046 | 0.789 +/- 0.120 | 0.162 +/- 0.062 |
| VOS energy score | 0.604 +/- 0.095 | 0.184 +/- 0.041 | 0.807 +/- 0.074 | 0.174 +/- 0.097 |
| EDL Dirichlet uncertainty | 0.520 +/- 0.225 | 0.166 +/- 0.063 | 0.862 +/- 0.115 | 0.225 +/- 0.187 |

Source note: OOD AUROC/AUPRC were read from `results/heldout_disease_mc_dropout_mean_p0.3_T100_cv_summary.csv`, `results/heldout_disease_energy_cv_summary.csv`, `results/heldout_disease_vos_cv_summary.csv`, and `results/heldout_disease_edl_mean_softplus_ann25_kl0p10_cv_summary.csv`. FPR@95TPR and ECE are not saved as fields in those summaries; they were derived from the corresponding fold prediction files: `results/heldout_disease_mc_dropout_fold{0,1,2}of3_mean_p0.3_T100_predictions.csv`, `results/heldout_disease_energy_fold{0,1,2}of3_predictions.csv`, `results/heldout_disease_vos_fold{0,1,2}of3_predictions.csv`, and `results/heldout_disease_edl_fold{0,1,2}of3_mean_softplus_ann25_kl0p10_predictions.csv`. ECE is ID-test uncalibrated ECE using `pred_proba_chd`.

## 4. PER-CONDITION OOD BREAKDOWN

Subject counts are constant across folds for the held-out conditions: ToF n=115, AVSD n=86, Aortic Stenosis n=26. Aortic Stenosis is low-n and should be treated as high-variance.

| Method | Condition | Subjects | OOD AUROC | OOD AUPRC | FPR@95TPR | Flag |
|---|---|---:|---:|---:|---:|---|
| MC Dropout entropy, T=100 | ToF | 115 | 0.598 +/- 0.187 | 0.137 +/- 0.070 | 0.842 +/- 0.088 |  |
| MC Dropout entropy, T=100 | AVSD | 86 | 0.617 +/- 0.180 | 0.118 +/- 0.056 | 0.744 +/- 0.156 |  |
| MC Dropout entropy, T=100 | Aortic Stenosis | 26 | 0.598 +/- 0.161 | 0.043 +/- 0.030 | 0.826 +/- 0.088 | LOW-N |
| Energy score | ToF | 115 | 0.626 +/- 0.087 | 0.124 +/- 0.043 | 0.795 +/- 0.137 |  |
| Energy score | AVSD | 86 | 0.628 +/- 0.080 | 0.087 +/- 0.022 | 0.776 +/- 0.100 |  |
| Energy score | Aortic Stenosis | 26 | 0.656 +/- 0.026 | 0.046 +/- 0.011 | 0.785 +/- 0.071 | LOW-N |
| VOS energy score | ToF | 115 | 0.590 +/- 0.121 | 0.105 +/- 0.031 | 0.834 +/- 0.061 |  |
| VOS energy score | AVSD | 86 | 0.621 +/- 0.079 | 0.085 +/- 0.022 | 0.763 +/- 0.070 |  |
| VOS energy score | Aortic Stenosis | 26 | 0.605 +/- 0.053 | 0.028 +/- 0.005 | 0.841 +/- 0.092 | LOW-N |
| EDL Dirichlet uncertainty | ToF | 115 | 0.508 +/- 0.243 | 0.096 +/- 0.042 | 0.859 +/- 0.115 |  |
| EDL Dirichlet uncertainty | AVSD | 86 | 0.527 +/- 0.219 | 0.076 +/- 0.032 | 0.858 +/- 0.150 |  |
| EDL Dirichlet uncertainty | Aortic Stenosis | 26 | 0.547 +/- 0.166 | 0.024 +/- 0.008 | 0.885 +/- 0.107 | LOW-N |

Source note: this table was derived from saved prediction files for the same four OOD methods listed in Section 3. Condition membership came from the per-condition columns `tetralogy`, `avsd`, and `a_stenosis` in those prediction CSVs.

## 5. GROUP-WISE ANALYSIS (PREVALENT DISEASE CLASSES)

The saved group-wise analysis is single-fold only. For subject-level methods, groups below 30 subjects are flagged `LOW-N`.

| Method | Unit | Disease | n per fold | OOD AUROC vs normal | Detect@tau | Recall | Mean p(CHD) | Flag |
|---|---|---|---:|---:|---:|---:|---:|---|
| baseline | video | raa | 1679 | 0.500 | 0.042 | 0.579 | 0.536 |  |
| baseline | video | coa | 1294 | 0.554 | 0.053 | 0.550 | 0.534 |  |
| baseline | video | tga | 820 | 0.497 | 0.055 | 0.639 | 0.573 |  |
| baseline | video | hlhs | 386 | 0.475 | 0.036 | 0.658 | 0.582 |  |
| baseline | video | p_stenosis | 403 | 0.488 | 0.042 | 0.586 | 0.559 |  |
| baseline | video | p_atresia | 343 | 0.591 | 0.061 | 0.531 | 0.517 |  |
| mc_dropout_mean_p0.3_T50 | subject | raa | 42 | 0.640 | 0.071 | 0.762 | 0.553 |  |
| mc_dropout_mean_p0.3_T50 | subject | coa | 34 | 0.652 | 0.029 | 0.765 | 0.554 |  |
| mc_dropout_mean_p0.3_T50 | subject | tga | 19 | 0.679 | 0.158 | 0.895 | 0.620 | LOW-N |
| mc_dropout_mean_p0.3_T50 | subject | hlhs | 13 | 0.703 | 0.077 | 0.692 | 0.544 | LOW-N |
| mc_dropout_mean_p0.3_T50 | subject | p_stenosis | 9 | 0.673 | 0.111 | 0.667 | 0.570 | LOW-N |
| mc_dropout_mean_p0.3_T50 | subject | p_atresia | 7 | 0.691 | 0.000 | 0.857 | 0.565 | LOW-N |
| energy | subject | raa | 42 | 0.679 | 0.071 | 0.738 | 0.519 |  |
| energy | subject | coa | 34 | 0.680 | 0.088 | 0.824 | 0.541 |  |
| energy | subject | tga | 19 | 0.681 | 0.105 | 0.895 | 0.608 | LOW-N |
| energy | subject | hlhs | 13 | 0.573 | 0.000 | 0.615 | 0.500 | LOW-N |
| energy | subject | p_stenosis | 9 | 0.597 | 0.000 | 0.667 | 0.558 | LOW-N |
| energy | subject | p_atresia | 7 | 0.677 | 0.000 | 0.857 | 0.622 | LOW-N |
| vos | subject | raa | 42 | 0.743 | 0.167 | 0.833 | 0.386 |  |
| vos | subject | coa | 34 | 0.744 | 0.118 | 0.765 | 0.360 |  |
| vos | subject | tga | 19 | 0.824 | 0.158 | 0.895 | 0.464 | LOW-N |
| vos | subject | hlhs | 13 | 0.716 | 0.000 | 0.846 | 0.292 | LOW-N |
| vos | subject | p_stenosis | 9 | 0.765 | 0.111 | 0.667 | 0.357 | LOW-N |
| vos | subject | p_atresia | 7 | 0.759 | 0.000 | 0.714 | 0.239 | LOW-N |
| edl_mean_relu_ann10 | subject | raa | 42 | 0.613 | 0.071 | 0.643 | 0.139 |  |
| edl_mean_relu_ann10 | subject | coa | 34 | 0.594 | 0.118 | 0.588 | 0.139 |  |
| edl_mean_relu_ann10 | subject | tga | 19 | 0.651 | 0.263 | 0.737 | 0.139 | LOW-N |
| edl_mean_relu_ann10 | subject | hlhs | 13 | 0.576 | 0.077 | 0.615 | 0.139 | LOW-N |
| edl_mean_relu_ann10 | subject | p_stenosis | 9 | 0.662 | 0.111 | 0.889 | 0.139 | LOW-N |
| edl_mean_relu_ann10 | subject | p_atresia | 7 | 0.580 | 0.143 | 0.714 | 0.139 | LOW-N |
| edl_mean_softplus_ann25_kl0p00 | subject | raa | 42 | 0.413 | 0.167 | 0.524 | 0.444 |  |
| edl_mean_softplus_ann25_kl0p00 | subject | coa | 34 | 0.359 | 0.118 | 0.529 | 0.443 |  |
| edl_mean_softplus_ann25_kl0p00 | subject | tga | 19 | 0.415 | 0.263 | 0.684 | 0.466 | LOW-N |
| edl_mean_softplus_ann25_kl0p00 | subject | hlhs | 13 | 0.347 | 0.154 | 0.462 | 0.446 | LOW-N |
| edl_mean_softplus_ann25_kl0p00 | subject | p_stenosis | 9 | 0.378 | 0.111 | 0.333 | 0.429 | LOW-N |
| edl_mean_softplus_ann25_kl0p00 | subject | p_atresia | 7 | 0.315 | 0.000 | 0.000 | 0.398 | LOW-N |
| edl_softplus_balanced_mean_softplus_ann25 | subject | raa | 42 | 0.411 | 0.024 | 0.786 | 0.495 |  |
| edl_softplus_balanced_mean_softplus_ann25 | subject | coa | 34 | 0.331 | 0.029 | 0.853 | 0.496 |  |
| edl_softplus_balanced_mean_softplus_ann25 | subject | tga | 19 | 0.342 | 0.000 | 0.842 | 0.496 | LOW-N |
| edl_softplus_balanced_mean_softplus_ann25 | subject | hlhs | 13 | 0.354 | 0.000 | 1.000 | 0.495 | LOW-N |
| edl_softplus_balanced_mean_softplus_ann25 | subject | p_stenosis | 9 | 0.407 | 0.000 | 1.000 | 0.495 | LOW-N |
| edl_softplus_balanced_mean_softplus_ann25 | subject | p_atresia | 7 | 0.486 | 0.000 | 0.857 | 0.494 | LOW-N |

Source note: values were read from `results/groupwise_per_disease.csv`. This file reports `n_folds=1`, so no fold standard deviation is available.

## 6. CALIBRATION

| Method | ID ECE uncalibrated | ID ECE calibrated | Combined ECE uncalibrated | Combined ECE calibrated | Temperature scaling status |
|---|---:|---:|---:|---:|---|
| LR baseline | 0.287 +/- 0.004 | NOT YET AVAILABLE | NOT YET AVAILABLE | NOT YET AVAILABLE | No calibrated probabilities saved |
| MLP head | 0.279 +/- 0.019 | NOT YET AVAILABLE | NOT YET AVAILABLE | NOT YET AVAILABLE | No calibrated probabilities saved |
| MC Dropout p=0.3 T=100 | 0.185 +/- 0.183 | NOT YET AVAILABLE | NOT YET AVAILABLE | NOT YET AVAILABLE | No calibrated probabilities saved; validation predictions/logits needed to fit temperature are not saved |
| Energy MLP | 0.162 +/- 0.062 | NOT YET AVAILABLE | NOT YET AVAILABLE | NOT YET AVAILABLE | Energy temperature is `1.0`; no classification temperature scaling saved |
| VOS MLP | 0.174 +/- 0.097 | NOT YET AVAILABLE | NOT YET AVAILABLE | NOT YET AVAILABLE | No calibrated probabilities saved |
| EDL softplus ann25 kl0.10 | 0.225 +/- 0.187 | 0.221 +/- 0.152 | 0.165 +/- 0.153 | 0.132 +/- 0.120 | Temperature scaling saved; mean temperature 6.875 +/- 9.925 in `results/heldout_disease_edl_mean_softplus_ann25_kl0p10_cv_summary.csv` |

Reliability-curve data: NOT YET AVAILABLE. I found ECE scalar fields for EDL, but no saved reliability bins or curve data files.

Source note: EDL ECE values were read from `results/heldout_disease_edl_mean_softplus_ann25_kl0p10_cv_summary.csv`. Non-EDL ID ECE values were derived from the fold prediction CSVs listed in Sections 2 and 3 using the repository ECE definition in `evaluate.py` over ID-test `label` and `pred_proba_chd`. Combined ECE for non-EDL methods is NOT YET AVAILABLE as a saved metric. Post-hoc temperature scaling for non-EDL methods is also NOT YET AVAILABLE from saved CSVs because all saved prediction CSVs contain only `id_test` and `heldout_disease`; no validation predictions/logits are saved.

## 7. MC DROPOUT DECOMPOSITION

Canonical MC Dropout row used here: p=0.3, T=100.

### MC Dropout T-sweep for RQ3

This table tests whether MC Dropout OOD/calibration metrics saturate as the number of stochastic forward passes increases. `T=1` is the deterministic single-pass MLP head baseline, recomputed at subject level by mean-pooling saved video-level `pred_proba_chd` per subject so that it matches the subject-level MC Dropout rows. For `T>=10`, OOD AUROC/AUPRC were read from saved MC Dropout fold summaries; ID ECE and FPR@95TPR were derived from saved prediction CSVs.

| Passes T | Method/source | OOD AUROC | OOD AUPRC | FPR@95TPR | ID ECE |
|---:|---|---:|---:|---:|---:|
| 1 | Deterministic MLP, subject-pooled from saved predictions | 0.557 +/- 0.036 | 0.169 +/- 0.020 | 0.881 +/- 0.065 | 0.306 +/- 0.017 |
| 10 | MC Dropout p=0.3 | 0.608 +/- 0.220 | 0.237 +/- 0.134 | 0.796 +/- 0.141 | 0.184 +/- 0.184 |
| 50 | MC Dropout p=0.3 | 0.606 +/- 0.221 | 0.229 +/- 0.123 | 0.801 +/- 0.140 | 0.184 +/- 0.184 |
| 100 | MC Dropout p=0.3 | 0.605 +/- 0.222 | 0.231 +/- 0.125 | 0.809 +/- 0.133 | 0.185 +/- 0.183 |
| 1000 | MC Dropout p=0.3 | 0.605 +/- 0.222 | 0.229 +/- 0.124 | 0.795 +/- 0.147 | 0.182 +/- 0.185 |

Source note: deterministic `T=1` values were derived from `results/heldout_disease_baseline_fold{0,1,2}of3_predictions.csv`. MC Dropout values were read/derived from `results/heldout_disease_mc_dropout_fold{0,1,2}of3_mean_p0.3_T{10,50,100,1000}_summary.csv` and corresponding `*_predictions.csv` files. The saved deterministic MLP fold summaries are video-level; their OOD AUROC is 0.528 +/- 0.019 and is not used as the main T=1 row above because it is not subject-pooled.

Interpretation for RQ3: the OOD AUROC curve is flat from T=10 onward within fold variance (`0.608 -> 0.606 -> 0.605 -> 0.605`), and ID ECE is also unchanged from T=10 onward (`0.184 -> 0.184 -> 0.185 -> 0.182`). The main change is from deterministic T=1 to stochastic T=10, especially in calibration: ECE drops from 0.306 +/- 0.017 to 0.184 +/- 0.184. The table supports a cautious saturation statement: in these saved folds, increasing MC passes beyond T=10 does not materially change OOD AUROC or ECE.

| Quantity | ID/normal/seen summary | Held-out summary | OOD metric if saved |
|---|---:|---:|---:|
| Predictive entropy, normal ID | 0.455 +/- 0.246 | N/A | N/A |
| Predictive entropy, seen disease ID | 0.613 +/- 0.084 | N/A | N/A |
| Predictive entropy, all ID | 0.472 +/- 0.222 | 0.610 +/- 0.066 | OOD AUROC 0.605 +/- 0.222; OOD AUPRC 0.231 +/- 0.125 |
| Mutual information | 0.049 +/- 0.011 | 0.063 +/- 0.024 | OOD AUROC 0.566 +/- 0.232; OOD AUPRC 0.203 +/- 0.114 |
| Predictive probability std | 0.084 +/- 0.029 | 0.107 +/- 0.023 | OOD AUROC 0.583 +/- 0.230; OOD AUPRC 0.213 +/- 0.117 |

Source note: values were read from `results/heldout_disease_mc_dropout_mean_p0.3_T100_cv_summary.csv`.

## 8. FETAL-CLIP RESULTS

These fetal-clip results now have 3/3 folds for LR/probe, MLP head, representation probe, MC Dropout, Energy, VOS, and EDL. Direct fetal-clip vs DINOv2 comparison is now available for LR/MLP/MC/Energy/VOS/EDL. Note: `results/heldout_disease_baseline_fetal_clip_fold*` now contains the deterministic MLP run; LR/probe values in this report come from the representation-probe LogisticRegression outputs. Fetal-clip Energy uses checkpoint/config id `fetalclip_weights_550f38e5_fetalclip_config_580f6925`, while the earlier fetal-clip LR/MC/VOS/EDL/probe runs use `fetalclip_weights_c3ea7e11_fetalclip_config_e5fc56d1`.

### Fetal-clip run inventory

| Method/result set | Folds available | Run date/time | Checkpoint/config provenance |
|---|---:|---|---|
| Fetal-clip LR/probe baseline | 3/3 | latest 2026-06-10 22:22:59 +0100 | `results/heldout_disease_representation_probe_fetal_clip_mean_cv_summary.csv` and fold files `results/heldout_disease_representation_probe_fetal_clip_fold{0,1,2}of3_mean_summary.csv`; checkpoints `checkpoints/heldout_disease_baseline_fetal_clip_fold{0,1,2}of3_mean_LogisticRegression.joblib`; checkpoint/config id `fetalclip_weights_c3ea7e11_fetalclip_config_e5fc56d1` |
| Fetal-clip MLP head | 3/3 | 2026-06-10 22:22:59 +0100 | `results/heldout_disease_baseline_fetal_clip_cv_summary.csv` and fold files `results/heldout_disease_baseline_fetal_clip_fold{0,1,2}of3_summary.csv`; checkpoints `checkpoints/heldout_disease_baseline_fetal_clip_fold{0,1,2}of3_mean_MLP.pt`; same fetal-clip checkpoint/config id |
| Fetal-clip representation probe | 3/3 | latest 2026-06-10 22:22:59 +0100 | `results/heldout_disease_representation_probe_fetal_clip_mean_cv_summary.csv` and fold files `results/heldout_disease_representation_probe_fetal_clip_fold{0,1,2}of3_mean_summary.csv`; same fetal-clip checkpoint/config id |
| Fetal-clip MC Dropout, p=0.3, T=100 | 3/3 | latest 2026-06-10 22:22:59 +0100 | `results/heldout_disease_mc_dropout_fetal_clip_mean_p0.3_T100_cv_summary.csv` and fold files `results/heldout_disease_mc_dropout_fetal_clip_fold{0,1,2}of3_mean_p0.3_T100_summary.csv`; checkpoints `checkpoints/heldout_disease_mc_dropout_fetal_clip_fold{0,1,2}of3_mean_p0.3_mlp.pt`; same fetal-clip checkpoint/config id |
| Fetal-clip Energy | 3/3 | latest 2026-06-10 22:22:59 +0100 | `results/heldout_disease_energy_fetal_clip_cv_summary.csv` and fold files `results/heldout_disease_energy_fetal_clip_fold{0,1,2}of3_summary.csv`; checkpoints `checkpoints/heldout_disease_energy_fetal_clip_fold{0,1,2}of3_mean_t1_energy_mlp.pt`; checkpoint/config id `fetalclip_weights_550f38e5_fetalclip_config_580f6925` |
| Fetal-clip VOS | 3/3 | latest 2026-06-10 22:22:59 +0100 | `results/heldout_disease_vos_fetal_clip_cv_summary.csv` and fold files `results/heldout_disease_vos_fetal_clip_fold{0,1,2}of3_summary.csv`; checkpoints `checkpoints/heldout_disease_vos_fetal_clip_fold{0,1,2}of3_mean_vos_mlp.pt`; Gaussian stats `checkpoints/heldout_disease_vos_fetal_clip_fold{0,1,2}of3_mean_vos_gaussian_stats.npz`; same fetal-clip checkpoint/config id |
| Fetal-clip EDL softplus ann25 kl0.10, T=100 | 3/3 | latest 2026-06-10 22:22:59 +0100 | `results/heldout_disease_edl_fetal_clip_mean_softplus_ann25_kl0p10_T100_cv_summary.csv` and fold files `results/heldout_disease_edl_fetal_clip_fold{0,1,2}of3_mean_softplus_ann25_kl0p10_T100_summary.csv`; checkpoints `checkpoints/heldout_disease_edl_fetal_clip_fold{0,1,2}of3_mean_softplus_ann25_kl0p10_edl_mlp.pt`; same fetal-clip checkpoint/config id |

Source note: run date/time is file modification time. All fetal-clip runs use held-out conditions `tetralogy,avsd,a_stenosis`, mean pooling, and `n_folds=3`, from their corresponding `*_split_info.json` files. ID-test subjects are 1298, 1299, and 1300 across folds 0-2; held-out subjects are 223 in each fold.

### Fetal-clip 3-fold ID and OOD summary

| Method | Folds used | ID AUPRC | ID AUROC | F1 | Sensitivity | Specificity | OOD AUROC | OOD AUPRC | FPR@95TPR | ID ECE | Held-out called CHD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fetal-clip LR/probe | 3/3 | 0.485 +/- 0.027 | 0.855 +/- 0.008 | 0.624 +/- 0.021 | 0.815 +/- 0.006 | 0.740 +/- 0.032 | 0.561 +/- 0.004 | 0.168 +/- 0.002 | 0.950 +/- 0.006 | 0.181 +/- 0.006 | 0.892 +/- 0.004 |
| Fetal-clip MLP | 3/3 | 0.481 +/- 0.030 | 0.852 +/- 0.012 | 0.643 +/- 0.073 | 0.776 +/- 0.083 | 0.769 +/- 0.119 | 0.565 +/- 0.068 | 0.175 +/- 0.022 | 0.854 +/- 0.082 | 0.183 +/- 0.047 | 0.803 +/- 0.103 |
| Fetal-clip MC Dropout T100 | 3/3 | 0.482 +/- 0.033 | 0.851 +/- 0.012 | 0.638 +/- 0.079 | 0.761 +/- 0.097 | 0.767 +/- 0.133 | 0.566 +/- 0.068 | 0.173 +/- 0.023 | 0.876 +/- 0.071 | 0.184 +/- 0.049 | 0.797 +/- 0.115 |
| Fetal-clip Energy | 3/3 | 0.485 +/- 0.019 | 0.854 +/- 0.008 | 0.638 +/- 0.050 | 0.800 +/- 0.050 | 0.760 +/- 0.071 | 0.509 +/- 0.099 | 0.147 +/- 0.029 | 0.849 +/- 0.095 | 0.224 +/- 0.062 | 0.837 +/- 0.053 |
| Fetal-clip VOS | 3/3 | 0.500 +/- 0.039 | 0.851 +/- 0.006 | 0.645 +/- 0.029 | 0.753 +/- 0.077 | 0.785 +/- 0.050 | 0.548 +/- 0.207 | 0.177 +/- 0.065 | 0.792 +/- 0.176 | 0.187 +/- 0.145 | 0.786 +/- 0.056 |
| Fetal-clip EDL T100 | 3/3 | 0.456 +/- 0.035 | 0.848 +/- 0.012 | 0.614 +/- 0.033 | 0.800 +/- 0.099 | 0.730 +/- 0.071 | 0.450 +/- 0.049 | 0.136 +/- 0.011 | 0.975 +/- 0.020 | 0.213 +/- 0.015 | 0.842 +/- 0.093 |

Source note: LR/probe classification values were read from `results/heldout_disease_representation_probe_fetal_clip_mean_cv_summary.csv`; LR/probe OOD values use `linear_probe_margin_distance` from `results/heldout_disease_representation_probe_fetal_clip_fold{0,1,2}of3_mean_predictions.csv`. MLP classification and OOD AUROC/AUPRC were read from `results/heldout_disease_baseline_fetal_clip_cv_summary.csv`; MLP FPR@95TPR and ID ECE were derived from `results/heldout_disease_baseline_fetal_clip_fold{0,1,2}of3_predictions.csv` using `uncertainty_entropy` and `pred_proba_chd`. Other classification and OOD AUROC/AUPRC values were read from the fetal-clip fold summary CSVs listed in the inventory above. FPR@95TPR and non-EDL ID ECE were read from the Energy summaries where saved, and otherwise derived from the corresponding fetal-clip prediction CSVs using MC Dropout `uncertainty_entropy`, Energy/VOS `ood_score`, EDL `uncertainty_dirichlet`, and `pred_proba_chd`.

### Fetal-clip representation probe OOD distances

| Probe distance score | Folds used | OOD AUROC | OOD AUPRC |
|---|---:|---:|---:|
| Probe probability | 3/3 | 0.863 +/- 0.006 | 0.469 +/- 0.015 |
| Nearest train distance | 3/3 | 0.508 +/- 0.004 | 0.156 +/- 0.002 |
| Healthy centroid distance | 3/3 | 0.517 +/- 0.007 | 0.159 +/- 0.004 |
| Seen disease centroid distance | 3/3 | 0.407 +/- 0.016 | 0.124 +/- 0.004 |
| Linear probe margin distance | 3/3 | 0.561 +/- 0.004 | 0.168 +/- 0.002 |

Source note: values were read from `results/heldout_disease_representation_probe_fetal_clip_fold{0,1,2}of3_mean_summary.csv`. The probe probability row is not a pure distance score; it is the CHD probability from the linear probe, with held-out disease treated as OOD positive.

### DINOv2 representation probe OOD distances

| Probe distance score | Folds used | OOD AUROC | OOD AUPRC |
|---|---:|---:|---:|
| Probe probability | 3/3 | 0.806 +/- 0.007 | 0.404 +/- 0.020 |
| Nearest train distance | 3/3 | 0.452 +/- 0.010 | 0.137 +/- 0.005 |
| Healthy centroid distance | 3/3 | 0.462 +/- 0.004 | 0.133 +/- 0.001 |
| Seen disease centroid distance | 3/3 | 0.441 +/- 0.008 | 0.128 +/- 0.001 |
| Linear probe margin distance | 3/3 | 0.546 +/- 0.014 | 0.159 +/- 0.005 |

Source note: values were read from `results/heldout_disease_representation_probe_fold{0,1,2}of3_mean_summary.csv`. The probe probability row is not a pure distance score; it is the CHD probability from the linear probe, with held-out disease treated as OOD positive.

### Direct 3-fold DINOv2 vs fetal-clip comparison

| Matched method | DINOv2 ID AUPRC | Fetal-clip ID AUPRC | Delta ID AUPRC | DINOv2 ID AUROC | Fetal-clip ID AUROC | DINOv2 OOD AUROC | Fetal-clip OOD AUROC | DINOv2 OOD AUPRC | Fetal-clip OOD AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LR/probe baseline | 0.237 +/- 0.017 | 0.485 +/- 0.027 | +0.248 | 0.681 +/- 0.003 | 0.855 +/- 0.008 | 0.520 +/- 0.005 | 0.561 +/- 0.004 | 0.191 +/- 0.003 | 0.168 +/- 0.002 |
| MLP head | 0.247 +/- 0.017 | 0.481 +/- 0.030 | +0.234 | 0.685 +/- 0.007 | 0.852 +/- 0.012 | 0.528 +/- 0.019 | 0.565 +/- 0.068 | 0.189 +/- 0.007 | 0.175 +/- 0.022 |
| MC Dropout T100 | 0.305 +/- 0.011 | 0.482 +/- 0.033 | +0.177 | 0.776 +/- 0.017 | 0.851 +/- 0.012 | 0.605 +/- 0.222 | 0.566 +/- 0.068 | 0.231 +/- 0.125 | 0.173 +/- 0.023 |
| Energy | 0.297 +/- 0.016 | 0.485 +/- 0.019 | +0.188 | 0.769 +/- 0.017 | 0.854 +/- 0.008 | 0.630 +/- 0.074 | 0.509 +/- 0.099 | 0.205 +/- 0.046 | 0.147 +/- 0.029 |
| VOS | 0.310 +/- 0.021 | 0.500 +/- 0.039 | +0.190 | 0.771 +/- 0.037 | 0.851 +/- 0.006 | 0.604 +/- 0.095 | 0.548 +/- 0.207 | 0.184 +/- 0.041 | 0.177 +/- 0.065 |
| EDL T100 | 0.312 +/- 0.034 | 0.456 +/- 0.035 | +0.144 | 0.772 +/- 0.033 | 0.848 +/- 0.012 | 0.526 +/- 0.228 | 0.450 +/- 0.049 | 0.172 +/- 0.068 | 0.136 +/- 0.011 |

Source note: DINOv2 values were read from `results/heldout_disease_logreg_fold{0,1,2}of3_summary.csv`, `results/heldout_disease_baseline_cv_summary.csv`, `results/heldout_disease_mc_dropout_fold{0,1,2}of3_mean_p0.3_T100_summary.csv`, `results/heldout_disease_energy_fold{0,1,2}of3_summary.csv`, `results/heldout_disease_vos_fold{0,1,2}of3_summary.csv`, and `results/heldout_disease_edl_fold{0,1,2}of3_mean_softplus_ann25_kl0p10_T100_summary.csv`. Fetal-clip MLP values were read from `results/heldout_disease_baseline_fetal_clip_cv_summary.csv`; fetal-clip LR/probe values were read from `results/heldout_disease_representation_probe_fetal_clip_mean_cv_summary.csv` and derived from the matching representation-probe prediction files for margin-distance OOD metrics. Other fetal-clip values were read from the matching fetal-clip fold summary files listed above.

### Fetal-clip per-condition held-out breakdown

| Method | Condition | Subjects | Mean p(CHD) | Recall as CHD | OOD AUROC | OOD AUPRC | FPR@95TPR | Flag |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Fetal-clip LR/probe | ToF | 115 | 0.692 +/- 0.014 | 0.875 +/- 0.013 | 0.573 +/- 0.020 | 0.098 +/- 0.007 | 0.911 +/- 0.039 |  |
| Fetal-clip LR/probe | AVSD | 86 | 0.735 +/- 0.023 | 0.911 +/- 0.013 | 0.538 +/- 0.025 | 0.069 +/- 0.005 | 0.978 +/- 0.015 |  |
| Fetal-clip LR/probe | Aortic Stenosis | 26 | 0.715 +/- 0.011 | 0.910 +/- 0.059 | 0.571 +/- 0.025 | 0.035 +/- 0.008 | 0.976 +/- 0.009 | LOW-N |
| Fetal-clip MLP | ToF | 115 | 0.709 +/- 0.028 | 0.806 +/- 0.098 | 0.576 +/- 0.079 | 0.105 +/- 0.025 | 0.836 +/- 0.075 |  |
| Fetal-clip MLP | AVSD | 86 | 0.739 +/- 0.031 | 0.798 +/- 0.108 | 0.539 +/- 0.071 | 0.075 +/- 0.010 | 0.888 +/- 0.077 |  |
| Fetal-clip MLP | Aortic Stenosis | 26 | 0.726 +/- 0.018 | 0.795 +/- 0.160 | 0.586 +/- 0.040 | 0.031 +/- 0.006 | 0.878 +/- 0.096 | LOW-N |
| Fetal-clip MC Dropout T100 | ToF | 115 | 0.690 +/- 0.030 | 0.791 +/- 0.118 | 0.579 +/- 0.078 | 0.105 +/- 0.024 | 0.844 +/- 0.068 |  |
| Fetal-clip MC Dropout T100 | AVSD | 86 | 0.722 +/- 0.033 | 0.798 +/- 0.106 | 0.539 +/- 0.071 | 0.072 +/- 0.007 | 0.908 +/- 0.053 |  |
| Fetal-clip MC Dropout T100 | Aortic Stenosis | 26 | 0.713 +/- 0.014 | 0.808 +/- 0.176 | 0.586 +/- 0.041 | 0.029 +/- 0.002 | 0.913 +/- 0.067 | LOW-N |
| Fetal-clip Energy | ToF | 115 | 0.756 +/- 0.070 | 0.812 +/- 0.058 | 0.526 +/- 0.103 | 0.090 +/- 0.023 | 0.834 +/- 0.094 |  |
| Fetal-clip Energy | AVSD | 86 | 0.802 +/- 0.040 | 0.860 +/- 0.081 | 0.473 +/- 0.087 | 0.056 +/- 0.009 | 0.882 +/- 0.103 |  |
| Fetal-clip Energy | Aortic Stenosis | 26 | 0.781 +/- 0.082 | 0.859 +/- 0.059 | 0.543 +/- 0.117 | 0.027 +/- 0.010 | 0.840 +/- 0.101 | LOW-N |
| Fetal-clip VOS | ToF | 115 | 0.674 +/- 0.142 | 0.783 +/- 0.060 | 0.568 +/- 0.216 | 0.106 +/- 0.037 | 0.769 +/- 0.190 |  |
| Fetal-clip VOS | AVSD | 86 | 0.727 +/- 0.121 | 0.795 +/- 0.055 | 0.523 +/- 0.204 | 0.081 +/- 0.039 | 0.807 +/- 0.170 |  |
| Fetal-clip VOS | Aortic Stenosis | 26 | 0.707 +/- 0.119 | 0.769 +/- 0.077 | 0.532 +/- 0.194 | 0.027 +/- 0.014 | 0.804 +/- 0.160 | LOW-N |
| Fetal-clip EDL T100 | ToF | 115 | 0.706 +/- 0.046 | 0.823 +/- 0.104 | 0.452 +/- 0.078 | 0.079 +/- 0.012 | 0.969 +/- 0.021 |  |
| Fetal-clip EDL T100 | AVSD | 86 | 0.740 +/- 0.015 | 0.860 +/- 0.081 | 0.435 +/- 0.017 | 0.056 +/- 0.002 | 0.985 +/- 0.020 |  |
| Fetal-clip EDL T100 | Aortic Stenosis | 26 | 0.739 +/- 0.035 | 0.859 +/- 0.080 | 0.480 +/- 0.046 | 0.022 +/- 0.004 | 0.993 +/- 0.006 | LOW-N |

Source note: this table was derived from the fetal-clip fold prediction CSVs listed in the inventory. LR/probe rows use `results/heldout_disease_representation_probe_fetal_clip_fold{0,1,2}of3_mean_predictions.csv` and `linear_probe_margin_distance`; MLP rows use `results/heldout_disease_baseline_fetal_clip_fold{0,1,2}of3_predictions.csv` and `uncertainty_entropy`. OOD negatives are each fold's `id_test` subjects; OOD positives are each held-out condition separately. Aortic Stenosis has only 26 held-out subjects per fold, so those rows are high-variance.

### Fetal-clip uncertainty decomposition

| Method | Quantity | ID mean | Held-out mean | OOD AUROC | OOD AUPRC |
|---|---|---:|---:|---:|---:|
| MC Dropout T100 | Predictive entropy | 0.640 +/- 0.070 | 0.640 +/- 0.081 | 0.566 +/- 0.068 | 0.173 +/- 0.023 |
| MC Dropout T100 | Mutual information | 0.056 +/- 0.009 | 0.060 +/- 0.020 | 0.513 +/- 0.091 | 0.158 +/- 0.034 |
| MC Dropout T100 | Predictive probability std | 0.095 +/- 0.008 | 0.105 +/- 0.017 | 0.537 +/- 0.082 | 0.165 +/- 0.035 |
| EDL T100 | Dirichlet uncertainty | 0.225 +/- 0.013 | 0.221 +/- 0.017 | 0.450 +/- 0.049 | 0.136 +/- 0.011 |
| EDL T100 | Predictive entropy | 0.620 +/- 0.023 | 0.611 +/- 0.009 | 0.484 +/- 0.043 | 0.150 +/- 0.007 |
| EDL T100 | Mutual information | 0.040 +/- 0.003 | 0.042 +/- 0.009 | 0.508 +/- 0.039 | 0.156 +/- 0.023 |
| EDL T100 | Predictive probability std | 0.081 +/- 0.003 | 0.084 +/- 0.010 | 0.504 +/- 0.040 | 0.154 +/- 0.021 |

Source note: MC Dropout decomposition values were read from `results/heldout_disease_mc_dropout_fetal_clip_fold{0,1,2}of3_mean_p0.3_T100_summary.csv`. EDL decomposition values were read from `results/heldout_disease_edl_fetal_clip_fold{0,1,2}of3_mean_softplus_ann25_kl0p10_T100_summary.csv`.

### Fetal-clip calibration

| Method | ID ECE uncalibrated | ID ECE calibrated | Combined ECE uncalibrated | Combined ECE calibrated | Temperature scaling status |
|---|---:|---:|---:|---:|---|
| Fetal-clip LR/probe | 0.181 +/- 0.006 | NOT YET AVAILABLE | NOT YET AVAILABLE | NOT YET AVAILABLE | No calibrated probabilities saved |
| Fetal-clip MLP | 0.183 +/- 0.047 | NOT YET AVAILABLE | NOT YET AVAILABLE | NOT YET AVAILABLE | No calibrated probabilities saved; validation predictions/logits needed to fit temperature are not saved |
| Fetal-clip MC Dropout T100 | 0.184 +/- 0.049 | NOT YET AVAILABLE | NOT YET AVAILABLE | NOT YET AVAILABLE | No calibrated probabilities saved |
| Fetal-clip Energy | 0.224 +/- 0.062 | NOT YET AVAILABLE | NOT YET AVAILABLE | NOT YET AVAILABLE | Energy temperature is `1.0`; no classification temperature scaling saved |
| Fetal-clip VOS | 0.187 +/- 0.145 | NOT YET AVAILABLE | NOT YET AVAILABLE | NOT YET AVAILABLE | No calibrated probabilities saved |
| Fetal-clip EDL T100 | 0.213 +/- 0.015 | 0.224 +/- 0.025 | 0.141 +/- 0.017 | 0.148 +/- 0.025 | Temperature scaling saved; mean temperature 1.126 +/- 0.123 |

Source note: EDL calibration values were read from `results/heldout_disease_edl_fetal_clip_fold{0,1,2}of3_mean_softplus_ann25_kl0p10_T100_summary.csv`. Fetal-clip Energy ID ECE was read from `results/heldout_disease_energy_fetal_clip_fold{0,1,2}of3_summary.csv`. LR/probe ECE was derived from `results/heldout_disease_representation_probe_fetal_clip_fold{0,1,2}of3_mean_predictions.csv`; MLP ECE was derived from `results/heldout_disease_baseline_fetal_clip_fold{0,1,2}of3_predictions.csv`. Other non-EDL ECE values were derived from fetal-clip prediction CSVs using ID-test `label` and `pred_proba_chd`.

### Fetal-clip interpretation

The fetal-clip backbone is currently the main classification improvement. Across all 3 folds, every matched fetal-clip classifier improves ID AUPRC over its DINOv2 counterpart. The largest simple representation signal is LR/probe: DINOv2 LR ID AUPRC is 0.237 +/- 0.017, while fetal-clip LR/probe ID AUPRC is 0.485 +/- 0.027. The deterministic MLP comparison now points the same way: DINOv2 MLP is 0.247 +/- 0.017, while fetal-clip MLP is 0.481 +/- 0.030. Fetal-clip Energy also improves ID AUPRC over DINOv2 Energy: 0.485 +/- 0.019 vs 0.297 +/- 0.016.

The OOD story is weaker. Fetal-clip LR/probe, MLP, MC Dropout, Energy, and VOS OOD AUROC values are 0.561, 0.565, 0.566, 0.509, and 0.548 respectively, and all OOD AUPRC values remain <= 0.177 for the UQ scores. Fetal-clip Energy underperforms DINOv2 Energy on OOD detection: OOD AUROC drops from 0.630 +/- 0.074 to 0.509 +/- 0.099. EDL's Dirichlet uncertainty is below random for aggregate OOD detection with AUROC 0.450 +/- 0.049. The representation-probe CHD probability has high held-out-vs-ID AUROC at 0.863 +/- 0.006 for fetal-clip and 0.806 +/- 0.007 for DINOv2, but that is because held-out diseases are confidently CHD-like, not because they are being identified as novel by an uncertainty/distance score.

The current working interpretation is: fetal-clip improves the CHD representation substantially, including on held-out disease classification, but held-out diseases often look like disease rather than like novel/OOD examples. This is good for broad CHD detection and still weak for novelty detection.

## 9. OPEN GAPS / SANITY CHECKS

### Sanity checks to investigate

| Issue | Evidence |
|---|---|
| OOD detection remains weak overall | Best aggregate OOD AUROC is Energy at 0.630 +/- 0.074; all OOD AUPRC values are <= 0.231. |
| FPR@95TPR is high for all OOD methods | Derived FPR@95TPR ranges from 0.789 +/- 0.120 to 0.862 +/- 0.115. |
| MC Dropout and EDL have very large OOD fold variance | MC Dropout T=100 OOD AUROC std is 0.222; EDL OOD AUROC std is 0.225. |
| EDL OOD detection is near-random on aggregate | EDL Dirichlet uncertainty OOD AUROC is 0.520 +/- 0.225. |
| Aortic Stenosis per-condition metrics are high-variance | Aortic Stenosis has only 26 subjects, and per-condition AUPRC is very low across methods: 0.024 to 0.046. |
| Energy stage drops ID AUPRC relative to MC Dropout | ID AUPRC changes from 0.305 +/- 0.011 to 0.297 +/- 0.016, a change of -0.008. |
| VOS/EDL improve ID AUPRC only marginally after Energy/VOS | VOS improves over Energy by +0.013 AUPRC; EDL improves over VOS by +0.001 AUPRC. |
| Several group-wise rows are very small-n | Subject-level `tga`, `hlhs`, `p_stenosis`, and `p_atresia` have n=19, 13, 9, and 7 respectively. |
| Fetal-clip improves ID classification but not OOD detection | Fetal-clip LR/probe over 3 folds has ID AUPRC 0.485 +/- 0.027, but margin-distance OOD AUPRC is only 0.168 +/- 0.002 and FPR@95TPR is 0.950 +/- 0.006. |
| Fetal-clip MLP improves ID classification but not OOD detection | ID AUPRC improves from DINOv2 MLP 0.247 +/- 0.017 to fetal-clip MLP 0.481 +/- 0.030, but OOD AUROC remains near chance at 0.565 +/- 0.068. |
| Fetal-clip baseline stem now contains MLP results | `results/heldout_disease_baseline_fetal_clip_fold{0,1,2}of3_summary.csv` now has `classifier=MLP`; LR/probe values are sourced from representation-probe LogisticRegression outputs. |
| Fetal-clip Energy improves ID classification but hurts OOD detection | ID AUPRC improves over DINOv2 Energy from 0.297 +/- 0.016 to 0.485 +/- 0.019, but OOD AUROC drops from 0.630 +/- 0.074 to 0.509 +/- 0.099 and OOD AUPRC drops from 0.205 +/- 0.046 to 0.147 +/- 0.029. |
| Fetal-clip Energy uses a different extractor/config id | Fetal-clip Energy uses `fetalclip_weights_550f38e5_fetalclip_config_580f6925`, while LR/MC/VOS/EDL/probe use `fetalclip_weights_c3ea7e11_fetalclip_config_e5fc56d1`; this may affect direct method comparisons within fetal-clip. |
| Fetal-clip VOS has huge OOD fold variance | Fetal-clip VOS OOD AUROC is 0.548 +/- 0.207 and FPR@95TPR is 0.792 +/- 0.176. |
| Fetal-clip EDL uncertainty is below random for held-out disease | Fetal-clip EDL aggregate OOD AUROC is 0.450 +/- 0.049 and OOD AUPRC is 0.136 +/- 0.011. |
| Fetal-clip representation distances are weak OOD scores | Across 3 folds, the best pure distance/margin AUROC is linear probe margin at 0.561 +/- 0.004; seen disease centroid distance is below random at 0.407 +/- 0.016. |

### NOT YET AVAILABLE items

| Item | Missing for method/stage |
|---|---|
| Final model result file/checkpoint | Final model |
| Reliability-curve bins/plots/data | All methods |
| Calibrated probabilities and calibrated ECE | LR baseline, MLP head, MC Dropout, Energy, VOS. Current prediction CSVs do not contain validation predictions/logits, so temperature scaling cannot be fit from saved CSVs without regenerating validation predictions from checkpoints. |
| Saved FPR@95TPR scalar metrics | DINOv2 MC Dropout, Energy, VOS, EDL and fetal-clip LR/probe, MLP, MC Dropout, VOS, EDL; values in this report are derived from prediction CSVs. Fetal-clip Energy has saved FPR@95TPR fields. |
| Saved per-condition 3-fold OOD summary table | DINOv2 MC Dropout, Energy, VOS, EDL and fetal-clip LR/probe, MLP, MC Dropout, VOS, EDL; values in this report are derived from prediction CSVs. Fetal-clip Energy has saved per-condition fields. |
| Combined ECE for non-EDL methods | LR baseline, MLP head, MC Dropout, Energy, VOS |
