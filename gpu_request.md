# GPU Compute Request — Fetal Echocardiography Classification

## Project Overview

Multi-label classification of congenital heart defects from fetal echocardiography videos across 9 cardiac conditions (AVSD, HLHS, TGA, Tetralogy of Fallot, RAA, CoA, Pulmonary Atresia, Aortic Stenosis, Pulmonary Stenosis). The dataset contains 4,127 subjects, each with variable-length ultrasound videos sampled at 16 frames per clip, resized to 224×224. Features are extracted using a frozen DINOv2 ViT-B/14 backbone (~86M parameters) and classified with lightweight trainable heads (~198K parameters). The research goal is to extend the baseline with calibrated uncertainty estimation to support safe clinical decision-making.

---

## Current Baseline Pipeline

| Phase | Detail | GPU Memory | Time |
|---|---|---|---|
| DINOv2 feature extraction | 4,127 videos × 16 frames = 66,032 frames | ~3–4 GB | ~5 min (one-time, cached) |
| MLP training (mean/max pooling) | 100 epochs, batch=64, Adam lr=1e-3 | ~300 MB | ~5–15 min per run |
| Attention + MLP joint training | 100 epochs, batch=64, pooling learned end-to-end | ~300 MB | ~10–20 min per run |
| Full baseline grid (6 experiments) | LogisticRegression, LinearSVC, kNN, MLP × 3 pooling strategies | 4 GB peak | ~35–70 min |

Feature cache after extraction: ~202 MB on disk. All subsequent training runs load cached embeddings, making them extremely lightweight.

---

## Planned Uncertainty Estimation Experiments

### MC Dropout
Stochastic forward passes (T=30) at inference with dropout active. No architectural change or extra training required.

- Extra training cost: **0**
- Extra inference cost: ~30× per evaluation sweep (~30 min per full eval)
- GPU memory: same as baseline (~300 MB)
- Planned runs: 10 configs (varying dropout rate 0.1–0.5, T=20–50)

### Deep Ensembles
Train N=10 independent MLP models from different random seeds; aggregate predictions for uncertainty.

- Training cost: **10× baseline** (~12 hours for 10 members across 9 conditions)
- GPU memory: same per run (~300 MB); runs are independent and can be parallelised
- Planned runs: 10 members × 3 random seeds = 30 training runs
- Most compute-intensive method in the project

### Evidential / Bayesian Head
Replace the sigmoid output with a Dirichlet (classification) or Normal-Inverse-Gamma (regression) head. Requires a custom loss function and a small architectural change to the classifier.

- Training overhead: ~1.2–1.5× per run (custom NIG/Dirichlet loss)
- GPU memory: ~300–400 MB
- Planned runs: 25 configs across learning rate (1e-4–1e-2), regularisation coefficient, and head variants

### Conformal Prediction
Post-hoc calibration on the held-out validation set using prediction sets with guaranteed coverage. No additional training required.

- GPU compute: **0** (CPU-only calibration, seconds per condition)
- GPU memory: 0 additional

---

## Full Experimental Budget

| Experiment Phase | Runs | Estimated GPU Hours |
|---|---|---|
| Baseline grid (6 experiments, 9 conditions) | 54 | ~9 h |
| MC Dropout sweeps | 10 configs | ~3 h |
| Deep Ensembles (10 members × 3 seeds) | 30 | ~35 h |
| Evidential/Bayesian head search | 25 configs | ~8 h |
| Conformal Prediction | post-hoc | ~0 h |
| Ablations, reruns, final evaluation | ~20 | ~10 h |
| **Total** | **~139 runs** | **~65 GPU hours** |

With 20% buffer for debugging and failed runs: **~78 GPU hours**.

---

## Hardware Requirements

| | Specification |
|---|---|
| **Minimum VRAM** | 8 GB (RTX 3070 / A10) |
| **Recommended VRAM** | 16–24 GB (RTX 3090 / A40) — fits DINOv2 + training batch comfortably |
| **Preferred allocation** | 1× A40 (48 GB) or 2× A10 (24 GB each) |
| **Duration** | 2–3 weeks |
| **Parallelism** | Ensemble members and condition-level runs are fully independent; can use 1–4 GPUs |

A single modern GPU (≥8 GB VRAM) is sufficient for all runs. Multiple GPUs would reduce wall-clock time for the Deep Ensembles phase from ~35 h to ~9 h (4 GPUs).

---

## Key Files

| File | Role |
|---|---|
| `config.yaml` | Training hyperparameters (epochs=100, lr=1e-3, batch=64, frames=16) |
| `run_baseline.py` | Experiment driver, EXPERIMENT_GRID |
| `data/dataset.py` | Video loading, 80/10/10 subject-level split |
| `classifiers/mlp.py` | MLP head (to be extended with uncertainty heads) |
| `aggregation/pooling.py` | Attention pooling (joint training with MLP) |
