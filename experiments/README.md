# Experiments

Runnable experiment entry points live here. Shared code should stay in the
top-level packages (`data`, `features`, `aggregation`, `classifiers`,
`evaluate`) so future uncertainty methods can reuse the same splits and metrics.

Suggested layout:

- `run_disease_holdout_baseline.py`: train a standard classifier without the
  held-out diseases and evaluate entropy-based baseline uncertainty.
- `run_disease_holdout_mc_dropout.py`: train an MLP on ID diseases and evaluate
  MC dropout entropy/mutual-information uncertainty on held-out diseases.
- `run_disease_holdout_energy.py`: train a deterministic two-logit MLP and
  evaluate post-hoc energy scores on held-out diseases.
- `run_ensemble_*.py`: deep ensemble variants.
- `run_evidential_*.py`: evidential/Bayesian head variants.
- `run_conformal_*.py`: conformal prediction calibration/evaluation.

The current disease-holdout setup excludes Tetralogy of Fallot, AVSD, and
Aortic Stenosis from training by default.
