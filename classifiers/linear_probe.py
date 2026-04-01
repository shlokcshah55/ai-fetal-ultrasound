from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


def get_logistic_regression(seed: int = 42) -> LogisticRegression:
    """Logistic Regression classifier with balanced class weights.

    Args:
        seed: Random state for reproducibility.

    Returns:
        Configured LogisticRegression instance.
    """
    return LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=seed,
    )


def get_linear_svc(seed: int = 42) -> CalibratedClassifierCV:
    """Linear SVM wrapped with calibration to enable predict_proba.

    LinearSVC does not expose predict_proba natively; wrapping with
    CalibratedClassifierCV (sigmoid calibration) enables AUROC/AUPRC
    computation.

    Args:
        seed: Random state for reproducibility.

    Returns:
        CalibratedClassifierCV wrapping a LinearSVC.
    """
    base = LinearSVC(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        random_state=seed,
    )
    return CalibratedClassifierCV(base, cv=5, method="sigmoid")
