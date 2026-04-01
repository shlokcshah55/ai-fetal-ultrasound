from __future__ import annotations

import random
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
import joblib

from data.dataset import get_dataloaders
from features.extract import load_dinov2, extract_features
from aggregation.pooling import mean_pool, max_pool, AttentionPooling, attention_pool
from classifiers.linear_probe import get_logistic_regression, get_linear_svc
from classifiers.knn import get_knn
from classifiers.mlp import MLP, train_mlp
from evaluate import evaluate, find_optimal_threshold


EXPERIMENT_GRID = [
    ("mean",      "LogisticRegression"),
    ("mean",      "LinearSVC"),
    ("mean",      "MLP"),
    ("mean",      "kNN"),
    ("max",       "MLP"),
    ("attention", "MLP"),
]


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str = "config.yaml") -> dict:
    repo_root = Path(__file__).parent
    with open(repo_root / config_path) as f:
        return yaml.safe_load(f)


def pool_features(
    feature_dict: dict[str, tuple[np.ndarray, int]],
    pooling: str,
    attention_head: AttentionPooling | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply pooling to a feature dict and return (X, y) arrays.

    For 'attention' pooling, attention_head must be provided and is used
    for inference (no gradient updates here).

    Returns:
        X of shape (M, 768), y of shape (M,)
    """
    X_list, y_list = [], []
    for video_path, (features, label) in feature_dict.items():
        if pooling == "mean":
            vec = mean_pool(features)
        elif pooling == "max":
            vec = max_pool(features)
        elif pooling == "attention":
            assert attention_head is not None
            vec = attention_pool(features, attention_head)
        else:
            raise ValueError(f"Unknown pooling: {pooling}")
        X_list.append(vec)
        y_list.append(label)
    return np.stack(X_list), np.array(y_list)


def train_attention_mlp(
    train_features: dict[str, tuple[np.ndarray, int]],
    val_features: dict[str, tuple[np.ndarray, int]],
    config: dict,
    device: torch.device,
    checkpoint_stem: str,
) -> tuple[MLP, AttentionPooling]:
    """Joint training of AttentionPooling + MLP for the attention experiment.

    The attention head is trained end-to-end with the MLP — both their
    parameters are passed to a single Adam optimizer.

    Returns:
        (trained MLP, trained AttentionPooling)
    """
    attention_head = AttentionPooling(embed_dim=768).to(device)
    model = MLP(input_dim=768).to(device)

    # Assemble dataset: each sample is a (N, 768) tensor
    def make_tensors(
        feature_dict: dict[str, tuple[np.ndarray, int]],
    ) -> tuple[list[torch.Tensor], list[int]]:
        tensors, labels = [], []
        for _, (features, label) in feature_dict.items():
            tensors.append(torch.from_numpy(features.astype(np.float32)))
            labels.append(label)
        return tensors, labels

    train_tensors, train_labels = make_tensors(train_features)
    val_tensors, val_labels = make_tensors(val_features)

    y_train_arr = np.array(train_labels)
    y_val_arr = np.array(val_labels)

    count_pos = int(y_train_arr.sum())
    count_neg = int(len(y_train_arr) - count_pos)
    pos_weight_val = count_neg / max(count_pos, 1)
    pos_weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float32, device=device)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(attention_head.parameters()),
        lr=config["training"]["mlp_lr"],
        weight_decay=config["training"]["mlp_weight_decay"],
    )

    patience = config["training"]["early_stopping_patience"]
    best_val_auroc = -1.0
    patience_counter = 0

    attn_ckpt = f"checkpoints/{checkpoint_stem}_attention_head.pt"
    mlp_ckpt = f"checkpoints/{checkpoint_stem}_mlp.pt"
    Path("checkpoints").mkdir(exist_ok=True)

    from sklearn.metrics import roc_auc_score, accuracy_score

    for epoch in range(config["training"]["mlp_epochs"]):
        # --- Training ---
        model.train()
        attention_head.train()

        # Shuffle training data each epoch
        indices = list(range(len(train_tensors)))
        random.shuffle(indices)

        batch_size = 64
        total_loss = 0.0
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            if len(batch_idx) <= 1:
                # Skip batches of size 1 (BatchNorm1d would fail)
                continue

            batch_frames = [train_tensors[j].to(device) for j in batch_idx]
            batch_labels = torch.tensor(
                [train_labels[j] for j in batch_idx], dtype=torch.float32, device=device
            )

            optimizer.zero_grad()
            # Pool each video with the attention head, then stack
            pooled = torch.stack(
                [attention_head(frames) for frames in batch_frames], dim=0
            )  # (B, 768)
            logits = model(pooled)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # --- Validation ---
        model.eval()
        attention_head.eval()

        val_proba_list: list[float] = []
        with torch.no_grad():
            for frames in val_tensors:
                pooled_v = attention_head(frames.to(device))  # (768,)
                logit = model(pooled_v.unsqueeze(0))          # (1,)
                val_proba_list.append(torch.sigmoid(logit).item())

        val_proba = np.array(val_proba_list)
        try:
            val_auroc = float(roc_auc_score(y_val_arr, val_proba))
        except ValueError:
            val_pred = (val_proba >= 0.5).astype(int)
            val_auroc = float(accuracy_score(y_val_arr, val_pred))

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            patience_counter = 0
            torch.save(model.state_dict(), mlp_ckpt)
            torch.save(attention_head.state_dict(), attn_ckpt)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Reload best checkpoints
    model.load_state_dict(torch.load(mlp_ckpt, map_location=device))
    attention_head.load_state_dict(torch.load(attn_ckpt, map_location=device))
    model.eval()
    attention_head.eval()

    return model, attention_head


def predict_attention_mlp(
    feature_dict: dict[str, tuple[np.ndarray, int]],
    model: MLP,
    attention_head: AttentionPooling,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference with trained attention + MLP on a feature dict."""
    y_true_list, y_proba_list = [], []
    model.eval()
    attention_head.eval()
    with torch.no_grad():
        for _, (features, label) in feature_dict.items():
            frames = torch.from_numpy(features.astype(np.float32)).to(device)
            pooled = attention_head(frames)               # (768,)
            logit = model(pooled.unsqueeze(0))            # (1,)
            proba = torch.sigmoid(logit).item()
            y_proba_list.append(proba)
            y_true_list.append(label)
    return np.array(y_true_list), np.array(y_proba_list)


def main() -> None:
    repo_root = Path(__file__).parent
    os.chdir(repo_root)

    config = load_config()
    seed = config["training"]["seed"]
    set_seeds(seed)

    device_str = config["features"]["device"]
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------ #
    # 1. Data loading                                                      #
    # ------------------------------------------------------------------ #
    print("\n[1/4] Loading data...")
    sononet_cfg = config.get("sononet", {})
    train_loader, val_loader, test_loader, split_info = get_dataloaders(
        csv_path=config["data"]["csv_path"],
        n_frames=config["data"]["n_frames"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        split=config["data"]["split"],
        seed=seed,
        sononet_dir=sononet_cfg.get("dir"),
        conf_threshold=sononet_cfg.get("conf_threshold", 0.5),
    )
    print(
        f"  Subjects — train: {len(split_info['train_subjects'])}, "
        f"val: {len(split_info['val_subjects'])}, "
        f"test: {len(split_info['test_subjects'])}"
    )

    # ------------------------------------------------------------------ #
    # 2. Feature extraction                                                #
    # ------------------------------------------------------------------ #
    print("\n[2/4] Extracting DINOv2 features...")
    dinov2 = load_dinov2(device)

    cache_dir = sononet_cfg.get("cache_dir") or config["features"]["cache_dir"]
    train_features = extract_features(train_loader, dinov2, device, cache_dir)
    val_features = extract_features(val_loader, dinov2, device, cache_dir)
    test_features = extract_features(test_loader, dinov2, device, cache_dir)
    print(
        f"  Videos — train: {len(train_features)}, "
        f"val: {len(val_features)}, test: {len(test_features)}"
    )

    # ------------------------------------------------------------------ #
    # 3. Pre-pool for mean/max experiments                                 #
    # ------------------------------------------------------------------ #
    print("\n[3/4] Pre-pooling features for mean/max experiments...")
    pooled_data: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for pooling in ("mean", "max"):
        X_train, y_train = pool_features(train_features, pooling)
        X_val, y_val = pool_features(val_features, pooling)
        X_test, y_test = pool_features(test_features, pooling)
        pooled_data[pooling] = {
            "train": (X_train, y_train),
            "val": (X_val, y_val),
            "test": (X_test, y_test),
        }

    # ------------------------------------------------------------------ #
    # 4. Experiment grid                                                   #
    # ------------------------------------------------------------------ #
    print("\n[4/4] Running experiment grid...")
    Path("results").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)

    results_rows: list[dict] = []

    for pooling, clf_name in EXPERIMENT_GRID:
        print(f"\n  [{pooling} + {clf_name}]")
        ckpt_stem = f"{pooling}_{clf_name}"

        # ---- attention + MLP: joint training path ---------------------- #
        if pooling == "attention" and clf_name == "MLP":
            mlp_model, attn_head = train_attention_mlp(
                train_features, val_features, config, device, ckpt_stem
            )

            y_val_true, y_val_proba = predict_attention_mlp(
                val_features, mlp_model, attn_head, device
            )
            opt_threshold = find_optimal_threshold(y_val_true, y_val_proba)

            y_test_true, y_test_proba = predict_attention_mlp(
                test_features, mlp_model, attn_head, device
            )
            metrics = evaluate(y_test_true, y_test_proba, threshold=opt_threshold)

            results_rows.append({
                "pooling": pooling,
                "classifier": clf_name,
                **{k: v for k, v in metrics.items()},
            })
            _print_row(pooling, clf_name, metrics)
            continue

        # ---- pre-pooled path (mean / max) ------------------------------ #
        X_train, y_train = pooled_data[pooling]["train"]
        X_val, y_val = pooled_data[pooling]["val"]
        X_test, y_test = pooled_data[pooling]["test"]

        if clf_name == "MLP":
            mlp_config = {
                "mlp_epochs": config["training"]["mlp_epochs"],
                "mlp_lr": config["training"]["mlp_lr"],
                "mlp_weight_decay": config["training"]["mlp_weight_decay"],
                "early_stopping_patience": config["training"]["early_stopping_patience"],
            }
            mlp_model, _ = train_mlp(
                X_train, y_train,
                X_val, y_val,
                config=mlp_config,
                device=device,
                checkpoint_path=f"checkpoints/{ckpt_stem}_mlp.pt",
            )
            mlp_model.eval()

            X_val_t = torch.from_numpy(X_val.astype(np.float32)).to(device)
            X_test_t = torch.from_numpy(X_test.astype(np.float32)).to(device)
            with torch.no_grad():
                val_proba = torch.sigmoid(mlp_model(X_val_t)).cpu().numpy()
                test_proba = torch.sigmoid(mlp_model(X_test_t)).cpu().numpy()

            opt_threshold = find_optimal_threshold(y_val, val_proba)
            metrics = evaluate(y_test, test_proba, threshold=opt_threshold)

        elif clf_name == "LogisticRegression":
            clf = get_logistic_regression(seed=seed)
            clf.fit(X_train, y_train)
            val_proba = clf.predict_proba(X_val)[:, 1]
            test_proba = clf.predict_proba(X_test)[:, 1]
            opt_threshold = find_optimal_threshold(y_val, val_proba)
            metrics = evaluate(y_test, test_proba, threshold=opt_threshold)
            joblib.dump(clf, f"checkpoints/{ckpt_stem}_clf.joblib")

        elif clf_name == "LinearSVC":
            clf = get_linear_svc(seed=seed)
            clf.fit(X_train, y_train)
            val_proba = clf.predict_proba(X_val)[:, 1]
            test_proba = clf.predict_proba(X_test)[:, 1]
            opt_threshold = find_optimal_threshold(y_val, val_proba)
            metrics = evaluate(y_test, test_proba, threshold=opt_threshold)
            joblib.dump(clf, f"checkpoints/{ckpt_stem}_clf.joblib")

        elif clf_name == "kNN":
            clf = get_knn()
            clf.fit(X_train, y_train)
            val_proba = clf.predict_proba(X_val)[:, 1]
            test_proba = clf.predict_proba(X_test)[:, 1]
            opt_threshold = find_optimal_threshold(y_val, val_proba)
            metrics = evaluate(y_test, test_proba, threshold=opt_threshold)
            joblib.dump(clf, f"checkpoints/{ckpt_stem}_clf.joblib")

        else:
            raise ValueError(f"Unknown classifier: {clf_name}")

        results_rows.append({
            "pooling": pooling,
            "classifier": clf_name,
            **{k: v for k, v in metrics.items()},
        })
        _print_row(pooling, clf_name, metrics)

    # ------------------------------------------------------------------ #
    # 5. Save results                                                      #
    # ------------------------------------------------------------------ #
    results_df = pd.DataFrame(results_rows)
    results_csv = "results/baseline_results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"\nResults saved to {results_csv}")
    print("\n" + "=" * 80)
    print(results_df[
        ["pooling", "classifier", "auroc", "auprc", "macro_f1", "sensitivity", "specificity"]
    ].to_string(index=False))
    print("=" * 80)


def _print_row(pooling: str, clf_name: str, metrics: dict) -> None:
    print(
        f"    AUROC={metrics['auroc']:.4f}  AUPRC={metrics['auprc']:.4f}  "
        f"F1={metrics['macro_f1']:.4f}  Sens={metrics['sensitivity']:.4f}  "
        f"Spec={metrics['specificity']:.4f}  Thr={metrics['threshold']:.3f}"
    )


if __name__ == "__main__":
    main()
