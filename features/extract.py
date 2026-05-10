from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def load_dinov2(device: torch.device) -> nn.Module:
    """Load DINOv2 ViT-B/14 from torch.hub, frozen in eval mode.

    The backbone is set to eval() and all parameters have requires_grad=False.
    An assertion verifies no parameters are trainable — this acts as a
    circuit breaker against accidental fine-tuning.

    Args:
        device: Target device for the model.

    Returns:
        Frozen DINOv2 model.
    """
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    assert not any(p.requires_grad for p in model.parameters()), (
        "DINOv2 backbone has trainable parameters — aborting. "
        "The frozen backbone constraint has been violated."
    )

    model = model.to(device)
    return model


def load_cached_features(
    video_paths: list[str],
    cache_dir: str,
    split_name: str | None = None,
) -> dict[str, tuple[np.ndarray, int]]:
    """Load previously cached DINOv2 features from disk (no model inference).

    This avoids decoding videos / loading frames. It expects that the cache
    was populated earlier by `extract_features(...)` using the same cache_dir.

    Args:
        video_paths: List of video paths (strings) used to derive cache keys.
        cache_dir: Directory containing cached .npy files.
        split_name: Optional label used in the progress bar.

    Returns:
        Dict mapping video_path → (features_array of shape (N, 768), label).
    """
    cache_path = Path(cache_dir)
    results: dict[str, tuple[np.ndarray, int]] = {}

    desc = "Loading cached features"
    if split_name is not None:
        desc = f"Loading cached features ({split_name})"

    missing: list[str] = []
    progress = tqdm(video_paths, desc=desc, unit="video")
    for video_path in progress:
        cache_key = hashlib.md5(video_path.encode()).hexdigest()
        feat_file = cache_path / f"{cache_key}.npy"
        label_file = cache_path / f"{cache_key}_label.npy"

        if not (feat_file.exists() and label_file.exists()):
            missing.append(video_path)
            progress.set_postfix(loaded=len(results), missing=len(missing))
            continue

        features = np.load(feat_file)
        label = int(np.load(label_file))
        results[video_path] = (features, label)
        progress.set_postfix(loaded=len(results), missing=len(missing))

    if missing:
        preview = "\n".join(f"  - {p}" for p in missing[:10])
        raise FileNotFoundError(
            "Missing cached DINOv2 features for some videos. "
            "Run extraction first (or use the correct --features-cache-dir).\n"
            f"Missing count: {len(missing)} (showing up to 10)\n{preview}"
        )

    return results


def _get_cls_token(model: nn.Module, frames: torch.Tensor) -> torch.Tensor:
    """Extract CLS token embeddings from DINOv2.

    Handles both dict-returning and tensor-returning forward methods
    across different DINOv2 hub versions.

    Args:
        model: Frozen DINOv2 model.
        frames: Input tensor of shape (N, 3, 224, 224).

    Returns:
        CLS token embeddings of shape (N, 768).
    """
    try:
        # Preferred: explicit CLS token extraction
        out = model.forward_features(frames)
        if isinstance(out, dict):
            return out["x_norm_clstoken"]
        # forward_features may return the full token sequence; take index 0
        return out[:, 0, :]
    except (AttributeError, KeyError):
        # Fallback: default forward (returns CLS token directly in most versions)
        out = model(frames)
        if isinstance(out, dict):
            return out["x_norm_clstoken"]
        return out


def extract_features(
    dataloader: DataLoader,
    model: nn.Module,
    device: torch.device,
    cache_dir: str,
) -> dict[str, tuple[np.ndarray, int]]:
    """Extract and cache DINOv2 CLS-token features for all videos.

    Features are saved to disk as .npy files keyed by the MD5 hash of the
    video path. On subsequent runs, cached files are loaded without re-running
    inference.

    Args:
        dataloader: DataLoader whose dataset returns (frames, label, video_path).
        model: Frozen DINOv2 model (output of load_dinov2).
        device: Device to run inference on.
        cache_dir: Directory for caching .npy feature files.

    Returns:
        Dict mapping video_path → (features_array of shape (N, 768), label).
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    results: dict[str, tuple[np.ndarray, int]] = {}
    use_amp = device.type == "cuda"

    total_videos = None
    try:
        total_videos = len(dataloader.dataset)  # type: ignore[arg-type]
    except Exception:
        total_videos = None

    cache_hits = 0
    computed = 0

    pbar = tqdm(total=total_videos, desc="Extracting features", unit="video")
    for batch in dataloader:
        frames_batch, labels_batch, video_paths = batch
        # frames_batch: list of B tensors, each (N_i, 3, 224, 224) — N_i may vary
        # labels_batch: (B,)
        # video_paths: list of B strings

        batch_size = len(frames_batch)

        for i in range(batch_size):
            video_path = video_paths[i]
            label = int(labels_batch[i].item())

            cache_key = hashlib.md5(video_path.encode()).hexdigest()
            feat_file = cache_path / f"{cache_key}.npy"
            label_file = cache_path / f"{cache_key}_label.npy"

            if feat_file.exists() and label_file.exists():
                features = np.load(feat_file)
                cached_label = int(np.load(label_file))
                results[video_path] = (features, cached_label)
                cache_hits += 1
                pbar.update(1)
                pbar.set_postfix(cache_hit=cache_hits, computed=computed)
                continue

            # (N, 3, 224, 224) → pass all frames as a batch through DINOv2
            frames = frames_batch[i].to(device)  # (N, 3, 224, 224)

            with torch.no_grad():
                if use_amp:
                    with torch.cuda.amp.autocast():
                        embeddings = _get_cls_token(model, frames)
                else:
                    embeddings = _get_cls_token(model, frames)

            # Cast to float32 — autocast may produce float16/bfloat16
            # which numpy does not support
            features = embeddings.float().cpu().numpy()  # (N, 768)

            np.save(feat_file, features)
            np.save(label_file, np.array(label, dtype=np.int64))
            results[video_path] = (features, label)
            computed += 1
            pbar.update(1)
            pbar.set_postfix(cache_hit=cache_hits, computed=computed)

    pbar.close()
    return results
