from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

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


def feature_cache_namespace(
    backbone: str,
    checkpoint_id: str | None = None,
) -> str:
    """Return a filesystem-safe feature-cache namespace."""
    raw = backbone if checkpoint_id in (None, "", "none") else f"{backbone}_{checkpoint_id}"
    safe = "".join(ch if ch.isalnum() else "_" for ch in raw.lower())
    safe = "_".join(part for part in safe.split("_") if part)
    return safe or "features"


def feature_cache_stem(video_path: str, cache_namespace: str | None = None) -> str:
    """Cache stem for a video path, preserving legacy DINO cache names by default."""
    video_hash = hashlib.md5(video_path.encode()).hexdigest()
    if cache_namespace is None:
        return video_hash
    return f"{cache_namespace}_{video_hash}"


def checkpoint_id_from_path(checkpoint_path: str | None) -> str:
    """Short stable identifier for checkpoint-specific cache separation."""
    if checkpoint_path is None:
        return "none"
    path = Path(checkpoint_path)
    label = path.stem or "checkpoint"
    digest_source = str(path.resolve())
    if path.exists():
        stat = path.stat()
        digest_source = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    digest = hashlib.md5(digest_source.encode()).hexdigest()[:8]
    safe_label = "".join(ch if ch.isalnum() else "_" for ch in label.lower())
    safe_label = "_".join(part for part in safe_label.split("_") if part)
    return f"{safe_label}_{digest}" if safe_label else digest


def resolve_fetal_clip_config(
    checkpoint_path: str | None,
    config_path: str | None = None,
) -> str | None:
    """Find the official FetalCLIP config when it is not passed explicitly."""
    if config_path is not None:
        return config_path

    candidates = [Path.cwd() / "FetalCLIP_config.json"]
    if checkpoint_path is not None:
        candidates.append(Path(checkpoint_path).parent / "FetalCLIP_config.json")
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def load_fetal_clip(
    checkpoint_path: str,
    device: torch.device,
    *,
    config_path: str | None = None,
) -> nn.Module:
    """Load a frozen FETAL-CLIP image encoder.

    The official FetalCLIP repository uses ``open_clip`` with
    ``FetalCLIP_config.json`` registered under the model name ``FetalCLIP`` and
    ``FetalCLIP_weights.pt`` passed as ``pretrained``. TorchScript/nn.Module
    checkpoints are also accepted as a fallback for locally exported encoders.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"FETAL-CLIP checkpoint not found: {checkpoint_path}")

    config_path = resolve_fetal_clip_config(checkpoint_path, config_path)
    if config_path is not None:
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"FETAL-CLIP config not found: {config_path}")
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "FETAL-CLIP extraction requires open_clip_torch. Install the "
                "FetalCLIP requirements or run `pip install open_clip_torch`."
            ) from exc

        with open(config_file) as f:
            config_fetalclip = json.load(f)
        open_clip.factory._MODEL_CONFIGS["FetalCLIP"] = config_fetalclip
        model, _, _ = open_clip.create_model_and_transforms(
            "FetalCLIP",
            pretrained=str(path),
        )
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        return model.to(device)

    try:
        model = torch.jit.load(str(path), map_location=device)
    except Exception:
        checkpoint: Any = torch.load(path, map_location=device)
        if isinstance(checkpoint, nn.Module):
            model = checkpoint
        elif isinstance(checkpoint, dict):
            candidate = checkpoint.get("model") or checkpoint.get("module")
            if isinstance(candidate, nn.Module):
                model = candidate
            else:
                raise TypeError(
                    "FETAL-CLIP checkpoint appears to be a state_dict/config-only "
                    "object. Save a TorchScript encoder or an nn.Module checkpoint "
                    "so the extractor can run without architecture-specific code."
                )
        else:
            raise TypeError(
                "Unsupported FETAL-CLIP checkpoint type. Expected TorchScript, "
                "nn.Module, or dict containing an nn.Module under 'model'."
            )

    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model.to(device)


def load_frozen_backbone(
    backbone: str,
    device: torch.device,
    *,
    fetal_clip_checkpoint: str | None = None,
    fetal_clip_config: str | None = None,
) -> tuple[nn.Module, dict[str, str]]:
    """Load a supported frozen feature backbone and return metadata."""
    if backbone == "dinov2":
        model = load_dinov2(device)
        checkpoint_id = "facebookresearch_dinov2_vitb14"
    elif backbone == "fetal_clip":
        if fetal_clip_checkpoint is None:
            raise ValueError("--fetal-clip-checkpoint is required for --backbone fetal_clip.")
        fetal_clip_config = resolve_fetal_clip_config(
            fetal_clip_checkpoint,
            fetal_clip_config,
        )
        model = load_fetal_clip(
            fetal_clip_checkpoint,
            device,
            config_path=fetal_clip_config,
        )
        config_id = checkpoint_id_from_path(fetal_clip_config) if fetal_clip_config else "no_config"
        checkpoint_id = (
            f"{checkpoint_id_from_path(fetal_clip_checkpoint)}_{config_id}"
        )
    else:
        raise ValueError("backbone must be 'dinov2' or 'fetal_clip'.")

    namespace = feature_cache_namespace(backbone, checkpoint_id)
    return model, {
        "backbone": backbone,
        "checkpoint_id": checkpoint_id,
        "cache_namespace": namespace,
    }


def load_cached_features(
    video_paths: list[str],
    cache_dir: str,
    split_name: str | None = None,
    cache_namespace: str | None = None,
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
        cache_key = feature_cache_stem(video_path, cache_namespace)
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


def _get_frame_embeddings(model: nn.Module, frames: torch.Tensor) -> torch.Tensor:
    """Extract one frozen embedding per frame from DINOv2 or CLIP-like encoders."""
    if hasattr(model, "forward_features"):
        try:
            out = model.forward_features(frames)
            if isinstance(out, dict):
                for key in ("x_norm_clstoken", "image_embeds", "pooler_output"):
                    if key in out:
                        return out[key]
            if isinstance(out, torch.Tensor):
                return out[:, 0, :] if out.ndim == 3 else out
        except (AttributeError, KeyError, TypeError):
            pass

    for method_name in ("encode_image", "get_image_features"):
        method = getattr(model, method_name, None)
        if callable(method):
            out = method(frames)
            if isinstance(out, dict):
                for key in ("image_embeds", "embeds", "features"):
                    if key in out:
                        return out[key]
            return out

    out = model(frames)
    if isinstance(out, dict):
        for key in ("x_norm_clstoken", "image_embeds", "pooler_output", "embeds", "features"):
            if key in out:
                return out[key]
        raise KeyError(
            "Backbone forward returned a dict without a known embedding key."
        )
    if isinstance(out, (tuple, list)):
        out = out[0]
    if not isinstance(out, torch.Tensor):
        raise TypeError("Backbone forward did not return tensor embeddings.")
    return out[:, 0, :] if out.ndim == 3 else out


def extract_features(
    dataloader: DataLoader,
    model: nn.Module,
    device: torch.device,
    cache_dir: str,
    cache_namespace: str | None = None,
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

            cache_key = feature_cache_stem(video_path, cache_namespace)
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
                        embeddings = _get_frame_embeddings(model, frames)
                else:
                    embeddings = _get_frame_embeddings(model, frames)

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
