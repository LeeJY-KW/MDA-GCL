"""Validated loading and split-independent normalization for precomputed features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import ExperimentConfig


@dataclass(frozen=True, slots=True)
class FeatureBundle:
    """Flattened feature rows plus the individual views used to build graphs."""

    features: np.ndarray
    labels: np.ndarray
    views: tuple[np.ndarray, ...]
    view_names: tuple[str, ...]


def _load_npy(base_dir: Path, relative_path: Path | None, name: str) -> np.ndarray:
    if relative_path is None:
        raise ValueError(f"{name} path is required by the selected feature configuration")
    path = base_dir / relative_path
    if path.suffix.lower() != ".npy":
        raise ValueError(f"{name} must be a precomputed .npy array; received {path}")
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {name} array at {path}; place the precomputed .npy file at "
            "the configured path (datasets and preprocessing are not included)"
        )
    try:
        array = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"could not load {name} array at {path}: {exc}") from exc
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise ValueError(f"{name} must contain real numeric values; received dtype {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def _flatten_feature(
    array: np.ndarray,
    config: ExperimentConfig,
    name: str,
    expected_dim: int,
    *,
    allow_eeg_axes: bool,
) -> np.ndarray:
    prefix = (config.n_subjects, config.n_samples_per_subject)
    flat_shape = prefix + (expected_dim,)
    legacy_eeg_shape = prefix + (5, 62)
    if array.shape == flat_shape:
        flattened = array.reshape(-1, expected_dim)
    elif allow_eeg_axes and expected_dim == 310 and array.shape == legacy_eeg_shape:
        flattened = array.reshape(-1, expected_dim)
    else:
        expected = f"{flat_shape}"
        if allow_eeg_axes and expected_dim == 310:
            expected += f" or {legacy_eeg_shape}"
        raise ValueError(
            f"malformed {name} axes: expected {expected}, received {array.shape}; "
            "only legacy subject/sample feature layouts are supported"
        )
    return flattened.astype(np.float32, copy=False)


def _load_labels(
    config: ExperimentConfig,
    base_dir: Path,
) -> np.ndarray:
    labels = _load_npy(base_dir, config.label_path, "emotion labels")
    expected = (config.n_subjects, config.n_samples_per_subject)
    expected_singleton = expected + (1,)
    if labels.shape == expected_singleton:
        labels = labels[..., 0]
    elif labels.shape != expected:
        raise ValueError(
            f"malformed emotion label axes: expected {expected} or "
            f"{expected_singleton}, received {labels.shape}; "
            "labels must have one row per configured subject/sample"
        )
    if not np.equal(labels, np.floor(labels)).all():
        raise ValueError("emotion labels must be integer class IDs")
    if ((labels < 0) | (labels >= config.n_classes)).any():
        raise ValueError(
            f"emotion labels must be in [0, {config.n_classes - 1}] for {config.dataset}"
        )
    return labels.reshape(-1).astype(np.int64, copy=False)


def load_feature_bundle(
    config: ExperimentConfig,
    base_dir: str | Path = ".",
) -> FeatureBundle:
    """Load the fixed main feature views for the configured dataset."""

    root = Path(base_dir)
    labels = _load_labels(config, root)
    de = _flatten_feature(
        _load_npy(root, config.de_path, "EEG DE"),
        config,
        "EEG DE",
        config.eeg_single_feature_dim,
        allow_eeg_axes=True,
    )
    views = [de]
    names = ["de"]
    if config.dataset == "seed_iv":
        views.append(
            _flatten_feature(
                _load_npy(root, config.psd_path, "EEG PSD"),
                config,
                "EEG PSD",
                config.eeg_single_feature_dim,
                allow_eeg_axes=True,
            )
        )
        names.append("psd")
    views.append(
        _flatten_feature(
            _load_npy(root, config.eye_path, "eye"),
            config,
            "eye",
            config.eye_feature_dim,
            allow_eeg_axes=False,
        )
    )
    names.append("eye")
    row_counts = {view.shape[0] for view in views} | {labels.shape[0]}
    if len(row_counts) != 1:
        received = {name: view.shape[0] for name, view in zip(names, views)}
        received["emotion labels"] = labels.shape[0]
        raise ValueError(f"incompatible feature rows and labels: {received}")
    features = np.concatenate(views, axis=1)
    return FeatureBundle(features, labels, tuple(views), tuple(names))


def _feature_matrix(features: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(features)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional feature matrix")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite numeric values")
    return array


def fit_source_standardizer(
    features: np.ndarray,
    source_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit mean and scale from selected feature rows without accepting labels."""

    array = _feature_matrix(features, "features")
    if source_indices is None:
        source = array
    else:
        indices = np.asarray(source_indices)
        if indices.dtype == np.bool_:
            if indices.ndim != 1 or indices.shape[0] != array.shape[0]:
                raise ValueError("source boolean mask must match the feature row count")
            source = array[indices]
        else:
            if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
                raise ValueError("source_indices must be a one-dimensional integer index array")
            if indices.size == 0 or (indices < 0).any() or (indices >= array.shape[0]).any():
                raise ValueError("source_indices must select at least one valid feature row")
            if np.unique(indices).size != indices.size:
                raise ValueError("source_indices must not contain duplicates")
            source = array[indices]
    if source.shape[0] == 0:
        raise ValueError("at least one source training row is required")
    mean = source.mean(axis=0, dtype=np.float64)
    scale = source.std(axis=0, dtype=np.float64)
    scale = np.where(scale == 0, 1.0, scale)
    return mean, scale


def apply_standardizer(
    features: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """Apply fitted parameters to feature rows."""

    array = _feature_matrix(features, "features")
    mean_array = np.asarray(mean)
    scale_array = np.asarray(scale)
    expected = (array.shape[1],)
    if mean_array.shape != expected or scale_array.shape != expected:
        raise ValueError(
            f"standardizer parameters must both have shape {expected}; received "
            f"{mean_array.shape} and {scale_array.shape}"
        )
    if not np.isfinite(mean_array).all() or not np.isfinite(scale_array).all():
        raise ValueError("standardizer parameters must be finite")
    if (scale_array <= 0).any():
        raise ValueError("standardizer scale values must be positive")
    transformed = (array - mean_array) / scale_array
    dtype = array.dtype if np.issubdtype(array.dtype, np.floating) else np.float64
    return transformed.astype(dtype, copy=False)


def standardize_partitions(
    features: np.ndarray,
    source_indices: np.ndarray,
    target_indices: np.ndarray,
) -> np.ndarray:
    """Z-score source and target rows independently using feature values only."""

    array = _feature_matrix(features, "features")
    partitions: list[np.ndarray] = []
    for name, values in (
        ("source_indices", source_indices),
        ("target_indices", target_indices),
    ):
        indices = np.asarray(values)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(f"{name} must be a one-dimensional integer index array")
        if indices.size == 0 or (indices < 0).any() or (indices >= array.shape[0]).any():
            raise ValueError(f"{name} must select at least one valid feature row")
        if np.unique(indices).size != indices.size:
            raise ValueError(f"{name} must not contain duplicates")
        partitions.append(indices.astype(np.int64, copy=False))
    if np.intersect1d(*partitions).size:
        raise ValueError("source_indices and target_indices must be disjoint")

    dtype = array.dtype if np.issubdtype(array.dtype, np.floating) else np.float64
    transformed = array.astype(dtype, copy=True)
    for indices in partitions:
        mean, scale = fit_source_standardizer(array, indices)
        transformed[indices] = apply_standardizer(array[indices], mean, scale)
    return transformed
