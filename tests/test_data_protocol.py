from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mda_gcl.config import ExperimentConfig, load_config
from mda_gcl.data import apply_standardizer, fit_source_standardizer, load_feature_bundle, standardize_partitions
from mda_gcl.protocols import build_domain_labels, iter_session_folds, iter_subject_folds


RELEASE_ROOT = Path(__file__).parents[1]


def make_config(*, dataset: str = "seed_iv", protocol: str = "cross_subject") -> ExperimentConfig:
    base = load_config(RELEASE_ROOT / "configs" / f"{dataset}_{protocol}.yaml")
    return replace(
        base,
        n_samples_per_subject=6,
        session_boundaries=((0, 2), (2, 4), (4, 6)),
        de_path=Path("de.npy"),
        psd_path=Path("psd.npy") if dataset == "seed_iv" else None,
        eye_path=Path("eye.npy"),
        label_path=Path("labels.npy"),
    )


def save_arrays(directory: Path, dataset: str) -> None:
    subjects = 15 if dataset == "seed_iv" else 16
    classes = 4 if dataset == "seed_iv" else 5
    eye_dim = 31 if dataset == "seed_iv" else 33
    de = np.arange(subjects * 6 * 310, dtype=np.float32).reshape(subjects, 6, 5, 62)
    np.save(directory / "de.npy", de)
    if dataset == "seed_iv":
        np.save(directory / "psd.npy", de + 0.5)
    np.save(directory / "eye.npy", np.arange(subjects * 6 * eye_dim, dtype=np.float32).reshape(subjects, 6, eye_dim))
    np.save(directory / "labels.npy", np.tile(np.arange(6) % classes, (subjects, 1)))


@pytest.mark.parametrize(
    ("dataset", "names", "feature_dim"),
    [("seed_iv", ("de", "psd", "eye"), 651), ("seed_v", ("de", "eye"), 343)],
)
def test_dataset_loads_only_main_views(
    tmp_path: Path, dataset: str, names: tuple[str, ...], feature_dim: int
) -> None:
    save_arrays(tmp_path, dataset)
    config = make_config(dataset=dataset)
    bundle = load_feature_bundle(config, tmp_path)

    assert bundle.features.shape == (config.n_subjects * 6, feature_dim)
    assert bundle.labels.shape == (config.n_subjects * 6,)
    assert bundle.view_names == names
    assert config.input_dim == feature_dim
    assert bundle.features.dtype == np.float32
    assert bundle.labels.dtype == np.int64


def test_source_target_partitions_are_independently_standardized_and_finite() -> None:
    features = np.array(
        [[1.0, 5.0, 7.0], [3.0, 5.0, 9.0], [101.0, 50.0, 19.0], [105.0, 50.0, 23.0]]
    )
    mean, scale = fit_source_standardizer(features, np.array([0, 1]))
    transformed = standardize_partitions(features, np.array([0, 1]), np.array([2, 3]))

    np.testing.assert_allclose(mean, [2.0, 5.0, 8.0])
    np.testing.assert_allclose(scale, [1.0, 1.0, 1.0])
    np.testing.assert_allclose(apply_standardizer(features[:2], mean, scale), transformed[:2])
    np.testing.assert_allclose(transformed[:2].mean(axis=0), 0.0, atol=1e-7)
    np.testing.assert_allclose(transformed[2:].mean(axis=0), 0.0, atol=1e-7)
    np.testing.assert_allclose(transformed[:, 1], 0.0)
    assert np.isfinite(transformed).all()


@pytest.mark.parametrize(
    ("source", "target", "message"),
    [
        (np.array([], dtype=np.int64), np.array([1]), "source_indices"),
        (np.array([0]), np.array([], dtype=np.int64), "target_indices"),
        (np.array([0, 0]), np.array([1]), "duplicates"),
        (np.array([0]), np.array([0]), "disjoint"),
    ],
)
def test_partition_standardization_rejects_invalid_indices(
    source: np.ndarray, target: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        standardize_partitions(np.ones((3, 2)), source, target)


def test_subject_folds_and_target_label_perturbation_are_label_safe(tmp_path: Path) -> None:
    save_arrays(tmp_path, "seed_iv")
    config = make_config()
    before = load_feature_bundle(config, tmp_path)
    first = next(iter_subject_folds(config, before.features.shape[0]))
    normalized_before = standardize_partitions(before.features, first.source_indices, first.target_indices)

    changed_labels = before.labels.reshape(15, 6).copy()
    changed_labels[0] = (changed_labels[0] + 1) % config.n_classes
    np.save(tmp_path / "labels.npy", changed_labels)
    after = load_feature_bundle(config, tmp_path)
    normalized_after = standardize_partitions(after.features, first.source_indices, first.target_indices)

    np.testing.assert_array_equal(first.target_indices, np.arange(6))
    assert first.source_indices.size == 14 * 6
    np.testing.assert_allclose(normalized_before, normalized_after)
    np.testing.assert_array_equal(before.labels[first.source_indices], after.labels[first.source_indices])
    assert not np.array_equal(before.labels[first.target_indices], after.labels[first.target_indices])


def test_session_folds_and_binary_domains_use_source_then_target_order() -> None:
    config = make_config(protocol="cross_session")
    folds = list(iter_session_folds(config, 15 * 6))
    first = folds[0]

    assert len(folds) == 45
    np.testing.assert_array_equal(first.source_indices, [2, 3, 4, 5])
    np.testing.assert_array_equal(first.target_indices, [0, 1])
    np.testing.assert_array_equal(build_domain_labels(config, first), [0, 0, 0, 0, 1, 1])
    subject_config = make_config()
    subject_fold = next(iter_subject_folds(subject_config))
    domains = build_domain_labels(subject_config, subject_fold)
    np.testing.assert_array_equal(domains[: 14 * 6], 0)
    np.testing.assert_array_equal(domains[14 * 6 :], 1)


@pytest.mark.parametrize(
    ("case", "message"),
    [("missing", "missing EEG DE array"), ("axes", "malformed EEG DE axes"), ("labels", "malformed emotion label axes")],
)
def test_missing_and_malformed_arrays_fail_actionably(tmp_path: Path, case: str, message: str) -> None:
    config = make_config()
    np.save(tmp_path / "labels.npy", np.zeros((15, 6), dtype=np.int64))
    if case == "axes":
        np.save(tmp_path / "de.npy", np.zeros((15, 6, 10, 31), dtype=np.float32))
    elif case == "labels":
        np.save(tmp_path / "de.npy", np.zeros((15, 6, 5, 62), dtype=np.float32))
        np.save(tmp_path / "labels.npy", np.zeros((15, 5), dtype=np.int64))

    with pytest.raises(FileNotFoundError if case == "missing" else ValueError, match=message):
        load_feature_bundle(config, tmp_path)


def test_impossible_fold_requests_fail_clearly() -> None:
    with pytest.raises(ValueError, match="require 90 rows"):
        list(iter_subject_folds(make_config(), 89))
    one_session = replace(make_config(protocol="cross_session"), session_boundaries=((0, 6),))
    with pytest.raises(ValueError, match="at least two session boundaries"):
        list(iter_session_folds(one_session))
