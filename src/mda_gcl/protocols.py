"""Deterministic label-safe subject and session protocols."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .config import ExperimentConfig


@dataclass(frozen=True, slots=True)
class Fold:
    """Source/target row indices; emotion labels are intentionally absent."""

    fold_id: str
    protocol: Literal["cross_subject", "cross_session"]
    source_indices: np.ndarray
    target_indices: np.ndarray
    target_subject: int
    target_session: int | None = None

    @property
    def indices(self) -> np.ndarray:
        """Rows in the source-then-target order expected by domain labels."""

        return np.concatenate((self.source_indices, self.target_indices))


def _expected_rows(config: ExperimentConfig, n_rows: int | None) -> int:
    expected = config.n_subjects * config.n_samples_per_subject
    actual = expected if n_rows is None else n_rows
    if isinstance(actual, bool) or not isinstance(actual, int) or actual != expected:
        raise ValueError(
            f"configured folds require {expected} rows "
            f"({config.n_subjects} subjects x {config.n_samples_per_subject} samples); "
            f"received {actual}"
        )
    return expected


def _immutable_indices(values: np.ndarray) -> np.ndarray:
    indices = np.asarray(values, dtype=np.int64)
    indices.setflags(write=False)
    return indices


def iter_subject_folds(
    config: ExperimentConfig,
    n_rows: int | None = None,
) -> Iterator[Fold]:
    """Yield leave-one-subject-out folds without accepting emotion labels."""

    if config.protocol != "cross_subject":
        raise ValueError("iter_subject_folds requires protocol='cross_subject'")
    total = _expected_rows(config, n_rows)
    if config.n_subjects < 2:
        raise ValueError("cross-subject evaluation requires at least two subjects")
    all_indices = np.arange(total, dtype=np.int64)
    samples = config.n_samples_per_subject
    for subject in range(config.n_subjects):
        start = subject * samples
        stop = start + samples
        target = all_indices[start:stop]
        source = np.concatenate((all_indices[:start], all_indices[stop:]))
        yield Fold(
            fold_id=f"subject-{subject + 1:02d}",
            protocol="cross_subject",
            source_indices=_immutable_indices(source),
            target_indices=_immutable_indices(target),
            target_subject=subject,
        )


def iter_session_folds(
    config: ExperimentConfig,
    n_rows: int | None = None,
) -> Iterator[Fold]:
    """Yield within-subject leave-one-session-out folds from configured boundaries."""

    if config.protocol != "cross_session":
        raise ValueError("iter_session_folds requires protocol='cross_session'")
    _expected_rows(config, n_rows)
    boundaries = config.session_boundaries
    if len(boundaries) < 2:
        raise ValueError("cross-session evaluation requires at least two session boundaries")
    samples = config.n_samples_per_subject
    local = np.arange(samples, dtype=np.int64)
    for subject in range(config.n_subjects):
        offset = subject * samples
        for session, (start, stop) in enumerate(boundaries):
            target_local = local[start:stop]
            source_local = np.concatenate((local[:start], local[stop:]))
            if source_local.size == 0 or target_local.size == 0:
                raise ValueError(
                    f"session {session + 1} creates an empty source or target partition"
                )
            yield Fold(
                fold_id=f"subject-{subject + 1:02d}-session-{session + 1:02d}",
                protocol="cross_session",
                source_indices=_immutable_indices(source_local + offset),
                target_indices=_immutable_indices(target_local + offset),
                target_subject=subject,
                target_session=session,
            )


def _validate_fold(config: ExperimentConfig, fold: Fold) -> None:
    if fold.protocol != config.protocol:
        raise ValueError(
            f"fold protocol {fold.protocol!r} does not match config protocol {config.protocol!r}"
        )
    source = fold.source_indices
    target = fold.target_indices
    if source.ndim != 1 or target.ndim != 1 or source.size == 0 or target.size == 0:
        raise ValueError("domain labels require non-empty one-dimensional source/target indices")
    if not np.issubdtype(source.dtype, np.integer) or not np.issubdtype(
        target.dtype, np.integer
    ):
        raise ValueError("fold indices must be integers")
    total = config.n_subjects * config.n_samples_per_subject
    combined = fold.indices
    if (combined < 0).any() or (combined >= total).any():
        raise ValueError("fold indices are outside the configured feature rows")
    if np.unique(combined).size != combined.size:
        raise ValueError("source and target fold indices must be unique and disjoint")


def build_domain_labels(config: ExperimentConfig, fold: Fold) -> np.ndarray:
    """Build DANN labels in ``fold.indices`` order, separate from emotion labels."""

    _validate_fold(config, fold)
    return np.concatenate(
        (
            np.zeros(fold.source_indices.size, dtype=np.int64),
            np.ones(fold.target_indices.size, dtype=np.int64),
        )
    )
