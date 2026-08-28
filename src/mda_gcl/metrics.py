"""Metrics computed from exact saved class-ID predictions."""

from __future__ import annotations

from numbers import Integral
from typing import Sequence

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score


def _class_id_arrays(
    targets: Sequence[int] | np.ndarray,
    predictions: Sequence[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    arrays: list[np.ndarray] = []
    for name, values in (("targets", targets), ("predictions", predictions)):
        array = np.asarray(values)
        if array.ndim != 1:
            raise ValueError(
                f"{name} must be one-dimensional exact class IDs, not logits or probabilities"
            )
        if array.size == 0:
            raise ValueError(f"{name} must be a non-empty class-ID array")
        if np.issubdtype(array.dtype, np.bool_) or not np.issubdtype(
            array.dtype, np.integer
        ):
            raise ValueError(
                f"{name} must contain exact integer class IDs, not logits or probabilities"
            )
        array = array.astype(np.int64, copy=False)
        if np.any(array < 0):
            raise ValueError(f"{name} class IDs must be non-negative")
        arrays.append(array)
    if arrays[0].shape != arrays[1].shape:
        raise ValueError(
            "targets and predictions must contain the same number of class IDs"
        )
    return arrays[0], arrays[1]


def accuracy(
    targets: Sequence[int] | np.ndarray,
    predictions: Sequence[int] | np.ndarray,
) -> float:
    """Return exact classification accuracy as a percentage in ``[0, 100]``."""

    target_array, prediction_array = _class_id_arrays(targets, predictions)
    return float(np.mean(target_array == prediction_array) * 100.0)


def weighted_f1(
    targets: Sequence[int] | np.ndarray,
    predictions: Sequence[int] | np.ndarray,
) -> float:
    """Return weighted F1 from exact class IDs."""

    target_array, prediction_array = _class_id_arrays(targets, predictions)
    return float(
        f1_score(target_array, prediction_array, average="weighted", zero_division=0)
    )


def normalized_confusion(
    targets: Sequence[int] | np.ndarray,
    predictions: Sequence[int] | np.ndarray,
    *,
    n_classes: int | None = None,
) -> np.ndarray:
    """Return a row-normalized confusion matrix with stable class dimensions."""

    target_array, prediction_array = _class_id_arrays(targets, predictions)
    observed_max = int(max(target_array.max(), prediction_array.max()))
    if n_classes is None:
        class_count = observed_max + 1
    elif (
        isinstance(n_classes, bool)
        or not isinstance(n_classes, Integral)
        or n_classes <= 0
    ):
        raise ValueError("n_classes must be a positive integer")
    else:
        class_count = int(n_classes)
    if observed_max >= class_count:
        raise ValueError(
            f"class ID {observed_max} is outside configured n_classes={class_count}"
        )
    return confusion_matrix(
        target_array,
        prediction_array,
        labels=np.arange(class_count),
        normalize="true",
    )
