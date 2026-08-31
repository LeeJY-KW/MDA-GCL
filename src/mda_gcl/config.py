"""Portable experiment configuration for the MDA-GCL release."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path, PureWindowsPath
from typing import Any, Literal

import yaml


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    dataset: Literal["seed_iv", "seed_v"]
    protocol: Literal["cross_subject", "cross_session"]
    de_path: Path | None
    psd_path: Path | None
    eye_path: Path | None
    label_path: Path
    n_subjects: int
    n_classes: int
    n_samples_per_subject: int
    session_boundaries: tuple[tuple[int, int], ...]
    eeg_single_feature_dim: int
    eye_feature_dim: int
    seed: int
    k1: int | Literal["dynamic"]
    k2: int
    fusion_steps: int
    hidden_dim: int
    embedding_dim: int
    learning_rate: float
    weight_decay: float
    epochs: int
    temperature: float
    feature_mask_probs: tuple[float, float]
    edge_drop_probs: tuple[float, float]
    contrastive_weight: float

    def __post_init__(self) -> None:
        choices = {
            "dataset": (self.dataset, {"seed_iv", "seed_v"}),
            "protocol": (
                self.protocol,
                {"cross_subject", "cross_session"},
            ),
        }
        for name, (value, allowed) in choices.items():
            if value not in allowed:
                raise ValueError(f"{name} must be one of {sorted(allowed)}")

        expected = {"seed_iv": (15, 4), "seed_v": (16, 5)}[self.dataset]
        if (self.n_subjects, self.n_classes) != expected:
            raise ValueError(
                f"{self.dataset} requires n_subjects={expected[0]} and "
                f"n_classes={expected[1]}"
            )
        positive_ints = {
            "n_samples_per_subject": self.n_samples_per_subject,
            "eeg_single_feature_dim": self.eeg_single_feature_dim,
            "eye_feature_dim": self.eye_feature_dim,
            "k2": self.k2,
            "fusion_steps": self.fusion_steps,
            "hidden_dim": self.hidden_dim,
            "embedding_dim": self.embedding_dim,
            "epochs": self.epochs,
        }
        for name, value in positive_ints.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if self.k1 != "dynamic" and (
            isinstance(self.k1, bool)
            or not isinstance(self.k1, int)
            or self.k1 <= 0
        ):
            raise ValueError("k1 must be 'dynamic' or a positive integer")

        for name, value in {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "temperature": self.temperature,
            "contrastive_weight": self.contrastive_weight,
        }.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if self.temperature <= 0 or self.contrastive_weight < 0:
            raise ValueError("temperature must be positive and contrastive_weight non-negative")
        for name, values in (
            ("feature_mask_probs", self.feature_mask_probs),
            ("edge_drop_probs", self.edge_drop_probs),
        ):
            if not isinstance(values, tuple) or len(values) != 2 or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 <= value < 1
                for value in values
            ):
                raise ValueError(f"{name} must contain two probabilities in [0, 1)")

        if not self.session_boundaries:
            raise ValueError("session_boundaries must not be empty")
        expected_start = 0
        for boundary in self.session_boundaries:
            if (
                not isinstance(boundary, tuple)
                or len(boundary) != 2
                or any(isinstance(value, bool) or not isinstance(value, int) for value in boundary)
                or boundary[0] != expected_start
                or boundary[1] <= boundary[0]
            ):
                raise ValueError("session_boundaries must be contiguous positive [start, end] pairs")
            expected_start = boundary[1]
        if expected_start != self.n_samples_per_subject:
            raise ValueError("session_boundaries must cover n_samples_per_subject")

        paths = {
            "de_path": self.de_path,
            "psd_path": self.psd_path,
            "eye_path": self.eye_path,
            "label_path": self.label_path,
        }
        if self.label_path is None:
            raise ValueError("label_path is required")
        for name, path in paths.items():
            if path is not None and (
                path == Path(".")
                or path.is_absolute()
                or PureWindowsPath(str(path)).is_absolute()
                or ".." in path.parts
            ):
                raise ValueError(f"{name} must be a release-relative path")
        if self.de_path is None or self.eye_path is None:
            raise ValueError("de_path and eye_path are required for the main model")
        if self.dataset == "seed_iv" and self.psd_path is None:
            raise ValueError("psd_path is required for the SEED-IV main model")
        if self.dataset == "seed_v" and self.psd_path is not None:
            raise ValueError("SEED-V PSD arrays are unavailable; psd_path must be null")

    @property
    def input_dim(self) -> int:
        eeg_views = 2 if self.dataset == "seed_iv" else 1
        return eeg_views * self.eeg_single_feature_dim + self.eye_feature_dim


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate one release-local YAML configuration."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw: Any = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError(f"configuration must be a YAML mapping: {config_path}")

    values = dict(raw)
    for name in ("de_path", "psd_path", "eye_path", "label_path"):
        value = values.get(name)
        if value is not None:
            if not isinstance(value, str):
                raise ValueError(f"{name} must be a path string or null")
            values[name] = Path(value)
    for name in ("session_boundaries", "feature_mask_probs", "edge_drop_probs"):
        value = values.get(name)
        if isinstance(value, list):
            values[name] = tuple(
                tuple(item) if isinstance(item, list) else item for item in value
            )
    try:
        config = ExperimentConfig(**values)
    except TypeError as exc:
        raise ValueError(f"invalid configuration structure in {config_path}: {exc}") from exc
    return config
