"""Fold-local training and deterministic clean-graph evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
import torch
from torch.nn import functional as F

from .config import ExperimentConfig
from .graph import augment_graph, normalize_adjacency
from .metrics import accuracy, normalized_confusion, weighted_f1
from .model import MDAGCL
from .protocols import Fold, build_domain_labels


@dataclass(frozen=True, slots=True)
class FoldResult:
    """Serializable fold provenance and exact final evaluation outputs.

    ``loss_trajectory`` stores, per configured epoch, total, supervised,
    contrastive, and domain losses in that order.
    """

    fold_id: str
    checkpoint_rule: str
    checkpoint_epoch: int
    evaluation_graph: str
    optimizer_learning_rate: float
    optimizer_weight_decay: float
    grl_schedule: str
    domain_class_count: int
    grl_cap: float
    gradient_clip_max_norm: float
    domain_discriminator_dropout: float
    loss_trajectory: tuple[tuple[float, float, float, float], ...]
    accuracy: float
    weighted_f1: float
    normalized_confusion: tuple[tuple[float, ...], ...]
    predictions: tuple[int, ...]
    targets: tuple[int, ...]
    training_seconds: float
    evaluation_seconds: float


def _float_tensor(
    values: np.ndarray | torch.Tensor,
    *,
    name: str,
    device: torch.device,
) -> torch.Tensor:
    try:
        tensor = (
            values.detach().to(device=device, dtype=torch.float32)
            if torch.is_tensor(values)
            else torch.as_tensor(values, dtype=torch.float32, device=device)
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{name} must be a numeric tensor or array") from exc
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be a 2-D matrix")
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} must contain only finite values")
    return tensor


def _raw_labels(values: np.ndarray | torch.Tensor, *, expected_rows: int) -> torch.Tensor:
    try:
        labels = (
            values.detach().to(device="cpu")
            if torch.is_tensor(values)
            else torch.as_tensor(values)
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("emotion labels must be a numeric tensor or array") from exc
    if labels.ndim != 1 or labels.shape[0] != expected_rows:
        raise ValueError(
            f"emotion labels must have shape [{expected_rows}]; received {tuple(labels.shape)}"
        )
    return labels


def _class_labels(
    values: torch.Tensor,
    *,
    n_classes: int,
    name: str,
    device: torch.device,
) -> torch.Tensor:
    if values.is_floating_point():
        if not torch.isfinite(values).all().item() or not torch.equal(
            values, values.round()
        ):
            raise ValueError(f"{name} must contain exact integer class IDs")
    labels = values.to(dtype=torch.long, device=device)
    if labels.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    if torch.any(labels < 0).item() or torch.any(labels >= n_classes).item():
        raise ValueError(f"{name} class IDs must be in [0, {n_classes - 1}]")
    return labels


def _validate_fold_rows(config: ExperimentConfig, fold: Fold, n_rows: int) -> np.ndarray:
    if fold.protocol != config.protocol:
        raise ValueError(
            f"fold protocol {fold.protocol!r} does not match config protocol {config.protocol!r}"
        )
    expected_rows = config.n_subjects * config.n_samples_per_subject
    if n_rows != expected_rows:
        raise ValueError(
            f"global features must have {expected_rows} rows in dataset order; received {n_rows}"
        )
    source = np.asarray(fold.source_indices)
    target = np.asarray(fold.target_indices)
    if (
        source.ndim != 1
        or target.ndim != 1
        or source.size == 0
        or target.size == 0
        or not np.issubdtype(source.dtype, np.integer)
        or not np.issubdtype(target.dtype, np.integer)
    ):
        raise ValueError("fold indices must be non-empty one-dimensional integer arrays")
    ordered = np.concatenate((source.astype(np.int64), target.astype(np.int64)))
    if (ordered < 0).any() or (ordered >= n_rows).any():
        raise ValueError("fold indices are outside the global feature rows")
    if np.unique(ordered).size != ordered.size:
        raise ValueError("source and target fold indices must be unique and disjoint")
    return ordered


def _domain_tensor(
    config: ExperimentConfig,
    fold: Fold,
    domain_labels: np.ndarray | torch.Tensor | None,
    *,
    device: torch.device,
) -> torch.Tensor:
    if domain_labels is None:
        raise ValueError(
            "DANN requires domain_labels in fold.indices source-then-target order"
        )
    expected = build_domain_labels(config, fold)
    try:
        provided = (
            domain_labels.detach().to(device="cpu").numpy()
            if torch.is_tensor(domain_labels)
            else np.asarray(domain_labels)
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("domain_labels must be a one-dimensional integer array") from exc
    if provided.ndim != 1 or provided.shape != expected.shape:
        raise ValueError(
            "domain_labels must match fold.indices length and source-then-target order"
        )
    if not np.issubdtype(provided.dtype, np.integer):
        raise ValueError("domain_labels must contain exact integer domain IDs")
    provided = provided.astype(np.int64, copy=False)
    if not np.array_equal(provided, expected):
        raise ValueError(
            "domain_labels do not match the configured fold.indices domain semantics"
        )
    return torch.as_tensor(provided, dtype=torch.long, device=device)


def _grl_scale(epoch: int, epochs: int, cap: float) -> float:
    progress = float(epoch) / float(epochs)
    return (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0) * cap


def _model_for_config(
    config: ExperimentConfig,
    *,
    device: torch.device,
) -> MDAGCL:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        model = MDAGCL(
            config.input_dim,
            config.hidden_dim,
            config.embedding_dim,
            config.n_classes,
            temperature=config.temperature,
        )
    return model.to(device)


def evaluate_fold(
    model: MDAGCL,
    clean_features: np.ndarray | torch.Tensor,
    clean_adjacency: np.ndarray | torch.Tensor,
    target_labels: np.ndarray | torch.Tensor,
    target_positions: np.ndarray | torch.Tensor,
) -> tuple[np.ndarray, float, float, np.ndarray]:
    """Evaluate one clean normalized fold graph with augmentations disabled.

    Target positions are fold-local positions. Target labels may contain either
    one value per target position or one value per clean graph node.
    """

    if not isinstance(model, MDAGCL):
        raise ValueError("model must be an MDAGCL instance")
    try:
        device = next(model.parameters()).device
    except StopIteration as exc:  # pragma: no cover - MDAGCL always has parameters
        raise ValueError("model has no parameters") from exc
    features = _float_tensor(clean_features, name="clean_features", device=device)
    adjacency = _float_tensor(clean_adjacency, name="clean_adjacency", device=device)
    node_count = features.shape[0]
    if adjacency.shape != (node_count, node_count):
        raise ValueError("clean_adjacency must be square and match clean_features rows")
    if torch.any(adjacency < 0).item() or not torch.allclose(adjacency, adjacency.T):
        raise ValueError("clean_adjacency must be nonnegative and symmetric")
    try:
        positions = (
            target_positions.detach().to(device="cpu", dtype=torch.long)
            if torch.is_tensor(target_positions)
            else torch.as_tensor(target_positions, dtype=torch.long)
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("target_positions must be a one-dimensional integer array") from exc
    if positions.ndim != 1 or positions.numel() == 0:
        raise ValueError("target_positions must be a non-empty one-dimensional array")
    if torch.any(positions < 0).item() or torch.any(positions >= node_count).item():
        raise ValueError("target_positions are outside the clean fold graph")
    if torch.unique(positions).numel() != positions.numel():
        raise ValueError("target_positions must be unique")

    raw_targets = _raw_labels(
        target_labels,
        expected_rows=(
            node_count
            if torch.as_tensor(target_labels).ndim == 1
            and torch.as_tensor(target_labels).shape[0] == node_count
            else positions.numel()
        ),
    )
    if raw_targets.shape[0] == node_count:
        raw_targets = raw_targets.index_select(0, positions)
    targets = _class_labels(
        raw_targets,
        n_classes=model.emotion_classifier.out_features,
        name="target labels",
        device=torch.device("cpu"),
    )
    positions_device = positions.to(device=device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        clean_embedding = model(features, adjacency)
        clean_pair = torch.cat((clean_embedding, clean_embedding), dim=1)
        logits = model.classify(clean_pair).index_select(0, positions_device)
        predictions = logits.argmax(dim=1).to(device="cpu").numpy().astype(np.int64)
    model.train(was_training)
    target_array = targets.numpy().astype(np.int64, copy=False)
    return (
        predictions,
        accuracy(target_array, predictions),
        weighted_f1(target_array, predictions),
        normalized_confusion(
            target_array,
            predictions,
            n_classes=model.emotion_classifier.out_features,
        ),
    )


def train_fold(
    config: ExperimentConfig,
    fold: Fold,
    features: np.ndarray | torch.Tensor,
    adjacency: np.ndarray | torch.Tensor,
    emotion_labels: np.ndarray | torch.Tensor,
    *,
    domain_labels: np.ndarray | torch.Tensor | None = None,
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[MDAGCL, FoldResult]:
    """Train one fold from global features and a fold-local raw adjacency.

    ``features`` and ``emotion_labels`` use global dataset row order. The
    adjacency uses ``fold.indices`` source-then-target order. Target emotion
    labels are read only after the fixed final epoch has been committed.
    """

    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    global_features = _float_tensor(features, name="features", device=resolved_device)
    ordered_indices = _validate_fold_rows(config, fold, global_features.shape[0])
    if global_features.shape[1] != config.input_dim:
        raise ValueError(
            f"features must have config.input_dim={config.input_dim} columns; "
            f"received {global_features.shape[1]}"
        )
    raw_labels = _raw_labels(emotion_labels, expected_rows=global_features.shape[0])
    source_global = torch.tensor(fold.source_indices, dtype=torch.long)
    source_labels = _class_labels(
        raw_labels.index_select(0, source_global),
        n_classes=config.n_classes,
        name="source emotion labels",
        device=resolved_device,
    )
    local_indices = torch.as_tensor(ordered_indices, dtype=torch.long, device=resolved_device)
    local_features = global_features.index_select(0, local_indices)
    raw_adjacency = _float_tensor(adjacency, name="adjacency", device=resolved_device)
    node_count = ordered_indices.size
    if raw_adjacency.shape != (node_count, node_count):
        raise ValueError(
            "adjacency must be fold-local with rows/columns in fold.indices source-then-target order"
        )
    if torch.any(raw_adjacency < 0).item() or not torch.allclose(
        raw_adjacency, raw_adjacency.T
    ):
        raise ValueError("adjacency must be nonnegative and symmetric")
    clean_adjacency = normalize_adjacency(raw_adjacency)
    assert isinstance(clean_adjacency, torch.Tensor)
    domains = _domain_tensor(config, fold, domain_labels, device=resolved_device)
    grl_cap = 0.02
    gradient_clip_max_norm = 0.1
    domain_discriminator_dropout = 0.5
    model = _model_for_config(config, device=resolved_device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    if generator is None:
        generator = torch.Generator(device=resolved_device).manual_seed(config.seed)
    elif not isinstance(generator, torch.Generator):
        raise ValueError("generator must be a torch.Generator")
    elif torch.device(generator.device) != resolved_device:
        raise ValueError("generator device must match the training device")

    source_positions = torch.arange(
        fold.source_indices.size, dtype=torch.long, device=resolved_device
    )
    target_positions = np.arange(
        fold.source_indices.size, node_count, dtype=np.int64
    )
    fork_devices: list[int] = []
    if resolved_device.type == "cuda":
        fork_devices = [
            resolved_device.index
            if resolved_device.index is not None
            else torch.cuda.current_device()
        ]
    trajectory: list[tuple[float, float, float, float]] = []
    training_started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad()
        grl_scale = _grl_scale(epoch, config.epochs, grl_cap)
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(config.seed + epoch)
            if resolved_device.type == "cuda":
                torch.cuda.manual_seed(config.seed + epoch)
            first_features, first_adjacency = augment_graph(
                local_features,
                raw_adjacency,
                config.feature_mask_probs[0],
                config.edge_drop_probs[0],
                generator=generator,
            )
            second_features, second_adjacency = augment_graph(
                local_features,
                raw_adjacency,
                config.feature_mask_probs[1],
                config.edge_drop_probs[1],
                generator=generator,
            )
            first_normalized = normalize_adjacency(first_adjacency)
            second_normalized = normalize_adjacency(second_adjacency)
            assert isinstance(first_normalized, torch.Tensor)
            assert isinstance(second_normalized, torch.Tensor)
            first_embedding = model(first_features, first_normalized)
            second_embedding = model(second_features, second_normalized)
            paired_embedding = torch.cat((first_embedding, second_embedding), dim=1)
            source_logits = model.classify(paired_embedding).index_select(
                0, source_positions
            )
            supervised_loss = F.cross_entropy(
                source_logits, source_labels, label_smoothing=0.1
            )
            contrastive_loss = model.contrastive_loss(first_embedding, second_embedding)
            domain_loss = F.cross_entropy(
                model.domain_logits(paired_embedding, grl_scale), domains
            )
            total_loss = (
                supervised_loss
                + config.contrastive_weight * contrastive_loss
                + domain_loss
            )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=gradient_clip_max_norm
            )
            optimizer.step()
        trajectory.append(
            (
                float(total_loss.detach().cpu()),
                float(supervised_loss.detach().cpu()),
                float(contrastive_loss.detach().cpu()),
                float(domain_loss.detach().cpu()),
            )
        )
    training_seconds = time.perf_counter() - training_started

    final_targets = raw_labels.index_select(
        0, torch.tensor(fold.target_indices, dtype=torch.long)
    )
    evaluation_started = time.perf_counter()
    predictions, fold_accuracy, fold_f1, confusion = evaluate_fold(
        model,
        local_features,
        clean_adjacency,
        final_targets,
        target_positions,
    )
    evaluation_seconds = time.perf_counter() - evaluation_started
    target_values = _class_labels(
        final_targets,
        n_classes=config.n_classes,
        name="target emotion labels",
        device=torch.device("cpu"),
    ).numpy()
    result = FoldResult(
        fold_id=fold.fold_id,
        checkpoint_rule="fixed-final",
        checkpoint_epoch=config.epochs,
        evaluation_graph="clean-normalized",
        optimizer_learning_rate=float(config.learning_rate),
        optimizer_weight_decay=float(config.weight_decay),
        grl_schedule="sigmoid-max-0.02-single-grl-scaling",
        domain_class_count=2,
        grl_cap=grl_cap,
        gradient_clip_max_norm=gradient_clip_max_norm,
        domain_discriminator_dropout=domain_discriminator_dropout,
        loss_trajectory=tuple(trajectory),
        accuracy=fold_accuracy,
        weighted_f1=fold_f1,
        normalized_confusion=tuple(
            tuple(float(value) for value in row) for row in confusion
        ),
        predictions=tuple(int(value) for value in predictions),
        targets=tuple(int(value) for value in target_values),
        training_seconds=training_seconds,
        evaluation_seconds=evaluation_seconds,
    )
    return model, result
