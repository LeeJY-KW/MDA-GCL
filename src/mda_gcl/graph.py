"""Paper-aligned SSNF graph construction and stochastic graph views."""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
import torch


def _features_array(features: np.ndarray) -> np.ndarray:
    try:
        array = np.asarray(features, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("features must be a numeric 2-D array") from exc
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 1:
        raise ValueError("features must have shape [at least 2 nodes, at least 1 feature]")
    if not np.isfinite(array).all():
        raise ValueError("features must contain only finite values")
    return array


def _affinity_array(affinity: np.ndarray, *, name: str = "affinity") -> np.ndarray:
    try:
        array = np.asarray(affinity, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric square matrix") from exc
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be a non-empty square matrix")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    if np.any(array < 0):
        raise ValueError(f"{name} must be nonnegative")
    return array


def _positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _probability(value: float, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"{name} must be a finite probability in [0, 1]")
    return float(value)


def gaussian_affinity(features: np.ndarray, scale: float | None = None) -> np.ndarray:
    """Return ``exp(-||x_i-x_j||^2 / scale)`` using one global scale.

    When ``scale`` is omitted, the legacy global heuristic is retained: five
    percent of the median positive off-diagonal squared distance. The diagonal
    is exactly one and no KNN sparsification is performed here.
    """
    array = _features_array(features)
    with np.errstate(over="ignore", invalid="ignore"):
        squared_norm = np.einsum("ij,ij->i", array, array)
        distances = squared_norm[:, None] + squared_norm[None, :] - 2.0 * (array @ array.T)
    if not np.isfinite(distances).all():
        raise ValueError("pairwise squared distances are not finite")
    np.maximum(distances, 0.0, out=distances)
    distances = (distances + distances.T) * 0.5
    positive = distances[np.triu_indices(array.shape[0], k=1)]
    positive = positive[positive > 0.0]
    if positive.size == 0:
        raise ValueError("features are degenerate: all rows are identical")
    if scale is None:
        resolved_scale = 0.05 * float(np.median(positive))
    elif (isinstance(scale, bool) or not isinstance(scale, Real)
          or not np.isfinite(scale) or scale <= 0.0):
        raise ValueError("scale must be a finite positive number")
    else:
        resolved_scale = float(scale)
    if not np.isfinite(resolved_scale) or resolved_scale <= 0.0:
        raise ValueError("global scale is degenerate for these features")

    affinity = np.exp(-distances / resolved_scale)
    affinity = (affinity + affinity.T) * 0.5
    np.fill_diagonal(affinity, 1.0)
    if not np.isfinite(affinity).all():
        raise ValueError("Gaussian affinity contains non-finite values")
    return affinity


def knn_transition(
    affinity: np.ndarray, k: int, *, include_self: bool = True
) -> np.ndarray:
    """Build a directed row-stochastic top-k transition matrix.

    ``k`` is clamped to the available node count and counts the mandatory
    diagonal entry when ``include_self`` is true. The result is intentionally
    not symmetrized; SSNF symmetrizes after cross-diffusion.
    """
    array = _affinity_array(affinity)
    requested_k = _positive_int(k, name="k")
    if not isinstance(include_self, bool):
        raise ValueError("include_self must be a boolean")
    node_count = array.shape[0]
    if not include_self and node_count == 1:
        raise ValueError("include_self=false requires at least two nodes")
    effective_k = min(requested_k, node_count if include_self else node_count - 1)
    candidates = array.copy()
    np.fill_diagonal(candidates, -np.inf)
    neighbor_count = effective_k - 1 if include_self else effective_k
    if neighbor_count:
        neighbors = np.argsort(-candidates, axis=1, kind="stable")[:, :neighbor_count]
    else:
        neighbors = np.empty((node_count, 0), dtype=np.int64)
    rows = np.arange(node_count)[:, None]
    transition = np.zeros_like(array)
    transition[rows, neighbors] = array[rows, neighbors]
    if include_self:
        diagonal = np.arange(node_count)
        transition[diagonal, diagonal] = array[diagonal, diagonal]
    row_sums = transition.sum(axis=1)
    if np.any(row_sums <= 0.0):
        bad_rows = np.flatnonzero(row_sums <= 0.0).tolist()
        raise ValueError(f"selected KNN neighborhood has zero mass in rows {bad_rows}")
    transition /= row_sums[:, None]
    return transition


def ssnf(
    *affinities: np.ndarray,
    k1: int,
    k2: int,
    t: int = 1,
    beta: float | None = None,
) -> np.ndarray:
    """Fuse affinity views with broad/local symmetric cross-diffusion."""

    if len(affinities) < 2:
        raise ValueError("SSNF requires at least two affinity views")
    views = [
        _affinity_array(affinity, name=f"affinity view {index}")
        for index, affinity in enumerate(affinities)
    ]
    shape = views[0].shape
    if any(view.shape != shape for view in views[1:]):
        raise ValueError("all SSNF affinity views must have the same shape")
    steps = _positive_int(t, name="t")
    if beta is not None:
        if len(views) != 2:
            raise ValueError("beta weighting is defined only for two SSNF views")
        beta = _probability(beta, name="beta")

    broad = [knn_transition(view, k1) for view in views]
    local = [knn_transition(view, k2) for view in views]
    current = broad
    for _ in range(steps):
        updated = []
        for index, gate in enumerate(local):
            other = sum(
                current[other_index]
                for other_index in range(len(current))
                if other_index != index
            ) / (len(current) - 1)
            diffused = gate @ other @ gate.T
            updated.append((diffused + diffused.T) * 0.5)
        current = updated

    if beta is None:
        fused = sum(current) / len(current)
    else:
        fused = beta * current[0] + (1.0 - beta) * current[1]
    fused = (fused + fused.T) * 0.5
    np.maximum(fused, 0.0, out=fused)
    if not np.isfinite(fused).all():
        raise ValueError("SSNF produced a non-finite fused adjacency")
    return fused


def normalize_adjacency(
    adjacency: np.ndarray | torch.Tensor,
    *,
    add_self_loops: bool = True,
) -> np.ndarray | torch.Tensor:
    """Apply safe symmetric ``D^-1/2 A D^-1/2`` normalization."""

    if not isinstance(add_self_loops, bool):
        raise ValueError("add_self_loops must be a boolean")
    if torch.is_tensor(adjacency):
        if adjacency.ndim != 2 or adjacency.shape[0] == 0 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("adjacency must be a non-empty square matrix")
        if not adjacency.is_floating_point():
            raise ValueError("torch adjacency must have a floating-point dtype")
        if not torch.isfinite(adjacency).all().item():
            raise ValueError("adjacency must contain only finite values")
        if torch.any(adjacency < 0).item():
            raise ValueError("adjacency must be nonnegative")
        symmetric = (adjacency + adjacency.T) * 0.5
        if add_self_loops:
            symmetric = symmetric + torch.eye(
                symmetric.shape[0], device=symmetric.device, dtype=symmetric.dtype
            )
        degree = symmetric.sum(dim=1)
        inverse_sqrt = torch.zeros_like(degree)
        positive = degree > 0
        inverse_sqrt[positive] = degree[positive].rsqrt()
        normalized = inverse_sqrt[:, None] * symmetric * inverse_sqrt[None, :]
        return (normalized + normalized.T) * 0.5

    array = _affinity_array(adjacency, name="adjacency")
    symmetric = (array + array.T) * 0.5
    if add_self_loops:
        symmetric = symmetric + np.eye(symmetric.shape[0], dtype=symmetric.dtype)
    degree = symmetric.sum(axis=1)
    inverse_sqrt = np.zeros_like(degree)
    positive = degree > 0.0
    inverse_sqrt[positive] = degree[positive] ** -0.5
    normalized = inverse_sqrt[:, None] * symmetric * inverse_sqrt[None, :]
    return (normalized + normalized.T) * 0.5


def augment_graph(
    features: torch.Tensor,
    adjacency: torch.Tensor,
    feature_mask_prob: float,
    edge_drop_prob: float,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask feature entries and undirected off-diagonal edges."""

    if not torch.is_tensor(features) or not torch.is_tensor(adjacency):
        raise ValueError("features and adjacency must be torch tensors")
    if features.ndim != 2 or not features.is_floating_point():
        raise ValueError("features must be a floating-point 2-D torch tensor")
    if adjacency.ndim != 2 or adjacency.shape != (features.shape[0], features.shape[0]):
        raise ValueError("adjacency must be square and match the feature node count")
    if not adjacency.is_floating_point():
        raise ValueError("adjacency must have a floating-point dtype")
    if features.device != adjacency.device:
        raise ValueError("features and adjacency must use the same device")
    if not torch.isfinite(features).all().item() or not torch.isfinite(adjacency).all().item():
        raise ValueError("features and adjacency must contain only finite values")
    if torch.any(adjacency < 0).item():
        raise ValueError("adjacency must be nonnegative")
    if not torch.equal(adjacency, adjacency.T):
        raise ValueError("adjacency must be symmetric for undirected edge dropping")
    if generator is not None:
        if not isinstance(generator, torch.Generator):
            raise ValueError("generator must be a torch.Generator")
        if torch.device(generator.device) != features.device:
            raise ValueError("generator device must match the graph device")
    feature_prob = _probability(feature_mask_prob, name="feature_mask_prob")
    edge_prob = _probability(edge_drop_prob, name="edge_drop_prob")

    augmented_features = features
    if feature_prob > 0.0:
        feature_keep = torch.rand(
            features.shape, device=features.device, generator=generator
        ) >= feature_prob
        augmented_features = features * feature_keep

    augmented_adjacency = adjacency
    if edge_prob > 0.0:
        node_count = adjacency.shape[0]
        upper = torch.triu_indices(node_count, node_count, offset=1, device=adjacency.device)
        upper_keep = torch.rand(
            upper.shape[1], device=adjacency.device, generator=generator
        ) >= edge_prob
        edge_keep = torch.eye(node_count, device=adjacency.device, dtype=torch.bool)
        edge_keep[upper[0], upper[1]] = upper_keep
        edge_keep[upper[1], upper[0]] = upper_keep
        augmented_adjacency = adjacency * edge_keep
    return augmented_features, augmented_adjacency
