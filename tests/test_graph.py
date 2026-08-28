from __future__ import annotations

import numpy as np
import pytest
import torch

from mda_gcl.graph import (
    augment_graph,
    gaussian_affinity,
    knn_transition,
    normalize_adjacency,
    ssnf,
)


def sample_features() -> np.ndarray:
    return np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [2.0, 1.0]],
        dtype=np.float64,
    )


def test_gaussian_affinity_matches_global_scale_kernel() -> None:
    features = sample_features()

    affinity = gaussian_affinity(features, scale=2.0)

    distances = ((features[:, None] - features[None, :]) ** 2).sum(axis=2)
    np.testing.assert_allclose(affinity, np.exp(-distances / 2.0))
    np.testing.assert_array_equal(np.diag(affinity), np.ones(4))
    np.testing.assert_allclose(affinity, affinity.T)
    assert np.isfinite(affinity).all()


def test_gaussian_default_scale_is_global_and_finite() -> None:
    affinity = gaussian_affinity(sample_features())

    assert affinity.shape == (4, 4)
    assert np.isfinite(affinity).all()
    np.testing.assert_allclose(affinity, affinity.T)


@pytest.mark.parametrize(
    ("features", "scale", "message"),
    [
        (np.ones(3), None, "shape"),
        (np.array([[0.0], [np.nan]]), None, "finite"),
        (np.ones((2, 2)), None, "degenerate"),
        (np.array([[0.0], [1.0]]), 0.0, "scale"),
    ],
)
def test_gaussian_affinity_rejects_invalid_inputs(
    features: np.ndarray, scale: float | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        gaussian_affinity(features, scale=scale)


def test_knn_transition_is_row_stochastic_and_explicitly_directed() -> None:
    affinity = np.array(
        [[1.0, 0.9, 0.1], [0.9, 1.0, 0.8], [0.1, 0.8, 1.0]]
    )

    transition = knn_transition(affinity, 2)

    np.testing.assert_allclose(transition.sum(axis=1), 1.0)
    assert np.count_nonzero(transition, axis=1).tolist() == [2, 2, 2]
    assert np.all(np.diag(transition) > 0.0)
    assert not np.allclose(transition, transition.T)


def test_knn_k_clamps_to_nodes_and_can_exclude_diagonal() -> None:
    affinity = gaussian_affinity(sample_features(), scale=4.0)

    clamped = knn_transition(affinity, 100)
    without_self = knn_transition(affinity, 2, include_self=False)

    np.testing.assert_allclose(clamped, affinity / affinity.sum(axis=1, keepdims=True))
    np.testing.assert_allclose(without_self.sum(axis=1), 1.0)
    np.testing.assert_array_equal(np.diag(without_self), np.zeros(4))
    assert np.count_nonzero(without_self, axis=1).tolist() == [2, 2, 2, 2]


@pytest.mark.parametrize(
    ("affinity", "k", "message"),
    [
        (np.ones((2, 3)), 1, "square"),
        (np.array([[1.0, -0.1], [-0.1, 1.0]]), 1, "nonnegative"),
        (np.eye(2), 0, "positive integer"),
        (np.zeros((2, 2)), 1, "zero mass"),
    ],
)
def test_knn_transition_rejects_invalid_inputs(
    affinity: np.ndarray, k: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        knn_transition(affinity, k)


def test_ssnf_is_symmetric_finite_and_honors_two_view_beta() -> None:
    first = gaussian_affinity(sample_features(), scale=3.0)
    second = gaussian_affinity(sample_features()[:, ::-1] * [1.0, 2.0], scale=3.0)

    first_only = ssnf(first, second, k1=4, k2=2, beta=1.0)
    second_only = ssnf(first, second, k1=4, k2=2, beta=0.0)
    mixed = ssnf(first, second, k1=4, k2=2, beta=0.25)

    assert mixed.shape == first.shape
    assert np.isfinite(mixed).all()
    assert np.all(mixed >= 0.0)
    np.testing.assert_allclose(mixed, mixed.T)
    np.testing.assert_allclose(mixed, 0.25 * first_only + 0.75 * second_only)


def test_ssnf_t_one_default_and_dynamic_k_clamping() -> None:
    first = gaussian_affinity(sample_features(), scale=2.0)
    second = gaussian_affinity(sample_features() * [2.0, 0.5], scale=2.0)

    default = ssnf(first, second, k1=100, k2=100)
    explicit = ssnf(first, second, k1=4, k2=4, t=1)

    np.testing.assert_allclose(default, explicit)


def test_ssnf_rejects_invalid_view_contracts() -> None:
    affinity = gaussian_affinity(sample_features(), scale=2.0)

    with pytest.raises(ValueError, match="at least two"):
        ssnf(affinity, k1=4, k2=2)
    with pytest.raises(ValueError, match="same shape"):
        ssnf(affinity, np.eye(3), k1=3, k2=2)
    with pytest.raises(ValueError, match="only for two"):
        ssnf(affinity, affinity, affinity, k1=4, k2=2, beta=0.5)
    with pytest.raises(ValueError, match="positive integer"):
        ssnf(affinity, affinity, k1=4, k2=2, t=0)


def test_normalize_adjacency_is_symmetric_and_isolated_node_safe() -> None:
    adjacency = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [0.0, 2.0, 0.0]])

    no_loops = normalize_adjacency(adjacency, add_self_loops=False)
    with_loops = normalize_adjacency(adjacency)

    np.testing.assert_allclose(no_loops, [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    np.testing.assert_allclose(no_loops, no_loops.T)
    np.testing.assert_allclose(with_loops, with_loops.T)
    assert with_loops[0, 0] == pytest.approx(1.0)
    assert np.isfinite(no_loops).all() and np.isfinite(with_loops).all()


def test_normalize_adjacency_supports_torch_dtype_and_degree_rule() -> None:
    adjacency = torch.tensor([[0.0, 2.0], [2.0, 0.0]], dtype=torch.float32)

    normalized = normalize_adjacency(adjacency, add_self_loops=False)

    assert isinstance(normalized, torch.Tensor)
    assert normalized.dtype == adjacency.dtype
    torch.testing.assert_close(normalized, torch.tensor([[0.0, 1.0], [1.0, 0.0]]))


def test_seeded_augmentation_is_deterministic_without_global_rng_mutation() -> None:
    features = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    adjacency = torch.ones((6, 6), dtype=torch.float32)
    global_state = torch.random.get_rng_state().clone()
    first_generator = torch.Generator().manual_seed(19)
    second_generator = torch.Generator().manual_seed(19)

    first = augment_graph(features, adjacency, 0.4, 0.5, generator=first_generator)
    second = augment_graph(features, adjacency, 0.4, 0.5, generator=second_generator)

    torch.testing.assert_close(first[0], second[0])
    torch.testing.assert_close(first[1], second[1])
    torch.testing.assert_close(torch.random.get_rng_state(), global_state)
    assert first[0].shape == features.shape and first[0].dtype == features.dtype
    assert first[1].shape == adjacency.shape and first[1].dtype == adjacency.dtype
    assert first[0].device == features.device and first[1].device == adjacency.device


def test_augmentation_keeps_undirected_mask_and_diagonal() -> None:
    features = torch.ones((6, 3), dtype=torch.float64)
    adjacency = torch.ones((6, 6), dtype=torch.float64)

    masked_features, masked_adjacency = augment_graph(
        features,
        adjacency,
        1.0,
        0.5,
        generator=torch.Generator().manual_seed(3),
    )

    torch.testing.assert_close(masked_features, torch.zeros_like(features))
    torch.testing.assert_close(masked_adjacency, masked_adjacency.T)
    torch.testing.assert_close(masked_adjacency.diag(), adjacency.diag())
    assert torch.count_nonzero(masked_adjacency - torch.eye(6)).item() < 30


def test_zero_probability_augmentation_is_exact_bypass() -> None:
    features = torch.randn((4, 3), generator=torch.Generator().manual_seed(1))
    adjacency = torch.eye(4)

    augmented_features, augmented_adjacency = augment_graph(
        features, adjacency, 0.0, 0.0, generator=torch.Generator().manual_seed(2)
    )

    assert augmented_features is features
    assert augmented_adjacency is adjacency
    torch.testing.assert_close(augmented_features, features)
    torch.testing.assert_close(augmented_adjacency, adjacency)


@pytest.mark.parametrize(
    ("features", "adjacency", "feature_prob", "edge_prob", "message"),
    [
        (torch.ones(3), torch.eye(3), 0.1, 0.1, "2-D"),
        (torch.ones((3, 2)), torch.eye(2), 0.1, 0.1, "node count"),
        (torch.ones((2, 2)), torch.tensor([[1.0, 1.0], [0.0, 1.0]]), 0.1, 0.1, "symmetric"),
        (torch.ones((2, 2)), torch.eye(2), -0.1, 0.1, "probability"),
        (torch.ones((2, 2)), torch.eye(2), 0.1, 1.1, "probability"),
    ],
)
def test_augmentation_rejects_invalid_inputs(
    features: torch.Tensor,
    adjacency: torch.Tensor,
    feature_prob: float,
    edge_prob: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        augment_graph(features, adjacency, feature_prob, edge_prob)
