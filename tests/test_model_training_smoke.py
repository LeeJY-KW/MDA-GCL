from __future__ import annotations

import inspect
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from mda_gcl.config import ExperimentConfig, load_config
from mda_gcl.graph import normalize_adjacency
from mda_gcl.metrics import accuracy, normalized_confusion, weighted_f1
from mda_gcl.model import GradientReversal, MDAGCL
from mda_gcl.protocols import Fold, build_domain_labels
from mda_gcl.training import evaluate_fold, train_fold


RELEASE_ROOT = Path(__file__).parents[1]


def make_config(*, protocol: str = "cross_subject", epochs: int = 1) -> ExperimentConfig:
    base = load_config(RELEASE_ROOT / "configs" / f"seed_iv_{protocol}.yaml")
    return replace(
        base,
        n_samples_per_subject=6,
        session_boundaries=((0, 2), (2, 4), (4, 6)),
        eeg_single_feature_dim=4,
        eye_feature_dim=3,
        hidden_dim=8,
        embedding_dim=4,
        epochs=epochs,
        feature_mask_probs=(0.1, 0.2),
        edge_drop_probs=(0.1, 0.2),
        contrastive_weight=0.05,
    )


def sample_fold(protocol: str) -> Fold:
    source = np.array([2, 3, 4, 5] if protocol == "cross_session" else [6, 7, 8, 9])
    return Fold(
        "tiny-fold",
        protocol,
        source,
        np.array([0, 1]),
        0,
        0 if protocol == "cross_session" else None,
    )


def sample_inputs() -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    features = torch.randn((90, 11), generator=torch.Generator().manual_seed(5))
    adjacency = torch.tensor(
        [
            [1.0, 0.4, 0.2, 0.0, 0.1, 0.3],
            [0.4, 1.0, 0.3, 0.1, 0.0, 0.2],
            [0.2, 0.3, 1.0, 0.5, 0.2, 0.0],
            [0.0, 0.1, 0.5, 1.0, 0.4, 0.2],
            [0.1, 0.0, 0.2, 0.4, 1.0, 0.5],
            [0.3, 0.2, 0.0, 0.2, 0.5, 1.0],
        ]
    )
    return features, adjacency, np.arange(90, dtype=np.int64) % 4


def train_tiny(config: ExperimentConfig) -> tuple[MDAGCL, object]:
    fold = sample_fold(config.protocol)
    features, adjacency, labels = sample_inputs()
    return train_fold(
        config,
        fold,
        features,
        adjacency,
        labels,
        domain_labels=build_domain_labels(config, fold),
        device="cpu",
        generator=torch.Generator().manual_seed(config.seed),
    )


def test_model_has_unconditional_projection_binary_domain_and_paired_heads() -> None:
    assert set(inspect.signature(MDAGCL).parameters) == {
        "input_dim", "hidden_dim", "embedding_dim", "emotion_classes", "temperature"
    }
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4)
        model = MDAGCL(5, 7, 3, 4, temperature=0.7)
    embeddings = model(
        torch.randn((6, 5), generator=torch.Generator().manual_seed(2)), torch.eye(6)
    )
    paired = torch.cat((embeddings, embeddings), dim=1)

    assert model.encoder.gcn2.weight.shape == (7, 3)
    assert (model.projection_head[0].in_features, model.projection_head[0].out_features) == (3, 6)
    assert (model.projection_head[2].in_features, model.projection_head[2].out_features) == (6, 3)
    assert model.emotion_classifier.in_features == 6
    assert model.domain_discriminator[0].in_features == 6
    assert model.domain_discriminator[3].p == 0.5
    assert model.classify(paired).shape == (6, 4)
    assert model.domain_logits(paired, 0.2).shape == (6, 2)
    with pytest.raises(ValueError, match="two concatenated"):
        model.classify(embeddings)
    variable = torch.tensor([1.0, 2.0], requires_grad=True)
    GradientReversal.apply(variable, 0.25).sum().backward()
    torch.testing.assert_close(variable.grad, torch.full_like(variable, -0.25))


@pytest.mark.parametrize("protocol", ["cross_subject", "cross_session"])
def test_main_training_is_cpu_safe_and_clean_evaluation_is_deterministic(protocol: str) -> None:
    config = make_config(protocol=protocol)
    global_rng_before = torch.random.get_rng_state().clone()
    model, result = train_tiny(config)

    torch.testing.assert_close(torch.random.get_rng_state(), global_rng_before)
    assert result.checkpoint_rule == "final-epoch"
    assert result.checkpoint_epoch == config.epochs
    assert result.evaluation_graph == "clean-normalized"
    assert result.domain_class_count == model.domain_classes == 2
    assert result.grl_cap == 0.02
    assert result.gradient_clip_max_norm == 0.1
    assert result.domain_discriminator_dropout == 0.5
    assert len(result.loss_trajectory) == config.epochs
    assert result.loss_trajectory[0][2] > 0.0
    assert result.loss_trajectory[0][3] > 0.0

    features, adjacency, labels = sample_inputs()
    fold = sample_fold(protocol)
    local_features = features[torch.as_tensor(fold.indices)]
    clean = normalize_adjacency(adjacency)
    first = evaluate_fold(model, local_features, clean, labels[fold.target_indices], np.array([4, 5]))
    second = evaluate_fold(model, local_features, clean, labels[fold.target_indices], np.array([4, 5]))
    np.testing.assert_array_equal(first[0], second[0])
    assert first[1:3] == second[1:3]
    np.testing.assert_array_equal(first[3], second[3])


def test_read_only_fold_indices_are_copied_without_warnings() -> None:
    config = make_config()
    fold = sample_fold(config.protocol)
    fold.source_indices.setflags(write=False)
    fold.target_indices.setflags(write=False)
    features, adjacency, labels = sample_inputs()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model, result = train_fold(
            config,
            fold,
            features,
            adjacency,
            labels,
            domain_labels=build_domain_labels(config, fold),
            generator=torch.Generator().manual_seed(config.seed),
        )

    assert caught == []
    assert result.checkpoint_rule == "final-epoch"
    assert len(result.predictions) == len(result.targets) == 2
    assert model.training is True
    assert fold.source_indices.flags.writeable is False
    assert fold.target_indices.flags.writeable is False


def test_adam_receives_learning_rate_and_weight_decay(monkeypatch: pytest.MonkeyPatch) -> None:
    config = replace(make_config(), learning_rate=0.0123, weight_decay=0.0456)
    captured: dict[str, float] = {}
    real_adam = torch.optim.Adam

    def recording_adam(parameters: object, **kwargs: float) -> torch.optim.Optimizer:
        captured.update(kwargs)
        return real_adam(parameters, **kwargs)

    monkeypatch.setattr(torch.optim, "Adam", recording_adam)
    _, result = train_tiny(config)

    assert captured == {"lr": 0.0123, "weight_decay": 0.0456}
    assert result.optimizer_learning_rate == 0.0123
    assert result.optimizer_weight_decay == 0.0456


@pytest.mark.parametrize("protocol", ["cross_subject", "cross_session"])
def test_target_label_perturbation_cannot_change_training_or_state(protocol: str) -> None:
    config = make_config(protocol=protocol, epochs=2)
    fold = sample_fold(protocol)
    features, adjacency, labels = sample_inputs()
    changed = labels.copy()
    changed[fold.target_indices] = (changed[fold.target_indices] + 2) % config.n_classes
    kwargs = {
        "domain_labels": build_domain_labels(config, fold),
        "generator": torch.Generator().manual_seed(config.seed),
    }
    first_model, first_result = train_fold(config, fold, features, adjacency, labels, **kwargs)
    kwargs["generator"] = torch.Generator().manual_seed(config.seed)
    second_model, second_result = train_fold(config, fold, features, adjacency, changed, **kwargs)

    assert first_result.loss_trajectory == second_result.loss_trajectory
    assert first_result.checkpoint_epoch == second_result.checkpoint_epoch == config.epochs
    assert first_result.predictions == second_result.predictions
    assert first_result.targets != second_result.targets
    for name, value in first_model.state_dict().items():
        torch.testing.assert_close(value, second_model.state_dict()[name], rtol=0.0, atol=0.0)


def test_domain_and_feature_contracts_fail_actionably() -> None:
    config = make_config()
    fold = sample_fold(config.protocol)
    features, adjacency, labels = sample_inputs()
    with pytest.raises(ValueError, match="DANN requires domain_labels"):
        train_fold(config, fold, features, adjacency, labels)
    with pytest.raises(ValueError, match="domain semantics"):
        train_fold(config, fold, features, adjacency, labels, domain_labels=np.zeros(6, dtype=np.int64))
    with pytest.raises(ValueError, match="config.input_dim"):
        train_fold(
            config,
            fold,
            features[:, :-1],
            adjacency,
            labels,
            domain_labels=build_domain_labels(config, fold),
        )


def test_exact_prediction_metrics_and_actionable_validation() -> None:
    targets = np.array([0, 0, 1, 2])
    predictions = np.array([0, 1, 1, 2])
    assert accuracy(targets, predictions) == pytest.approx(75.0)
    assert weighted_f1(targets, predictions) == pytest.approx(0.75)
    confusion = normalized_confusion(targets, predictions, n_classes=4)
    assert confusion.shape == (4, 4)
    np.testing.assert_allclose(confusion[3], 0.0)
    with pytest.raises(ValueError, match="not logits or probabilities"):
        accuracy(targets, np.eye(4))
