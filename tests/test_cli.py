from __future__ import annotations

from dataclasses import replace
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from mda_gcl import __version__, cli
from mda_gcl.config import load_config
from mda_gcl.data import FeatureBundle
from mda_gcl.protocols import Fold


RELEASE_ROOT = Path(__file__).parents[1]
CONFIGS = RELEASE_ROOT / "configs"


def _config(name: str = "seed_iv_cross_subject.yaml") -> Path:
    return CONFIGS / name


def _without_timing(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _without_timing(item)
            for key, item in value.items()
            if key not in {"training_seconds", "evaluation_seconds", "result_files"}
        }
    if isinstance(value, list):
        return [_without_timing(item) for item in value]
    return value


def _module_command(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(RELEASE_ROOT / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "mda_gcl.cli", *arguments],
        cwd=RELEASE_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_module_help_exposes_only_operational_options() -> None:
    assert __version__ == "3.0.0"
    completed = _module_command("--help")

    assert completed.returncode == 0
    for option in (
        "--config", "--data-root", "--output-dir", "--device",
        "--dry-run", "--preflight-only", "--max-folds",
    ):
        assert option in completed.stdout
    for removed in ("--mode", "--modality", "--eeg-feature", "--ssnf", "--gcl", "--dann"):
        assert removed not in completed.stdout
    assert set(inspect.signature(cli.run_experiment).parameters) == {
        "config_path", "data_root", "output_dir", "device", "dry_run",
        "preflight_only", "max_folds",
    }


@pytest.mark.parametrize(
    "name",
    [
        "seed_iv_cross_subject.yaml",
        "seed_iv_cross_session.yaml",
        "seed_v_cross_subject.yaml",
        "seed_v_cross_session.yaml",
    ],
)
def test_four_main_dry_runs_write_only_json_and_csv(tmp_path: Path, name: str) -> None:
    output = tmp_path / name.removesuffix(".yaml")
    result = cli.run_experiment(_config(name), dry_run=True, output_dir=output)

    assert result["dry_run"] is True
    assert result["evidence_status"] == "synthetic-smoke"
    assert result["selected_fold_count"] == 1
    assert result["checkpoint_rule"] == "final-epoch"
    assert result["domain_class_count"] == 2
    assert result["grl_cap"] == 0.02
    assert result["gradient_clip_max_norm"] == 0.1
    assert result["domain_discriminator_dropout"] == 0.5
    assert result["normalization"]["target_emotion_labels"] == (
        "loaded-and-validated-as-evaluation-input-but-not-indexed-for-"
        "training-or-checkpoint-selection"
    )
    assert result["fold_results"][0]["graph_provenance"]["normalization"] == (
        "source-target-independent-per-view"
    )
    assert {path.name for path in output.iterdir()} == {"results.json", "fold_results.csv"}
    assert json.loads((output / "results.json").read_text(encoding="utf-8"))["fold_results"] == result["fold_results"]
    assert result["fold_ids"][0] in (output / "fold_results.csv").read_text(encoding="utf-8")
    assert "attribution" not in json.dumps(result).lower()
    assert "plot" not in json.dumps(result).lower()


def test_dry_run_is_deterministic_except_timings_and_paths(tmp_path: Path) -> None:
    first = cli.run_experiment(_config(), dry_run=True, output_dir=tmp_path / "first")
    second = cli.run_experiment(_config(), dry_run=True, output_dir=tmp_path / "second")
    assert _without_timing(first) == _without_timing(second)


@pytest.mark.parametrize("option", ["--mode", "--modality", "--eeg-feature", "--no-ssnf", "--no-gcl", "--no-dann"])
def test_removed_cli_options_are_unrecognized(option: str, tmp_path: Path) -> None:
    completed = _module_command(
        "--config", str(_config()), "--dry-run", "--output-dir", str(tmp_path), option
    )
    assert completed.returncode == 2
    assert "unrecognized arguments" in completed.stderr


def test_removed_yaml_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "old.yaml"
    path.write_text(_config().read_text(encoding="utf-8") + "\nmode: alternate\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected keyword argument 'mode'"):
        load_config(path)


@pytest.mark.parametrize("protocol", ["cross_subject", "cross_session"])
def test_fold_inputs_use_label_free_independent_partition_statistics(protocol: str) -> None:
    config = replace(
        load_config(_config(f"seed_iv_{protocol}.yaml")),
        n_samples_per_subject=6,
        session_boundaries=((0, 2), (2, 4), (4, 6)),
        eeg_single_feature_dim=2,
        eye_feature_dim=1,
        k2=2,
    )
    rows = config.n_subjects * 6
    source = np.array([2, 3, 4, 5]) if protocol == "cross_session" else np.array([6, 7, 8, 9])
    target = np.array([0, 1])
    views = tuple(np.zeros((rows, dimension), dtype=np.float32) for dimension in (2, 2, 1))
    for index, view in enumerate(views):
        view[source] = np.arange(source.size)[:, None] + 10.0 * index
        view[target] = np.arange(target.size)[:, None] + 100.0 * (index + 1)
    labels = np.arange(rows, dtype=np.int64) % 4
    fold = Fold("tiny", protocol, source, target, 0, 0 if protocol == "cross_session" else None)
    bundle = FeatureBundle(np.concatenate(views, axis=1), labels, views, ("de", "psd", "eye"))

    features, adjacency, provenance = cli._fold_inputs(config, bundle, fold)
    changed = FeatureBundle(bundle.features, (labels + 1) % 4, views, bundle.view_names)
    changed_features, changed_adjacency, _ = cli._fold_inputs(config, changed, fold)

    np.testing.assert_allclose(features[source].mean(axis=0), 0.0, atol=1e-7)
    np.testing.assert_allclose(features[target].mean(axis=0), 0.0, atol=1e-7)
    np.testing.assert_allclose(features, changed_features)
    np.testing.assert_allclose(adjacency, changed_adjacency)
    assert provenance["topology"] == "intra-eeg-ssnf+inter-view-ssnf-balanced"


def test_view_contract_and_hierarchical_topology(monkeypatch: pytest.MonkeyPatch) -> None:
    config = replace(
        load_config(_config()),
        n_samples_per_subject=2,
        session_boundaries=((0, 1), (1, 2)),
        eeg_single_feature_dim=2,
        eye_feature_dim=1,
        k2=2,
    )
    rows = config.n_subjects * 2
    rng = np.random.default_rng(3)
    views = tuple(rng.normal(size=(rows, dimension)).astype(np.float32) for dimension in (2, 2, 1))
    bundle = FeatureBundle(np.concatenate(views, axis=1), np.arange(rows) % 4, views, ("de", "psd", "eye"))
    fold = Fold("tiny", "cross_subject", np.arange(2, rows), np.arange(2), 0)
    calls: list[tuple[tuple[np.ndarray, ...], dict[str, object]]] = []

    def recording_ssnf(*affinities: np.ndarray, **kwargs: object) -> np.ndarray:
        calls.append((affinities, kwargs))
        return np.full_like(affinities[0], 0.25 * len(calls))

    monkeypatch.setattr(cli, "ssnf", recording_ssnf)
    _, adjacency, provenance = cli._fold_inputs(config, bundle, fold)

    assert len(calls) == 2
    assert calls[1][1]["beta"] == 0.5
    np.testing.assert_allclose(adjacency, 0.5)
    assert provenance["topology"] == "intra-eeg-ssnf+inter-view-ssnf-balanced"
    bad = FeatureBundle(bundle.features[:, :3], bundle.labels, views[:2], ("de", "eye"))
    with pytest.raises(ValueError, match="requires feature views"):
        cli._fold_inputs(config, bad, fold)


def test_real_preflight_missing_data_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as error:
        cli.run_experiment(
            _config(), data_root=tmp_path / "missing", output_dir=tmp_path / "out", preflight_only=True
        )
    assert "missing emotion labels array" in str(error.value)
    assert "datasets and preprocessing are not included" in str(error.value)


@pytest.mark.parametrize("value", [0, -1, True])
def test_max_folds_must_be_positive(tmp_path: Path, value: object) -> None:
    with pytest.raises(ValueError, match="max_folds must be a positive integer"):
        cli.run_experiment(_config(), dry_run=True, output_dir=tmp_path, max_folds=value)  # type: ignore[arg-type]


def test_main_returns_clear_error_code(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    code = cli.main([
        "--config", str(_config()), "--preflight-only", "--data-root",
        str(tmp_path / "missing"), "--output-dir", str(tmp_path / "out"),
    ])
    assert code == 2
    assert "missing emotion labels array" in capsys.readouterr().err
