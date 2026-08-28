"""Single configuration-driven command line entry point for MDA-GCL."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, fields, is_dataclass, replace
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import torch

from .config import ExperimentConfig, load_config
from .data import FeatureBundle, load_feature_bundle, standardize_partitions
from .graph import gaussian_affinity, knn_transition, ssnf
from .protocols import Fold, build_domain_labels, iter_session_folds, iter_subject_folds
from .training import FoldResult, train_fold


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one performance-only MDA-GCL experiment from a release YAML config."
    )
    parser.add_argument("--config", required=True, type=Path, help="release YAML config path")
    parser.add_argument("--data-root", type=Path, default=Path("."), help="root for config data paths")
    parser.add_argument("--output-dir", type=Path, default=Path("results"), help="result directory")
    parser.add_argument("--device", default="cpu", help="torch device (default: cpu)")
    parser.add_argument("--dry-run", action="store_true", help="run one tiny deterministic CPU fold")
    parser.add_argument("--preflight-only", action="store_true", help="validate config, arrays, and folds only")
    parser.add_argument("--max-folds", type=int, help="positive upper bound on executed folds")
    return parser


def _synthetic(config: ExperimentConfig) -> tuple[ExperimentConfig, FeatureBundle]:
    samples = 6 if config.protocol == "cross_session" else 2
    boundaries = ((0, 2), (2, 4), (4, 6)) if samples == 6 else ((0, 1), (1, 2))
    config = replace(
        config,
        n_samples_per_subject=samples,
        session_boundaries=boundaries,
        eeg_single_feature_dim=4,
        eye_feature_dim=3,
        k2=2,
        fusion_steps=1,
        hidden_dim=8,
        embedding_dim=4,
        epochs=1,
    )
    rows = config.n_subjects * samples
    names = ["de", "psd", "eye"] if config.dataset == "seed_iv" else ["de", "eye"]
    dimensions = [config.eeg_single_feature_dim] * (len(names) - 1) + [config.eye_feature_dim]
    rng = np.random.default_rng(config.seed)
    views = tuple(
        rng.normal(loc=index * 0.2, size=(rows, dimension)).astype(np.float32)
        for index, dimension in enumerate(dimensions)
    )
    features = np.concatenate(views, axis=1)
    labels = np.arange(rows, dtype=np.int64) % config.n_classes
    return config, FeatureBundle(features, labels, views, tuple(names))


def _broad(affinity: np.ndarray, k1: int) -> np.ndarray:
    transition = knn_transition(affinity, k1)
    return (transition + transition.T) * 0.5


def _fold_inputs(
    config: ExperimentConfig, bundle: FeatureBundle, fold: Fold
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    expected_names = ("de", "psd", "eye") if config.dataset == "seed_iv" else ("de", "eye")
    if bundle.view_names != expected_names or len(bundle.views) != len(expected_names):
        raise ValueError(
            f"{config.dataset} requires fixed feature views {expected_names}; "
            f"received {bundle.view_names}"
        )
    standardized: dict[str, np.ndarray] = {}
    for name, view in zip(bundle.view_names, bundle.views):
        standardized[name] = standardize_partitions(
            view, fold.source_indices, fold.target_indices
        )
    features = np.concatenate([standardized[name] for name in bundle.view_names], axis=1)
    affinities = {
        name: gaussian_affinity(view[fold.indices]) for name, view in standardized.items()
    }
    node_count = fold.indices.size
    k1 = max(1, node_count // config.n_classes) if config.k1 == "dynamic" else config.k1
    fusion = {"k1": k1, "k2": config.k2, "t": config.fusion_steps}
    if config.dataset == "seed_iv":
        eeg = ssnf(affinities["de"], affinities["psd"], **fusion)
        topology = "intra-eeg-ssnf+inter-view-ssnf-balanced"
    else:
        eeg = _broad(affinities["de"], k1)
        topology = "single-eeg-broad+inter-view-ssnf-balanced"
    adjacency = ssnf(eeg, affinities["eye"], beta=0.5, **fusion)
    provenance = {
        "normalization": "source-target-independent-per-view",
        "index_order": "source-then-target",
        "topology": topology,
        "resolved_k1": k1,
        "k2": config.k2,
        "fusion_steps": config.fusion_steps,
        "source_rows": int(fold.source_indices.size),
        "target_rows": int(fold.target_indices.size),
    }
    return features.astype(np.float32, copy=False), adjacency.astype(np.float32), provenance


def _json_default(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_results(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results.json"
    csv_path = output_dir / "fold_results.csv"
    payload["result_files"] = {"json": json_path.name, "csv": csv_path.name}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fieldnames = [field.name for field in fields(FoldResult)] + ["graph_provenance"]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for result in payload["fold_results"]:
            writer.writerow({
                name: json.dumps(result[name], separators=(",", ":"))
                if isinstance(result.get(name), (list, dict)) else result.get(name)
                for name in fieldnames
            })


def run_experiment(
    config_path: str | Path,
    *,
    data_root: str | Path = ".",
    output_dir: str | Path = "results",
    device: str = "cpu",
    dry_run: bool = False,
    preflight_only: bool = False,
    max_folds: int | None = None,
) -> dict[str, Any]:
    """Validate, execute, and serialize one configured experiment."""
    if dry_run and preflight_only:
        raise ValueError("dry_run and preflight_only are mutually exclusive")
    if max_folds is not None and (
        isinstance(max_folds, bool) or not isinstance(max_folds, int) or max_folds < 1
    ):
        raise ValueError("max_folds must be a positive integer")
    resolved_device = torch.device(device)
    if dry_run and resolved_device.type != "cpu":
        raise ValueError("dry_run requires device='cpu'")
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    config = load_config(config_path)
    config, bundle = _synthetic(config) if dry_run else (
        config, load_feature_bundle(config, data_root)
    )
    fold_factory = iter_subject_folds if config.protocol == "cross_subject" else iter_session_folds
    available = list(fold_factory(config, bundle.features.shape[0]))
    if max_folds is not None and max_folds > len(available):
        raise ValueError(f"max_folds must not exceed available fold count {len(available)}")
    limit = max_folds if max_folds is not None else (1 if dry_run else len(available))
    selected = available[:limit]
    results: list[dict[str, Any]] = []
    if not preflight_only:
        for fold in selected:
            features, adjacency, provenance = _fold_inputs(config, bundle, fold)
            domains = build_domain_labels(config, fold)
            model, fold_result = train_fold(
                config, fold, features, adjacency, bundle.labels,
                domain_labels=domains, device=resolved_device,
            )
            result = asdict(fold_result)
            result["graph_provenance"] = provenance
            results.append(result)
    aggregate = {}
    for name in ("accuracy", "weighted_f1"):
        values = np.asarray([result[name] for result in results], dtype=np.float64)
        aggregate[name] = {
            "mean": float(values.mean()) if values.size else None,
            "std": float(values.std()) if values.size else None,
        }
    payload = {
        "schema_version": 1,
        "config_path": Path(config_path),
        "effective_config": config,
        "protocol": config.protocol,
        "feature_views": ["de", "psd", "eye"] if config.dataset == "seed_iv" else ["de", "eye"],
        "normalization": {
            "rule": "source-target-independent-per-view",
            "target_feature_statistics": "used-transductively",
            "target_emotion_labels": (
                "loaded-and-validated-as-evaluation-input-but-not-indexed-for-"
                "training-or-checkpoint-selection"
            ),
        },
        "checkpoint_rule": "fixed-final",
        "domain_class_count": 2,
        "grl_cap": 0.02,
        "gradient_clip_max_norm": 0.1,
        "domain_discriminator_dropout": 0.5,
        "device": str(resolved_device),
        "dry_run": dry_run,
        "preflight_only": preflight_only,
        "evidence_status": "synthetic-smoke" if dry_run else ("not-run" if preflight_only else "executed"),
        "available_fold_count": len(available),
        "selected_fold_count": len(selected),
        "fold_ids": [fold.fold_id for fold in selected],
        "fold_results": results,
        "aggregate": aggregate,
    }
    payload = json.loads(json.dumps(payload, default=_json_default))
    _write_results(payload, Path(output_dir))
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = vars(parser.parse_args(argv))
    config_path = args.pop("config")
    try:
        payload = run_experiment(config_path, **args)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        parser.print_usage(sys.stderr)
        print(f"{parser.prog}: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "evidence_status": payload["evidence_status"],
        "selected_fold_count": payload["selected_fold_count"],
        "result_files": payload["result_files"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
