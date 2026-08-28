# MDA-GCL fixed-model release

This directory is the GitHub-facing implementation of one fixed MDA-GCL model for four dataset/protocol configurations. It contains the model, training pipeline, YAML configurations, tests, and usage documentation. Feature arrays are external inputs. Datasets, preprocessing code, checkpoints, generated artifacts, and experimental records are not bundled.

## Fixed model

Every run uses the same contract:

- SEED-IV views are DE, PSD, and eye features; SEED-V views are DE and eye features.
- Each view is standardized independently for the source and target partitions. Target feature statistics are transductive, but target emotion-label values are not used for training or checkpoint selection.
- The graph follows the paper hierarchy: DE and PSD are fused first when both exist, then the EEG graph is fused with eye features by SSNF.
- Two stochastic projected graph views drive the paired GCL objective.
- A binary DANN head distinguishes source from target samples.
- The configured final epoch is evaluated once using a classifier input formed by duplicating the clean normalized-graph projected embedding; evaluation uses no stochastic augmentation.

The command surface selects only the configuration, external data root, output directory, device, preflight or synthetic execution, and an optional bounded fold count.

## Install and inspect

From this directory:

```bash
python -m pip install -e .
python -m mda_gcl.cli --help
```

CPU synthetic runs are deterministic; real training requires a CUDA-capable PyTorch installation.

## Four dataset/protocol configurations

```bash
python -m mda_gcl.cli --config configs/seed_iv_cross_subject.yaml --data-root /path/to/data --output-dir /path/to/output --device cuda:0
python -m mda_gcl.cli --config configs/seed_iv_cross_session.yaml --data-root /path/to/data --output-dir /path/to/output --device cuda:0
python -m mda_gcl.cli --config configs/seed_v_cross_subject.yaml --data-root /path/to/data --output-dir /path/to/output --device cuda:0
python -m mda_gcl.cli --config configs/seed_v_cross_session.yaml --data-root /path/to/data --output-dir /path/to/output --device cuda:0
```

Use `--preflight-only` to validate arrays and fold compatibility without training. Use `--dry-run --device cpu` for a tiny in-memory pipeline smoke test that never reads dataset arrays. `--max-folds N` bounds an otherwise unchanged fold sequence.

Each completed run writes exactly:

- `results.json`: effective configuration, fixed-model provenance, fold outputs, and aggregate metrics.
- `fold_results.csv`: one row per fold with exact metrics, predictions, targets, and provenance.

Existing files are overwritten only inside the explicitly requested output directory. The process never changes the working directory.
