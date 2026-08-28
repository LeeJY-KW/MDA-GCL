# Reproducibility workflow

All commands below run from the release root. The release never changes the working directory and writes only to the requested output directory.

## 1. Environment

```bash
python -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m mda_gcl.cli --help
```

Real training requires a CUDA-capable PyTorch environment. Synthetic validation runs on CPU.

## 2. Supply and validate arrays

Place external arrays under the layout in [DATA.md](DATA.md), then preflight each intended cell:

```bash
python -m mda_gcl.cli --config configs/seed_iv_cross_subject.yaml --data-root /path/to/data --output-dir /tmp/seed-iv-subject-preflight --device cpu --preflight-only
python -m mda_gcl.cli --config configs/seed_iv_cross_session.yaml --data-root /path/to/data --output-dir /tmp/seed-iv-session-preflight --device cpu --preflight-only
python -m mda_gcl.cli --config configs/seed_v_cross_subject.yaml --data-root /path/to/data --output-dir /tmp/seed-v-subject-preflight --device cpu --preflight-only
python -m mda_gcl.cli --config configs/seed_v_cross_session.yaml --data-root /path/to/data --output-dir /tmp/seed-v-session-preflight --device cpu --preflight-only
```

## 3. Run the four fixed cells

```bash
python -m mda_gcl.cli --config configs/seed_iv_cross_subject.yaml --data-root /path/to/data --output-dir /path/to/seed-iv-subject --device cuda:0
python -m mda_gcl.cli --config configs/seed_iv_cross_session.yaml --data-root /path/to/data --output-dir /path/to/seed-iv-session --device cuda:0
python -m mda_gcl.cli --config configs/seed_v_cross_subject.yaml --data-root /path/to/data --output-dir /path/to/seed-v-subject --device cuda:0
python -m mda_gcl.cli --config configs/seed_v_cross_session.yaml --data-root /path/to/data --output-dir /path/to/seed-v-session --device cuda:0
```

Use `--max-folds N` only to bound the prefix of the protocol fold sequence for resource checks. It does not change the selected folds or model.

## 4. Fixed execution contract

For every fold, the runner:

1. Takes source and target indices from the protocol.
2. Independently standardizes each source and target feature view.
3. Builds fold-local affinities in source-then-target order and applies hierarchical SSNF.
4. Trains paired stochastic projected graph views with GCL and binary source/target DANN.
5. Selects the configured final epoch without indexing target emotion-label values.
6. Evaluates the final state once using a classifier input formed by duplicating the clean normalized-graph projected embedding; no stochastic augmentation is used at evaluation.
7. Uses target emotion-label values only to compute final metrics.

## 5. Runtime artifacts

Every completed output directory contains only `results.json` and `fold_results.csv`. The JSON includes the effective configuration, fixed checkpoint rule, normalization provenance, exact fold outputs, and aggregate mean/std metrics. The CSV contains one row per fold with exact predictions and targets.

## 6. Optional local check

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -p no:cacheprovider -W error
```
