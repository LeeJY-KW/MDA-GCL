# External data contract

The release does not bundle raw recordings, processed arrays, or preprocessing code. Supply NumPy arrays under a data root while keeping the configuration-relative layout below.

## Required files

```text
<data-root>/data/seed_iv/SEED_IV_4S_DE.npy
<data-root>/data/seed_iv/SEED_IV_4S_PSD.npy
<data-root>/data/seed_iv/SEED_IV_4S_EYE.npy
<data-root>/data/seed_iv/SEED_IV_4S_LABEL.npy

<data-root>/data/seed_v/subject_de.npy
<data-root>/data/seed_v/subject_eye.npy
<data-root>/data/seed_v/subject_label.npy
```

The loader rejects object arrays, non-numeric features, non-integral labels, unsupported label encodings, inconsistent row counts, wrong feature widths, and nonfinite values.

| Dataset | Subjects | Samples per subject | DE | PSD | Eye |
|---|---:|---:|---:|---:|---:|
| SEED-IV | 15 | 2,505 | 310 | 310 | 31 |
| SEED-V | 16 | 1,823 | 310 | not used | 33 |

DE and PSD are each ordered as five frequency bands by 62 EEG channels. The eye block follows the dataset width without reordering. Labels are integer emotion classes: SEED-IV uses 0–3 and SEED-V uses 0–4.

## Fold use and normalization

Protocol folds preserve the configured row order. Every fold arranges graph rows source first and target second. Each feature view is standardized separately in the source and target partitions; the source statistics are never applied to target rows, and target statistics are computed from target features only. This is transductive feature use.

The complete label array is loaded and validated as evaluation input before training. Target emotion-label values are not indexed for graph construction, normalization, optimization, or checkpoint selection. They are indexed only after the configured final model state for metric computation.

Cross-subject runs construct one held-out-subject fold per subject. Cross-session runs construct one fold per subject/session pair.

Run preflight before allocating training resources:

```bash
python -m mda_gcl.cli --config configs/seed_v_cross_session.yaml --data-root /path/to/data --output-dir /tmp/mda-gcl-preflight --device cpu --preflight-only
```
