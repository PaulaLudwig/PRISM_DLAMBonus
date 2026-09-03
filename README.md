# PRISM: Two-Stage Patch-TCN + Cross-Variate Attention

Deep-learning project for **multivariate long-horizon time-series forecasting**. PRISM separates forecasting into two stages:

1. **Stage 1 – temporal representation learning:** a channel-independent Patch-TCN encodes each variable independently.
2. **Stage 2 – cross-variate modeling:** an iTransformer-style Transformer encoder models dependencies between variables.

The project additionally studies the information bottleneck created by the bridge between these stages through three bridge variants: **mean pooling**, **attention pooling**, and a **patch-preserving bridge**.

The architecture is evaluated on the course operations dataset and on **ETTh1** as an additional generalization dataset.

---

## Project Structure

```text
PRISM_DLAMBonus/
├── data/
│   ├── train.csv
│   ├── validation_input.csv
│   ├── forecast_index_validation.csv
│   ├── metadata.json
│   └── etth1/
│       ├── raw/
│       │   └── ETTh1.csv
│       └── prepared/
│           ├── train.csv
│           └── validation_input.csv
├── scripts/
│   ├── download_data.py
│   └── pre_dataset2.py
├── src/
│   ├── data.py
│   ├── metrics.py
│   ├── model.py
│   └── prism.py
├── checkpoints/
├── baseline/
├── submission_template/
├── train.py
├── eval_holdout.py
├── eval_baselines.py
├── predict_validation.py
├── requirements.txt
└── README.md
```

---

## Environment Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

The minimal requirements are:

```text
numpy
pandas
torch
huggingface_hub
```

The training code can run on CPU, CUDA, or Apple Silicon via MPS. To explicitly select Apple MPS, for example:

```bash
--device mps
```

---

## Data

### Course dataset

The course dataset is stored in `data/`:

```text
data/
├── train.csv
├── validation_input.csv
├── forecast_index_validation.csv
└── metadata.json
```

If the course data is not present, it can be downloaded with:

```bash
python scripts/download_data.py
```

The forecasting setup uses:

- **168 hours** of history
- **336 hours** forecast horizon
- known future covariates where available

### ETTh1

ETTh1 is used as a second dataset to evaluate whether the architecture and bridge behavior generalize beyond the course dataset. The forecasting target is the oil-temperature variable `OT`.

The raw ETTh1 file is already included in the repository at:

```text
data/etth1/raw/ETTh1.csv
```

Prepare the dataset once with:

```bash
python scripts/prepare_etth1.py
```

This creates:

```text
data/etth1/prepared/train.csv
data/etth1/prepared/validation_input.csv
```

---

## Model

### Stage 1: Patch-TCN

Each variable is processed independently by a shared temporal encoder.

Main settings:

- history: 168 hours
- forecast horizon: 336 hours
- patch length: 24 hours
- target history: 7 patches
- channel-independent Patch-TCN
- dilated causal TCN with dilations 1, 2, and 4
- ReZero residual scaling for training stability

For the course dataset, known-future covariates span the full history + forecast interval and therefore contain 21 patches.

### Stage 2: Cross-Variate Attention

Stage 2 uses a Transformer encoder to model dependencies between variables. Following the iTransformer idea, attention is applied across variable representations rather than ordinary time-step tokens.

### Bridge variants

The `--pooling` argument controls how Stage 1 representations are passed to Stage 2:

- `mean` – mean-pools all patch representations into one token per variable.
- `attention` – learns attention weights over patches, but still compresses each variable to one token.
- `patches` – preserves all patch-level representations and lets Stage 2 attend over variable × patch tokens.

The patch-preserving bridge additionally uses variable embeddings, patch-position embeddings, and a learned forecast token.

### Architecture ablations

The `--mode` argument controls which stages are active:

- `full` – Stage 1 followed by Stage 2.
- `stage1` – Patch-TCN representation without cross-variate Transformer attention.
- `stage2` – cross-variate Transformer without the Patch-TCN encoder.

`--pooling patches` is supported only with `--mode full`.

---

## Training Configuration

Unless otherwise stated, experiments use:

| Hyperparameter | Value |
|---|---:|
| History | 168 h |
| Horizon | 336 h |
| Patch length | 24 h |
| Hidden dimension | 128 |
| Transformer layers | 2 |
| Attention heads | 8 |
| Dropout | 0.2 |
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Gradient clipping | 1.0 |
| Scheduler | Cosine annealing |
| Batch size | 128 |
| Window stride | 6 h |
| Early-stopping patience | 6 |
| Seed | 42 |

---

## Course-Dataset Experiments

### Full PRISM with mean pooling

```bash
python train.py \
  --model prism \
  --mode full \
  --pooling mean \
  --epochs 30 \
  --stride 6 \
  --batch-size 128
```

### Full PRISM with attention pooling

```bash
python train.py \
  --model prism \
  --mode full \
  --pooling attention \
  --epochs 30 \
  --stride 6 \
  --batch-size 128
```

### Full PRISM with patch-preserving bridge

```bash
python train.py \
  --model prism \
  --mode full \
  --pooling patches \
  --epochs 30 \
  --stride 6 \
  --batch-size 128
```

### Stage 1 only

```bash
python train.py \
  --model prism \
  --mode stage1 \
  --epochs 30 \
  --stride 6 \
  --batch-size 128
```

### Stage 2 only

```bash
python train.py \
  --model prism \
  --mode stage2 \
  --epochs 30 \
  --stride 6 \
  --batch-size 128
```

---

## ETTh1 Experiments

First prepare ETTh1:

```bash
python scripts/prepare_etth1.py
```

Use `data/etth1/prepared` as the data directory. Separate checkpoint directories are used below to avoid overwriting models from other experiments.

### Mean pooling

```bash
python train.py \
  --data-dir data/etth1/prepared \
  --output-dir checkpoints/etth1_mean \
  --model prism \
  --mode full \
  --pooling mean \
  --epochs 30 \
  --stride 6 \
  --batch-size 128
```

### Attention pooling

```bash
python train.py \
  --data-dir data/etth1/prepared \
  --output-dir checkpoints/etth1_attention \
  --model prism \
  --mode full \
  --pooling attention \
  --epochs 30 \
  --stride 6 \
  --batch-size 128
```

### Patch-preserving bridge

```bash
python train.py \
  --data-dir data/etth1/prepared \
  --output-dir checkpoints/etth1_patches \
  --model prism \
  --mode full \
  --pooling patches \
  --epochs 30 \
  --stride 6 \
  --batch-size 128
```

### Stage 1 only

```bash
python train.py \
  --data-dir data/etth1/prepared \
  --output-dir checkpoints/etth1_stage1 \
  --model prism \
  --mode stage1 \
  --epochs 30 \
  --stride 6 \
  --batch-size 128
```

### Stage 2 only

```bash
python train.py \
  --data-dir data/etth1/prepared \
  --output-dir checkpoints/etth1_stage2 \
  --model prism \
  --mode stage2 \
  --epochs 30 \
  --stride 6 \
  --batch-size 128
```

---

## Evaluation

The local holdout is the final 336 labelled hours excluded from training.

Evaluate a course-dataset checkpoint with:

```bash
python eval_holdout.py \
  --checkpoint checkpoints/prism_full_patches.pt \
  --scaler checkpoints/scaler.npz \
  --device mps
```

The evaluation reports:

- WAPE
- MAE
- MSE
- RMSE
- MAPE
- sMAPE

Lower values are better for all metrics. **WAPE is the primary model-selection metric.**

---

## Results

### Course dataset

| Model / bridge | WAPE ↓ |
|---|---:|
| Full PRISM – mean pooling | 0.2429 |
| Full PRISM – attention pooling | 0.2436 |
| Stage 2 only | 0.2322* |
| **Full PRISM – patch preserving** | **0.2076** |

`*` The Stage-2-only score is a previously documented project result. Its original checkpoint was not available in the current checkpoint directory when the bridge experiments were reproduced.

The independently re-evaluated patch-preserving checkpoint gives:

| Metric | Value |
|---|---:|
| WAPE | **0.2076** |
| MAE | 2.2170 |
| MSE | 11.1609 |
| RMSE | 3.3408 |
| MAPE | 0.2620 |
| sMAPE | 0.2336 |

Compared with mean pooling, patch preservation reduces WAPE from **0.2429 to 0.2076**, corresponding to an approximately **14.5% relative reduction**.

Attention pooling performs similarly to mean pooling. This suggests that, on the course dataset, the main limitation is not uniform averaging specifically, but the loss of temporal structure caused by compressing the complete Stage-1 patch sequence into a single token before cross-variate attention.

### ETTh1

| Model / bridge | WAPE ↓ | MAE | RMSE | MAPE | sMAPE |
|---|---:|---:|---:|---:|---:|
| **Full PRISM – mean pooling** | **0.0943** | 0.9328 | 1.3295 | 0.1101 | 0.0982 |
| Full PRISM – attention pooling | 0.0977 | 0.9664 | 1.4476 | 0.1183 | 0.1014 |
| Stage 2 only | 0.0980 | 0.9690 | 1.3594 | 0.1124 | 0.1019 |
| Full PRISM – patch preserving | 0.1059 | 1.0468 | 1.4267 | 0.1201 | 0.1102 |
| Stage 1 only | 0.1124 | 1.1117 | 1.4729 | 0.1214 | 0.1171 |

ETTh1 does **not** reproduce the same ranking as the course dataset. Mean pooling performs best, while preserving all patch representations performs worse.

This indicates that the preferred bridge is **dataset-dependent**. On the course dataset, retaining fine-grained temporal structure appears to remove an information bottleneck. On ETTh1, stronger temporal compression instead performs better and may provide a useful inductive bias or regularization effect.

---

## Training on All Labelled Course Data

After selecting an architecture using the local holdout, `--full` trains on all labelled course-dataset hours.

For the patch-preserving PRISM model:

```bash
python train.py \
  --model prism \
  --mode full \
  --pooling patches \
  --epochs 30 \
  --stride 6 \
  --batch-size 128 \
  --full
```

This produces a full-data checkpoint and scaler with filenames containing `_full`.

---

## Generating Validation Predictions

Example:

```bash
python predict_validation.py \
  --checkpoint checkpoints/prism_full_patches_full.pt \
  --scaler checkpoints/scaler_full.npz \
  --output-file predictions/prism_full_patches.csv \
  --device mps
```

The expected output format is:

```csv
series_id,timestamp,prediction
```

---

## Reproducibility Notes

- Python, NumPy, and PyTorch seeds are set through `--seed` (default: `42`).
- Apple MPS operations are not guaranteed to be bit-reproducible, so small run-to-run differences may occur even with fixed seeds.
- Scaler statistics are fitted on the training region only.
- Training windows cannot access held-out target labels.
- Missing-value handling is causal.
- Model selection is based on holdout WAPE with early stopping.
- Architecture comparisons should use the same split, stride, batch size, optimizer settings, and maximum epoch budget.

---

## Main Experimental Finding

The experiments support a **dataset-dependent view of temporal aggregation** between the two PRISM stages.

On the course dataset, preserving Stage-1 patch representations substantially improves performance, providing evidence that collapsing each variable to a single token can create an information bottleneck. On ETTh1, however, the original mean-pooled bridge achieves the best result.

The experiments therefore do not support a universal preference for patch preservation. Instead, they show that the appropriate degree of temporal compression between temporal representation learning and cross-variate modeling depends on the structure of the forecasting dataset.
