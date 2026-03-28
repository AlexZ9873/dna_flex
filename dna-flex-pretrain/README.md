# DNA Flex Pretrain → PBM Fine-Tuning (Transformer Prototype)

This repo contains a **DNA transformer prototype** that is:
1) **Pretrained** on hg38 genome windows using *self-supervised DNA flexibility targets* (and optional MLM), and  
2) **Fine-tuned** to predict TF binding scores using **gcPBM** data (example: `Max.txt`).

The goal is to learn sequence representations that encode DNA mechanics/flexibility and transfer to downstream TF binding tasks with fewer labeled examples.

---

## Repository layout (what each folder is for)

### `configs/`
YAML configuration files that control key settings (k-mer size, model size, window size, features, training).
- `configs/pretrain.yaml`  
  Controls pretraining: tokenizer `k`, model architecture, window size, which flexibility features to use, and training settings.
- `configs/finetune_pbm.yaml`  
  Controls PBM fine-tuning: dataset path, pretrained checkpoint path, train/val/test split, learning rate, etc.

### `src/`
Reusable core code (the “library”):
- `src/tokenization.py`  
  Converts a DNA sequence into overlapping **k-mers** and encodes each k-mer as **6×4 one-hot** (flattened to length 24 when k=6).
- `src/flex_features.py` (or equivalent in your repo)  
  Loads di-/tri-nucleotide lookup tables and computes flex feature targets for each k-mer token.
- `src/genome_dataset.py` (or equivalent)  
  Loads genome windows, builds model inputs (`x`, `attention_mask`, `mlm_labels`) and regression targets (`flex_targets`).
- `src/pbm_dataset.py`  
  Loads PBM data (`sequence<TAB>score`) and outputs tensors for fine-tuning.
- `src/model.py`  
  Defines the transformer model:
  - encoder that processes k-mer one-hot inputs
  - MLM head (optional) to predict masked k-mers
  - flex regression head to predict flexibility features per token

### `scripts/`
(Scripts guide)
Primary scripts (use these):
Runnable scripts (the “entry points”):
**Pretraining / data prep**
- `scripts/make_windows_from_yaml.py`  
  Sample genome windows from hg38 (filters out Ns) using settings from `configs/pretrain.yaml`.
- `scripts/split_windows_from_yaml.py`  
  Split windows into train/val lists.
- `scripts/compute_flex_norm_stats.py`  
  Computes global mean/std for each flex feature across many windows and saves to `data/processed/flex_norm_stats.yaml`.
- `scripts/pretrain_hg38_tiny_trainval.py`  
  Main pretraining run: trains on genome windows and saves checkpoints.

**PBM fine-tuning / evaluation**
- `scripts/finetune_pbm_max_flex_maxpool_betterhead.py`  
  Best-performing PBM fine-tune pipeline:
  - freeze pretrained encoder
  - compute flex predictions per token (12 dims)
  - **max pool across token positions**
  - train an MLP head (with dropout + early stopping) to predict PBM score
- `scripts/eval_pbm_r2_flex_maxpool.py`  
  Evaluates PBM performance with Pearson + R² and prints **calibrated R²** (fit y ≈ a*yhat + b on val).

**Experimental/validation scripts (optional reference)**
- `'validate_*`  
  internal checks (shape checks, loss checks, ablations)
- `finetune_pbm_max_*baseline`  
  baseline comparisons.
- `finetune_pbm_max_hidden_* / *_unfreeze_*`  
  alternative fine-tuning experiments.
  
### `data/`
Local datasets (not committed to GitHub if large):
- `data/raw/`  
  Raw inputs (hg38 FASTA, window lists, PBM files, flex lookup tables)
- `data/processed/`  
  Processed artifacts like global normalization stats for flex features.

### `checkpoints/`
Saved model weights (excluded from GitHub because they can be large).
- Pretraining checkpoints (last / best)
- Fine-tuning head checkpoints

### `logs/`
Training logs (CSV or text). excluded from GitHub.

---

## Setup

### 1) Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```
### 2) PBM data is included, data/raw/pbm/Max.txt is included in this repo for reproducing PBM fine-tuning.
hg38 is not included, Download hg38 locally using:
```
./scripts/download_hg38_ucsc.sh
gunzip -k data/raw/hg38.fa.gz

```
### 3) Install dependencies
```
pip install --upgrade pip
pip install torch pyyaml numpy pandas tqdm
pip install matplotlib
```
### 4) Run PBM fine-tuning (best pipeline)
```
PYTHONPATH="$(pwd)" python scripts/finetune_pbm_max_flex_maxpool_betterhead.py
```
Expected:
1. prints epoch-by-epoch training lines
2. prints final metrics: best_val_rmse, test_rmse, test_pearson, test_r2
3. saves head checkpoint: checkpoints/pbm_max_best_head_flex_maxpool_betterhead.pt

### 5) Evaluate Pearson + R² (including calibrated R²)
```
PYTHONPATH="$(pwd)" python scripts/eval_pbm_r2_flex_maxpool.py
```
1. test_pearson (trend / ranking quality)
2. test_r2 (variance explained; sensitive to scale/offset)
3. test_r2_calibrated (fits y ≈ a*yhat + b on validation then evaluates on test)


---

## Recent updates (2026-03)

This repo recently added PBM baselines + paper-style plots to better compare the transformer features against classic sequence models.

### What changed / what was added

**1) New PBM baselines (paper-style comparisons)**
- `scripts/baseline_pbm_ridge_1mer.py`  
  Ridge regression on **positional 1-mer** features (A/C/G/T at each position).  
  This provides a strong “classic PBM baseline” and helps validate that the PBM file parsing + splits + metrics are correct.

**2) New transformer fine-tune heads (to improve R² and interpretability)**
- `scripts/finetune_pbm_hidden_poslinear.py`  
  Uses **frozen pretrained encoder hidden states** and trains a **position-aware linear head** (flatten per-position features) to predict PBM score.
- `scripts/finetune_pbm_hidden_plus_flex_poslinear.py`  
  Same as above but concatenates **hidden + flex_pred** features (still frozen encoder) before the position-aware linear head.

These “position-aware linear” heads tend to perform much better than pooled heads on PBM because PBM binding is strongly motif/position dependent.

**3) Figure/plot generation scripts (paper-style panels)**
- `scripts/plot_panelA_r2_ridge_vs_transformer.py`  
  Panel A-style scatter comparing model performance metrics.
- `scripts/plot_panelB_pred_vs_obs.py`  
  Panel B-style Predicted vs Observed scatter (test set) for ridge vs transformer head.
- `scripts/plot_panelC_sample_size_vs_r2.py`  
  Panel C-style sample size vs R² curves (mean ± std across seeds).
- `scripts/panel_c_ridge_on_transformer_features.py`  
  Utility to run Panel C using transformer features (frozen encoder) + ridge.

**4) Generated outputs saved for reproducibility**
- `plots/`  
  CSVs + PNGs saved from the plotting scripts (so results can be reproduced/checked).
- `figures/`  
  Figure PNG exports (final “paper-style” images).

**5) Repo hygiene**
- Updated `.gitignore` to exclude OS/Python cache files (e.g., `.DS_Store`, `__pycache__`, `*.pyc`, `*.bak`) to keep commits clean.

### Why this matters
These additions make it easier to:
- verify PBM evaluation with a strong classic baseline (ridge 1-mer),
- demonstrate what the pretrained transformer learned (via hidden-state linear heads),
- produce paper-style plots that directly compare against the paper’s reported PBM R² trends.

