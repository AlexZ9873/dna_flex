# scripts/

This folder contains the active analysis scripts for the DNA flex pretraining project.

Scripts are organized by workflow stage:

```text
scripts/
├── data_prep/
├── pretrain/
├── finetune/
├── benchmark/
├── plot/
└── archive/
```

---

## data_prep/

| Script | Purpose |
|---|---|
| `compute_flex_norm_stats.py` | Compute normalization statistics for flexibility targets. |
| `make_windows_from_yaml.py` | Generate genome windows from config settings. |
| `split_windows_from_yaml.py` | Split genome windows into train/validation sets. |
| `download_hg38_ucsc.sh` | Helper script for downloading hg38. |

---

## pretrain/

| Script | Purpose |
|---|---|
| `pretrain_hg38_tiny_trainval.py` | Main hg38 pretraining script using flexibility regression and masked 6-mer prediction. |
| `pretrain_bendability_stage1.py` | Continued pretraining with bendability regression, optionally with flexibility loss. |
| `pretrain_bendability_stage2_mlm.py` | Continued pretraining with flex+MLM or bend+flex+MLM on bendability sequences. |

---

## finetune/

| Script | Purpose |
|---|---|
| `baseline_pbm_ridge_1mer.py` | Positional 1-mer Ridge baseline for PBM. |
| `finetune_pbm_hidden_poslinear.py` | Downstream PBM readout using pretrained hidden states. |
| `finetune_pbm_hidden_plus_flex_poslinear.py` | Downstream PBM readout using hidden states plus predicted flexibility features. |

---

## benchmark/

| Script | Purpose |
|---|---|
| `benchmark_htselex_option2_panelA.py` | HT-SELEX benchmark comparing pretrained representation against 1-mer, 2-mer, and 3-mer baselines. |
| `add_htselex_1mer12flex.py` | Adds custom 1-mer + 12-flex baseline to HT-SELEX benchmark outputs. |
| `benchmark_gcpbm_four_checkpoints.py` | Compares four bendability-stage checkpoints on gcPBM. |
| `panelC_gcpbm_sample_efficiency.py` | gcPBM sample-efficiency / Panel C benchmark across training percentages. |

---

## plot/

| Script | Purpose |
|---|---|
| `plot_gcpbm_4model_boxplot.py` | Box plot comparing bend only, bend+flex, flex+MLM, and bend+flex+MLM gcPBM performance. |
| `replot_panelC_gcpbm_no_errorbar.py` | Replots Panel C without error bars. |
| `replot_htselex_2x2_clean.py` | Replots clean 2x2 HT-SELEX benchmark figures. |

---

## archive/

Legacy, validation, exploratory, and earlier-version scripts are preserved here for reference but are not part of the main reproducible workflow.

Typical contents:

```text
archive/validation/
archive/legacy_pbm/
archive/legacy_pretrain/
archive/legacy_data_prep/
archive/legacy_htselex/
archive/legacy_plots/
archive/_trash_candidates/
```

Do not delete archived scripts until the current workflow has been stable for a while.

---

## Recommended execution pattern

Always run scripts from the repository root:

```bash
cd /path/to/dna-flex-pretrain
source .venv/bin/activate
export PYTHONPATH="$(pwd)"
```

Example:

```bash
PYTHONPATH="$(pwd)" python scripts/benchmark/panelC_gcpbm_sample_efficiency.py \
  --pcts 0.3 1 3 10 30 100 \
  --seeds 0 1 2 \
  --out_csv plots/panelC_gcpbm_sample_efficiency.csv \
  --out_png plots/panelC_gcpbm_sample_efficiency.png
```
