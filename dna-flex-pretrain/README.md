# DNA Flex Pretrain

Transformer-based DNA representation learning for transcription factor binding prediction.

This project tests whether a pretrained DNA encoder can learn a reusable representation that combines:

1. **sequence context**, learned through masked 6-mer prediction, and
2. **DNA mechanical / flexibility information**, learned through lookup-table-derived flexibility targets.

The goal is to build a pretrained encoder that can transfer to TF-binding tasks where labeled data are limited, including gcPBM, HT-SELEX, and future in-vivo TF-binding benchmarks.

---

## Project motivation

Transcription factor (TF) binding is not determined by motif sequence alone. Binding specificity also depends on local sequence context and DNA structural / mechanical properties such as flexibility, bendability, torsional variability, and stiffness.

Previous feature-engineering work showed that adding DNA flexibility descriptors to sequence baselines can improve TF-binding prediction. This project asks a related representation-learning question:

> Can a transformer encoder be pretrained to learn reusable DNA sequence and mechanics-aware representations, rather than hand-engineering task-specific features each time?

---

## Core model

### Input representation

DNA sequences are tokenized into overlapping **6-mers**.

For a sequence of length `L`, the number of tokens is:

```text
L - 6 + 1
```

Each 6-mer is represented as a `6 x 4` one-hot matrix and flattened into a 24-dimensional vector.

```text
DNA sequence -> overlapping 6-mers -> 24D one-hot token vectors
```

### Encoder

The encoder is a small transformer encoder.

Current architecture:

```text
d_model  = 64
n_heads  = 4
n_layers = 2
```

The encoder maps each token into a contextual hidden state:

```text
input tokens [T, 24] -> transformer encoder -> hidden states [T, 64]
```

---

## Pretraining objectives

The main pretrained model uses two self-supervised objectives.

### 1. Flexibility regression

For each 6-mer token, the model predicts a 12-dimensional flexibility vector.

The 12 features are computed from lookup tables:

- 8 dinucleotide features
- 4 trinucleotide features

For each 6-mer, the relevant internal 2-mer or 3-mer values are averaged to produce the token-level target.

### 2. Masked 6-mer prediction

A subset of 6-mer tokens is masked. The MLM head predicts the original 6-mer identity from surrounding context.

For `k = 6`:

```text
4^6 = 4096 possible DNA 6-mers
```

plus special tokens, giving a vocabulary size of about 4100.

### Total pretraining loss

```text
L_total = lambda_flex * L_flex + lambda_mlm * L_mlm
```

Default setting:

```text
lambda_flex = 1.0
lambda_mlm  = 0.1
```

---

## Downstream transfer

After pretraining, the encoder is reused for TF-binding prediction.

The downstream learning problem is:

```text
DNA sequence -> measured TF binding score
```

Current downstream datasets:

- **gcPBM**: Max, Mad, Myc
- **HT-SELEX**: large multi-family TF benchmark

---

## Downstream readouts

Several readout strategies are tested.

| Readout | Description |
|---|---|
| `flex only` | uses only the predicted flexibility vector |
| `hidden only` | uses contextual transformer hidden states |
| `hidden + flex` | concatenates hidden state and predicted flexibility at each token |

The strongest current downstream readout is generally:

```text
hidden state + predicted flexibility
```

---

## Baselines

The pretrained representations are compared against classical sequence and flexibility baselines.

| Baseline | Description |
|---|---|
| `1-mer ridge` | positional A/C/G/T one-hot features with Ridge regression |
| `1-mer + 12-flex` | positional 1-mer plus this project's 12 lookup-table flexibility features |
| `2-mer ridge` | positional dinucleotide one-hot features |
| `3-mer ridge` | positional trinucleotide one-hot features |

---

## Bendability-stage experiments

The project also tests whether a sequence-level bendability objective improves transfer.

The bendability data are 50-bp DNA sequences with one bendability score per sequence. Four continued-pretraining strategies are compared:

| Strategy | Active objectives |
|---|---|
| `bend only` | bendability regression |
| `bend + flex` | bendability regression + flexibility regression |
| `flex + MLM` | flexibility regression + masked 6-mer prediction |
| `bend + flex + MLM` | bendability regression + flexibility regression + masked 6-mer prediction |

Current result:

> Bendability supervision is learnable, but in the current setup it does not improve transfer beyond flex+MLM. The flex+MLM checkpoint gives the strongest and most consistent gcPBM transfer performance among the four bendability-stage strategies.

---

## Current findings

### gcPBM

- `hidden + flex` readout outperforms flex-only readout.
- `flex + MLM` is the strongest continued-pretraining strategy in the current gcPBM setup.
- Bendability-only pretraining performs worst among the four bendability-stage strategies.
- Adding bendability on top of flex+MLM does not improve gcPBM transfer in the current setup.
- Sample-efficiency analysis suggests that pretrained representations are useful in low-data regimes.

### HT-SELEX

- The pretrained representation generally improves over a simple 1-mer baseline.
- It is competitive with sequence / flexibility baselines.
- Strong explicit 3-mer baselines remain difficult to beat.
- Current frozen encoder features do not fully capture all explicit trinucleotide binding signal.

### Interpretation

The model learns useful sequence-context and mechanics-related representations without TF-binding labels. However, stronger pretraining, fine-tuning, hybrid models, or larger model capacity may be needed to consistently outperform strong k-mer baselines on HT-SELEX.

---

## Repository structure

```text
dna-flex-pretrain/
├── configs/                 # YAML configuration files
├── data/                    # Local data files; large external datasets are mostly ignored by git
├── docs/                    # Documentation, script inventory, data summaries
├── scripts/
│   ├── data_prep/           # Data preparation scripts
│   ├── pretrain/            # hg38 and bendability-stage pretraining
│   ├── finetune/            # PBM fine-tuning / downstream heads
│   ├── benchmark/           # HT-SELEX and gcPBM benchmark scripts
│   ├── plot/                # Final plotting / replotting scripts
│   └── archive/             # Legacy, exploratory, debug, and validation scripts
├── src/                     # Model, dataset, tokenization, and utility code
├── requirements.txt
├── setup.py
└── README.md
```

---

## Important source files

| File | Purpose |
|---|---|
| `src/model.py` | transformer encoder, MLM head, flexibility head |
| `src/tokenization.py` | overlapping k-mer tokenization and one-hot encoding |
| `src/genome_dataset.py` | hg38 pretraining dataset with MLM/flex targets |
| `src/bendability_dataset.py` | 50-bp bendability dataset loader |
| `src/pbm_dataset.py` | PBM / gcPBM dataset loader |
| `src/flex_features.py` | lookup-table-based flexibility feature computation |
| `src/collate.py` | batching / padding utilities |
| `src/utils.py` | configuration helpers |

---

## Active scripts

### Data preparation

```text
scripts/data_prep/compute_flex_norm_stats.py
scripts/data_prep/make_windows_from_yaml.py
scripts/data_prep/split_windows_from_yaml.py
scripts/data_prep/download_hg38_ucsc.sh
```

### Pretraining

```text
scripts/pretrain/pretrain_hg38_tiny_trainval.py
scripts/pretrain/pretrain_bendability_stage1.py
scripts/pretrain/pretrain_bendability_stage2_mlm.py
```

### Downstream fine-tuning

```text
scripts/finetune/baseline_pbm_ridge_1mer.py
scripts/finetune/finetune_pbm_hidden_poslinear.py
scripts/finetune/finetune_pbm_hidden_plus_flex_poslinear.py
```

### Benchmarks

```text
scripts/benchmark/benchmark_htselex_option2_panelA.py
scripts/benchmark/add_htselex_1mer12flex.py
scripts/benchmark/benchmark_gcpbm_four_checkpoints.py
scripts/benchmark/panelC_gcpbm_sample_efficiency.py
```

### Plotting

```text
scripts/plot/plot_gcpbm_4model_boxplot.py
scripts/plot/replot_panelC_gcpbm_no_errorbar.py
scripts/plot/replot_htselex_2x2_clean.py
```

---

## Environment setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
export PYTHONPATH="$(pwd)"
```

---

## Data layout

Large external datasets are expected locally and are generally not committed to git.

Expected local layout:

```text
data/raw/flex_tables/lookup.yaml

data/raw/pbm/
├── Max.txt
├── Mad.txt
└── Myc.txt

data/raw/htselex/
└── *.txt

data/raw/bendability/
├── Data1/
│   ├── train_set.txt
│   ├── vaild_set.txt
│   └── test_set.txt
├── Data2/
│   ├── train_set.txt
│   ├── vaild_set.txt
│   └── test_set.txt
├── minidata/
└── no_replicate.txt
```

The filename `vaild_set.txt` is intentionally kept because that is the spelling used in the upstream bendability dataset.

---

## Checkpoints

Model checkpoints are expected locally under:

```text
checkpoints/
```

Important checkpoint names used in current benchmark scripts include:

```text
checkpoints/hg38_256_chr1-22_200k_di8_tri4_best_by_val_flex.pt
checkpoints/bendability_stage1_data1_bendonly.pt
checkpoints/bendability_stage1_data1_flex0p2.pt
checkpoints/bendstage_flexmlm_data1.pt
checkpoints/bendstage_bendflexmlm_data1.pt
```

Checkpoints are usually not committed to git because they can be large. If sharing checkpoints publicly, use GitHub Releases, Zenodo, Google Drive, or another artifact storage mechanism.

---

## Example workflows

Always run from the repository root:

```bash
export PYTHONPATH="$(pwd)"
```

### hg38 flex+MLM pretraining

```bash
PYTHONPATH="$(pwd)" python scripts/pretrain/pretrain_hg38_tiny_trainval.py
```

### Bendability stage 1: bend only

```bash
PYTHONPATH="$(pwd)" python scripts/pretrain/pretrain_bendability_stage1.py \
  --split_dir data/raw/bendability/Data1 \
  --epochs 15 \
  --patience 4 \
  --batch_size 256 \
  --encoder_lr 1e-5 \
  --head_lr 1e-3 \
  --lambda_flex 0.0 \
  --out checkpoints/bendability_stage1_data1_bendonly.pt
```

### Bendability stage 2: bend+flex+MLM

```bash
PYTHONPATH="$(pwd)" python scripts/pretrain/pretrain_bendability_stage2_mlm.py \
  --split_dir data/raw/bendability/Data1 \
  --epochs 10 \
  --patience 3 \
  --batch_size 256 \
  --encoder_lr 1e-5 \
  --head_lr 1e-3 \
  --lambda_mlm 0.1 \
  --lambda_flex 1.0 \
  --lambda_bend 1.0 \
  --out checkpoints/bendstage_bendflexmlm_data1.pt
```

### HT-SELEX benchmark

```bash
PYTHONPATH="$(pwd)" python scripts/benchmark/benchmark_htselex_option2_panelA.py \
  --limit_files 215 \
  --outer_folds 10 \
  --inner_folds 5 \
  --seed 0 \
  --out_prefix plots/htselex_option2_all215_seed0
```

Add the custom 1-mer + 12-flex baseline:

```bash
PYTHONPATH="$(pwd)" python scripts/benchmark/add_htselex_1mer12flex.py \
  --existing_csv plots/htselex_option2_all215_seed0.csv \
  --lookup_yaml data/raw/flex_tables/lookup.yaml \
  --folder data/raw/htselex \
  --seed 0 \
  --outer_folds 10 \
  --inner_folds 5 \
  --out_prefix plots/htselex_option2_all215_seed0_plus12flex
```

### gcPBM four-checkpoint benchmark

```bash
PYTHONPATH="$(pwd)" python scripts/benchmark/benchmark_gcpbm_four_checkpoints.py
```

Plot the four-model comparison:

```bash
PYTHONPATH="$(pwd)" python scripts/plot/plot_gcpbm_4model_boxplot.py \
  --bend_only_csv plots/gcpbm_bendonly.csv \
  --bend_flex_csv plots/gcpbm_bendflex.csv \
  --flex_mlm_csv plots/gcpbm_flexmlm.csv \
  --bend_flex_mlm_csv plots/gcpbm_bendflexmlm.csv \
  --out_csv plots/gcpbm_4model_merged.csv \
  --out_png plots/gcpbm_4model_boxplot.png
```

### gcPBM sample-efficiency / Panel C

```bash
PYTHONPATH="$(pwd)" python scripts/benchmark/panelC_gcpbm_sample_efficiency.py \
  --pcts 0.3 1 3 10 30 100 \
  --seeds 0 1 2 \
  --out_csv plots/panelC_gcpbm_sample_efficiency.csv \
  --out_png plots/panelC_gcpbm_sample_efficiency.png
```

Optional no-error-bar replot:

```bash
PYTHONPATH="$(pwd)" python scripts/plot/replot_panelC_gcpbm_no_errorbar.py \
  --csv plots/panelC_gcpbm_sample_efficiency.csv \
  --out_png plots/panelC_gcpbm_sample_efficiency_noerr.png
```

---

## Current limitations

- HT-SELEX transfer is promising but does not consistently beat strong explicit 3-mer baselines.
- Bendability supervision is biologically meaningful but does not yet improve transfer beyond flex+MLM in the current setup.
- Most experiments use frozen encoder features with simple downstream readouts; full fine-tuning may improve results.
- Model size, tokenization strategy, and loss weighting have not been fully optimized.
- Formal statistical testing for all comparisons is still needed.

---

## Future directions

1. Add paired statistical testing for HT-SELEX and gcPBM.
2. Test low-data transfer more directly on HT-SELEX.
3. Test hybrid models that combine pretrained embeddings with explicit k-mer or flexibility features.
4. Tune flex / MLM / bendability loss weights.
5. Try larger transformer encoders or alternative tokenization.
6. Add biological interpretability by mapping motif and flank positions important in the pretrained representation.
7. Extend to in-vivo TF binding using ChIP-seq / DNase-seq benchmarks.
8. Explore richer mechanics targets such as shape, stiffness, bendability, and DNA breathing.

---

## References

- Vaswani et al. 2017. Attention Is All You Need.
- Devlin et al. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
- Ji et al. 2021. DNABERT: Pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome.
- Yang et al. 2017. Transcription factor family-specific DNA shape readout revealed by quantitative specificity models.
- Zhou et al. 2015. Quantitative modeling of transcription factor binding specificities using DNA shape.
- Basu et al. 2021. Measuring DNA mechanics on the genome scale.
- Jiang et al. 2023. Assessing base-resolution DNA mechanics on the genome scale.
- Dey, Yella, and Kumar. DNA conformational flexibility descriptors improve transcription factor binding prediction across diverse transcription factor families.
