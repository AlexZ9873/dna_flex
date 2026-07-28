# DNA Biophysical Auxiliary Pretraining

Systematic DNA representation learning for in vitro transcription
factor–DNA binding prediction.

## Overview

This project tests whether sequence-derived DNA biophysical supervision can
improve the transferability, labeled-data efficiency, and interpretability of
DNA foundation models. The active systematic study compares models that
receive the same DNA sequence input and differ in their pretraining objective.
TF-binding labels and TF protein information remain absent from pretraining,
and the primary downstream comparison uses DNA sequence only. This design is
intended to isolate whether biophysical auxiliary supervision changes the
quality of the learned DNA representation.

## Core research question

> Does sequence-derived DNA biophysical auxiliary pretraining improve
> transferability, data efficiency, and interpretability for in vitro TF-DNA
> binding prediction?

The scientific motivation is that TF binding depends on motif sequence and
local sequence context, while DNA shape, flexibility, bendability, and
stiffness can contribute to molecular recognition. Rather than adding these
properties only as task-specific downstream features, this study asks whether
predicting them during pretraining helps a sequence encoder learn a more useful
and interpretable representation.

## Study workflow

```text
hg38 sequence
→ reproducible genomic split
→ tokenizer-independent sequence and feature alignment
→ S0/S1/S2 pretraining
→ in vitro TF-binding evaluation
→ transferability, data-efficiency, and interpretability analysis
```

## Current status

| Stage | Status |
|---|---|
| Completed | Scientific design, canonical base and base-step coordinates, coordinate-aware tokenization, the 12-feature lookup-table provider, reproducible hg38 splitting and leakage audits, training-only feature normalization, and supporting tests |
| In progress | Transition from the data and feature foundation to the true S0 sequence-only masked-language-model implementation |
| Next | S0 baseline, S1 individual-feature supervision, S2a fixed PCA targets, and S2b frozen learned physical components |
| Later | Controlled tokenizer comparisons, common downstream evaluation, reduced-label and transfer experiments, interpretability analyses, and future physical-feature providers |

The completed work establishes a reproducible experimental foundation. The
systematic S0, S1, S2a, and S2b pretraining comparisons have **not** yet been
run, so this branch does not claim that biophysical supervision improves
TF-binding prediction.

## Controlled pretraining conditions

All primary conditions use corrupted DNA sequence as the encoder input.
Biophysical values are supervision targets, not additional inputs, and are not
silently supplied during primary downstream inference.

| Condition | Pretraining target | Purpose |
|---|---|---|
| **S0** | Original DNA bases at corrupted positions | Sequence-only masked-language-model baseline |
| **S1** | S0 targets plus individually standardized physical features | Test whether named raw biophysical targets improve representation learning |
| **S2a** | S0 targets plus fixed PCA-derived physical components | Test a compact, fixed biophysical subspace fit on training data only |
| **S2b** | S0 targets plus codes from a separately trained and frozen physical-feature compressor | Test a learned but independently defined biophysical subspace |

S0, S1, S2a, and S2b will use matched encoder capacity, pretraining data,
optimization budgets, downstream splits, and evaluation budgets. Physical
losses will be applied only where the complete sequence support of a target is
hidden from the encoder.

Direct physical-feature input fusion is deferred until the primary
sequence-only comparison is complete.

## Tokenization and physical features

The planned primary representation is a **1-mer tokenizer**. Overlapping
stride-one **3-mer** and legacy **6-mer** tokenizers are controlled ablations.
Tokenizer size and physical-feature context length are treated as separate
experimental choices.

The current feature source is a lookup-table provider with:

- eight dinucleotide features aligned to base steps; and
- four trinucleotide features aligned to their middle bases.

The systematic representation preserves these native coordinates instead of
averaging all feature values inside tokenizer tokens. Each feature track
retains validity masks, context-support spans, and reverse-complement metadata.
Unavailable or ambiguous values are masked rather than silently imputed.

Offline DeepDNAshape outputs and processed hexABC features are planned future
extensions. Their provider interfaces are defined, but the providers
themselves are not completed current functionality.

## Reproducible hg38 pretraining split

The active pretraining dataset uses a coordinate-preserving,
whole-chromosome split of 256-base hg38 windows.

| Split | Chromosomes | Sequences |
|---|---|---:|
| Training | chr1–chr20 | 180,000 |
| Validation | chr21 | 10,000 |
| Test | chr22 | 10,000 |

Stored audits report zero cross-split genomic-interval, same-locus,
exact-sequence, and reverse-complement-equivalent overlap. Repeated sequences
within a split are recorded separately rather than hidden. Preserving genomic
coordinates makes overlap checks possible and prevents a random sequence-level
split from placing nearby or equivalent genomic windows in different
evaluation partitions.

## Training-only normalization

A versioned normalization artifact has been generated for the 12 current
lookup-table features using the 180,000-sequence training split only. It records
feature order, coordinate type, valid observation counts, means, standard
deviations, provider identity, and the associated split identity.

The same training-derived statistics must be used for training, validation,
and test data. Missing or incompatible artifacts fail closed; per-window or
per-sequence normalization is not used as a fallback.

## Planned downstream evaluation

The primary downstream comparison remains sequence-only so that any difference
can be attributed to pretraining rather than the direct availability of
physical features at inference time. Planned evaluation includes:

- per-TF performance within in vitro binding assays;
- transfer to held-out TFs and TF families;
- compatible cross-assay transfer;
- nested labeled-data fractions from 1% to 100%;
- frozen embeddings from S0, S1, S2a, and S2b;
- raw positional 1-mer, 2-mer, and 3-mer baselines;
- a common XGBoost probe as the primary downstream predictor; and
- neural downstream evaluation as an explicitly matched secondary analysis.

The comparisons will use identical examples, folds, training fractions,
hyperparameter-search budgets, metrics, and random seeds. TF protein embeddings
are excluded from the primary study and may only be introduced later as a
separate downstream condition.

## Repository organization

```text
dna-flex-pretrain/
├── configs/                 # Experiment and systematic split configuration
├── data/
│   ├── raw/                 # Local source data
│   ├── generated/           # Generated split records and manifests
│   └── processed/           # Versioned derived artifacts
├── docs/                    # Project notes, inventories, and progress reports
├── scripts/
│   ├── data_prep/           # Systematic split, audit, and normalization tools
│   ├── pretrain/            # Currently legacy prototype training scripts
│   ├── finetune/            # Legacy/prototype downstream scripts
│   ├── benchmark/           # Existing benchmark scripts
│   ├── plot/                # Plotting utilities
│   └── archive/             # Archived exploratory and validation scripts
├── src/
│   ├── coordinates.py       # Canonical base, base-step, and token spans
│   ├── feature_schema.py    # Physical-feature metadata and provider contracts
│   ├── feature_providers.py # Native-coordinate lookup-table provider
│   ├── genomic_splits.py    # Coordinate-preserving split construction/audits
│   └── ...                  # Legacy models plus systematic data utilities
├── tests/                   # Coordinate, feature, artifact, and split tests
├── PROJECT_SPEC.md          # Scientific design and experimental controls
└── README.md
```

## Environment setup

From the project directory:

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
export PYTHONPATH="$(pwd)"
```

Large external datasets and model checkpoints are expected to remain local
and are generally not committed to Git. The current lookup table is expected
at:

```text
data/raw/flex_tables/lookup.yaml
```

The hg38 reference used to regenerate the systematic split is expected at:

```text
data/raw/hg38.fa
```

Existing PBM, gcPBM, HT-SELEX, and bendability data layouts are retained for
the legacy prototype workflows. They are not presented here as the active
systematic pretraining workflow.

## Roadmap

1. Implement and validate the true S0 sequence-only MLM baseline.
2. Add S1 masked prediction of individual normalized physical features.
3. Fit separate training-only base and base-step PCA artifacts for S2a.
4. Train, validate, and freeze the independent physical compressor for S2b.
5. Run matched multi-seed S0/S1/S2a/S2b pretraining comparisons.
6. Compare 1-mer, 3-mer, and 6-mer tokenizers under matched corruption.
7. Run common downstream, reduced-data, and transfer evaluations.
8. Analyze learned sequence representations and biophysical components.
9. Consider offline DeepDNAshape and processed hexABC providers.

## Legacy prototype code and results

The repository retains an earlier overlapping 6-mer flex+MLM prototype,
bendability-stage experiments, and gcPBM/HT-SELEX benchmark scripts. Those
experiments explored useful ideas, but they used a different target alignment,
modeling protocol, and downstream setup from the active systematic study.

Legacy checkpoints, scripts, plots, and prior observations are kept for
reference and comparison. They must not be interpreted as results from the
S0/S1/S2 experimental design, and old training scripts should not be used as
the new systematic workflow without being updated to the current
specification.

## Progress report

A concise lab-meeting summary of the current branch is available in
[Advisor progress report — July 27, 2026](docs/advisor_progress_2026-07-27.md).

## References

- Vaswani et al. 2017. *Attention Is All You Need.*
- Devlin et al. 2019. *BERT: Pre-training of Deep Bidirectional Transformers
  for Language Understanding.*
- Ji et al. 2021. *DNABERT: Pre-trained Bidirectional Encoder Representations
  from Transformers model for DNA-language in genome.*
- Yang et al. 2017. *Transcription factor family-specific DNA shape readout
  revealed by quantitative specificity models.*
- Zhou et al. 2015. *Quantitative modeling of transcription factor binding
  specificities using DNA shape.*
- Basu et al. 2021. *Measuring DNA mechanics on the genome scale.*
- Jiang et al. 2023. *Assessing base-resolution DNA mechanics on the genome
  scale.*
- Dey, Yella, and Kumar. *DNA conformational flexibility descriptors improve
  transcription factor binding prediction across diverse transcription factor
  families.*
