# Scientific question

Does DNA biophysical auxiliary pretraining improve transferability, data
efficiency, and interpretability of DNA foundation models for in vitro
TF-DNA binding prediction when downstream inference uses DNA sequence only?

# Experimental axes

The study must keep the following axes independent:

1. Transformer tokenizer:
   - primary: 1-mer;
   - ablation: overlapping stride-one 3-mer;
   - legacy ablation: overlapping stride-one 6-mer.

2. Physical-feature source:
   - initial: existing dinucleotide and trinucleotide lookup tables;
   - future: offline DeepDNAshape outputs;
   - future: processed hexABC features.

3. Pretraining objective:
   - S0;
   - S1;
   - S2a;
   - S2b;
   - deferred S3;
   - deferred S4.

4. Downstream predictor:
   - primary: XGBoost;
   - secondary diagnostic: Ridge or another fixed linear probe.

Do not conflate tokenizer k, physical-feature context length, pretraining
objective, or downstream predictor.

# Primary controlled comparison

## S0: sequence-only MLM

Input:

- corrupted DNA sequence only.

Targets:

- original DNA base identities at corrupted base coordinates.

Loss:

- base-level cross-entropy evaluated only at corrupted bases.

Downstream use:

- frozen sequence embeddings derived from DNA sequence only.

## S1: individual physical-feature auxiliary prediction

Input:

- corrupted DNA sequence only;
- physical features are targets and are never encoder inputs.

Targets:

- original bases at corrupted base coordinates;
- individually standardized raw physical features at native base or base-step
  coordinates.

Loss:

- base-level MLM cross-entropy;
- one named masked regression loss per physical feature.

Physical-feature losses are evaluated only where the complete sequence support
of the target feature lies inside a corrupted base span.

Downstream use:

- frozen sequence embeddings derived from DNA sequence only;
- auxiliary predictions are not primary downstream inputs.

## S2a: fixed PCA physical components

Input:

- corrupted DNA sequence only.

Targets:

- original bases;
- fixed PCA component scores derived from standardized physical features.

PCA must be fit using pretraining training-split physical values only.
Base-centered and base-step-centered features must initially use separate PCA
models.

Loss:

- base-level MLM cross-entropy;
- masked regression to fixed PCA scores.

Downstream use:

- frozen sequence embeddings derived from DNA sequence only.

## S2b: frozen learned physical components

Input:

- corrupted DNA sequence only.

Targets:

- original bases;
- codes produced by a separately trained and frozen physical-feature
  compressor.

The compressor:

- is trained only on standardized physical-feature vectors from the
  pretraining training split;
- receives no DNA sequence, TF labels, TF protein information, validation-test
  statistics, or downstream labels;
- is frozen before sequence-model pretraining;
- has no gradient path from the sequence model.

Loss:

- base-level MLM cross-entropy;
- masked regression to frozen compressor codes.

Downstream use:

- frozen sequence embeddings derived from DNA sequence only.

# Deferred secondary conditions

## S3: physical-feature input projection

Input:

- corrupted DNA sequence;
- explicitly supplied physical-feature projection.

Target:

- original bases.

Loss:

- MLM cross-entropy.

S3 is deferred until S0, S1, S2a, and S2b work. It is not part of the primary
sequence-only downstream comparison because it may require physical features
during downstream inference.

## S4: physical input plus auxiliary prediction

Input:

- corrupted DNA sequence;
- explicitly supplied physical-feature projection.

Targets:

- original bases;
- physical features or frozen physical components.

Loss:

- MLM cross-entropy;
- masked physical auxiliary loss.

S4 is deferred and must be reported as a separate inference condition.

# Shared controls

S0, S1, S2a, and S2b must use the same:

- encoder architecture;
- tokenizer for comparisons within a tokenizer condition;
- pretraining sequences;
- base-coordinate corruption manifests;
- number of optimization steps;
- optimizer and learning-rate schedule;
- model dimension;
- approximate parameter count;
- checkpoint-selection budget;
- downstream split manifests;
- downstream training fractions;
- XGBoost outer folds and inner validation folds;
- XGBoost hyperparameter-search budget;
- downstream metrics;
- random seeds.

The main tokenizer comparison must use the same base-coordinate corruption
pattern so that 1-mer, 3-mer, and 6-mer models see the same corrupted bases.

# Canonical coordinate system

For a sequence of length L:

- base B_i is sequence position i for i in [0, L);
- base-step S_i lies between B_i and B_(i+1) for i in [0, L-1);
- overlapping token T_(k,t) covers base span [t, t+k).

Physical features are stored in their native coordinate system.

Initial feature alignment:

- dinucleotide value for sequence[i:i+2] is anchored at S_i;
- trinucleotide value for sequence[i:i+3] is anchored at B_(i+1).

Do not average native features inside tokenizer tokens as the primary
representation.

A tokenizer-independent base lattice is constructed from token embeddings
using a fixed uniform aggregation over tokens covering each base. A base-step
lattice is constructed from adjacent base representations. Learned
token-to-position weighting is not part of the primary pretraining study.

# Base-span corruption

Corruption is defined on the original base sequence before tokenization.

For every sampled base span:

1. replace all bases in the span with an explicit mask state;
2. tokenize the corrupted sequence;
3. ensure that every overlapping token containing a corrupted base sees the
   mask at that relative base position;
4. predict original bases at corrupted base coordinates.

For physical feature f at coordinate q, apply physical loss only when:

support_f(q) is fully contained in the corrupted base set.

The support span is defined by the physical-feature provider and may be larger
than the named feature granularity.

This rule prevents neighboring overlapping k-mer tokens or an unmasked local
sequence from trivially revealing the supervised target.

A legacy token-ID MLM may be reported only as a separate ablation.

# Physical-feature provider interface

Every provider must declare:

- stable feature identifier;
- display name;
- source, citation, and version;
- unit;
- native base or base-step coordinate;
- sequence granularity and full context span;
- alignment rule;
- reverse-complement transformation;
- ambiguous-base rule;
- missing-value rule;
- feature order;
- schema version.

Every provider must return:

- feature values;
- validity masks;
- native coordinate indices;
- sequence-support spans;
- orientation metadata when needed.

Initial provider:

- existing dinucleotide and trinucleotide lookup tables.

Future DeepDNAshape provider:

- consumes offline, previously processed predictions;
- preserves base-centered versus base-step-centered outputs;
- must not install or call DeepDNAshape during the initial implementation.

Future hexABC provider:

- consumes processed offline features;
- explicitly distinguishes 4,096 oriented hexamers from 2,080
  reverse-complement classes;
- implements any required orientation-dependent sign or coordinate
  transformation.

Do not invent, approximate, or silently impute missing physical values.

# Normalization

Fit normalization statistics after creating the pretraining split.

For every raw feature, save:

- mean;
- standard deviation;
- valid observation count;
- feature name and order;
- native coordinate space;
- provider and schema version;
- training split hash;
- data fingerprint.

Use training-split statistics for training, validation, and test data.

Do not use per-sequence or per-window normalization as a fallback.
Incompatible or missing normalization artifacts must fail closed.

# Interpretable physical subspace

Use separate low-dimensional projections for:

- base-centered sequence representations;
- base-step-centered sequence representations.

The projections are global and shared across sequence positions.

Distinguish:

- global feature loadings, which define physical components;
- sequence-position weights, which determine which positions matter for a
  particular TF.

Use global physical components during pretraining.
Defer learned sequence-position weighting to downstream TF-specific models.

# Fixed PCA components

Fit PCA separately for base-centered and base-step-centered feature groups
using standardized training-split values only.

Save:

- raw feature order;
- normalization statistics;
- PCA loadings;
- explained variance;
- cumulative explained variance;
- component-score scales;
- training split hash;
- provider/schema hash;
- solver;
- seed;
- sign convention;
- component-stability metadata across seeds or fit samples.

Use a deterministic sign convention.
Align components across seeds before reporting stability.
For degenerate components, report subspace stability rather than only
individual loading correlation.

# Frozen physical-feature compressor

Train the compressor separately from the sequence model.

The initial compressor should:

- use separate base and step feature groups;
- have a bottleneck smaller than the raw feature dimension;
- reconstruct every valid standardized input feature;
- include bottleneck variance and decorrelation diagnostics;
- select checkpoints using only a physical-feature validation split;
- run with multiple seeds.

Freeze the selected compressor before sequence-model pretraining.

Save:

- encoder and decoder weights;
- architecture;
- bottleneck size;
- raw feature order;
- normalization and code-normalization statistics;
- per-feature reconstruction metrics;
- split and schema hashes;
- seed;
- component-stability metadata.

Do not jointly learn the component definition with the sequence model in S2a
or S2b.

# Physical auxiliary losses

Use named losses and metrics for each raw feature or component.

Apply a target only when:

- the provider marks it valid;
- its complete sequence support is corrupted;
- its native coordinate is represented by the batch.

Do not treat padding, ambiguous bases, unavailable provider values, or
out-of-range coordinates as valid zero-valued targets.

# Downstream evaluation

TF-binding labels must remain completely absent from pretraining.

Primary downstream inputs for S0, S1, S2a, and S2b are DNA sequence only.
Physical targets, physical predictions, PCA scores, and compressor codes are
not supplied to the primary downstream predictor.

## Common XGBoost probe

Use XGBoost as the primary common downstream predictor for:

- raw positional 1-mer features;
- raw positional 2-mer features;
- raw positional 3-mer features;
- frozen sequence embeddings from S0;
- frozen sequence embeddings from S1;
- frozen sequence embeddings from S2a;
- frozen sequence embeddings from S2b.

The primary frozen-embedding representation is the concatenation of fixed
masked mean and masked maximum pooling over the canonical base lattice.

A position-aware flattened base-lattice probe may be reported separately for
fixed-length assays.

Ridge may be retained as a secondary linear diagnostic.

## Shared downstream protocol

All compared conditions must use identical:

- example IDs;
- sequence and reverse-complement-canonical hashes;
- outer folds;
- inner validation folds;
- sampled training fractions;
- random seeds;
- label preprocessing;
- XGBoost hyperparameter candidates;
- hyperparameter-search count;
- maximum boosting rounds;
- early-stopping rule;
- evaluation metrics.

The outer test fold must not influence preprocessing, early stopping,
hyperparameter selection, thresholds, or feature construction.

Initial labeled-data fractions:

- 1%;
- 5%;
- 10%;
- 25%;
- 50%;
- 100%.

Fractions must be nested within each outer training fold.

Primary metrics:

- R2;
- Pearson correlation;
- Spearman correlation;
- RMSE.

Use paired fold- and seed-level comparisons.

# Transferability protocols

Evaluate:

- per-TF within-dataset performance;
- protocol generalization to held-out TFs;
- protocol generalization to held-out TF families;
- same-TF cross-assay transfer where label semantics are compatible;
- HT-SELEX, PBM, gcPBM, and later SELEX-seq.

A sequence-only predictor trained across unrelated TF labels must not be
described as zero-shot TF transfer without an explicit TF-conditioning
mechanism.

TF protein embeddings remain excluded from the primary study. They may be
introduced later only as an explicitly separate downstream condition.

# Data-leakage controls

Persist and audit:

- exact sequence overlap;
- reverse-complement-equivalent overlap;
- genomic-window overlap;
- TF overlap;
- TF-family overlap;
- assay overlap;
- repeated probe overlap;
- near-duplicate sequence groups where relevant.

Preserve genomic coordinates when creating pretraining windows.
Do not split sampled genome sequence strings after discarding their source
coordinates.

# Initial implementation order

1. Factorized experiment configuration.
2. Canonical base/base-step coordinates and split manifests.
3. 1-mer, 3-mer, and 6-mer tokenizers with base-span corruption.
4. Existing lookup-table feature provider.
5. Training-only normalization artifacts.
6. Separate base and step PCA artifacts.
7. Separately trained and frozen physical compressor.
8. S0, S1, S2a, and S2b model/loss framework.
9. Tiny synthetic-data and leakage tests.
10. Common XGBoost downstream benchmark.
11. Multi-seed lookup-feature study.
12. Future offline DeepDNAshape and hexABC providers.

S3 and S4 remain deferred until the primary sequence-only comparison is
complete.
