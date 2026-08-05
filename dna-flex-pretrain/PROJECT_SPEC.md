# Active scientific question

Can a DNA Transformer pretrained with true sequence-only MLM (S0) or with
matched individual biophysical auxiliary targets (S1) reach the performance of
a CNN-RC trained from scratch while using fewer labeled Exd-Hox SELEX-seq
examples?

The secondary question is whether S1 improves low-data performance over S0
when both conditions use the same encoder, pretraining data, corruption,
optimization budget, and downstream protocol. Downstream inference for S0 and
S1 uses DNA sequence only.

# Active Milestone 3A scope

The active experimental choices are deliberately narrow:

1. Transformer tokenizer:
   - active: 1-mer only;
   - deferred: overlapping stride-one 3-mer and legacy 6-mer comparisons.

2. Physical-feature source:
   - active: the existing dinucleotide and trinucleotide lookup provider;
   - deferred: PCA components, learned compressors, DeepDNAshape, and hexABC.

3. Pretraining objective:
   - active: S0 and S1;
   - deferred: S2a, S2b, S3, S4, and other component or input-fusion methods.

4. Downstream comparison:
   - primary external scratch baseline: artifact-backed 100-filter CNN-RC;
   - controlled Transformer conditions: random initialization, S0, and S1;
   - primary adaptation: frozen encoder with a shared sequence-regression head;
   - secondary adaptation: end-to-end Transformer fine-tuning;
   - deferred diagnostics: XGBoost, Ridge, raw positional k-mers, and
     32-filter CNN-RC.

Tokenizer k, physical-feature context length, pretraining objective, model
family, and downstream adaptation must remain independently identified.

# Active controlled pretraining comparison

## S0: sequence-only MLM

Input:

- corrupted DNA sequence only.

Targets:

- original DNA base identities at corrupted base coordinates.

Loss:

- base-level cross-entropy evaluated only at corrupted bases.

Downstream use:

- primary: frozen sequence embeddings derived from DNA sequence only;
- secondary: end-to-end fine-tuning with the shared Transformer regression
  head;
- no non-sequence downstream input.

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

- primary: frozen sequence embeddings derived from DNA sequence only;
- secondary: end-to-end fine-tuning with the shared Transformer regression
  head;
- auxiliary predictions are not downstream inputs.

# Shared controls

S0 and S1 must use the same:

- encoder architecture;
- 1-mer tokenizer and vocabulary;
- pretraining sequences;
- base-coordinate corruption manifests;
- number of optimization steps;
- optimizer and learning-rate schedule;
- model dimension;
- approximate parameter count;
- checkpoint-selection budget;
- downstream split manifests;
- immutable nested downstream subsets;
- downstream regression-head architecture;
- validation access and hyperparameter-search count;
- downstream metrics;
- random seeds.

The S1 auxiliary heads may add parameters outside the shared encoder, but the
encoder itself must be identical to S0. TF-binding labels and TF protein
information remain completely absent from pretraining. S0 and S1 checkpoints
must be selected without downstream TF labels.

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

# Deferred physical-component expansion

S2a, S2b, PCA fitting, learned compressors, and other physical-component
experiments are not active Milestone 3A work. The following constraints are
retained for a later, separately approved phase.

## Interpretable physical subspace

Use separate low-dimensional projections for:

- base-centered sequence representations;
- base-step-centered sequence representations.

The projections are global and shared across sequence positions.

Distinguish:

- global feature loadings, which define physical components;
- sequence-position weights, which determine which positions matter for a
  particular TF.

Future component conditions may use global physical components during
pretraining.
Defer learned sequence-position weighting to downstream TF-specific models.

## S2a: fixed PCA components

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

## S2b: frozen physical-feature compressor

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

## S3 and S4: physical-feature input fusion

S3 and S4 remain separate, deferred inference conditions because they supply
physical information to the sequence model input. They must not replace the
primary sequence-only downstream comparison or be described as directly
comparable to S0/S1 without disclosing the additional inference input.

# Physical auxiliary losses

Active S1 uses named losses and metrics for each raw feature. Any future
component condition must apply the same validity and support rules to its named
component losses.

Apply a target only when:

- the provider marks it valid;
- its complete sequence support is corrupted;
- its native coordinate is represented by the batch.

Do not treat padding, ambiguous bases, unavailable provider values, or
out-of-range coordinates as valid zero-valued targets.

# Active Exd-Hox SELEX benchmark

The first downstream benchmark is the eight-TF Exd-Hox SELEX-seq dataset from
Wang et al. It contains AbdA, AbdB, Antp, Dfd, Lab, Pb, Scr, and Ubx relative
affinities for unaligned 14-mer DNA sequences.

Source requirements:

- official repository commit:
  `9e6d6ef0355558c98855b83a9c21fe11999f65d9`;
- source directory: `CNN_models/data/SELEX_canonical`;
- imported source files must be bound to exact byte sizes and SHA-256 hashes;
- HDF5 sequence, one-hot, target, dtype, range, and validity checks must fail
  closed.

`SELEX_canonical` and `SELEX_RCmodel` are byte-identical for every train/test
file. `SELEX_RCmodel` is only a legacy architecture-routing duplicate and must
not be treated as independent data.

The supplied split contains 91 exact labeled-row train/test overlaps: AbdA 2,
Pb 85, and Ubx 4. The overlaps have identical labels. There are no RC-only
train/test overlaps and no exact- or RC-equivalent label conflicts.

## Paper-reproduction protocol

The supplied train/test split may be used unchanged only in a result namespace
labeled `paper_split_reproduction`. Every table and figure from this protocol
must disclose the 91 exact overlaps. It is for paper comparison, not the
primary scientific claim.

The paper-facing CNN-RC uses the artifact-backed 100-filter architecture. The
32-filter implementation is an ablation. The original Python 2/Keras stack is
not the maintained implementation; any legacy validation use requires separate
approval, explicit labeling, and isolation from the project environment.

## Primary scientific split

The primary protocol must:

1. combine the TF-specific canonical source partitions while retaining source
   occurrence provenance;
2. assign stable example IDs, exact sequence hashes, and RC-canonical hashes;
3. keep every exact or RC-equivalent sequence group in exactly one of train,
   validation, or test;
4. preserve all conflicting-label evidence and fail closed rather than average
   or impute it;
5. create the split before any RC augmentation;
6. use the same immutable split manifest for every model condition;
7. seal the test set after the manifest and evaluation policy are fixed.

The split proportions, stratification rule, treatment of exact duplicate
occurrences, and seed must be approved and recorded before the split is
generated. No downstream result may be used to choose among candidate split
policies.

# Controlled downstream models

The required comparison matrix is:

| Model | Initialization | Primary adaptation | Secondary adaptation |
|---|---|---|---|
| CNN-RC | random | train from scratch | none |
| Random Transformer | random | frozen-encoder sanity control | train end to end |
| S0 Transformer | S0 checkpoint | freeze encoder | fine-tune encoder and head |
| S1 Transformer | S1 checkpoint | freeze encoder | fine-tune encoder and head |

All Transformer conditions use the same encoder architecture, base-lattice
pooling, and sequence-regression head. The primary frozen representation is
the concatenation of masked mean and masked maximum pooling over the canonical
base lattice.

TF-binding labels remain absent from pretraining. Primary downstream inputs
are DNA sequence only. Physical targets, predictions, PCA scores, compressor
codes, and TF protein information are not downstream inputs.

XGBoost, Ridge, raw positional k-mer probes, and other predictors are deferred
optional diagnostics. They do not replace the active neural comparison and
must not add a production dependency without explicit approval.

# Low-data subsets

Low-data subsets must be immutable manifests generated from primary training
groups only. For each TF and subset seed, create one deterministic ordered list
and define every smaller subset as a prefix. CNN-RC, random Transformer, S0,
and S1 must use the same example IDs at every requested size.

The absolute counts or fractions must be approved from the audited primary
training-set sizes and fixed before test access. Every result must report both
the requested size and the actual number of labeled rows and unique RC groups.
Generated reverse complements do not count as new labels.

# Random seeds and model selection

- Use multiple paired experiment seeds fixed before evaluation.
- Pair S0 and S1 pretraining seeds and base-corruption manifests.
- Pair downstream subset, initialization, data-order, and augmentation seeds
  across model conditions where the concepts apply.
- Seed Python, NumPy, model-framework, data-loader, and accelerator RNGs.
- Record deterministic-kernel settings and known nondeterministic operations.
- Give every model family the same number of hyperparameter candidates,
  validation evaluations, allowed restarts, and checkpoint-selection access.
- Architecture-specific search spaces are permitted, but their identities and
  budgets must be fixed before test access.
- Training and validation may control preprocessing, early stopping, and model
  selection. Test labels may not.

The primary test set is sealed. Final test evaluation must be a separate
operation that requires frozen data, split, subset, configuration, selection,
and run identities and writes a test-access record. A later protocol change
creates a new study identity; it does not overwrite or silently reevaluate the
old study.

# Reverse-complement policy

- Group exact and RC-equivalent sequences before splitting or sampling.
- Apply the same documented on-the-fly orientation policy to every downstream
  condition.
- At inference, use the same forward/RC prediction aggregation for all models.
- CNN-RC must pass an explicit `f(x) == f(RC(x))` numerical-tolerance test.
- Handle palindromes without double-counting.
- RC augmentation must not alter labeled-example counts or search budgets.

# Metrics and success criteria

The primary paper-comparable metric is R-squared, calculated as
`1 - SSE / SST` on the sealed test data. It must not be replaced by squared
Pearson correlation.

Also report:

- Pearson correlation;
- Spearman correlation;
- RMSE;
- low-affinity-stratified performance using thresholds defined from training
  data only;
- per-seed values and paired uncertainty;
- per-TF values and equal-TF-weighted aggregate summaries.

Before test access, pre-specify:

- the non-inferiority margin defining "same performance as CNN-RC";
- the full-data CNN-RC reference for each TF;
- the smallest nested label count that qualifies as a match;
- the data-efficiency ratio;
- the paired S1-minus-S0 primary statistic and uncertainty method;
- multiplicity handling for secondary per-size comparisons.

If a model never meets the CNN-RC criterion, report the label requirement as
right-censored. Do not infer success from a noisy one-point learning-curve
crossing.

# Experiment results and figures

Training, evaluation, aggregation, and plotting must be separate operations.
Plotting reads finalized machine-readable tables and must not load checkpoints
or recompute test predictions.

Every run and metric row must identify:

- schema, study, run, data, split, subset-set, and subset IDs;
- project and external-source commits;
- TF, model family, pretraining condition, and adaptation mode;
- tokenizer, RC policy, and checkpoint hash;
- requested and actual label counts;
- split, subset, pretraining, downstream, and search seeds;
- search-space identity, budget, selected hyperparameters, and config hash;
- evaluation split, stratum, metric name, value, and sample count.

Required figure families are:

1. dataset counts and affinity distributions;
2. CNN-RC low-data learning curves;
3. CNN-RC, random Transformer, S0, and S1 learning curves;
4. paired S1-minus-S0 difference curves;
5. labels required to match the full-data CNN-RC reference;
6. per-TF and equal-TF-weighted aggregate summaries;
7. means and uncertainty across paired seeds.

Each figure must have machine-readable source data and a provenance sidecar
binding it to the config, data, split, subset, run, metric-table, and code
identities.

Run and figure paths are immutable and content-addressed. Exclusive creation is
the default. A repeated identity with different content fails; a changed input
or configuration creates a new identity. Completed results, figures,
checkpoints, and prior experiment outputs must not be overwritten or deleted
without approval.

# Data-leakage controls

Persist and audit:

- exact and RC-equivalent sequence overlap;
- duplicate labeled rows and conflicting labels;
- genomic-window and locus overlap in pretraining data;
- TF, TF-family, and assay overlap;
- repeated probes and near-duplicate sequence groups where relevant.

Preserve genomic coordinates when creating pretraining windows. Do not split
sampled genome strings after discarding the coordinates required to detect
overlap.

# Completed foundation and legacy boundary

The canonical base/base-step coordinate system, coordinate-aware tokenization,
lookup-table provider, coordinate-preserving hg38 split and leakage audits, and
training-only normalization artifact are completed foundations and remain in
force.

The current overlapping 6-mer flex+MLM model, token-level corruption,
token-averaged physical targets, permissive checkpoint loads, existing
checkpoints, and legacy PBM/HT-SELEX/gcPBM results are not true S0 or S1. They
must remain labeled legacy and cannot support Milestone 3A performance claims.

No controlled S0, S1, or CNN-RC Milestone 3A experiment has been run. The
project must not claim that biophysical supervision improves binding
prediction until the specified experiments are complete.

# Deferred work

The following remain outside active Milestone 3A scope:

- S2a fixed PCA components;
- S2b frozen learned physical components;
- DeepDNAshape and hexABC providers;
- S3/S4 physical-feature input fusion;
- 3-mer and 6-mer tokenizer comparisons;
- 32-filter CNN-RC;
- XGBoost, Ridge, and raw positional k-mer diagnostics;
- held-out-TF, held-out-family, and cross-assay transfer;
- TF protein embeddings or other explicit TF-conditioning mechanisms.

A sequence-only predictor trained across unrelated TF labels must not be
described as zero-shot TF transfer without explicit TF conditioning.

# Milestone 3A implementation order

1. Documentation pivot.
2. External-data import and audit.
3. Downstream split and subset manifests.
4. CNN-RC reproduction.
5. True S0.
6. S1.
7. Controlled low-data experiments and figures.
