# Project purpose

This repository tests whether individual biophysical auxiliary targets during
matched S1 pretraining improve low-data transfer over true sequence-only S0
pretraining for in vitro TF-DNA affinity prediction. The active Milestone 3A
benchmark uses the unaligned 14-mer Exd-Hox SELEX-seq data from Wang et al. and
a 100-filter CNN-RC trained from scratch as the primary external baseline.

# Scientific constraints

- Pretraining may use DNA sequence and sequence-derived biophysical targets.
- Keep Transformer tokenizer k, physical-feature source, pretraining objective,
  and downstream predictor as independent experimental axes.
- Use a 1-mer tokenizer as the only active Milestone 3A representation.
- Active pretraining conditions are:
  - S0: true sequence-only, base-level MLM;
  - S1: the matched S0 encoder and MLM objective plus named individual raw
    physical-feature targets.
- S0 and S1 must use the same encoder, 1-mer tokenizer, pretraining sequences,
  base-coordinate corruption manifests, optimization steps, optimizer and
  schedule, seeds, checkpoint-selection budget, and downstream protocol.
- Overlapping stride-one 3-mer and legacy 6-mer comparisons are deferred.
- Do not assume tokenizer k equals physical-feature context length.
- S2a, S2b, PCA components, learned physical compressors, DeepDNAshape,
  hexABC, S3, S4, and physical-feature input fusion are deferred.
- TF-binding labels must remain completely absent from pretraining.
- Do not use TF protein sequence or TF protein embeddings during pretraining.
- TF protein embeddings may only be introduced as an explicitly separate
  downstream experimental condition.
- Always preserve S0 as the sequence-only MLM baseline matched to S1.
- The primary downstream comparison must use DNA sequence only.
- Primary downstream inputs for S0 and S1 must remain DNA sequence only.
- Auxiliary physical predictions, PCA scores, and compressor codes must not be
  silently supplied to the primary downstream predictor.
- Physical-feature losses must be evaluated only at corrupted local sites
  whose complete sequence support is hidden from the encoder input.
- Do not silently provide physical features during downstream inference.
- Do not invent, approximate, or silently impute missing physical-feature values.
- Record the source, unit, sequence granularity, alignment rule, and
  reverse-complement rule for every physical feature.
- Calculate normalization statistics using the training split only.
- Never fit scalers, feature weights, or thresholds using validation or test data.
- Use identical train, validation, and test splits across model conditions.
- Use multiple random seeds for reported comparisons.
- Keep TF, TF-family, assay, and sequence-level leakage checks explicit.
- Do not claim that S0 or S1 improves TF-binding prediction until the controlled
  Milestone 3A experiments have actually been run.

# Active Milestone 3A downstream benchmark

- The first active downstream benchmark is the eight-TF Exd-Hox unaligned
  14-mer `SELEX_canonical` dataset from Wang et al.
- Pin the external source to official repository commit
  `9e6d6ef0355558c98855b83a9c21fe11999f65d9` and record every imported file
  hash.
- `SELEX_canonical` and `SELEX_RCmodel` are byte-identical. Treat
  `SELEX_RCmodel` only as a legacy architecture-routing duplicate, not as a
  distinct dataset.
- The supplied train/test split contains 91 exact labeled-row overlaps: AbdA
  2, Pb 85, and Ubx 4. It has no RC-only overlaps and no conflicting labels.
- Use the supplied split only for a clearly labeled paper-reproduction
  protocol. It must not support the primary scientific claim.
- Build the primary train/validation/test split from exact and
  reverse-complement-canonical sequence groups. Split before any RC
  augmentation.
- Persist immutable, nested low-data subset manifests and use the same labeled
  examples for CNN-RC, random Transformer, S0 Transformer, and S1 Transformer.
- Use the artifact-backed 100-filter CNN-RC as the primary external model
  trained from scratch. Treat the 32-filter CNN-RC as an ablation.
- Use a shared sequence-regression architecture for the random, S0, and S1
  Transformer conditions.
- Frozen-encoder evaluation is primary. End-to-end Transformer fine-tuning is
  secondary.
- Use paired seeds, equal hyperparameter-search counts and selection access,
  fixed validation data, and a sealed test set.
- Apply a documented RC policy fairly across all conditions. RC augmentation
  must not change the labeled-example count.
- Use R-squared as the primary paper-comparable metric. Also report Pearson
  correlation, Spearman correlation, RMSE, low-affinity-stratified results,
  and variability across seeds.
- Write machine-readable metrics with immutable data, split, subset, config,
  model, checkpoint, code, and seed identities.
- Keep plotting separate from training and evaluation. Every figure must have
  machine-readable source data and provenance.
- Completed runs and figures must fail on accidental overwrite. A changed
  input or configuration must produce a new identity.

# Feature rules

- Physical features may have nucleotide, dinucleotide, trinucleotide, tetramer,
  hexamer, or whole-sequence granularity.
- Alignment between sequence tokens and physical features must be explicit.
- Ambiguous bases such as N must be masked or handled through a documented rule.
- Reverse-complement handling must be tested.
- Do not assume that all 4,096 oriented hexamers can be mapped to 2,080
  double-stranded hexamers without an explicit orientation transformation.
- Standardize individual features before creating weighted combinations.
- Use canonical zero-based base coordinates B_i and base-step coordinates S_i,
  where S_i lies between B_i and B_(i+1).
- Every tokenizer token must retain its half-open base span [start, end).
- Every physical-feature value must declare its native base or base-step
  coordinate and its complete sequence-support span.
- Preserve base-centered and base-step-centered features as separate native
  coordinate groups.
- Initial dinucleotide targets are base-step-centered.
- Initial trinucleotide targets are centered on the middle base.
- Do not average physical features inside tokenizer tokens as the primary
  alignment rule.
- Align token embeddings to bases using a fixed, documented aggregation before
  applying physical heads.
- Define MLM corruption in base coordinates before tokenization.
- For overlapping k-mer tokenizers, corrupt every token representation that
  contains a corrupted base so neighboring tokens cannot reveal the original
  base.
- Fit raw-feature normalization using the pretraining training split only.
- Any future DeepDNAshape provider must consume offline processed values and
  preserve the provider's base-centered versus base-step-centered alignment.
- Do not install or call DeepDNAshape without explicit approval.
- A future hexABC provider must document its oriented-to-double-stranded
  mapping and any orientation-dependent transformation.

# Deferred expansion rules

- S2a may use only fixed PCA components fit on standardized pretraining
  training-split values. Base-centered and base-step-centered PCA must remain
  separate unless a scientifically justified coordinate transformation is
  defined.
- S2b may use only a separately trained and frozen physical-feature compressor
  that receives no sequence input, TF labels, downstream labels,
  validation-test statistics, or gradients from the sequence model.
- Do not jointly learn the physical-component definition in S2a or S2b.
- Future component definitions must be global and shared across sequence
  positions. Learned TF-specific position weighting remains downstream.
- Save component loadings or compressor weights, raw feature order,
  normalization and code-normalization statistics, explained variance where
  applicable, seed, split/schema hashes, and stability metadata.
- Any future jointly learned global softmax feature weights must be a separate
  ablation and must not replace fixed S2a or frozen S2b.
- S3 and S4 must remain separate inference conditions that disclose their
  additional physical-feature inputs. They must not replace the primary
  sequence-only S0/S1 comparison.

# Engineering rules

- Inspect relevant files before changing them.
- Make small, reviewable changes.
- Do not rewrite working modules unless the task requires it.
- Ask before adding a new production dependency.
- Do not download large datasets or model checkpoints without approval.
- Do not run full pretraining jobs on the local computer.
- Add unit tests for new data mappings, losses, feature heads, and model shapes.
- Persist pretraining and downstream split manifests with example IDs, sequence
  hashes, reverse-complement-canonical hashes, fold assignments, sampled
  training subset sizes, and seeds.
- Preserve genomic source coordinates when generating pretraining windows.
- Do not create genome train/validation splits after discarding the source
  coordinates needed to detect overlap.
- Use identical split and subset manifests, validation access, sampled training
  examples, and hyperparameter-search budgets across downstream conditions.
- CNN-RC is the active primary external baseline. XGBoost and Ridge are
  deferred optional diagnostics and must not replace the controlled neural
  comparison.
- Do not add or install XGBoost or any other production dependency until the
  user approves it.
- Validate Exd-Hox HDF5 schemas, dtypes, one-hot orientation, target ranges,
  source hashes, exact duplicates, RC-equivalent duplicates, and label
  conflicts during import.
- Add unit tests for base/base-step coordinates, tokenizer-independent
  alignment, base-span corruption, feature-support masking, reverse
  complements, normalization, per-feature losses, model shapes, Exd-Hox
  imports, grouped splits, nested subsets, CNN-RC invariance, metric
  definitions, result schemas, sealed-test behavior, and overwrite protection.
- Checkpoint loading must validate model condition, tokenizer, feature schema,
  normalization artifact, split metadata, and component artifacts when
  applicable; do not rely on permissive loading to hide incompatibilities.
- Keep the current legacy 6-mer flex+MLM model and existing checkpoints labeled
  as legacy. They are not true S0 or S1 and must not be used as such.
- Run the relevant tests after every implementation task.
- Explain every changed file and every test result.
- Never delete data, checkpoints, or previous experiment outputs without approval.

# Python style

- Prefer explicit and readable Python over clever or highly compressed code.
- Avoid unnecessary one-line expressions and list comprehensions.
- Avoid f-strings, context managers, break/continue, while True, and broad
  try/except blocks unless they are clearly necessary.
- Use descriptive variable names.
- Add type hints and short docstrings where they improve clarity.
- Keep data processing, model definitions, training, and evaluation separate.

# Version control

- Work only on the active feature branch.
- Do not force-push.
- Do not commit raw data, checkpoints, temporary files, credentials, or secrets.
- Keep each commit focused on one milestone.
