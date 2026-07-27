# Project purpose

This repository tests whether DNA biophysical auxiliary pretraining improves
transferability, data efficiency, and interpretability for in vitro TF-DNA
binding prediction.

# Scientific constraints

- Pretraining may use DNA sequence and sequence-derived biophysical targets.
- Keep Transformer tokenizer k, physical-feature source, pretraining objective,
  and downstream predictor as independent experimental axes.
- Use a 1-mer tokenizer as the primary representation.
- Treat overlapping stride-one 3-mer and legacy 6-mer tokenizers as controlled
  ablations.
- Do not assume tokenizer k equals physical-feature context length.
- Primary pretraining conditions are:
  - S0: sequence-only MLM;
  - S1: sequence-only input with individual raw physical-feature targets;
  - S2a: sequence-only input with fixed PCA physical-component targets;
  - S2b: sequence-only input with targets from a separately trained and frozen
    physical-feature compressor.
- Defer physical-feature input fusion conditions S3 and S4 until S0, S1, S2a,
  and S2b are complete.
- TF-binding labels must remain completely absent from pretraining.
- Do not use TF protein sequence or TF protein embeddings during pretraining.
- TF protein embeddings may only be introduced as an explicitly separate
  downstream experimental condition.
- Always preserve a sequence-only MLM baseline using the same encoder,
  tokenizer, pretraining data, optimization budget, and evaluation splits.
- The primary downstream comparison must use DNA sequence only.
- Primary downstream inputs for S0, S1, S2a, and S2b must remain DNA sequence
  only.
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

# Feature rules

- Physical features may have nucleotide, dinucleotide, trinucleotide, tetramer,
  hexamer, or whole-sequence granularity.
- Alignment between sequence tokens and physical features must be explicit.
- Ambiguous bases such as N must be masked or handled through a documented rule.
- Reverse-complement handling must be tested.
- Do not assume that all 4,096 oriented hexamers can be mapped to 2,080
  double-stranded hexamers without an explicit orientation transformation.
- Standardize individual features before creating weighted combinations.
- Primary physical-component targets must use fixed PCA components or outputs
  from a separately trained and frozen physical-feature compressor.
- Do not jointly learn the physical-component definition with the sequence
  model in the primary S2a or S2b conditions.
- Global physical-feature combinations must be shared across sequence
  positions.
- Defer learned sequence-position weighting to explicitly downstream,
  TF-specific models.
- Any future jointly learned global softmax feature weights must be a separate
  ablation and must not replace S2a or S2b.
- Save component loadings or compressor weights, raw feature order,
  normalization statistics, explained variance where applicable, seed,
  split/schema hashes, and component-stability metadata.
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
- Fit raw-feature normalization, PCA, and compressor code normalization using
  the pretraining training split only.
- Fit base-centered and base-step-centered PCA models separately unless a
  scientifically justified coordinate transformation is explicitly defined.
- A separately trained physical-feature compressor must receive no sequence
  input, TF labels, downstream labels, validation-test statistics, or
  gradients from the sequence model.
- A future DeepDNAshape provider must consume offline processed values and
  preserve the provider's base-centered versus base-step-centered alignment.
- Do not install or call DeepDNAshape without explicit approval.
- A future hexABC provider must document its oriented-to-double-stranded
  mapping and any orientation-dependent transformation.

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
  training fractions, and seeds.
- Preserve genomic source coordinates when generating pretraining windows.
- Do not create genome train/validation splits after discarding the source
  coordinates needed to detect overlap.
- Use identical outer folds, inner validation folds, sampled training examples,
  and hyperparameter-search budgets across downstream model conditions.
- Use XGBoost as the planned primary common downstream predictor for raw
  positional k-mer baselines and frozen sequence embeddings.
- Do not add or install XGBoost until the user approves the new production
  dependency.
- Keep Ridge as an optional secondary linear diagnostic rather than the sole
  downstream comparison.
- Add unit tests for base/base-step coordinates, tokenizer-independent
  alignment, base-span corruption, feature-support masking, reverse
  complements, normalization, PCA artifacts, compressor non-collapse,
  per-feature losses, component losses, and model shapes.
- Checkpoint loading must validate model condition, tokenizer, feature schema,
  normalization artifact, component artifact, and split metadata; do not rely
  on permissive loading to hide incompatibilities.
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
