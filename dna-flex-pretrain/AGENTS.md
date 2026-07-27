# Project purpose

This repository tests whether DNA biophysical auxiliary pretraining improves
transferability, data efficiency, and interpretability for in vitro TF-DNA
binding prediction.

# Scientific constraints

- Pretraining may use DNA sequence and sequence-derived biophysical targets.
- Do not use TF protein sequence or TF protein embeddings during pretraining.
- TF protein embeddings may only be introduced as an explicitly separate
  downstream experimental condition.
- Always preserve a sequence-only MLM baseline using the same encoder,
  tokenizer, pretraining data, optimization budget, and evaluation splits.
- The primary downstream comparison must use DNA sequence only.
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
- Initial learned feature combinations should use global softmax-normalized
  weights for interpretability.
- Save learned weights and their variation across random seeds.

# Engineering rules

- Inspect relevant files before changing them.
- Make small, reviewable changes.
- Do not rewrite working modules unless the task requires it.
- Ask before adding a new production dependency.
- Do not download large datasets or model checkpoints without approval.
- Do not run full pretraining jobs on the local computer.
- Add unit tests for new data mappings, losses, feature heads, and model shapes.
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
