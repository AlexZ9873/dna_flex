# Scientific question

Does DNA biophysical auxiliary pretraining improve transferability, data
efficiency, and interpretability of DNA foundation models for in vitro
TF-DNA binding prediction?

# Primary controlled comparison

S0: Sequence-only MLM.

S1: MLM plus separate auxiliary heads for individual physical features.

S2: MLM plus one or more learned weighted biophysical components.

All three conditions must use the same:

- sequence encoder;
- tokenizer;
- pretraining sequences;
- number of optimization steps;
- optimizer and learning-rate schedule;
- model dimension and approximate parameter count;
- downstream data splits;
- downstream fine-tuning procedure;
- evaluation metrics;
- random seeds.

# Secondary comparisons

S3: Physical-feature projection and fusion with sequence embeddings.

S4: Feature projection plus auxiliary feature prediction.

External baselines such as DNABERT-2 and Nucleotide Transformer will be added
after the internal controlled comparison works.

# Learned biophysical component

Each raw feature is standardized first.

Global feature weights are calculated using softmax so that they are
nonnegative and sum to one.

The model must save:

- raw learned parameters;
- normalized feature weights;
- feature names and order;
- learned weights for every random seed;
- weight stability across seeds.

# Downstream evaluation

## Transferability

- held-out TFs;
- held-out TF families;
- cross-assay transfer;
- transfer between HT-SELEX, PBM, gcPBM, and SELEX-seq where appropriate.

## Data efficiency

Evaluate at fixed labeled-data fractions, initially:

- 1%;
- 5%;
- 10%;
- 25%;
- 50%;
- 100%.

Use the same subsampled examples for every model condition.

## Interpretability

- prediction performance for each physical feature;
- linear probes from frozen sequence embeddings;
- learned composite-feature weights;
- stability of weights across seeds;
- attribution around known binding motifs;
- controlled sequence mutations and their effects on embeddings and predictions.

# Initial implementation scope

1. Feature schema and feature registry.
2. K-mer lookup and sequence-position alignment.
3. Reverse-complement handling.
4. Training-only feature normalization.
5. Individual auxiliary prediction heads.
6. Configurable loss composition.
7. Learned weighted feature components.
8. Tiny synthetic-data sanity test.
9. One downstream TF-binding task.
10. Multi-TF and multi-assay benchmark.