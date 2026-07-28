# Progress Report — July 28, 2026

## Big picture

The project asks whether sequence-derived DNA biophysical supervision can make
DNA foundation-model representations more transferable, more useful when
labeled data are scarce, and easier to interpret for in vitro transcription
factor (TF) binding prediction.

The primary experiment is designed to isolate the effect of the pretraining
objective. Each model will receive DNA sequence only, use the same encoder and
pretraining data, and be evaluated with the same downstream splits and
budgets. TF-binding labels and TF protein information will remain absent from
pretraining. The main downstream comparison will also use DNA sequence only,
so a difference between models cannot be explained by supplying physical
features directly at inference time.

The current branch establishes the data, feature, and reproducibility
foundation needed for that comparison. It does not yet prove the scientific
hypothesis: the systematic model-pretraining experiments have not been run.

## Work completed

The scientific design has been organized into four primary conditions. S0 is a
sequence-only masked-language-model baseline. S1 adds prediction of individual
DNA physical features. S2a replaces the individual targets with fixed
principal-component-analysis (PCA) targets, and S2b uses codes from a
separately trained and frozen physical-feature compressor. Keeping these
conditions separate will allow us to ask whether any effect comes from
biophysical supervision in general, from particular named features, or from a
lower-dimensional physical subspace.

A tokenizer-independent coordinate system is now implemented. It distinguishes
base positions from the steps between adjacent bases and records the exact
sequence span covered by every token. This matters because physical properties
do not all live at the same type of position: the initial dinucleotide features
are base-step-centered, while the trinucleotide features are centered on their
middle base. The new representation preserves those native positions rather
than averaging all values inside a 6-mer token.

The first physical-feature provider exposes 12 existing lookup-table features:
eight dinucleotide and four trinucleotide features. It records feature order,
coordinate type, required sequence context, validity, and reverse-complement
handling. Ambiguous or missing values are masked rather than silently
invented. Interfaces have been reserved for future offline DeepDNAshape and
processed hexABC features, but those extensions have not yet been implemented.

The previous random hg38 sequence split has been replaced for the systematic
study by a coordinate-preserving, whole-chromosome split. Training contains
180,000 sequences from chromosomes 1–20, validation contains 10,000 sequences
from chromosome 21, and testing contains 10,000 sequences from chromosome 22.
The stored audits report zero cross-split genomic-interval, same-locus,
exact-sequence, and reverse-complement-equivalent overlap. This reduces the
risk that nearly identical genomic material appears on both sides of an
evaluation.

A versioned normalization artifact has also been produced for the 12 lookup
features using the training split only. Validation and test sequences did not
contribute to the fitted means or standard deviations. The artifact is linked
to the expected feature provider and split so that incompatible inputs fail
clearly instead of being normalized with the wrong statistics.

Automated tests now cover the coordinate system, tokenizer alignment, feature
providers, reverse complements, data fingerprints, normalization
compatibility, genomic splitting, overlap audits, reproducibility, and artifact
integrity. Together, these additions create a trustworthy experimental
foundation, but they are infrastructure rather than evidence of improved
TF-binding performance.

## Current stage

The project is transitioning from infrastructure to modeling. The immediate
coding milestone is the true S0 sequence-only masked-language model. The
planned primary tokenizer is 1-mer, with overlapping 3-mer and legacy 6-mer
representations retained as controlled ablations.

The repository still contains earlier 6-mer flex+MLM, bendability, gcPBM, and
HT-SELEX prototype workflows. They are useful historical references, but their
results were produced under a different design and should not be treated as
results from the systematic S0/S1/S2 study.

## Immediate next steps

1. Implement the true S0 sequence-only MLM with corruption defined in base
   coordinates. For overlapping tokenizers, every token representation that
   contains a corrupted base must also hide that base.
2. Implement S1 using the 12 individually normalized lookup-table features.
   A physical target will contribute to the loss only when its full sequence
   support is hidden.
3. Build S2a with separate training-only PCA models for base-centered and
   base-step-centered feature groups. Then build S2b using a separately
   trained, validated, and frozen physical-feature compressor.
4. Compare 1-mer, overlapping 3-mer, and legacy 6-mer tokenizers while keeping
   base corruption, model capacity, optimization budget, and data fixed.
5. Establish matched downstream evaluation using XGBoost as the primary common
   probe, together with explicitly controlled neural downstream evaluation.
6. Run reduced-data experiments and transfer experiments across TFs, TF
   families, and compatible in vitro assays using shared folds and multiple
   random seeds.

Small synthetic and smoke tests should precede expensive training. The full
S0/S1/S2 comparison should begin only after model shapes, masking behavior,
checkpoint compatibility, and loss selection have been validated.

## Expected final study

The final study should provide a paired, multi-seed comparison of S0, S1, S2a,
and S2b under matched experimental conditions. The main outputs will include
per-TF prediction metrics, transfer to held-out TFs or families, performance
across nested labeled-data fractions, and comparisons with raw positional
k-mer baselines.

Interpretability outputs should include per-feature prediction errors, PCA
loadings or frozen-compressor metadata, component stability, and analyses of
how learned sequence representations relate to the physical targets. The
central conclusion must remain conditional on the experiments: the study may
show a benefit, no measurable effect, or an effect limited to particular
assays, feature groups, or data regimes.

## Discussion points

1. **Primary tokenizer:** Should 1-mer remain the primary representation, with
   3-mer and 6-mer restricted to controlled ablations, or is there a strong
   scientific reason to elevate an overlapping tokenizer?
2. **Next feature families:** After the 12-feature lookup baseline, should the
   project prioritize offline DeepDNAshape outputs, processed hexABC features,
   or another physical-feature family?
3. **First downstream assay:** Which in vitro TF-binding assay should be the
   first systematic benchmark—HT-SELEX for TF-family breadth, PBM/gcPBM for
   continuity with the prototype work, or another dataset with stronger
   transfer and replicate structure?
