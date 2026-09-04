"""Plot and validate finalized-table-only Exd-Hox primary-split figures.

Generation requires the exact clean commit that contains the plot generator::

    python -m scripts.plot.plot_exd_hox_primary_split \
        --expected-plot-generator-commit <full-commit>

Historical v2 outputs can be validated from a later checkout without checking
out their producer commit::

    python -m scripts.plot.plot_exd_hox_primary_split \
        --validate-manifest \
        plots/exd_hox_primary_split_v2/exd_hox_primary_split_plot_manifest_v2.json \
        --expected-plot-generator-commit <historical-full-commit>

The plotter deliberately has no HDF5, split-generation, or sealed-target
dependency. Test affinities are represented only by aggregate test counts.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Dict, Mapping, Sequence, Tuple

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "dna_flex_pretrain_matplotlib"),
)
os.environ.setdefault(
    "XDG_CACHE_HOME",
    str(Path(tempfile.gettempdir()) / "dna_flex_pretrain_cache"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.downstream_fingerprints import (
    build_hashed_manifest,
    fingerprint_file,
    hash_file_bytes,
    repository_relative_path,
    validate_hashed_manifest,
    validate_repository_relative_path,
    write_json_exclusive,
    write_tsv_exclusive,
)


PRIMARY_CONFIG_SCHEMA_VERSION = "exd_hox_primary_split_config.v1"
SPLIT_MANIFEST_SCHEMA_VERSION = "exd_hox_primary_split_manifest.v1"
SUBSET_MANIFEST_SCHEMA_VERSION = "exd_hox_subset_set_manifest.v1"
PLOTTING_ENTRY_POINT = "scripts/plot/plot_exd_hox_primary_split.py"
PRIMARY_CONFIG_LOGICAL_PATH = "configs/exd_hox_primary_split_v1.yaml"
EXTERNAL_SOURCE_COMMIT = "9e6d6ef0355558c98855b83a9c21fe11999f65d9"
SOURCE_FOUNDATION_COMMIT = "62a99688cbd3af97f081df500223ba6f55cd0fe0"
SPLIT_PIPELINE_COMMIT = "65c6bddc0f4570e7a6cf5a90f5f6ef5801e01d27"

RAW_FILE_SHA256 = {
    "data/raw/exd_hox_selex_canonical_v1/AbdA/AbdA_test.h5": (
        "b8d1ed87591725de6eb8892c9d033eb4d21c57ec78083cf06890d20a22d5e0d3"
    ),
    "data/raw/exd_hox_selex_canonical_v1/AbdA/AbdA_train.h5": (
        "8941ac376a9af4da31aa439ea224c74b012199808e65c8101ce10e0d92f965b6"
    ),
    "data/raw/exd_hox_selex_canonical_v1/AbdB/AbdB_test.h5": (
        "42d48167c4784daa2758c9274bb0816ea21b0b7c70b3b0e450372b1003854a0c"
    ),
    "data/raw/exd_hox_selex_canonical_v1/AbdB/AbdB_train.h5": (
        "601ec9f1ab802ea56ddf9b03017172fd049fcfa95c794bf4e3b656fd825aff7a"
    ),
    "data/raw/exd_hox_selex_canonical_v1/Antp/Antp_test.h5": (
        "a27eec70d02304f95d333718ac82e96d220bb3da6295beb680eb582e189e88c0"
    ),
    "data/raw/exd_hox_selex_canonical_v1/Antp/Antp_train.h5": (
        "bf8ac08213c302ffc0e9c4357a90d9470abe5842ffd00f984a5dfbb763c249ae"
    ),
    "data/raw/exd_hox_selex_canonical_v1/Dfd/Dfd_test.h5": (
        "ab25a4de43974515cf1b3ac45aeff8d3c38792cb4d1101324068fefaa99afcdb"
    ),
    "data/raw/exd_hox_selex_canonical_v1/Dfd/Dfd_train.h5": (
        "30f7130ff3f7822acd91e77337885b31e69bd96106daf790539af23031ac60f8"
    ),
    "data/raw/exd_hox_selex_canonical_v1/Lab/Lab_test.h5": (
        "f7877379397316d56e2406774afb8ad144f92ce272b57f61765b721d19ec814a"
    ),
    "data/raw/exd_hox_selex_canonical_v1/Lab/Lab_train.h5": (
        "231fa9475b575c046aa69a239a39f13d1669edc69f61c0714f3625cb5436c3d6"
    ),
    "data/raw/exd_hox_selex_canonical_v1/Pb/Pb_test.h5": (
        "7b639e6f970d1c5d9953d45a597e4f792161486ab3986c7af4c44e86235b46ce"
    ),
    "data/raw/exd_hox_selex_canonical_v1/Pb/Pb_train.h5": (
        "13943e14e52f1b4573cb595a500049d77b736315fb21321d949b8bb557a172d8"
    ),
    "data/raw/exd_hox_selex_canonical_v1/Scr/Scr_test.h5": (
        "fbd63bba7472bdf3ee655057a6c4b8c077813705952b4b13f77afdb691c10934"
    ),
    "data/raw/exd_hox_selex_canonical_v1/Scr/Scr_train.h5": (
        "ba67be0d76de57cb735e41ffe2367229b3dfc14263d588368330494143b7c2cc"
    ),
    "data/raw/exd_hox_selex_canonical_v1/Ubx/Ubx_test.h5": (
        "f676f7c5c0675011190ecafd0e7d7557dcf957a5791a97aa830531a44fc88826"
    ),
    "data/raw/exd_hox_selex_canonical_v1/Ubx/Ubx_train.h5": (
        "00ae0186ac0aabb393f77c3101b5c011d8b9b2b2e54b018417327071a6241dfb"
    ),
}

COUNT_INPUT_FILENAME = "exd_hox_primary_split_count_summary_v1.tsv"
AFFINITY_INPUT_FILENAME = "exd_hox_primary_split_affinity_histogram_v1.tsv"
LEAKAGE_INPUT_FILENAME = "exd_hox_primary_split_leakage_audit_v1.tsv"
SUBSET_INPUT_FILENAME = "exd_hox_nested_subset_levels_v1.tsv"
SPLIT_MANIFEST_FILENAME = "exd_hox_primary_split_manifest_v1.json"
SUBSET_MANIFEST_FILENAME = "exd_hox_subset_set_manifest_v1.json"

@dataclass(frozen=True)
class PlotContract:
    """Canonical paths and filenames for one plot-contract schema."""

    config_schema_version: str
    config_logical_path: str
    manifest_schema_version: str
    manifest_filename: str
    plot_logical_directory: str
    count_source_filename: str
    affinity_source_filename: str
    subset_source_filename: str
    leakage_source_filename: str
    comparison_source_filename: str
    count_plot_stem: str
    affinity_plot_stem: str
    subset_plot_stem: str
    leakage_plot_stem: str
    comparison_plot_stem: str
    creator_metadata: str

    @property
    def output_filenames(self) -> Tuple[str, ...]:
        """Return the five source tables and five PNG/PDF pairs."""

        return (
            self.count_source_filename,
            self.affinity_source_filename,
            self.subset_source_filename,
            self.leakage_source_filename,
            self.comparison_source_filename,
            "{0}.png".format(self.count_plot_stem),
            "{0}.pdf".format(self.count_plot_stem),
            "{0}.png".format(self.affinity_plot_stem),
            "{0}.pdf".format(self.affinity_plot_stem),
            "{0}.png".format(self.subset_plot_stem),
            "{0}.pdf".format(self.subset_plot_stem),
            "{0}.png".format(self.leakage_plot_stem),
            "{0}.pdf".format(self.leakage_plot_stem),
            "{0}.png".format(self.comparison_plot_stem),
            "{0}.pdf".format(self.comparison_plot_stem),
        )


V2_PLOT_CONTRACT = PlotContract(
    config_schema_version="exd_hox_primary_split_plot_config.v2",
    config_logical_path="configs/exd_hox_primary_split_plot_v2.yaml",
    manifest_schema_version="exd_hox_primary_split_plot_manifest.v2",
    manifest_filename="exd_hox_primary_split_plot_manifest_v2.json",
    plot_logical_directory="plots/exd_hox_primary_split_v2",
    count_source_filename="exd_hox_primary_split_counts_plot_source_v2.tsv",
    affinity_source_filename="exd_hox_primary_split_affinity_plot_source_v2.tsv",
    subset_source_filename="exd_hox_nested_subset_counts_plot_source_v2.tsv",
    leakage_source_filename="exd_hox_primary_split_leakage_plot_source_v2.tsv",
    comparison_source_filename=(
        "exd_hox_paper_vs_primary_split_plot_source_v2.tsv"
    ),
    count_plot_stem="exd_hox_primary_split_counts_v2",
    affinity_plot_stem="exd_hox_primary_split_affinity_distributions_v2",
    subset_plot_stem="exd_hox_nested_subset_counts_v2",
    leakage_plot_stem="exd_hox_primary_split_leakage_v2",
    comparison_plot_stem="exd_hox_paper_vs_primary_split_counts_v2",
    creator_metadata="dna-flex-pretrain Milestone 3D-B.1",
)
V3_PLOT_CONTRACT = PlotContract(
    config_schema_version="exd_hox_primary_split_plot_config.v3",
    config_logical_path="configs/exd_hox_primary_split_plot_v3.yaml",
    manifest_schema_version="exd_hox_primary_split_plot_manifest.v3",
    manifest_filename="exd_hox_primary_split_plot_manifest_v3.json",
    plot_logical_directory="plots/exd_hox_primary_split_v3",
    count_source_filename="exd_hox_primary_split_counts_plot_source_v3.tsv",
    affinity_source_filename="exd_hox_primary_split_affinity_plot_source_v3.tsv",
    subset_source_filename="exd_hox_nested_subset_counts_plot_source_v3.tsv",
    leakage_source_filename="exd_hox_primary_split_leakage_plot_source_v3.tsv",
    comparison_source_filename=(
        "exd_hox_paper_vs_primary_split_plot_source_v3.tsv"
    ),
    count_plot_stem="exd_hox_primary_split_counts_v3",
    affinity_plot_stem="exd_hox_primary_split_affinity_distributions_v3",
    subset_plot_stem="exd_hox_nested_subset_counts_v3",
    leakage_plot_stem="exd_hox_primary_split_leakage_v3",
    comparison_plot_stem="exd_hox_paper_vs_primary_split_counts_v3",
    creator_metadata="dna-flex-pretrain Milestone 3D-B.2",
)

PLOT_CONTRACTS_BY_CONFIG_SCHEMA = {
    V2_PLOT_CONTRACT.config_schema_version: V2_PLOT_CONTRACT,
    V3_PLOT_CONTRACT.config_schema_version: V3_PLOT_CONTRACT,
}
PLOT_CONTRACTS_BY_MANIFEST_SCHEMA = {
    V2_PLOT_CONTRACT.manifest_schema_version: V2_PLOT_CONTRACT,
    V3_PLOT_CONTRACT.manifest_schema_version: V3_PLOT_CONTRACT,
}
DEFAULT_PLOT_CONTRACT = V3_PLOT_CONTRACT

# These names retain the public API while identifying the generation default.
PLOT_CONFIG_SCHEMA_VERSION = DEFAULT_PLOT_CONTRACT.config_schema_version
PLOT_MANIFEST_SCHEMA_VERSION = DEFAULT_PLOT_CONTRACT.manifest_schema_version
PLOT_CONFIG_LOGICAL_PATH = DEFAULT_PLOT_CONTRACT.config_logical_path
PLOT_LOGICAL_DIRECTORY = DEFAULT_PLOT_CONTRACT.plot_logical_directory
COUNT_SOURCE_FILENAME = DEFAULT_PLOT_CONTRACT.count_source_filename
AFFINITY_SOURCE_FILENAME = DEFAULT_PLOT_CONTRACT.affinity_source_filename
SUBSET_SOURCE_FILENAME = DEFAULT_PLOT_CONTRACT.subset_source_filename
LEAKAGE_SOURCE_FILENAME = DEFAULT_PLOT_CONTRACT.leakage_source_filename
COMPARISON_SOURCE_FILENAME = DEFAULT_PLOT_CONTRACT.comparison_source_filename
PLOT_MANIFEST_FILENAME = DEFAULT_PLOT_CONTRACT.manifest_filename
COUNT_PLOT_STEM = DEFAULT_PLOT_CONTRACT.count_plot_stem
AFFINITY_PLOT_STEM = DEFAULT_PLOT_CONTRACT.affinity_plot_stem
SUBSET_PLOT_STEM = DEFAULT_PLOT_CONTRACT.subset_plot_stem
LEAKAGE_PLOT_STEM = DEFAULT_PLOT_CONTRACT.leakage_plot_stem
COMPARISON_PLOT_STEM = DEFAULT_PLOT_CONTRACT.comparison_plot_stem
OUTPUT_FILENAMES = DEFAULT_PLOT_CONTRACT.output_filenames

PLOT_MANIFEST_FIELDS = frozenset(
    (
        "schema_version",
        "study_identifier",
        "dataset_identifier",
        "external_source_commit",
        "source_foundation_commit",
        "split_pipeline_commit",
        "plot_generator_commit",
        "plot_generator_tracked_worktree_clean",
        "plotting_entry_point_path",
        "plotting_entry_point_byte_size",
        "plotting_entry_point_sha256",
        "plot_config_path",
        "plot_config_sha256",
        "primary_split_config_path",
        "primary_split_config_sha256",
        "primary_split_id",
        "primary_split_manifest_path",
        "primary_split_manifest_hash",
        "primary_split_manifest_file_sha256",
        "subset_set_id",
        "subset_set_manifest_path",
        "subset_set_manifest_hash",
        "subset_set_manifest_file_sha256",
        "plot_directory",
        "inputs",
        "outputs",
        "test_target_policy",
        "manifest_hash",
    )
)

COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

COUNT_INPUT_FIELDS = (
    "protocol",
    "transcription_factor",
    "split",
    "row_count",
    "logical_example_count",
    "global_rc_group_count",
    "exact_cross_split_overlap_occurrence_count",
)
AFFINITY_INPUT_FIELDS = (
    "transcription_factor",
    "split",
    "bin_index",
    "bin_left",
    "bin_right",
    "logical_example_count",
)
LEAKAGE_INPUT_FIELDS = (
    "comparison",
    "left_split",
    "right_split",
    "exact_sequence_overlap_group_count",
    "reverse_complement_equivalent_overlap_group_count",
    "reverse_complement_only_overlap_group_count",
    "logical_example_overlap_count",
)
SUBSET_INPUT_FIELDS = (
    "transcription_factor",
    "level_id",
    "request_type",
    "request_value",
    "unaliased_requested_logical_example_count",
    "alias_absolute_anchor",
    "canonical_requested_logical_example_count",
    "actual_logical_example_count",
    "actual_rc_group_count",
    "inclusive_maximum_rank",
)

COUNT_SOURCE_FIELDS = (
    "transcription_factor",
    "split",
    "logical_example_count",
    "global_rc_group_count",
)
AFFINITY_SOURCE_FIELDS = (
    "record_type",
    "transcription_factor",
    "split",
    "bin_index",
    "bin_left",
    "bin_right",
    "logical_example_count",
)
COMPARISON_SOURCE_FIELDS = (
    "protocol",
    "transcription_factor",
    "split",
    "logical_example_count",
)

PRIMARY_SPLITS = ("training", "validation", "test")
PAPER_SPLITS = ("train", "test")
AFFINITY_SPLITS = ("training", "validation")
FORBIDDEN_PLAINTEXT_FIELDS = frozenset(
    (
        "affinity",
        "affinity_value",
        "plaintext_target",
        "target",
        "target_bits",
        "target_float32_bits",
        "target_value",
    )
)

SUBSET_FIGURE_TITLE = "Requested versus actual nested low-data counts"
SUBSET_FIGURE_CAPTION = (
    "Absolute levels are requested counts; percentage levels are fractions of "
    "the full primary training split.\nPercentage requests may alias an "
    "absolute canonical level."
)
LEAKAGE_FIGURE_CAPTION = (
    "RC-equivalent overlap is inclusive of exact-sequence matches; the three "
    "series are not additive.\nRC-only counts exclude exact matches."
)
COMPARISON_FIGURE_TITLE = (
    "Supplied split row occurrences versus primary logical examples"
)
COMPARISON_FIGURE_CAPTION = (
    "For each TF, supplied-split bars count labeled row occurrences in the "
    "supplied train/test files,\nwhereas primary-split bars count reconciled "
    "logical labeled examples after exact/RC grouping.\nThe two bar families "
    "therefore use different counting units."
)


def parse_arguments(argv=None):
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate or validate versioned Exd-Hox plots with commit-bound "
            "code provenance."
        )
    )
    parser.add_argument(
        "--config",
        default=PLOT_CONFIG_LOGICAL_PATH,
    )
    parser.add_argument("--repository-root", default=".")
    parser.add_argument(
        "--expected-plot-generator-commit",
        required=True,
        help="Full commit expected to contain the plot generator.",
    )
    parser.add_argument(
        "--validate-manifest",
        metavar="PATH",
        help=(
            "Validate an existing v2 or v3 plot manifest and its historical "
            "producer blob instead of generating plots."
        ),
    )
    return parser.parse_args(argv)


def plot_primary_split_tables(
    config_path: Path | str,
    repository_root: Path | str,
    expected_plot_generator_commit: str,
) -> Dict[str, Any]:
    """Create five immutable plot families from finalized public tables."""

    root = Path(os.path.abspath(repository_root))
    config_file = _resolve_repository_file(root, config_path)
    _reject_protected_input_path(repository_relative_path(config_file, root))
    if config_file.is_symlink() or not config_file.is_file():
        raise FileNotFoundError("Plot config must be a regular public file.")
    config = _load_plot_config(config_file)
    contract = _plot_contract_for_config_schema(config["schema_version"])
    config_logical_path = repository_relative_path(config_file, root)
    if config_logical_path != contract.config_logical_path:
        raise ValueError("The plot config must use its contract's canonical path.")
    bindings = _load_bound_plot_inputs(config, config_file, root)
    plot_logical_directory = str(config["outputs"]["plot_directory"])
    plot_directory = _resolve_repository_path(root, plot_logical_directory)
    _require_new_plot_directory(plot_directory)

    runtime_head = _verify_generation_git_binding(
        repository_root=root,
        config=config,
        expected_plot_generator_commit=expected_plot_generator_commit,
    )
    entry_point_path = _resolve_repository_path(root, PLOTTING_ENTRY_POINT)
    entry_point_fingerprint = _verify_tracked_entry_point(
        repository_root=root,
        runtime_head=runtime_head,
        entry_point_path=entry_point_path,
    )
    snapshot_paths = (
        entry_point_path,
        config_file,
        bindings["primary_config_path"],
        *bindings["input_paths"],
    )
    initial_snapshots = _fingerprint_paths(snapshot_paths, root)
    _recheck_generation_state(
        repository_root=root,
        runtime_head=runtime_head,
        snapshot_paths=snapshot_paths,
        initial_snapshots=initial_snapshots,
    )
    revalidated_config = _load_plot_config(config_file, contract)
    if revalidated_config != config:
        raise ValueError("Plot config changed before generation snapshot validation.")
    revalidated_bindings = _load_bound_plot_inputs(
        revalidated_config,
        config_file,
        root,
    )
    if tuple(revalidated_bindings["input_paths"]) != tuple(bindings["input_paths"]):
        raise ValueError("Public plot input bindings changed before generation.")
    bindings = revalidated_bindings
    _recheck_generation_state(
        repository_root=root,
        runtime_head=runtime_head,
        snapshot_paths=snapshot_paths,
        initial_snapshots=initial_snapshots,
    )
    initial_input_fingerprints = _fingerprint_paths(
        bindings["input_paths"],
        root,
    )

    count_rows = _read_tsv(bindings["count_path"], COUNT_INPUT_FIELDS)
    affinity_rows = _read_tsv(bindings["affinity_path"], AFFINITY_INPUT_FIELDS)
    leakage_rows = _read_tsv(bindings["leakage_path"], LEAKAGE_INPUT_FIELDS)
    subset_rows = _read_tsv(bindings["subset_path"], SUBSET_INPUT_FIELDS)
    transcription_factors = tuple(
        bindings["primary_config"]["dataset"]["transcription_factors"]
    )

    _validate_count_rows(transcription_factors, count_rows)
    _validate_affinity_rows(transcription_factors, affinity_rows)
    _validate_leakage_rows(leakage_rows)
    _validate_subset_rows(transcription_factors, subset_rows)

    count_source_rows = _count_source_rows(transcription_factors, count_rows)
    affinity_source_rows = _affinity_source_rows(
        transcription_factors,
        affinity_rows,
        count_rows,
    )
    subset_source_rows = _subset_source_rows(transcription_factors, subset_rows)
    leakage_source_rows = _leakage_source_rows(leakage_rows)
    comparison_source_rows = _comparison_source_rows(
        transcription_factors,
        count_rows,
    )

    plot_directory.parent.mkdir(parents=True, exist_ok=True)
    staging_context = tempfile.TemporaryDirectory(
        prefix=".exd_hox_primary_split_plot_staging_",
        dir=plot_directory.parent,
    )
    staging_directory = Path(staging_context.name) / "plots"
    staging_directory.mkdir()
    try:
        write_tsv_exclusive(
            staging_directory / contract.count_source_filename,
            COUNT_SOURCE_FIELDS,
            count_source_rows,
        )
        write_tsv_exclusive(
            staging_directory / contract.affinity_source_filename,
            AFFINITY_SOURCE_FIELDS,
            affinity_source_rows,
        )
        write_tsv_exclusive(
            staging_directory / contract.subset_source_filename,
            SUBSET_INPUT_FIELDS,
            subset_source_rows,
        )
        write_tsv_exclusive(
            staging_directory / contract.leakage_source_filename,
            LEAKAGE_INPUT_FIELDS,
            leakage_source_rows,
        )
        write_tsv_exclusive(
            staging_directory / contract.comparison_source_filename,
            COMPARISON_SOURCE_FIELDS,
            comparison_source_rows,
        )

        _plot_primary_counts(staging_directory, count_source_rows, contract)
        _plot_affinity_distributions(
            staging_directory,
            transcription_factors,
            affinity_source_rows,
            contract,
        )
        _plot_subset_counts(
            staging_directory,
            transcription_factors,
            subset_source_rows,
            contract,
        )
        _plot_leakage(staging_directory, leakage_source_rows, contract)
        _plot_protocol_comparison(
            staging_directory,
            transcription_factors,
            comparison_source_rows,
            contract,
        )

        manifest = _write_plot_manifest(
            config=config,
            config_file=config_file,
            repository_root=root,
            plot_logical_directory=plot_logical_directory,
            staging_directory=staging_directory,
            input_fingerprints=initial_input_fingerprints,
            bindings=bindings,
            runtime_head=runtime_head,
            entry_point_fingerprint=entry_point_fingerprint,
            contract=contract,
        )
        _recheck_generation_state(
            repository_root=root,
            runtime_head=runtime_head,
            snapshot_paths=snapshot_paths,
            initial_snapshots=initial_snapshots,
        )
        _require_new_plot_directory(plot_directory)
        os.rename(staging_directory, plot_directory)
    finally:
        staging_context.cleanup()
    return manifest


def _plot_contract_for_config_schema(schema_version: Any) -> PlotContract:
    if not isinstance(schema_version, str):
        raise ValueError("Unsupported Exd-Hox primary split plot config schema.")
    contract = PLOT_CONTRACTS_BY_CONFIG_SCHEMA.get(schema_version)
    if contract is None:
        raise ValueError("Unsupported Exd-Hox primary split plot config schema.")
    return contract


def _plot_contract_for_manifest_schema(schema_version: Any) -> PlotContract:
    if not isinstance(schema_version, str):
        raise ValueError("Unsupported Exd-Hox primary split plot manifest schema.")
    contract = PLOT_CONTRACTS_BY_MANIFEST_SCHEMA.get(schema_version)
    if contract is None:
        raise ValueError("Unsupported Exd-Hox primary split plot manifest schema.")
    return contract


def _load_plot_config(
    path: Path,
    expected_contract: PlotContract | None = None,
) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as input_file:
        payload = yaml.safe_load(input_file)
    if not isinstance(payload, Mapping):
        raise ValueError("Primary split plot config must be a mapping.")
    contract = _plot_contract_for_config_schema(payload.get("schema_version"))
    if expected_contract is not None and contract != expected_contract:
        raise ValueError("Plot config schema differs from the manifest contract.")
    _require_exact_fields(
        payload,
        ("schema_version", "study", "provenance", "inputs", "outputs"),
        "Plot config",
    )
    study = payload.get("study")
    provenance = payload.get("provenance")
    inputs = payload.get("inputs")
    outputs = payload.get("outputs")
    if not isinstance(study, Mapping):
        raise ValueError("Plot config study must be a mapping.")
    if not isinstance(provenance, Mapping):
        raise ValueError("Plot config provenance must be a mapping.")
    if not isinstance(inputs, Mapping):
        raise ValueError("Plot config inputs must be a mapping.")
    if not isinstance(outputs, Mapping):
        raise ValueError("Plot config outputs must be a mapping.")

    _require_exact_fields(
        study,
        ("identifier", "dataset_identifier"),
        "Plot config study",
    )
    _require_nonempty_string(study.get("identifier"), "Study identifier")
    _require_nonempty_string(
        study.get("dataset_identifier"),
        "Dataset identifier",
    )

    _require_exact_fields(
        provenance,
        (
            "external_source_commit",
            "source_foundation_commit",
            "split_pipeline_commit",
            "plotting_entry_point",
        ),
        "Plot config provenance",
    )
    for field in (
        "external_source_commit",
        "source_foundation_commit",
        "split_pipeline_commit",
    ):
        _require_full_commit(provenance.get(field), field)
    if provenance.get("external_source_commit") != EXTERNAL_SOURCE_COMMIT:
        raise ValueError("Plot config external source commit differs.")
    if provenance.get("source_foundation_commit") != SOURCE_FOUNDATION_COMMIT:
        raise ValueError("Plot config source-foundation commit differs.")
    if provenance.get("split_pipeline_commit") != SPLIT_PIPELINE_COMMIT:
        raise ValueError("Plot config split-pipeline commit differs.")
    if provenance.get("plotting_entry_point") != PLOTTING_ENTRY_POINT:
        raise ValueError("Plot config plotting entry point differs.")

    required_input_fields = (
        "primary_split_config_path",
        "primary_split_config_sha256",
        "primary_split_id",
        "primary_split_manifest_path",
        "primary_split_manifest_hash",
        "primary_split_manifest_file_sha256",
        "subset_set_id",
        "subset_set_manifest_path",
        "subset_set_manifest_hash",
        "subset_set_manifest_file_sha256",
        "count_summary_path",
        "affinity_histogram_path",
        "leakage_audit_path",
        "subset_levels_path",
        "raw_file_sha256",
    )
    _require_exact_fields(inputs, required_input_fields, "Plot config inputs")
    path_fields = (
        "primary_split_config_path",
        "primary_split_manifest_path",
        "subset_set_manifest_path",
        "count_summary_path",
        "affinity_histogram_path",
        "leakage_audit_path",
        "subset_levels_path",
    )
    for field in path_fields:
        validate_repository_relative_path(str(inputs[field]))
    if inputs["primary_split_config_path"] != PRIMARY_CONFIG_LOGICAL_PATH:
        raise ValueError("Plot config primary split config path differs.")
    for field in (
        "primary_split_config_sha256",
        "primary_split_id",
        "primary_split_manifest_hash",
        "primary_split_manifest_file_sha256",
        "subset_set_manifest_hash",
        "subset_set_manifest_file_sha256",
    ):
        _require_sha256(inputs.get(field), field)
    _require_nonempty_string(inputs.get("subset_set_id"), "Subset-set ID")
    raw_file_sha256 = inputs.get("raw_file_sha256")
    if not isinstance(raw_file_sha256, Mapping) or not raw_file_sha256:
        raise ValueError("Plot config raw-file SHA-256 identities must be a mapping.")
    for raw_path, raw_sha256 in raw_file_sha256.items():
        normalized_raw_path = validate_repository_relative_path(str(raw_path))
        if Path(normalized_raw_path).suffix.lower() not in (".h5", ".hdf5"):
            raise ValueError("Raw-file identity path must identify an HDF5 file.")
        _require_sha256(raw_sha256, "Raw-file SHA-256")
    normalized_raw_identities = {}
    for raw_path, raw_sha256 in raw_file_sha256.items():
        normalized_raw_identities[str(raw_path)] = str(raw_sha256)
    if normalized_raw_identities != RAW_FILE_SHA256:
        raise ValueError("Plot config raw-file SHA-256 identities differ.")

    _require_exact_fields(outputs, ("plot_directory",), "Plot config outputs")
    validate_repository_relative_path(str(outputs["plot_directory"]))
    if outputs["plot_directory"] != contract.plot_logical_directory:
        raise ValueError("Plot config output directory differs from its contract.")
    return dict(payload)


def _load_primary_split_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as input_file:
        payload = yaml.safe_load(input_file)
    if not isinstance(payload, Mapping):
        raise ValueError("Primary split config must be a mapping.")
    if payload.get("schema_version") != PRIMARY_CONFIG_SCHEMA_VERSION:
        raise ValueError("Unsupported Exd-Hox primary split config schema.")
    study = payload.get("study")
    dataset = payload.get("dataset")
    outputs = payload.get("outputs")
    if not isinstance(study, Mapping):
        raise ValueError("Primary split config study must be a mapping.")
    if not isinstance(dataset, Mapping):
        raise ValueError("Primary split config dataset must be a mapping.")
    if not isinstance(outputs, Mapping):
        raise ValueError("Primary split config outputs must be a mapping.")
    transcription_factors = dataset.get("transcription_factors")
    if not isinstance(transcription_factors, Sequence) or isinstance(
        transcription_factors,
        (str, bytes),
    ):
        raise ValueError("Dataset transcription_factors must be a sequence.")
    normalized_tfs = tuple(str(value) for value in transcription_factors)
    if not normalized_tfs or len(set(normalized_tfs)) != len(normalized_tfs):
        raise ValueError("Dataset transcription factors must be nonempty and unique.")
    for field in ("identifier", "dataset_identifier", "project_commit"):
        _require_nonempty_string(study.get(field), "Primary study {0}".format(field))
    _require_full_commit(study.get("project_commit"), "project_commit")
    _require_full_commit(
        study.get("external_source_commit"),
        "external_source_commit",
    )
    for field in ("split_directory", "subset_directory"):
        if field not in outputs:
            raise ValueError("Primary split config outputs.{0} is missing.".format(field))
        validate_repository_relative_path(str(outputs[field]))
    return dict(payload)


def _load_bound_plot_inputs(
    config: Mapping[str, Any],
    config_file: Path,
    repository_root: Path,
) -> Dict[str, Any]:
    inputs = config["inputs"]
    provenance = config["provenance"]
    primary_config_path = _resolve_repository_path(
        repository_root,
        str(inputs["primary_split_config_path"]),
    )
    _require_public_regular_file(primary_config_path, repository_root)
    primary_config_sha256 = hash_file_bytes(primary_config_path)
    if primary_config_sha256 != inputs["primary_split_config_sha256"]:
        raise ValueError("Primary split config fingerprint mismatch.")
    primary_config = _load_primary_split_config(primary_config_path)
    if primary_config["study"]["identifier"] != config["study"]["identifier"]:
        raise ValueError("Primary split study identifier differs from plot config.")
    if (
        primary_config["study"]["dataset_identifier"]
        != config["study"]["dataset_identifier"]
    ):
        raise ValueError("Primary split dataset identifier differs from plot config.")
    if (
        primary_config["study"]["external_source_commit"]
        != provenance["external_source_commit"]
    ):
        raise ValueError("Primary split external source commit differs.")
    if (
        primary_config["study"]["project_commit"]
        != provenance["source_foundation_commit"]
    ):
        raise ValueError("Primary split source-foundation commit differs.")

    split_manifest_path = _resolve_repository_path(
        repository_root,
        str(inputs["primary_split_manifest_path"]),
    )
    subset_manifest_path = _resolve_repository_path(
        repository_root,
        str(inputs["subset_set_manifest_path"]),
    )
    _require_public_regular_file(split_manifest_path, repository_root)
    _require_public_regular_file(subset_manifest_path, repository_root)
    split_manifest = _load_hashed_manifest(
        split_manifest_path,
        SPLIT_MANIFEST_SCHEMA_VERSION,
    )
    subset_manifest = _load_hashed_manifest(
        subset_manifest_path,
        SUBSET_MANIFEST_SCHEMA_VERSION,
    )

    if hash_file_bytes(split_manifest_path) != inputs[
        "primary_split_manifest_file_sha256"
    ]:
        raise ValueError("Primary split manifest file fingerprint mismatch.")
    if hash_file_bytes(subset_manifest_path) != inputs[
        "subset_set_manifest_file_sha256"
    ]:
        raise ValueError("Subset-set manifest file fingerprint mismatch.")
    if split_manifest["manifest_hash"] != inputs["primary_split_manifest_hash"]:
        raise ValueError("Primary split manifest identity differs from plot config.")
    if subset_manifest["manifest_hash"] != inputs["subset_set_manifest_hash"]:
        raise ValueError("Subset-set manifest identity differs from plot config.")
    if split_manifest.get("split_identity_hash") != inputs["primary_split_id"]:
        raise ValueError("Primary split identity differs from plot config.")
    if subset_manifest["manifest_hash"] != inputs["subset_set_id"]:
        raise ValueError("Subset-set identity differs from plot config.")
    if subset_manifest.get("split_manifest_hash") != split_manifest["manifest_hash"]:
        raise ValueError("Subset manifest does not bind the split manifest identity.")
    if subset_manifest.get("split_identity_hash") != split_manifest.get(
        "split_identity_hash"
    ):
        raise ValueError("Subset manifest does not bind the primary split identity.")

    for manifest, label in (
        (split_manifest, "Primary split"),
        (subset_manifest, "Subset-set"),
    ):
        if manifest.get("study_identifier") != config["study"]["identifier"]:
            raise ValueError("{0} study identifier differs.".format(label))
        if (
            manifest.get("dataset_identifier")
            != config["study"]["dataset_identifier"]
        ):
            raise ValueError("{0} dataset identifier differs.".format(label))
        if manifest.get("config_path") != inputs["primary_split_config_path"]:
            raise ValueError("{0} primary config path differs.".format(label))
        if manifest.get("config_sha256") != inputs["primary_split_config_sha256"]:
            raise ValueError("{0} primary config fingerprint differs.".format(label))

    split_directory = str(split_manifest["split_directory"])
    subset_directory = str(subset_manifest["subset_directory"])
    if primary_config["outputs"]["split_directory"] != split_directory:
        raise ValueError("Primary split directory binding differs.")
    if primary_config["outputs"]["subset_directory"] != subset_directory:
        raise ValueError("Subset directory binding differs.")
    if repository_relative_path(split_manifest_path.parent, repository_root) != split_directory:
        raise ValueError("Primary split manifest is outside its bound directory.")
    if repository_relative_path(subset_manifest_path.parent, repository_root) != subset_directory:
        raise ValueError("Subset-set manifest is outside its bound directory.")

    expected_public_paths = {
        "count_summary_path": Path(split_directory, COUNT_INPUT_FILENAME).as_posix(),
        "affinity_histogram_path": Path(
            split_directory,
            AFFINITY_INPUT_FILENAME,
        ).as_posix(),
        "leakage_audit_path": Path(
            split_directory,
            LEAKAGE_INPUT_FILENAME,
        ).as_posix(),
        "subset_levels_path": Path(
            subset_directory,
            SUBSET_INPUT_FILENAME,
        ).as_posix(),
    }
    for field, expected_path in expected_public_paths.items():
        if inputs[field] != expected_path:
            raise ValueError("Configured public plot input path differs: {0}.".format(field))

    count_path = _resolve_repository_path(repository_root, inputs["count_summary_path"])
    affinity_path = _resolve_repository_path(
        repository_root,
        inputs["affinity_histogram_path"],
    )
    leakage_path = _resolve_repository_path(
        repository_root,
        inputs["leakage_audit_path"],
    )
    subset_path = _resolve_repository_path(repository_root, inputs["subset_levels_path"])
    for public_input_path in (count_path, affinity_path, leakage_path, subset_path):
        _require_public_regular_file(public_input_path, repository_root)
    for split_input_path in (count_path, affinity_path, leakage_path):
        _validate_manifest_artifact(split_manifest, split_input_path, repository_root)
    _validate_manifest_artifact(subset_manifest, subset_path, repository_root)

    input_paths = (
        count_path,
        affinity_path,
        leakage_path,
        subset_path,
        split_manifest_path,
        subset_manifest_path,
    )
    return {
        "primary_config_path": primary_config_path,
        "primary_config": primary_config,
        "primary_config_sha256": primary_config_sha256,
        "split_manifest_path": split_manifest_path,
        "split_manifest": split_manifest,
        "subset_manifest_path": subset_manifest_path,
        "subset_manifest": subset_manifest,
        "count_path": count_path,
        "affinity_path": affinity_path,
        "leakage_path": leakage_path,
        "subset_path": subset_path,
        "input_paths": input_paths,
        "plot_config_path": config_file,
    }


def _require_exact_fields(
    payload: Mapping[str, Any],
    expected_fields: Sequence[str],
    label: str,
) -> None:
    if any(not isinstance(field, str) for field in payload):
        raise ValueError("{0} field names must be strings.".format(label))
    observed = set(payload)
    expected = set(expected_fields)
    if observed != expected:
        missing = sorted(expected.difference(observed))
        extra = sorted(observed.difference(expected))
        message = "{0} fields differ; missing={1}, extra={2}."
        raise ValueError(message.format(label, missing, extra))


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("{0} must be a nonempty string.".format(label))
    return value


def _require_full_commit(value: Any, label: str) -> str:
    if not isinstance(value, str) or COMMIT_PATTERN.fullmatch(value) is None:
        message = "{0} must be a full 40-character lowercase Git commit."
        raise ValueError(message.format(label))
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError("{0} must be a lowercase SHA-256 digest.".format(label))
    return value


def _resolve_repository_file(
    repository_root: Path,
    path: Path | str,
) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repository_root / candidate
    normalized = Path(os.path.abspath(candidate))
    try:
        normalized.relative_to(repository_root)
    except ValueError as error:
        raise ValueError("Path is outside the repository root.") from error
    _reject_symlink_components(normalized, repository_root)
    return normalized


def _run_git(
    repository_root: Path,
    arguments: Sequence[str],
) -> subprocess.CompletedProcess[bytes]:
    command = ["git", "-C", str(repository_root)]
    command.extend(arguments)
    return subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _git_text(
    repository_root: Path,
    arguments: Sequence[str],
    description: str,
) -> str:
    result = _run_git(repository_root, arguments)
    if result.returncode != 0:
        raise ValueError("Unable to {0}.".format(description))
    try:
        output = result.stdout.decode("ascii").strip()
    except UnicodeDecodeError as error:
        raise ValueError("Git returned non-ASCII output for {0}.".format(description)) from error
    return output


def _resolve_runtime_head(repository_root: Path) -> str:
    runtime_head = _git_text(
        repository_root,
        ("rev-parse", "--verify", "HEAD^{commit}"),
        "resolve runtime Git HEAD",
    )
    return _require_full_commit(runtime_head, "Runtime Git HEAD")


def _require_clean_tracked_worktree(repository_root: Path) -> None:
    staged = _run_git(
        repository_root,
        ("diff", "--cached", "--quiet", "--no-ext-diff", "--ignore-submodules=none"),
    )
    if staged.returncode == 1:
        raise ValueError("Staged tracked changes are not allowed during generation.")
    if staged.returncode != 0:
        raise ValueError("Unable to inspect staged tracked changes.")
    unstaged = _run_git(
        repository_root,
        ("diff", "--quiet", "--no-ext-diff", "--ignore-submodules=none"),
    )
    if unstaged.returncode == 1:
        raise ValueError("Unstaged tracked changes are not allowed during generation.")
    if unstaged.returncode != 0:
        raise ValueError("Unable to inspect unstaged tracked changes.")


def _require_local_commit(repository_root: Path, commit: str, label: str) -> None:
    _require_full_commit(commit, label)
    resolved = _git_text(
        repository_root,
        ("rev-parse", "--verify", "{0}^{{commit}}".format(commit)),
        "resolve {0}".format(label),
    )
    if resolved != commit:
        raise ValueError("{0} does not resolve to its pinned commit.".format(label))


def _require_ancestor(
    repository_root: Path,
    ancestor: str,
    descendant: str,
    label: str,
) -> None:
    result = _run_git(
        repository_root,
        ("merge-base", "--is-ancestor", ancestor, descendant),
    )
    if result.returncode == 1:
        raise ValueError("Required Git ancestry is absent: {0}.".format(label))
    if result.returncode != 0:
        raise ValueError("Unable to verify Git ancestry: {0}.".format(label))


def _verify_commit_chain(
    repository_root: Path,
    source_foundation_commit: str,
    split_pipeline_commit: str,
    plot_generator_commit: str,
) -> None:
    for commit, label in (
        (source_foundation_commit, "source-foundation commit"),
        (split_pipeline_commit, "split-pipeline commit"),
        (plot_generator_commit, "plot-generator commit"),
    ):
        _require_local_commit(repository_root, commit, label)
    _require_ancestor(
        repository_root,
        source_foundation_commit,
        split_pipeline_commit,
        "source foundation -> split pipeline",
    )
    _require_ancestor(
        repository_root,
        split_pipeline_commit,
        plot_generator_commit,
        "split pipeline -> plot generator",
    )


def _verify_generation_git_binding(
    repository_root: Path,
    config: Mapping[str, Any],
    expected_plot_generator_commit: str,
) -> str:
    expected_commit = _require_full_commit(
        expected_plot_generator_commit,
        "Expected plot-generator commit",
    )
    runtime_head = _resolve_runtime_head(repository_root)
    if runtime_head != expected_commit:
        raise ValueError("Expected plot-generator commit does not match runtime HEAD.")
    _require_clean_tracked_worktree(repository_root)
    provenance = config["provenance"]
    _verify_commit_chain(
        repository_root,
        provenance["source_foundation_commit"],
        provenance["split_pipeline_commit"],
        runtime_head,
    )
    return runtime_head


def _git_tree_path(repository_root: Path, logical_path: str) -> str:
    git_top_text = _git_text(
        repository_root,
        ("rev-parse", "--show-toplevel"),
        "resolve Git top-level",
    )
    git_top = Path(git_top_text).resolve()
    try:
        project_prefix = repository_root.resolve().relative_to(git_top)
    except ValueError as error:
        raise ValueError("Repository root is outside the Git worktree.") from error
    candidate = project_prefix / validate_repository_relative_path(logical_path)
    return candidate.as_posix()


def _git_blob_bytes(
    repository_root: Path,
    commit: str,
    logical_path: str,
) -> bytes:
    tree_path = _git_tree_path(repository_root, logical_path)
    object_specification = "{0}:{1}".format(commit, tree_path)
    result = _run_git(repository_root, ("cat-file", "blob", object_specification))
    if result.returncode != 0:
        message = "Plotting entry point is not tracked at the required commit."
        raise ValueError(message)
    return result.stdout


def _verify_tracked_entry_point(
    repository_root: Path,
    runtime_head: str,
    entry_point_path: Path,
) -> Dict[str, Any]:
    _require_public_regular_file(entry_point_path, repository_root)
    logical_path = repository_relative_path(entry_point_path, repository_root)
    blob = _git_blob_bytes(repository_root, runtime_head, logical_path)
    fingerprint = fingerprint_file(entry_point_path, logical_path).to_dict()
    blob_sha256 = hashlib.sha256(blob).hexdigest()
    if len(blob) != fingerprint["byte_size"] or blob_sha256 != fingerprint["sha256"]:
        raise ValueError("Tracked plotting entry-point blob differs from worktree bytes.")
    return fingerprint


def _recheck_generation_state(
    repository_root: Path,
    runtime_head: str,
    snapshot_paths: Sequence[Path],
    initial_snapshots: Sequence[Mapping[str, Any]],
) -> None:
    current_head = _resolve_runtime_head(repository_root)
    if current_head != runtime_head:
        raise ValueError("Runtime Git HEAD changed while plotting.")
    _require_clean_tracked_worktree(repository_root)
    for snapshot_path in snapshot_paths:
        if snapshot_path.is_symlink() or not snapshot_path.is_file():
            raise ValueError("Snapshotted plot provenance file changed while plotting.")
    try:
        current_snapshots = _fingerprint_paths(snapshot_paths, repository_root)
    except OSError as error:
        raise ValueError("Snapshotted plot provenance file changed while plotting.") from error
    if tuple(current_snapshots) != tuple(initial_snapshots):
        raise ValueError("Snapshotted plot provenance file changed while plotting.")


def _reject_protected_input_path(logical_path: str) -> None:
    path = Path(logical_path)
    if path.suffix.lower() in (".h5", ".hdf5"):
        raise ValueError("Plot inputs must not use HDF5 files.")
    for component in path.parts:
        lowered = component.lower()
        if "sealed" in lowered or lowered == "test_targets":
            raise ValueError("Plot inputs must not use a sealed-target directory.")


def _require_public_regular_file(path: Path, repository_root: Path) -> None:
    logical_path = repository_relative_path(path, repository_root)
    _reject_protected_input_path(logical_path)
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(
            "Finalized plot input must be a regular public file: {0}".format(path)
        )


def _resolve_repository_path(repository_root: Path, relative_path: str) -> Path:
    normalized = validate_repository_relative_path(relative_path)
    candidate = Path(os.path.abspath(repository_root / normalized))
    try:
        candidate.relative_to(repository_root)
    except ValueError as error:
        raise ValueError("Path is outside the repository root.") from error
    _reject_symlink_components(candidate, repository_root)
    return candidate


def _reject_symlink_components(path: Path, repository_root: Path) -> None:
    try:
        relative_path = path.relative_to(repository_root)
    except ValueError as error:
        raise ValueError("Path is outside the repository root.") from error
    current = repository_root
    for component in relative_path.parts:
        current = current / component
        if current.is_symlink():
            raise ValueError("Repository artifact paths must not contain symlinks.")


def _require_finalized_directory(path: Path, label: str) -> None:
    if not path.is_dir() or path.is_symlink():
        message = "Finalized {0} directory does not exist: {1}"
        raise FileNotFoundError(message.format(label, path))


def _require_new_plot_directory(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(
            "Refusing to overwrite existing plot directory: {0}".format(path)
        )


def _load_hashed_manifest(path: Path, expected_schema: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if not isinstance(payload, Mapping):
        raise ValueError("Finalized manifest must be a mapping.")
    if payload.get("schema_version") != expected_schema:
        raise ValueError("Unsupported finalized manifest schema.")
    validate_hashed_manifest(payload)
    if not isinstance(payload.get("policy"), Mapping):
        raise ValueError("Finalized manifest policy must be a mapping.")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, (Mapping, Sequence)) or isinstance(
        artifacts,
        (str, bytes),
    ):
        raise ValueError("Finalized manifest artifacts must be a collection.")
    return dict(payload)


def _artifact_entries(manifest: Mapping[str, Any]) -> Tuple[Mapping[str, Any], ...]:
    artifacts = manifest["artifacts"]
    if isinstance(artifacts, Mapping):
        candidates = tuple(artifacts.values())
    else:
        candidates = tuple(artifacts)
    entries = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("Manifest artifact entry must be a mapping.")
        required = ("path", "byte_size", "sha256")
        if any(field not in candidate for field in required):
            raise ValueError("Manifest artifact entry is incomplete.")
        entries.append(candidate)
    return tuple(entries)


def _validate_manifest_artifact(
    manifest: Mapping[str, Any],
    physical_path: Path,
    repository_root: Path,
) -> None:
    logical_path = repository_relative_path(physical_path, repository_root)
    matches = []
    for artifact in _artifact_entries(manifest):
        if artifact["path"] == logical_path:
            matches.append(artifact)
    if len(matches) != 1:
        message = "Manifest must bind exactly one artifact for {0}."
        raise ValueError(message.format(logical_path))
    expected = matches[0]
    observed = fingerprint_file(physical_path, logical_path).to_dict()
    normalized_expected = {
        "path": str(expected["path"]),
        "byte_size": int(expected["byte_size"]),
        "sha256": str(expected["sha256"]),
    }
    if observed != normalized_expected:
        raise ValueError(
            "Finalized artifact fingerprint mismatch: {0}".format(logical_path)
        )


def _fingerprint_paths(
    paths: Sequence[Path],
    repository_root: Path,
) -> Tuple[Dict[str, Any], ...]:
    rows = []
    for path in paths:
        logical_path = repository_relative_path(path, repository_root)
        rows.append(fingerprint_file(path, logical_path).to_dict())
    rows.sort(key=lambda row: row["path"])
    return tuple(rows)


def _read_tsv(
    path: Path,
    expected_fields: Sequence[str],
) -> Tuple[Dict[str, str], ...]:
    with open(path, "r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file, delimiter="\t")
        if tuple(reader.fieldnames or ()) != tuple(expected_fields):
            message = "Unexpected finalized TSV schema for {0}."
            raise ValueError(message.format(path.name))
        forbidden = set(reader.fieldnames or ()).intersection(
            FORBIDDEN_PLAINTEXT_FIELDS
        )
        if forbidden:
            raise ValueError("Public plot input contains plaintext target fields.")
        return tuple(dict(row) for row in reader)


def _nonnegative_integer(value: str, label: str) -> int:
    try:
        converted = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("{0} must be an integer.".format(label)) from error
    if converted < 0:
        raise ValueError("{0} must be nonnegative.".format(label))
    return converted


def _finite_float(value: str, label: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("{0} must be numeric.".format(label)) from error
    if not math.isfinite(converted):
        raise ValueError("{0} must be finite.".format(label))
    return converted


def _validate_count_rows(
    transcription_factors: Sequence[str],
    rows: Sequence[Mapping[str, str]],
) -> None:
    expected_pairs = set()
    for transcription_factor in transcription_factors:
        for split in PRIMARY_SPLITS:
            expected_pairs.add(("primary", transcription_factor, split))
        for split in PAPER_SPLITS:
            expected_pairs.add(
                ("paper_split_reproduction", transcription_factor, split)
            )
    observed_pairs = set()
    for row in rows:
        key = (row["protocol"], row["transcription_factor"], row["split"])
        if key in observed_pairs:
            raise ValueError("Duplicate count-summary row.")
        observed_pairs.add(key)
        for field in COUNT_INPUT_FIELDS[3:]:
            _nonnegative_integer(row[field], "Count summary {0}".format(field))
    if observed_pairs != expected_pairs:
        raise ValueError("Count summary protocol, TF, or split coverage mismatch.")


def _validate_affinity_rows(
    transcription_factors: Sequence[str],
    rows: Sequence[Mapping[str, str]],
) -> None:
    expected_tfs = set(transcription_factors)
    observed_pairs = set()
    indices_by_pair: Dict[Tuple[str, str], list[int]] = {}
    for row in rows:
        transcription_factor = row["transcription_factor"]
        split = row["split"]
        if transcription_factor not in expected_tfs:
            raise ValueError("Affinity histogram contains an unexpected TF.")
        if split not in AFFINITY_SPLITS:
            raise ValueError(
                "Affinity histogram may contain only training and validation rows."
            )
        pair = (transcription_factor, split)
        observed_pairs.add(pair)
        bin_index = _nonnegative_integer(row["bin_index"], "Affinity bin index")
        left = _finite_float(row["bin_left"], "Affinity bin left edge")
        right = _finite_float(row["bin_right"], "Affinity bin right edge")
        if right <= left:
            raise ValueError("Affinity histogram bins must have positive width.")
        _nonnegative_integer(
            row["logical_example_count"],
            "Affinity histogram logical-example count",
        )
        indices_by_pair.setdefault(pair, []).append(bin_index)
    expected_pairs = set()
    for transcription_factor in transcription_factors:
        for split in AFFINITY_SPLITS:
            expected_pairs.add((transcription_factor, split))
    if observed_pairs != expected_pairs:
        raise ValueError("Affinity histogram TF/split coverage mismatch.")
    for indices in indices_by_pair.values():
        ordered = sorted(indices)
        if ordered != list(range(len(ordered))):
            raise ValueError("Affinity histogram bin indices must be contiguous.")


def _validate_leakage_rows(rows: Sequence[Mapping[str, str]]) -> None:
    if not rows:
        raise ValueError("Leakage audit must not be empty.")
    seen = set()
    for row in rows:
        key = (row["comparison"], row["left_split"], row["right_split"])
        if key in seen:
            raise ValueError("Duplicate leakage-audit row.")
        seen.add(key)
        for field in LEAKAGE_INPUT_FIELDS[3:]:
            _nonnegative_integer(row[field], "Leakage {0}".format(field))


def _low_data_display_label(request_type: Any, request_value: Any) -> str:
    """Return a strict public label without consulting an internal level ID."""

    if request_type == "absolute":
        if isinstance(request_value, bool):
            raise ValueError("Absolute low-data request must be a positive integer.")
        text = str(request_value)
        if re.fullmatch(r"[1-9][0-9]*", text) is None:
            raise ValueError("Absolute low-data request must be a positive integer.")
        return "n={0}".format(int(text))
    if request_type == "fractional":
        if isinstance(request_value, bool):
            raise ValueError("Fractional low-data request must be in (0, 1].")
        try:
            fraction = float(request_value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Fractional low-data request must be in (0, 1]."
            ) from error
        if not math.isfinite(fraction) or fraction <= 0.0 or fraction > 1.0:
            raise ValueError("Fractional low-data request must be in (0, 1].")
        percentage = fraction * 100.0
        nearest_integer = round(percentage)
        if math.isclose(percentage, nearest_integer, abs_tol=1e-12):
            percentage_text = str(int(nearest_integer))
        else:
            percentage_text = "{0:.12g}".format(percentage)
        return "{0}%".format(percentage_text)
    raise ValueError("Unknown low-data request type: {0}".format(request_type))


def _validate_subset_rows(
    transcription_factors: Sequence[str],
    rows: Sequence[Mapping[str, str]],
) -> None:
    expected_tfs = set(transcription_factors)
    observed_tfs = set()
    seen = set()
    for row in rows:
        transcription_factor = row["transcription_factor"]
        if transcription_factor not in expected_tfs:
            raise ValueError("Subset levels contain an unexpected TF.")
        observed_tfs.add(transcription_factor)
        key = (transcription_factor, row["level_id"])
        if key in seen:
            raise ValueError("Duplicate subset level row.")
        seen.add(key)
        _low_data_display_label(row["request_type"], row["request_value"])
        for field in (
            "unaliased_requested_logical_example_count",
            "canonical_requested_logical_example_count",
            "actual_logical_example_count",
            "actual_rc_group_count",
            "inclusive_maximum_rank",
        ):
            _nonnegative_integer(row[field], "Subset {0}".format(field))
    if observed_tfs != expected_tfs:
        raise ValueError("Subset-level TF coverage mismatch.")


def _count_source_rows(
    transcription_factors: Sequence[str],
    rows: Sequence[Mapping[str, str]],
) -> Tuple[Dict[str, Any], ...]:
    by_key = {}
    for row in rows:
        by_key[(row["protocol"], row["transcription_factor"], row["split"])] = row
    selected = []
    for transcription_factor in transcription_factors:
        for split in PRIMARY_SPLITS:
            row = by_key[("primary", transcription_factor, split)]
            selected.append(
                {
                    "transcription_factor": transcription_factor,
                    "split": split,
                    "logical_example_count": int(row["logical_example_count"]),
                    "global_rc_group_count": int(row["global_rc_group_count"]),
                }
            )
    return tuple(selected)


def _affinity_source_rows(
    transcription_factors: Sequence[str],
    affinity_rows: Sequence[Mapping[str, str]],
    count_rows: Sequence[Mapping[str, str]],
) -> Tuple[Dict[str, Any], ...]:
    tf_order = {}
    for index, transcription_factor in enumerate(transcription_factors):
        tf_order[transcription_factor] = index
    split_order = {"training": 0, "validation": 1}
    selected = []
    for row in affinity_rows:
        selected.append(
            {
                "record_type": "affinity_histogram",
                "transcription_factor": row["transcription_factor"],
                "split": row["split"],
                "bin_index": int(row["bin_index"]),
                "bin_left": row["bin_left"],
                "bin_right": row["bin_right"],
                "logical_example_count": int(row["logical_example_count"]),
            }
        )
    selected.sort(
        key=lambda row: (
            tf_order[row["transcription_factor"]],
            split_order[row["split"]],
            row["bin_index"],
        )
    )
    count_by_key = {}
    for row in count_rows:
        count_by_key[(row["protocol"], row["transcription_factor"], row["split"])] = row
    for transcription_factor in transcription_factors:
        test_row = count_by_key[("primary", transcription_factor, "test")]
        selected.append(
            {
                "record_type": "test_count",
                "transcription_factor": transcription_factor,
                "split": "test",
                "bin_index": "",
                "bin_left": "",
                "bin_right": "",
                "logical_example_count": int(test_row["logical_example_count"]),
            }
        )
    return tuple(selected)


def _subset_source_rows(
    transcription_factors: Sequence[str],
    rows: Sequence[Mapping[str, str]],
) -> Tuple[Dict[str, Any], ...]:
    tf_order = {}
    for index, transcription_factor in enumerate(transcription_factors):
        tf_order[transcription_factor] = index
    selected = [dict(row) for row in rows]
    selected.sort(
        key=lambda row: (
            tf_order[row["transcription_factor"]],
            int(row["canonical_requested_logical_example_count"]),
            row["level_id"],
        )
    )
    return tuple(selected)


def _leakage_source_rows(
    rows: Sequence[Mapping[str, str]],
) -> Tuple[Dict[str, Any], ...]:
    selected = [dict(row) for row in rows]
    selected.sort(
        key=lambda row: (
            row["comparison"],
            row["left_split"],
            row["right_split"],
        )
    )
    return tuple(selected)


def _comparison_source_rows(
    transcription_factors: Sequence[str],
    rows: Sequence[Mapping[str, str]],
) -> Tuple[Dict[str, Any], ...]:
    by_key = {}
    for row in rows:
        by_key[(row["protocol"], row["transcription_factor"], row["split"])] = row
    selected = []
    for transcription_factor in transcription_factors:
        for protocol, splits in (
            ("paper_split_reproduction", PAPER_SPLITS),
            ("primary", PRIMARY_SPLITS),
        ):
            for split in splits:
                row = by_key[(protocol, transcription_factor, split)]
                selected.append(
                    {
                        "protocol": protocol,
                        "transcription_factor": transcription_factor,
                        "split": split,
                        "logical_example_count": int(
                            row["logical_example_count"]
                        ),
                    }
                )
    return tuple(selected)


def _expand_limits_to_major_ticks(axis: plt.Axes) -> None:
    """Keep every visible major tick label inside the figure geometry."""

    x_ticks = axis.get_xticks()
    if len(x_ticks) > 0:
        left_limit, right_limit = axis.get_xlim()
        expanded_left = min(left_limit, float(np.min(x_ticks)))
        expanded_right = max(right_limit, float(np.max(x_ticks)))
        if expanded_left != left_limit or expanded_right != right_limit:
            axis.set_xlim(expanded_left, expanded_right)
    y_ticks = axis.get_yticks()
    if len(y_ticks) > 0:
        lower_limit, upper_limit = axis.get_ylim()
        expanded_lower = min(lower_limit, float(np.min(y_ticks)))
        expanded_upper = max(upper_limit, float(np.max(y_ticks)))
        if expanded_lower != lower_limit or expanded_upper != upper_limit:
            axis.set_ylim(expanded_lower, expanded_upper)


def _plot_primary_counts(
    output_directory: Path,
    rows: Sequence[Mapping[str, Any]],
    contract: PlotContract = DEFAULT_PLOT_CONTRACT,
) -> None:
    labels = []
    for row in rows:
        if row["transcription_factor"] not in labels:
            labels.append(row["transcription_factor"])
    by_key = {}
    for row in rows:
        by_key[(row["transcription_factor"], row["split"])] = int(
            row["logical_example_count"]
        )
    positions = np.arange(len(labels))
    width = 0.25
    colors = {"training": "#3366AA", "validation": "#EE7733", "test": "#009988"}
    figure, axis = plt.subplots(figsize=(9.4, 5.4), layout="constrained")
    for split_index, split in enumerate(PRIMARY_SPLITS):
        values = [by_key[(label, split)] for label in labels]
        offset = (split_index - 1) * width
        axis.bar(
            positions + offset,
            values,
            width,
            color=colors[split],
            label=split.capitalize(),
        )
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Number of logical labeled examples")
    axis.set_title("Exd-Hox primary split counts")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_ylim(bottom=0.0)
    _expand_limits_to_major_ticks(axis)
    _save_figure_pair(figure, output_directory, contract.count_plot_stem, contract)
    plt.close(figure)


def _panel_layout(
    panel_count: int,
    figure_width: float = 10.0,
    row_height: float = 3.1,
) -> Tuple[plt.Figure, np.ndarray]:
    column_count = 1 if panel_count == 1 else 2
    row_count = int(math.ceil(panel_count / column_count))
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(figure_width, max(4.0, row_height * row_count)),
        squeeze=False,
        sharex=False,
        layout="constrained",
    )
    return figure, axes


def _plot_affinity_distributions(
    output_directory: Path,
    transcription_factors: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    contract: PlotContract = DEFAULT_PLOT_CONTRACT,
) -> None:
    figure, axes = _panel_layout(len(transcription_factors))
    colors = {"training": "#3366AA", "validation": "#EE7733"}
    flat_axes = axes.flat
    for axis, transcription_factor in zip(flat_axes, transcription_factors):
        for split in AFFINITY_SPLITS:
            selected = []
            for row in rows:
                if (
                    row["record_type"] == "affinity_histogram"
                    and row["transcription_factor"] == transcription_factor
                    and row["split"] == split
                ):
                    selected.append(row)
            left = np.asarray(
                [float(row["bin_left"]) for row in selected],
                dtype=np.float64,
            )
            right = np.asarray(
                [float(row["bin_right"]) for row in selected],
                dtype=np.float64,
            )
            counts = np.asarray(
                [int(row["logical_example_count"]) for row in selected],
                dtype=np.float64,
            )
            widths = right - left
            total = np.sum(counts)
            density = counts if total == 0 else counts / total / widths
            axis.plot(
                (left + right) / 2.0,
                density,
                color=colors[split],
                label=split.capitalize(),
                linewidth=1.5,
            )
        test_counts = []
        for row in rows:
            if (
                row["record_type"] == "test_count"
                and row["transcription_factor"] == transcription_factor
            ):
                test_counts.append(int(row["logical_example_count"]))
        axis.text(
            0.98,
            0.96,
            "Test count: {0}".format(test_counts[0]),
            transform=axis.transAxes,
            horizontalalignment="right",
            verticalalignment="top",
            fontsize=9,
        )
        axis.set_title(transcription_factor)
        axis.set_xlabel("Relative binding affinity (0–1)")
        axis.set_ylabel("Normalized density")
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(bottom=0.0)
        _expand_limits_to_major_ticks(axis)
    for unused_index in range(len(transcription_factors), axes.size):
        axes.flat[unused_index].set_visible(False)
    axes.flat[0].legend(frameon=False)
    figure.suptitle(
        "Training/validation affinity distributions; test targets sealed",
    )
    _save_figure_pair(
        figure,
        output_directory,
        contract.affinity_plot_stem,
        contract,
    )
    plt.close(figure)


def _plot_subset_counts(
    output_directory: Path,
    transcription_factors: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    contract: PlotContract = DEFAULT_PLOT_CONTRACT,
) -> None:
    figure, axes = _panel_layout(
        len(transcription_factors),
        figure_width=13.0,
        row_height=3.8,
    )
    for axis, transcription_factor in zip(axes.flat, transcription_factors):
        selected = []
        for row in rows:
            if row["transcription_factor"] == transcription_factor:
                selected.append(row)
        positions = np.arange(len(selected))
        requested = [
            int(row["canonical_requested_logical_example_count"])
            for row in selected
        ]
        actual = [int(row["actual_logical_example_count"]) for row in selected]
        axis.plot(
            positions,
            requested,
            marker="o",
            linewidth=1.2,
            color="#777777",
            label="Canonical requested count",
        )
        axis.plot(
            positions,
            actual,
            marker="s",
            linewidth=1.2,
            color="#AA4499",
            label="Actual subset count",
        )
        axis.set_xticks(
            positions,
            [
                _low_data_display_label(
                    row["request_type"],
                    row["request_value"],
                )
                for row in selected
            ],
            rotation=40,
            horizontalalignment="right",
        )
        axis.set_title(transcription_factor)
        axis.set_xlabel("Low-data training level")
        axis.set_ylabel("Number of labeled training examples")
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_ylim(bottom=0.0)
        _expand_limits_to_major_ticks(axis)
    for unused_index in range(len(transcription_factors), axes.size):
        axes.flat[unused_index].set_visible(False)
    axes.flat[0].legend(frameon=False)
    figure.suptitle(
        "{0}\n{1}".format(SUBSET_FIGURE_TITLE, SUBSET_FIGURE_CAPTION),
        fontsize=11,
    )
    _save_figure_pair(figure, output_directory, contract.subset_plot_stem, contract)
    plt.close(figure)


def _plot_leakage(
    output_directory: Path,
    rows: Sequence[Mapping[str, Any]],
    contract: PlotContract = DEFAULT_PLOT_CONTRACT,
) -> None:
    comparison_labels = {
        "paper_split_reproduction": "Supplied paper split",
        "primary": "Primary split",
    }
    labels = [
        "{0}\n{1} vs {2}".format(
            comparison_labels.get(row["comparison"], row["comparison"]),
            str(row["left_split"]).capitalize(),
            str(row["right_split"]).capitalize(),
        )
        for row in rows
    ]
    positions = np.arange(len(rows))
    width = 0.25
    series = (
        (
            "Exact-sequence overlap groups",
            "exact_sequence_overlap_group_count",
            "#CC6677",
        ),
        (
            "RC-equivalent overlap groups (includes exact)",
            "reverse_complement_equivalent_overlap_group_count",
            "#4477AA",
        ),
        (
            "RC-only overlap groups",
            "reverse_complement_only_overlap_group_count",
            "#228833",
        ),
    )
    figure, axis = plt.subplots(
        figsize=(max(10.0, 2.3 * len(rows)), 6.2),
        layout="constrained",
    )
    for series_index, (label, field, color) in enumerate(series):
        values = [int(row[field]) for row in rows]
        offset = (series_index - 1) * width
        axis.bar(positions + offset, values, width, color=color, label=label)
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Cross-split overlap groups")
    axis.set_title("Exact and reverse-complement leakage audit")
    axis.legend(frameon=False, fontsize=9)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_ylim(bottom=0.0)
    _expand_limits_to_major_ticks(axis)
    figure.suptitle(LEAKAGE_FIGURE_CAPTION, fontsize=10)
    _save_figure_pair(figure, output_directory, contract.leakage_plot_stem, contract)
    plt.close(figure)


def _plot_protocol_comparison(
    output_directory: Path,
    transcription_factors: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    contract: PlotContract = DEFAULT_PLOT_CONTRACT,
) -> None:
    by_key = {}
    for row in rows:
        by_key[(row["protocol"], row["transcription_factor"], row["split"])] = int(
            row["logical_example_count"]
        )
    positions = np.arange(len(transcription_factors))
    categories = (
        (
            "paper_split_reproduction",
            "train",
            "Supplied train (row occurrences)",
            "#88CCEE",
        ),
        (
            "paper_split_reproduction",
            "test",
            "Supplied test (row occurrences)",
            "#CC6677",
        ),
        (
            "primary",
            "training",
            "Primary training (logical labeled examples)",
            "#3366AA",
        ),
        (
            "primary",
            "validation",
            "Primary validation (logical labeled examples)",
            "#EE7733",
        ),
        (
            "primary",
            "test",
            "Primary test (logical labeled examples)",
            "#009988",
        ),
    )
    width = 0.16
    figure, axis = plt.subplots(figsize=(13.0, 6.8), layout="constrained")
    center = (len(categories) - 1) / 2.0
    for category_index, (protocol, split, label, color) in enumerate(categories):
        values = [
            by_key[(protocol, transcription_factor, split)]
            for transcription_factor in transcription_factors
        ]
        offset = (category_index - center) * width
        axis.bar(positions + offset, values, width, color=color, label=label)
    axis.set_xticks(positions, transcription_factors)
    axis.set_ylabel("Count")
    axis.set_title(COMPARISON_FIGURE_TITLE)
    axis.legend(frameon=False, ncol=2, fontsize=8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_ylim(bottom=0.0)
    _expand_limits_to_major_ticks(axis)
    figure.suptitle(COMPARISON_FIGURE_CAPTION, fontsize=9)
    _save_figure_pair(
        figure,
        output_directory,
        contract.comparison_plot_stem,
        contract,
    )
    plt.close(figure)


def _save_figure_pair(
    figure: plt.Figure,
    output_directory: Path,
    stem: str,
    contract: PlotContract = DEFAULT_PLOT_CONTRACT,
) -> None:
    png_path = output_directory / "{0}.png".format(stem)
    pdf_path = output_directory / "{0}.pdf".format(stem)
    with open(png_path, "xb") as png_file:
        figure.savefig(
            png_file,
            format="png",
            dpi=180,
            metadata={"Creator": contract.creator_metadata},
        )
    with open(pdf_path, "xb") as pdf_file:
        figure.savefig(
            pdf_file,
            format="pdf",
            metadata={
                "Creator": contract.creator_metadata,
                "Producer": "matplotlib",
                "CreationDate": None,
                "ModDate": None,
            },
        )


def _write_plot_manifest(
    config: Mapping[str, Any],
    config_file: Path,
    repository_root: Path,
    plot_logical_directory: str,
    staging_directory: Path,
    input_fingerprints: Sequence[Mapping[str, Any]],
    bindings: Mapping[str, Any],
    runtime_head: str,
    entry_point_fingerprint: Mapping[str, Any],
    contract: PlotContract,
) -> Dict[str, Any]:
    output_fingerprints = []
    for filename in contract.output_filenames:
        logical_path = Path(plot_logical_directory, filename).as_posix()
        fingerprint = fingerprint_file(
            staging_directory / filename,
            logical_path,
        )
        output_fingerprints.append(fingerprint.to_dict())
    output_fingerprints.sort(key=lambda row: row["path"])

    manifest = build_hashed_manifest(
        contract.manifest_schema_version,
        {
            "study_identifier": config["study"]["identifier"],
            "dataset_identifier": config["study"]["dataset_identifier"],
            "external_source_commit": config["provenance"][
                "external_source_commit"
            ],
            "source_foundation_commit": config["provenance"][
                "source_foundation_commit"
            ],
            "split_pipeline_commit": config["provenance"][
                "split_pipeline_commit"
            ],
            "plot_generator_commit": runtime_head,
            "plot_generator_tracked_worktree_clean": True,
            "plotting_entry_point_path": entry_point_fingerprint["path"],
            "plotting_entry_point_byte_size": entry_point_fingerprint[
                "byte_size"
            ],
            "plotting_entry_point_sha256": entry_point_fingerprint["sha256"],
            "plot_config_path": repository_relative_path(
                config_file,
                repository_root,
            ),
            "plot_config_sha256": hash_file_bytes(config_file),
            "primary_split_config_path": repository_relative_path(
                bindings["primary_config_path"],
                repository_root,
            ),
            "primary_split_config_sha256": bindings["primary_config_sha256"],
            "primary_split_id": config["inputs"]["primary_split_id"],
            "primary_split_manifest_path": repository_relative_path(
                bindings["split_manifest_path"],
                repository_root,
            ),
            "primary_split_manifest_hash": bindings["split_manifest"][
                "manifest_hash"
            ],
            "primary_split_manifest_file_sha256": hash_file_bytes(
                bindings["split_manifest_path"]
            ),
            "subset_set_id": config["inputs"]["subset_set_id"],
            "subset_set_manifest_path": repository_relative_path(
                bindings["subset_manifest_path"],
                repository_root,
            ),
            "subset_set_manifest_hash": bindings["subset_manifest"][
                "manifest_hash"
            ],
            "subset_set_manifest_file_sha256": hash_file_bytes(
                bindings["subset_manifest_path"]
            ),
            "plot_directory": plot_logical_directory,
            "inputs": list(input_fingerprints),
            "outputs": output_fingerprints,
            "test_target_policy": (
                "aggregate_test_counts_only_no_test_affinity_distribution"
            ),
        },
    )
    write_json_exclusive(staging_directory / contract.manifest_filename, manifest)
    return manifest


def _load_json_mapping(path: Path, label: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if not isinstance(payload, Mapping):
        raise ValueError("{0} must be a mapping.".format(label))
    return dict(payload)


def _validate_fingerprint_collection(
    entries: Any,
    expected_paths: Sequence[str],
    repository_root: Path,
    label: str,
) -> None:
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise ValueError("{0} fingerprints must be a sequence.".format(label))
    expected = set(expected_paths)
    observed_paths = []
    normalized_entries = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("{0} fingerprint must be a mapping.".format(label))
        _require_exact_fields(
            entry,
            ("path", "byte_size", "sha256"),
            "{0} fingerprint".format(label),
        )
        logical_path = validate_repository_relative_path(str(entry["path"]))
        byte_size = entry["byte_size"]
        if isinstance(byte_size, bool) or not isinstance(byte_size, int):
            raise ValueError("{0} byte size must be an integer.".format(label))
        if byte_size < 0:
            raise ValueError("{0} byte size must be nonnegative.".format(label))
        sha256 = _require_sha256(entry["sha256"], "{0} SHA-256".format(label))
        observed_paths.append(logical_path)
        normalized_entries.append(
            {
                "path": logical_path,
                "byte_size": byte_size,
                "sha256": sha256,
            }
        )
    if observed_paths != sorted(observed_paths):
        raise ValueError("{0} fingerprints must be path-sorted.".format(label))
    if len(observed_paths) != len(set(observed_paths)):
        raise ValueError("{0} fingerprint paths must be unique.".format(label))
    if set(observed_paths) != expected:
        raise ValueError("{0} fingerprint path set differs.".format(label))
    for entry in normalized_entries:
        physical_path = _resolve_repository_path(repository_root, entry["path"])
        if physical_path.is_symlink() or not physical_path.is_file():
            raise ValueError("{0} fingerprint target must be a regular file.".format(label))
        observed = fingerprint_file(physical_path, entry["path"]).to_dict()
        if observed != entry:
            raise ValueError("{0} fingerprint mismatch: {1}".format(label, entry["path"]))


def _verify_historical_entry_point(
    repository_root: Path,
    manifest: Mapping[str, Any],
) -> None:
    blob = _git_blob_bytes(
        repository_root,
        manifest["plot_generator_commit"],
        manifest["plotting_entry_point_path"],
    )
    if len(blob) != manifest["plotting_entry_point_byte_size"]:
        raise ValueError("Historical plotting entry-point byte size differs.")
    if hashlib.sha256(blob).hexdigest() != manifest["plotting_entry_point_sha256"]:
        raise ValueError("Historical plotting entry-point fingerprint differs.")


def validate_primary_split_plot_manifest(
    manifest_path: Path | str,
    repository_root: Path | str,
    expected_plot_generator_commit: str,
) -> Dict[str, Any]:
    """Validate versioned plot bytes and the historical producer blob.

    The current checkout may be newer than, and need not be clean like, the
    explicitly expected historical generator commit.
    """

    root = Path(os.path.abspath(repository_root))
    expected_generator = _require_full_commit(
        expected_plot_generator_commit,
        "Expected historical plot-generator commit",
    )
    manifest_file = _resolve_repository_file(root, manifest_path)
    if manifest_file.is_symlink() or not manifest_file.is_file():
        raise FileNotFoundError("Plot manifest must be a regular file.")
    manifest = _load_json_mapping(manifest_file, "Plot manifest")
    contract = _plot_contract_for_manifest_schema(manifest.get("schema_version"))
    _require_exact_fields(manifest, tuple(PLOT_MANIFEST_FIELDS), "Plot manifest")
    _require_sha256(manifest.get("manifest_hash"), "Manifest hash")
    validate_hashed_manifest(manifest)

    for field in (
        "external_source_commit",
        "source_foundation_commit",
        "split_pipeline_commit",
        "plot_generator_commit",
    ):
        _require_full_commit(manifest.get(field), field)
    if manifest["plot_generator_commit"] != expected_generator:
        raise ValueError("Expected historical plot-generator commit differs.")
    if manifest["plot_generator_tracked_worktree_clean"] is not True:
        raise ValueError("Generation tracked-worktree state was not clean.")
    if manifest["plotting_entry_point_path"] != PLOTTING_ENTRY_POINT:
        raise ValueError("Plotting entry-point path differs.")
    entry_point_byte_size = manifest["plotting_entry_point_byte_size"]
    if isinstance(entry_point_byte_size, bool) or not isinstance(
        entry_point_byte_size,
        int,
    ):
        raise ValueError("Plotting entry-point byte size must be an integer.")
    if entry_point_byte_size < 0:
        raise ValueError("Plotting entry-point byte size must be nonnegative.")
    _require_sha256(
        manifest["plotting_entry_point_sha256"],
        "Plotting entry-point SHA-256",
    )

    if manifest["plot_config_path"] != contract.config_logical_path:
        raise ValueError("Plot config path differs from the contract's canonical path.")
    if manifest["plot_directory"] != contract.plot_logical_directory:
        raise ValueError("Plot directory differs from the manifest contract.")
    config_file = _resolve_repository_path(root, manifest["plot_config_path"])
    _require_public_regular_file(config_file, root)
    if hash_file_bytes(config_file) != manifest["plot_config_sha256"]:
        raise ValueError("Plot config fingerprint mismatch.")
    config = _load_plot_config(config_file, contract)
    bindings = _load_bound_plot_inputs(config, config_file, root)

    expected_scalar_fields = {
        "study_identifier": config["study"]["identifier"],
        "dataset_identifier": config["study"]["dataset_identifier"],
        "external_source_commit": config["provenance"]["external_source_commit"],
        "source_foundation_commit": config["provenance"][
            "source_foundation_commit"
        ],
        "split_pipeline_commit": config["provenance"]["split_pipeline_commit"],
        "plotting_entry_point_path": config["provenance"][
            "plotting_entry_point"
        ],
        "plot_config_path": repository_relative_path(config_file, root),
        "plot_config_sha256": hash_file_bytes(config_file),
        "primary_split_config_path": repository_relative_path(
            bindings["primary_config_path"],
            root,
        ),
        "primary_split_config_sha256": bindings["primary_config_sha256"],
        "primary_split_id": config["inputs"]["primary_split_id"],
        "primary_split_manifest_path": repository_relative_path(
            bindings["split_manifest_path"],
            root,
        ),
        "primary_split_manifest_hash": bindings["split_manifest"][
            "manifest_hash"
        ],
        "primary_split_manifest_file_sha256": hash_file_bytes(
            bindings["split_manifest_path"]
        ),
        "subset_set_id": config["inputs"]["subset_set_id"],
        "subset_set_manifest_path": repository_relative_path(
            bindings["subset_manifest_path"],
            root,
        ),
        "subset_set_manifest_hash": bindings["subset_manifest"]["manifest_hash"],
        "subset_set_manifest_file_sha256": hash_file_bytes(
            bindings["subset_manifest_path"]
        ),
        "plot_directory": config["outputs"]["plot_directory"],
        "test_target_policy": (
            "aggregate_test_counts_only_no_test_affinity_distribution"
        ),
    }
    for field, expected_value in expected_scalar_fields.items():
        if manifest[field] != expected_value:
            raise ValueError("Plot manifest {0} differs from bound config.".format(field))

    _verify_commit_chain(
        root,
        manifest["source_foundation_commit"],
        manifest["split_pipeline_commit"],
        manifest["plot_generator_commit"],
    )
    _verify_historical_entry_point(root, manifest)

    expected_input_paths = []
    for input_path in bindings["input_paths"]:
        expected_input_paths.append(repository_relative_path(input_path, root))
    _validate_fingerprint_collection(
        manifest["inputs"],
        expected_input_paths,
        root,
        "Input",
    )

    plot_directory = _resolve_repository_path(root, manifest["plot_directory"])
    if plot_directory.is_symlink() or not plot_directory.is_dir():
        raise ValueError("Plot directory must be a regular directory.")
    expected_output_paths = []
    for filename in contract.output_filenames:
        expected_output_paths.append(
            Path(manifest["plot_directory"], filename).as_posix()
        )
    _validate_fingerprint_collection(
        manifest["outputs"],
        expected_output_paths,
        root,
        "Output",
    )
    expected_directory_names = set(contract.output_filenames)
    expected_directory_names.add(contract.manifest_filename)
    observed_directory_names = set()
    for path in plot_directory.iterdir():
        observed_directory_names.add(path.name)
    if observed_directory_names != expected_directory_names:
        raise ValueError(
            "Plot directory contains missing, extra, or non-contract files."
        )
    expected_manifest_path = plot_directory / contract.manifest_filename
    if manifest_file.resolve() != expected_manifest_path.resolve():
        raise ValueError("Plot manifest is relocated from its canonical directory.")
    return manifest


def main(argv=None):
    """Generate immutable v3 plots or validate a supported manifest."""

    arguments = parse_arguments(argv)
    if arguments.validate_manifest:
        manifest = validate_primary_split_plot_manifest(
            manifest_path=arguments.validate_manifest,
            repository_root=arguments.repository_root,
            expected_plot_generator_commit=(
                arguments.expected_plot_generator_commit
            ),
        )
    else:
        manifest = plot_primary_split_tables(
            config_path=arguments.config,
            repository_root=arguments.repository_root,
            expected_plot_generator_commit=(
                arguments.expected_plot_generator_commit
            ),
        )
    print(
        json.dumps(
            manifest,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
    )
    return manifest


if __name__ == "__main__":
    main()
