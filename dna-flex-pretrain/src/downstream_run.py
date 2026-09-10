"""B3a scientific identities and deterministic primitives; no training loop.

B3b must freeze the production source inventory. This module verifies supplied
inventories without assuming that future training modules or CLIs exist.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Any, Sequence

from src.cnn_rc import CONTRACT_ID, architecture_settings
from src.downstream_fingerprints import canonical_json_bytes
from src.exd_hox_dataset import ExdHoxDataset, TF_NAMES, open_public_tf_data


class RunContractError(ValueError):
    """A scientific or provenance contract is invalid."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RunContractError(message)


def keys(value: Any, expected: Sequence[str], name: str) -> None:
    require(type(value) is dict and set(value) == set(expected), name + " fields differ.")


def integer(value: Any, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def hex_digest(value: Any, length: int = 64) -> bool:
    return type(value) is str and re.fullmatch("[0-9a-f]{" + str(length) + "}", value) is not None


def identifier(value: Any, prefix: str) -> bool:
    return type(value) is str and value.startswith(prefix) and hex_digest(value[len(prefix):])


def strict_json(data: bytes) -> dict[str, Any]:
    """Reject duplicate keys, nonfinite constants and non-object roots."""
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key.")
            result[key] = value
        return result

    def invalid(value):
        raise RunContractError("Nonfinite JSON constant: " + value)

    try:
        result = json.loads(data.decode("utf-8"), object_pairs_hook=pairs, parse_constant=invalid)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RunContractError("Malformed JSON.") from error
    require(type(result) is dict, "JSON root must be an object.")
    _plain_json(result)
    return result


def _plain_json(value: Any) -> None:
    if type(value) is dict:
        for key, item in value.items():
            require(type(key) is str, "JSON keys must be strings.")
            _plain_json(item)
    elif type(value) is list:
        for item in value:
            _plain_json(item)
    else:
        require(type(value) in (str, int, float, bool, type(None)), "Unsupported JSON type.")
        if type(value) is float:
            require(math.isfinite(value), "Nonfinite JSON value.")


def domain_hash(domain: str, value: Any) -> str:
    """H(domain, value) uses the approved canonical JSON encoding."""
    _plain_json(value)
    require(type(domain) is str and bool(domain) and "\0" not in domain, "Invalid hash domain.")
    return hashlib.sha256(domain.encode("utf-8") + b"\0" + canonical_json_bytes(value)).hexdigest()


def default_config() -> dict[str, Any]:
    """Return the approved provisional smoke contract, independent of files."""
    return {
        "schema_version": "exd_hox_cnn_rc_training_config.v1",
        "study_identifier": "exd_hox_primary_low_data.v1", "model_family": "cnn_rc",
        "model_contract": CONTRACT_ID,
        "data": {
            "dataset_identifier": "wang_etal_exd_hox_selex_canonical.v1",
            "split_identity_hash": "a684fd4fd863709d4a59e8925a2f76d95255e0f33a9996216fd896ce098c393f",
            "split_manifest_hash": "fb595729defc1f140637f0a75d2beb78694a0a36e0fa04446727483bc121e564",
            "subset_set_manifest_hash": "ce75331e6bf5db939ff70df06a1cda07028e31664ef7a8d579c9363d00fc4125",
            "staging_policy_identifier": "exd_hox_public_training_stage.v1",
            "staging_config_sha256": "50a7ba5ae1bd7219b20cdc71b511dfbce8b38855a1b065ac4777eca71a18833b",
        },
        "supported_tfs": list(TF_NAMES),
        "supported_downstream_seeds": [33001, 33002, 33003, 33004, 33005],
        "seeds": {"policy": "downstream_named_sha256.v1"},
        "orientation": {
            "training": "logical_example_sha256_orientation.v1",
            "validation": "forward_rc_mean_with_diagnostic.v1",
            "cnn_rc_atol": 1e-6, "cnn_rc_rtol": 1e-5,
        },
        "batching": {
            "order_policy": "epoch_sha256_sort.v1", "num_workers": 0,
            "drop_last": False, "persistent_workers": False, "persistent_cache": False,
            "gradient_accumulation_steps": 1,
        },
        "numerics": {
            "precision": "float32", "amp": False, "compile": False,
            "deterministic_algorithms": True, "deterministic_warn_only": False,
            "cudnn_benchmark": False, "cudnn_deterministic": True, "allow_tf32": False,
            "float32_matmul_precision": "highest", "cublas_workspace_config": ":4096:8",
        },
        "candidates": {"smoke_adam_v1": {
            "status": "provisional_smoke_candidate",
            "loss": "batch_mean_mse_plus_independent_convolution_penalty.v1",
            "batch_size": 128, "regularization": {"l1": 5e-6, "l2": 1e-5},
            "optimizer": {
                "name": "Adam", "learning_rate": 5e-5, "betas": [0.9, 0.999],
                "epsilon": 1e-8, "weight_decay": 0.0, "amsgrad": False,
                "foreach": False, "fused": False, "maximize": False,
                "capturable": False, "differentiable": False,
            },
            "scheduler": None, "budget": {"maximum_epochs": 2},
            "validation": {"every_epochs": 1, "batch_size": 128,
                           "initial_evaluation": False, "selection_evaluations_maximum": 2},
            "checkpoint": {"every_updates": 100, "before_epoch_validation": True,
                           "after_epoch_validation": True},
            "selection": {"policy": "maximum_r2_minimum_rmse_earliest_update.v1",
                          "r2_tolerance": 0.0, "rmse_tolerance": 0.0},
            "early_stopping": {"enabled": False},
        }},
        "artifact_schemas": {
            "run": "downstream_resolved_run.v1", "attempt": "downstream_execution_attempt.v1",
            "checkpoint": "downstream_checkpoint.v1", "checkpoint_index": "downstream_checkpoint_index.v1",
            "epoch_metrics": "cnn_rc_epoch_metrics.v1", "validation_summary": "cnn_rc_validation_summary.v1",
            "completion": "downstream_completion.v1",
        },
    }


def _same_shape(value: Any, template: Any) -> None:
    require(type(value) is type(template), "Configuration field type differs.")
    if type(template) is dict:
        keys(value, tuple(template), "Configuration")
        for key in template:
            _same_shape(value[key], template[key])
    elif type(template) is list:
        require(len(value) == len(template), "Configuration list length differs.")
        for actual, expected in zip(value, template):
            _same_shape(actual, expected)


def validate_config(config: Any) -> None:
    """v1 accepts precisely the approved smoke choices; changes need approval."""
    expected = default_config()
    _same_shape(config, expected)
    _plain_json(config)
    require(config == expected, "Unsupported scientific configuration value.")


def load_config(path: Path) -> dict[str, Any]:
    config = strict_json(read_regular(path))
    validate_config(config)
    return config


def logical_path(value: Any) -> str:
    require(type(value) is str and bool(value), "Invalid source path.")
    require(not value.startswith("/") and "\\" not in value, "Nonrelative source path.")
    require(all(part not in ("", ".", "..") for part in value.split("/")), "Escaping source path.")
    require(all(ord(char) >= 32 for char in value), "Control character in path.")
    return value


def read_regular(path: Path) -> bytes:
    """Read a regular file without following symlinks, including ancestors."""
    absolute = Path(os.path.abspath(path))
    parent = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in absolute.parts[1:-1]:
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            os.close(parent)
            parent = child
        descriptor = os.open(absolute.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        try:
            before = os.fstat(descriptor)
            require(stat.S_ISREG(before.st_mode), "Not a regular file.")
            chunks = []
            block = os.read(descriptor, 1024 * 1024)
            while block:
                chunks.append(block)
                block = os.read(descriptor, 1024 * 1024)
            after = os.fstat(descriptor)
            current = os.stat(absolute.name, dir_fd=parent, follow_symlinks=False)
            fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
            require(all(getattr(before, field) == getattr(after, field) == getattr(current, field)
                        for field in fields), "File changed while reading.")
            return b"".join(chunks)
        finally:
            os.close(descriptor)
    finally:
        os.close(parent)


def _git(root: Path, *arguments: str) -> bytes:
    result = subprocess.run(["git", "-C", str(root), *arguments], capture_output=True, check=False)
    require(result.returncode == 0, "Git evidence unavailable or invalid.")
    return result.stdout


def validate_software(software: Any) -> None:
    keys(software, ("runtime_commit", "source_inventory"), "Software")
    require(hex_digest(software["runtime_commit"], 40), "Invalid software commit.")
    inventory = software["source_inventory"]
    require(type(inventory) is list and bool(inventory), "Empty source inventory.")
    paths = []
    for record in inventory:
        keys(record, ("path", "git_blob", "sha256", "byte_size"), "Source record")
        paths.append(logical_path(record["path"]))
        require(hex_digest(record["git_blob"], 40) and hex_digest(record["sha256"])
                and integer(record["byte_size"]), "Malformed source fingerprint.")
    require(paths == sorted(set(paths)), "Source paths must be sorted and unique.")


def _source_record(root: Path, commit: str, path: str) -> dict[str, Any]:
    prefix = _git(root, "rev-parse", "--show-prefix").decode().strip()
    tree_path = prefix + logical_path(path)
    # ls-tree paths are root-relative when -C points at a subdirectory only
    # with --full-tree; explicitly request it to avoid prefix ambiguity.
    entry = _git(root, "ls-tree", "--full-tree", "-z", commit, "--", tree_path)
    require(bool(entry), "Source is not tracked at commit.")
    fields = entry.rstrip(b"\0").split(b"\t")
    require(len(fields) == 2 and fields[1].decode() == tree_path, "Source tree path differs.")
    mode, kind, blob = fields[0].decode().split(" ")
    require(mode in ("100644", "100755") and kind == "blob", "Source must be a regular tracked blob.")
    content = _git(root, "cat-file", "blob", blob)
    return {"path": path, "git_blob": blob, "sha256": hashlib.sha256(content).hexdigest(),
            "byte_size": len(content)}


def verify_historical_sources(root: Path, software: dict[str, Any]) -> None:
    """Check recorded commit objects without consulting current checkout bytes."""
    validate_software(software)
    for record in software["source_inventory"]:
        require(_source_record(root, software["runtime_commit"], record["path"]) == record,
                "Historical source blob mismatch.")


def verify_runtime_sources(
    root: Path, expected_commit: str, source_paths: Sequence[str],
    *, executing_paths: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """Verify HEAD, tracked cleanliness and explicit inventory bytes.

    B3b must supply its frozen complete inventory and actual imported module
    paths. B3a deliberately has no production inventory or future-file list.
    """
    require(hex_digest(expected_commit, 40), "Expected commit must be a full SHA-1.")
    require(_git(root, "rev-parse", "HEAD").decode().strip() == expected_commit, "Wrong runtime HEAD.")
    top = Path(_git(root, "rev-parse", "--show-toplevel").decode().strip())
    require(not _git(top, "status", "--porcelain=v1", "--untracked-files=no"), "Dirty tracked state.")
    require(type(source_paths) in (tuple, list) and bool(source_paths), "Missing source inventory.")
    require(len(set(source_paths)) == len(source_paths), "Duplicate source paths.")
    inventory = []
    for path in sorted(source_paths):
        record = _source_record(root, expected_commit, logical_path(path))
        content = read_regular(root / path)
        require(len(content) == record["byte_size"] and hashlib.sha256(content).hexdigest() == record["sha256"],
                "Executing source bytes differ from commit.")
        inventory.append(record)
    if executing_paths is not None:
        require(type(executing_paths) is dict and set(executing_paths).issubset(source_paths),
                "Executing module absent from source inventory.")
        for name, physical in executing_paths.items():
            require(Path(os.path.abspath(physical)) == Path(os.path.abspath(root / name)), "Shadowed project import.")
            read_regular(physical)
    require(_git(root, "rev-parse", "HEAD").decode().strip() == expected_commit
            and not _git(top, "status", "--porcelain=v1", "--untracked-files=no"), "Checkout changed.")
    result = {"runtime_commit": expected_commit, "source_inventory": inventory}
    validate_software(result)
    return result


COMPONENTS = (
    "model_initialization", "training_data_order", "training_orientation", "dataloader_worker",
    "dropout", "checkpoint_verification", "synthetic_fixture", "python_runtime", "numpy_runtime", "torch_runtime",
)


def named_seed(parent_seed: int, component: str) -> int:
    require(integer(parent_seed) and parent_seed < 2**63 and component in COMPONENTS, "Invalid seed component.")
    digest = domain_hash("downstream_named_seed.v1", {"parent_seed": parent_seed, "component": component})
    return int(digest[:16], 16) % 2**63


def seed_record(parent_seed: int) -> dict[str, Any]:
    require(parent_seed in (33001, 33002, 33003, 33004, 33005) and type(parent_seed) is int,
            "Unsupported downstream seed.")
    derived = {}
    for component in COMPONENTS:
        derived[component] = named_seed(parent_seed, component)
    return {"policy": "downstream_named_sha256.v1", "parent_seed": parent_seed, "derived_seeds": derived}


def worker_seed(worker_parent_seed: int, epoch: int, worker_index: int) -> int:
    require(integer(worker_parent_seed) and worker_parent_seed < 2**63
            and integer(epoch) and integer(worker_index), "Invalid worker seed input.")
    digest = domain_hash("downstream_worker_seed.v1", {
        "worker_seed": worker_parent_seed, "epoch": epoch, "worker_index": worker_index})
    return int(digest[:16], 16) % 2**63


def training_orientation(
    orientation_seed: int, training_membership_hash: str, epoch: int, logical_example_id: str,
) -> bool:
    """True means RC; decisions are paired across model families and aliases."""
    require(integer(orientation_seed) and orientation_seed < 2**63 and integer(epoch)
            and hex_digest(training_membership_hash) and identifier(logical_example_id, "lex_"),
            "Invalid orientation input.")
    digest = domain_hash("downstream_training_orientation.v1", {
        "orientation_seed": orientation_seed, "training_membership_hash": training_membership_hash,
        "epoch": epoch, "logical_example_id": logical_example_id,
    })
    return int(digest, 16) % 2 == 1


def epoch_order(logical_ids: Sequence[str], data_order_seed: int, epoch: int) -> tuple[int, ...]:
    require(type(logical_ids) in (tuple, list) and bool(logical_ids), "Empty membership.")
    require(all(identifier(value, "lex_") for value in logical_ids)
            and len(set(logical_ids)) == len(logical_ids), "Invalid membership IDs.")
    require(integer(data_order_seed) and data_order_seed < 2**63 and integer(epoch), "Invalid order inputs.")
    decorated = []
    for index, logical_id in enumerate(logical_ids):
        digest = domain_hash("downstream_epoch_order.v1", {
            "data_order_seed": data_order_seed, "epoch": epoch, "logical_example_id": logical_id})
        decorated.append((digest, logical_id, index))
    return tuple(item[2] for item in sorted(decorated))


def permutation_hash(logical_ids: Sequence[str], permutation: Sequence[int]) -> str:
    require(sorted(permutation) == list(range(len(logical_ids)))
            and all(type(index) is int for index in permutation), "Invalid permutation.")
    return domain_hash("downstream_epoch_permutation.v1", [logical_ids[index] for index in permutation])


def remaining_batches(permutation: Sequence[int], batch_size: int, next_batch_index: int = 0) -> tuple:
    require(integer(batch_size, 1) and integer(next_batch_index), "Invalid batch position.")
    require(bool(permutation) and sorted(permutation) == list(range(len(permutation)))
            and all(type(index) is int for index in permutation), "Invalid permutation.")
    count = (len(permutation) + batch_size - 1) // batch_size
    require(next_batch_index <= count, "Batch position past epoch.")
    return tuple(tuple(permutation[index * batch_size:(index + 1) * batch_size])
                 for index in range(next_batch_index, count))


def membership_hash(dataset: ExdHoxDataset) -> str:
    records = []
    for index in range(len(dataset)):
        metadata = dataset[index].metadata
        records.append({"logical_example_id": metadata.logical_example_id,
                        "global_rc_group_id": metadata.global_rc_group_id,
                        "target_float32_bits": metadata.target_bits_big_endian_hex})
    return domain_hash("downstream_membership.v1", records)


def _assemble_run(config, data, level_id, downstream_seed, candidate_id, software):
    """Pure assembly after public-stage and runtime provenance verification."""
    validate_config(config)
    validate_software(software)
    require(candidate_id in config["candidates"], "Unknown candidate.")
    training = data.dataset("training", level_id=level_id)
    validation = data.dataset("validation")
    level = next(item for item in data.levels() if item.level_id == level_id)
    metadata = training[0].metadata
    validation_metadata = validation[0].metadata
    for name in ("split_identity_hash", "split_manifest_hash", "subset_set_manifest_hash"):
        require(getattr(metadata, name) == config["data"][name], "Stage differs from scientific configuration.")
    candidate = copy.deepcopy(config["candidates"][candidate_id])
    common = copy.deepcopy(config)
    del common["candidates"]
    count = len(training)
    batches = (count + candidate["batch_size"] - 1) // candidate["batch_size"]
    selection = {
        "transcription_factor": data.transcription_factor,
        "requested_level_id": level.level_id, "canonical_level_id": level.canonical_level_id,
        "request_type": level.request_type, "request_value": level.request_value,
        "unaliased_requested_logical_example_count": level.unaliased_requested_logical_example_count,
        "canonical_requested_logical_example_count": level.canonical_requested_logical_example_count,
        "actual_logical_example_count": count, "actual_rc_group_count": training.unique_rc_group_count,
        "inclusive_maximum_rank": level.inclusive_maximum_rank,
        "training_membership_hash": membership_hash(training), "validation_id": validation_metadata.validation_id,
        "validation_membership_hash": membership_hash(validation), "validation_row_count": len(validation),
        "validation_rc_group_count": validation.unique_rc_group_count,
    }
    identity = {
        "configuration": {"schema_version": config["schema_version"],
                          "canonical_config_hash": domain_hash("downstream_config.v1", config),
                          "resolved_contract": common, "candidate_id": candidate_id,
                          "resolved_candidate": candidate},
        "model": {"family": config["model_family"], "contract_id": CONTRACT_ID,
                  "architecture_settings": architecture_settings(),
                  "architecture_hash": domain_hash("downstream_architecture.v1", architecture_settings())},
        "data": {**copy.deepcopy(config["data"]), "stage_id": data.stage_id, "split_seed": 31001, "subset_seed": 32001},
        "selection": selection, "seeds": seed_record(downstream_seed),
        "budget": {"maximum_epochs": candidate["budget"]["maximum_epochs"], "batches_per_epoch": batches,
                   "maximum_updates": batches * candidate["budget"]["maximum_epochs"],
                   "maximum_selection_evaluations": candidate["validation"]["selection_evaluations_maximum"]},
        "software": copy.deepcopy(software),
    }
    result = {"schema_version": "downstream_resolved_run.v1", "run_id": "run_" + domain_hash("downstream_scientific_run.v1", identity),
              "identity": identity}
    result["manifest_hash"] = domain_hash("downstream_resolved_run_manifest.v1", result)
    validate_run(result)
    return result


def resolve_run(
    *, config_path: Path, stage_root: Path, transcription_factor: str, level_id: str,
    expected_stage_id: str, downstream_seed: int, candidate_id: str,
    checkout_root: Path, expected_software_commit: str, source_paths: Sequence[str],
    executing_paths: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """Resolve one run using only verified code and B2's public constructor."""
    relative = Path(os.path.abspath(config_path)).relative_to(Path(os.path.abspath(checkout_root))).as_posix()
    require(relative in source_paths, "Tracked config absent from source inventory.")
    software = verify_runtime_sources(checkout_root, expected_software_commit, source_paths, executing_paths=executing_paths)
    config = load_config(config_path)
    data = open_public_tf_data(stage_root, transcription_factor=transcription_factor, expected_stage_id=expected_stage_id)
    result = _assemble_run(config, data, level_id, downstream_seed, candidate_id, software)
    require(verify_runtime_sources(checkout_root, expected_software_commit, source_paths,
                                   executing_paths=executing_paths) == software, "Runtime provenance changed.")
    return result


def validate_run(run: Any) -> None:
    """Validate historical run structure and all internally derivable values."""
    keys(run, ("schema_version", "run_id", "identity", "manifest_hash"), "Resolved run")
    require(run["schema_version"] == "downstream_resolved_run.v1", "Unsupported run schema.")
    identity = run["identity"]
    keys(identity, ("configuration", "model", "data", "selection", "seeds", "budget", "software"), "Run identity")
    configuration = identity["configuration"]
    keys(configuration, ("schema_version", "canonical_config_hash", "resolved_contract", "candidate_id", "resolved_candidate"), "Resolved config")
    config = copy.deepcopy(configuration["resolved_contract"])
    require(type(config) is dict and "candidates" not in config, "Malformed resolved contract.")
    require(configuration["candidate_id"] == "smoke_adam_v1", "Unknown candidate ID.")
    config["candidates"] = {configuration["candidate_id"]: configuration["resolved_candidate"]}
    validate_config(config)
    require(configuration["schema_version"] == config["schema_version"] and configuration["canonical_config_hash"] == domain_hash("downstream_config.v1", config), "Config hash differs.")
    expected_model = {"family": "cnn_rc", "contract_id": CONTRACT_ID, "architecture_settings": architecture_settings(),
                      "architecture_hash": domain_hash("downstream_architecture.v1", architecture_settings())}
    require(canonical_json_bytes(identity["model"]) == canonical_json_bytes(expected_model), "Model identity differs.")
    data = identity["data"]
    keys(data, (*config["data"], "stage_id", "split_seed", "subset_seed"), "Run data")
    require(identifier(data["stage_id"], "exd_hox_training_stage_"), "Invalid stage ID.")
    require(type(data["split_seed"]) is int and data["split_seed"] == 31001
            and type(data["subset_seed"]) is int and data["subset_seed"] == 32001, "Data seeds differ.")
    for name, value in config["data"].items():
        require(data[name] == value, "Run data identity differs.")
    selection = identity["selection"]
    keys(selection, ("transcription_factor", "requested_level_id", "canonical_level_id", "request_type", "request_value",
                     "unaliased_requested_logical_example_count", "canonical_requested_logical_example_count",
                     "actual_logical_example_count", "actual_rc_group_count", "inclusive_maximum_rank", "training_membership_hash",
                     "validation_id", "validation_membership_hash", "validation_row_count", "validation_rc_group_count"), "Selection")
    require(selection["transcription_factor"] in TF_NAMES, "Invalid TF.")
    for field in ("requested_level_id", "canonical_level_id"):
        require(identifier(selection[field], "lvl_"), "Invalid level ID.")
    for field in ("training_membership_hash", "validation_membership_hash"):
        require(hex_digest(selection[field]), "Invalid membership hash.")
    validation_digest = hashlib.sha256(("exd_hox_fixed_validation.v1\0" + data["split_identity_hash"] + "\0" + selection["transcription_factor"]).encode()).hexdigest()
    require(selection["validation_id"] == "validation_" + validation_digest, "Validation identity differs.")
    for field in ("unaliased_requested_logical_example_count", "canonical_requested_logical_example_count", "actual_logical_example_count",
                  "actual_rc_group_count", "inclusive_maximum_rank", "validation_row_count", "validation_rc_group_count"):
        require(integer(selection[field], 1), "Invalid selection count.")
    require(selection["actual_rc_group_count"] == selection["inclusive_maximum_rank"]
            and selection["actual_rc_group_count"] <= selection["actual_logical_example_count"]
            and selection["canonical_requested_logical_example_count"] <= selection["actual_logical_example_count"]
            and selection["validation_rc_group_count"] <= selection["validation_row_count"], "Inconsistent selection counts.")
    require(selection["request_type"] in ("absolute", "fractional") and type(selection["request_value"]) is str, "Invalid level request.")
    if selection["request_type"] == "absolute":
        require(selection["request_value"] == str(selection["unaliased_requested_logical_example_count"])
                and selection["unaliased_requested_logical_example_count"] == selection["canonical_requested_logical_example_count"]
                and selection["requested_level_id"] == selection["canonical_level_id"], "Invalid absolute level.")
    else:
        require(re.fullmatch(r"0\.[0-9]+|1\.0", selection["request_value"]) is not None
                and 0 < float(selection["request_value"]) <= 1, "Invalid fraction.")
    seeds = identity["seeds"]
    require(type(seeds) is dict and "parent_seed" in seeds, "Missing seeds.")
    require(canonical_json_bytes(seeds) == canonical_json_bytes(seed_record(seeds["parent_seed"])), "Derived seeds differ.")
    candidate = configuration["resolved_candidate"]
    batches = (selection["actual_logical_example_count"] + candidate["batch_size"] - 1) // candidate["batch_size"]
    expected_budget = {"maximum_epochs": 2, "batches_per_epoch": batches, "maximum_updates": 2 * batches, "maximum_selection_evaluations": 2}
    require(canonical_json_bytes(identity["budget"]) == canonical_json_bytes(expected_budget), "Run budget differs.")
    validate_software(identity["software"])
    require(run["run_id"] == "run_" + domain_hash("downstream_scientific_run.v1", identity), "Run ID differs.")
    content = {key: run[key] for key in ("schema_version", "run_id", "identity")}
    require(run["manifest_hash"] == domain_hash("downstream_resolved_run_manifest.v1", content), "Run manifest hash differs.")


def attempt_id(run_id: str, nonce: str) -> str:
    require(identifier(run_id, "run_") and hex_digest(nonce, 32), "Invalid attempt identity input.")
    return "attempt_" + domain_hash("downstream_execution_attempt.v1", {"run_id": run_id, "nonce": nonce})


def verify_run_sources(run: dict, checkout_root: Path) -> None:
    """Recheck recorded runtime inventory before publication or execution."""
    validate_run(run)
    software = run["identity"]["software"]
    paths = [record["path"] for record in software["source_inventory"]]
    require(verify_runtime_sources(checkout_root, software["runtime_commit"], paths) == software,
            "Runtime source inventory differs from resolved run.")


ATTEMPT_FIELDS = (
    "schema_version", "run_id", "attempt_id", "nonce", "operation", "parent_attempt_id", "resume_checkpoint_id",
    "physical_roots", "host", "device", "hardware", "requested_resources", "observed_resources", "slurm",
    "environment", "environment_hash", "resume_compatibility_hash", "started_at", "ended_at", "status",
    "exit_code", "failure_reason", "manifest_hash",
)


def validate_attempt(attempt: dict) -> None:
    """Validate immutable start or terminal receipts; execution facts stay here."""
    keys(attempt, ATTEMPT_FIELDS, "Execution attempt")
    require(attempt["schema_version"] == "downstream_execution_attempt.v1", "Attempt schema differs.")
    require(attempt["attempt_id"] == attempt_id(attempt["run_id"], attempt["nonce"]), "Attempt ID differs.")
    require(attempt["operation"] in ("training", "validation"), "Invalid attempt operation.")
    for name, prefix in (("parent_attempt_id", "attempt_"), ("resume_checkpoint_id", "ckpt_")):
        require(attempt[name] is None or identifier(attempt[name], prefix), "Invalid attempt parent.")
    keys(attempt["physical_roots"], ("stage", "output", "attempt"), "Execution roots")
    for path in attempt["physical_roots"].values():
        require(type(path) is str and Path(path).is_absolute(), "Execution root must be absolute.")
    for field in ("host", "device", "hardware"):
        require(type(attempt[field]) is str and bool(attempt[field]), "Missing execution observation.")
    for field in ("requested_resources", "observed_resources"):
        keys(attempt[field], ("tasks", "cpus", "gpus", "memory_bytes"), "Resources")
        for value in attempt[field].values():
            require(value is None or integer(value), "Invalid resource observation.")
    keys(attempt["slurm"], ("job_id", "array_job_id", "array_task_id", "account", "partition"), "Slurm")
    for value in attempt["slurm"].values():
        require(value is None or (type(value) is str and bool(value)), "Invalid Slurm value.")
    require(type(attempt["environment"]) is dict, "Missing environment.")
    require(attempt["environment_hash"] == domain_hash("downstream_attempt_environment.v1", attempt["environment"])
            and hex_digest(attempt["resume_compatibility_hash"]), "Environment fingerprint differs.")
    require(type(attempt["started_at"]) is str and bool(attempt["started_at"]), "Missing start time.")
    require(attempt["status"] in ("running", "succeeded", "failed", "interrupted"), "Invalid attempt status.")
    if attempt["status"] == "running":
        require(attempt["ended_at"] is None and attempt["exit_code"] is None and attempt["failure_reason"] is None, "Running attempt has terminal facts.")
    else:
        require(type(attempt["ended_at"]) is str and bool(attempt["ended_at"])
                and type(attempt["exit_code"]) is int, "Missing terminal facts.")
        if attempt["status"] == "succeeded":
            require(attempt["exit_code"] == 0 and attempt["failure_reason"] is None, "Invalid success receipt.")
        else:
            require(type(attempt["failure_reason"]) is str and bool(attempt["failure_reason"]), "Missing failure reason.")
    content = dict(attempt)
    del content["manifest_hash"]
    require(attempt["manifest_hash"] == domain_hash("downstream_execution_attempt_manifest.v1", content), "Attempt manifest hash differs.")
