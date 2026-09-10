"""One deterministic CNN-RC run over an accepted public B2 stage.

All checkpoint storage and locks belong to B3a. Results use the same exclusive
bundle publisher. There are no data, numerical-policy, or provenance bypasses.
Context managers and exception cleanup are necessary at state/IO boundaries.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import math
import os
from pathlib import Path
import platform
import secrets
import stat
import sys
from typing import Any

import numpy as np
import torch

from src import cnn_rc as models
from src import downstream_checkpoint as checkpoints
from src import downstream_metrics as metrics
from src import downstream_run as runs
from src import exd_hox_dataset as datasets
from src.downstream_fingerprints import canonical_json_bytes


B3A_ACCEPTED_COMMIT = "14d727906981fc1178b1d35735ea627f4b2c0982"
SOURCE_PATHS = (
    "configs/carc_exd_hox_training_staging_v1.json",
    "configs/exd_hox_cnn_rc_v1.json",
    "scripts/downstream/train_cnn_rc.py",
    "scripts/downstream/validate_cnn_rc.py",
    "src/__init__.py",
    "src/cnn_rc.py",
    "src/cnn_rc_training.py",
    "src/downstream_checkpoint.py",
    "src/downstream_fingerprints.py",
    "src/downstream_metrics.py",
    "src/downstream_run.py",
    "src/exd_hox_dataset.py",
)
CONFIG_PATH = "configs/exd_hox_cnn_rc_v1.json"
ACCUMULATOR_FIELDS = ("example_count", "update_count", "squared_error_sum",
                      "regularization_sum", "total_loss_sum")


def configure_runtime(device: str) -> dict:
    """Fix one-device float32 numerical settings before any model execution."""
    runs.require(device in ("cpu", "cuda:0"), "Only cpu or cuda:0 is supported.")
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if device == "cuda:0":
        runs.require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") in (None, ":4096:8"),
                     "Incompatible CUBLAS_WORKSPACE_CONFIG.")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        runs.require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
                     "Exactly one visible CUDA device is required.")
        torch.cuda.init()
    else:
        runs.require(not torch.cuda.is_initialized(), "CPU execution has active CUDA state.")
    environment = checkpoints.capture_environment()
    checkpoints.validate_environment(environment)
    return environment


def _verify_sources(expected_commit: str) -> tuple[Path, dict]:
    root = Path(os.path.abspath(__file__)).parents[1]
    for namespace in ("scripts", "scripts.downstream"):
        runs.require(getattr(sys.modules.get(namespace), "__file__", None) is None,
                     "Unexpected package initializer outside the production source inventory.")
        runs.require(not os.path.lexists(root / namespace.replace(".", "/") / "__init__.py"),
                     "Unexpected package initializer outside the production source inventory.")
    executing = {}
    for name, module in tuple(sys.modules.items()):
        if name == "src" or name.startswith("src.") or name.startswith("scripts.downstream."):
            path = getattr(module, "__file__", None)
            if path is not None:
                relative = "src/__init__.py" if name == "src" else name.replace(".", "/") + ".py"
                runs.require(relative in SOURCE_PATHS, "Executing project module absent from production source inventory.")
                executing[relative] = Path(path)
    main = sys.modules.get("__main__")
    specification = getattr(main, "__spec__", None)
    if specification is not None and specification.name in (
        "scripts.downstream.train_cnn_rc", "scripts.downstream.validate_cnn_rc",
    ):
        executing[specification.name.replace(".", "/") + ".py"] = Path(main.__file__)
    software = runs.verify_runtime_sources(root, expected_commit, SOURCE_PATHS,
                                           executing_paths=executing)
    return root, software


def _absolute_directory(path: Path) -> Path:
    absolute = Path(os.path.abspath(path))
    runs.require(absolute != Path("/"), "Filesystem root cannot be an execution root.")
    for parent in reversed((absolute, *absolute.parents)):
        if os.path.lexists(parent):
            mode = parent.lstat().st_mode
            runs.require(stat.S_ISDIR(mode) and not stat.S_ISLNK(mode),
                         "Execution roots require physical directories without symlinks.")
    return absolute


def _separate(*paths: Path) -> None:
    for index, first in enumerate(paths):
        for second in paths[index + 1:]:
            runs.require(first != second and first not in second.parents and second not in first.parents,
                         "Stage, output and attempt roots must not overlap.")


def _mkdir(path: Path) -> None:
    """Create directories through anchored descriptors without symlink traversal."""
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in path.parts[1:]:
            try:
                os.mkdir(component, 0o700, dir_fd=descriptor)
            except FileExistsError:
                pass
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
    finally:
        os.close(descriptor)


def orient_training_batch(batch: datasets.ExdHoxBatch, run: dict, epoch: int) -> datasets.ExdHoxBatch:
    """Orient already collated B2 samples, retaining all labels and metadata."""
    identity = run["identity"]
    selection = identity["selection"]
    seed = identity["seeds"]["derived_seeds"]["training_orientation"]
    oriented = batch.x.clone()
    for index, item in enumerate(batch.metadata):
        runs.require(item.primary_split == "training"
                     and item.level_id == selection["requested_level_id"]
                     and item.transcription_factor == selection["transcription_factor"]
                     and item.stage_id == identity["data"]["stage_id"], "Wrong training batch identity.")
        if runs.training_orientation(seed, selection["training_membership_hash"], epoch,
                                     item.logical_example_id):
            oriented[index] = models.reverse_complement(batch.x[index:index + 1])[0]
    return datasets.ExdHoxBatch(oriented, batch.y, batch.metadata)


def _finite_state(model: models.CNNRC, optimizer: torch.optim.Adam) -> None:
    for name, parameter in model.named_parameters():
        runs.require(bool(torch.isfinite(parameter).all()), "Nonfinite parameter: " + name)
    for name, buffer in model.named_buffers():
        runs.require(bool(torch.isfinite(buffer).all()), "Nonfinite BN buffer: " + name)
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value) and value.is_floating_point():
                runs.require(bool(torch.isfinite(value).all()), "Nonfinite optimizer state.")
            elif type(value) is float:
                runs.require(math.isfinite(value), "Nonfinite optimizer state.")


def training_step(model, optimizer, batch, run: dict, epoch: int) -> dict:
    """One update; any failure invalidates the caller's in-memory attempt."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    oriented = orient_training_batch(batch, run, epoch)
    device = model.W.device
    predictions = model(oriented.x.to(device))
    mse = (predictions - oriented.y.to(device)).square().mean()
    coefficients = run["identity"]["configuration"]["resolved_candidate"]["regularization"]
    penalty = model.convolution_kernel_penalty(**coefficients)
    total = mse + penalty
    for name, value in (("MSE", mse), ("regularization", penalty), ("total loss", total)):
        runs.require(bool(torch.isfinite(value).all()), "Nonfinite " + name + ".")
    total.backward()
    for name, parameter in model.named_parameters():
        runs.require(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()),
                     "Missing or nonfinite gradient: " + name)
    optimizer.step()
    _finite_state(model, optimizer)
    optimizer.zero_grad(set_to_none=True)
    count = len(batch.metadata)
    return {"example_count": count, "update_count": 1,
            "squared_error_sum": float(mse.item()) * count,
            "regularization_sum": float(penalty.item()), "total_loss_sum": float(total.item())}


def _validation_snapshot(model, optimizer) -> str:
    return checkpoints.state_fingerprint({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "gradients": [parameter.grad for parameter in model.parameters()],
        "rng": checkpoints.capture_rng(),
    })


def evaluate_validation(model, optimizer, validation, run: dict) -> dict:
    """Complete fixed arrays with an unaveraged invariance gate and no mutation."""
    selection = run["identity"]["selection"]
    runs.require(len(validation) == selection["validation_row_count"]
                 and runs.membership_hash(validation) == selection["validation_membership_hash"],
                 "Validation membership differs.")
    before = _validation_snapshot(model, optimizer)
    previous_mode = model.training
    forwards = []
    reverse = []
    targets = []
    groups = []
    batch_size = run["identity"]["configuration"]["resolved_candidate"]["validation"]["batch_size"]
    try:
        model.eval()
        with torch.no_grad():
            for start in range(0, len(validation), batch_size):
                samples = [validation[index] for index in range(start, min(start + batch_size, len(validation)))]
                batch = datasets.collate_exd_hox(samples)
                for offset, item in enumerate(batch.metadata):
                    runs.require(item.primary_split == "validation" and item.level_id is None
                                 and item.validation_id == selection["validation_id"]
                                 and item.sample_rank_one_based == start + offset + 1,
                                 "Validation must use deterministic full B2 membership.")
                inputs = batch.x.to(model.W.device)
                forwards.append(model(inputs).detach().cpu().numpy().reshape(-1).astype(np.float64))
                reverse.append(model(models.reverse_complement(inputs)).detach().cpu().numpy().reshape(-1).astype(np.float64))
                targets.append(batch.y.numpy().reshape(-1).astype(np.float64))
                groups.extend(item.global_rc_group_id for item in batch.metadata)
        forward = np.concatenate(forwards)
        rc = np.concatenate(reverse)
        target = np.concatenate(targets)
        runs.require(np.isfinite(forward).all() and np.isfinite(rc).all(), "Nonfinite validation prediction.")
        difference = np.abs(rc - forward)
        tolerance = 1e-6 + 1e-5 * np.abs(forward)
        diagnostic = {"atol": 1e-6, "rtol": 1e-5,
                      "maximum_absolute_difference": float(difference.max()),
                      "maximum_normalized_difference": float((difference / tolerance).max()),
                      "violation_count": int(np.count_nonzero(difference > tolerance))}
        runs.require(diagnostic["violation_count"] == 0, "CNN-RC invariance failed before averaging.")
        mean = (forward + rc) / np.float64(2.0)
        result = {"metrics": {}, "rc_diagnostic": diagnostic}
        for name, predictions in (("forward", forward), ("reverse_complement", rc), ("mean", mean)):
            result["metrics"][name] = metrics.compute_regression_metrics(target, predictions, groups)
        return result
    finally:
        model.train(previous_mode)
        runs.require(_validation_snapshot(model, optimizer) == before,
                     "Validation mutated model, optimizer, gradients, BN or RNG state; abort attempt.")


def select_best(history: list[dict]) -> dict | None:
    best = None
    best_key = None
    for event in history:
        score = event["metrics"]["mean"]
        if score["r2"] is not None:
            runs.require(type(score["r2"]) is float and math.isfinite(score["r2"])
                         and type(score["rmse"]) is float and math.isfinite(score["rmse"]),
                         "Selection requires finite float64 metrics.")
            key = (score["r2"], -score["rmse"], -event["global_update"], -event["epoch"])
            if best_key is None or key > best_key:
                best_key = key
                best = {"epoch": event["epoch"], "global_update": event["global_update"],
                        "metrics": copy.deepcopy(score), "checkpoint_ref": copy.deepcopy(event["checkpoint_ref"])}
    return best


def _hash_record(record: dict) -> dict:
    result = copy.deepcopy(record)
    runs.require("manifest_hash" not in result, "Self hash supplied as hash input.")
    result["manifest_hash"] = runs.domain_hash(result["schema_version"], result)
    return result


def _fingerprint(path: str, raw: bytes) -> dict:
    runs.logical_path(path)
    return {"path": path, "byte_size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _check_fingerprints(records: Any) -> None:
    runs.require(type(records) is list, "Invalid artifact fingerprints.")
    paths = []
    for record in records:
        runs.keys(record, ("path", "byte_size", "sha256"), "Artifact fingerprint")
        paths.append(runs.logical_path(record["path"]))
        runs.require(runs.integer(record["byte_size"], 1) and runs.hex_digest(record["sha256"]),
                     "Malformed artifact fingerprint.")
    runs.require(paths == sorted(set(paths)), "Artifact inventory must be sorted and unique.")


ARTIFACT_FIELDS = {
    "cnn_rc_epoch_metrics.v1": ("epoch", "global_update", "training", "validation_event_id", "pending_checkpoint_id", "epoch_complete_checkpoint_id"),
    "cnn_rc_validation_event.v1": ("event_id", "epoch", "global_update", "pending_checkpoint_id", "epoch_complete_checkpoint_id", "history_event"),
    "cnn_rc_validation_started.v1": ("event_id", "epoch", "pending_checkpoint_id", "attempt_id"),
    "cnn_rc_validation_summary.v1": ("selected_checkpoint_id", "selection", "seeds", "data", "software", "environment_compatibility", "environment_compatibility_hash", "producing_attempt_id", "best_epoch", "best_global_update", "validation_event_id", "metrics", "rc_diagnostic", "artifact_fingerprints", "status"),
    "cnn_rc_durable_inventory.v1": ("files",),
    "downstream_completion.v1": ("selected_checkpoint_id", "terminal_recovery_checkpoint_id", "validation_event_id", "global_update", "consumed_validation_event_count", "consumed_selection_count", "artifact_fingerprints", "status"),
    "cnn_rc_validation_verification.v1": ("checkpoint_id", "validation_event_id", "recorded_event_hash", "source_completion_hash", "producing_attempt_id", "selection_unchanged", "additional_selection_evaluations", "artifact_fingerprints", "status"),
}


def _validate_history_event(event: dict, run: dict) -> None:
    runs.keys(event, ("event_id", "epoch", "global_update", "checkpoint_ref", "metrics", "rc_diagnostic"), "History event")
    runs.require(runs.integer(event["epoch"]) and event["epoch"] < 2
                 and event["global_update"] == (event["epoch"] + 1) * run["identity"]["budget"]["batches_per_epoch"],
                 "Validation event position differs.")
    runs.require(event["event_id"] == _event_id(run, event["epoch"]), "Validation event identity differs.")
    runs.require(event["checkpoint_ref"] == {"kind": "self"}, "Committed event must refer to its epoch-complete snapshot.")
    runs.keys(event["metrics"], ("forward", "reverse_complement", "mean"), "Orientation metrics")
    for score in event["metrics"].values():
        checkpoints.validate_metrics(score, run)
    diagnostic = event["rc_diagnostic"]
    runs.keys(diagnostic, ("atol", "rtol", "maximum_absolute_difference", "maximum_normalized_difference", "violation_count"), "RC diagnostic")
    runs.require(type(diagnostic["atol"]) is float and diagnostic["atol"] == 1e-6
                 and type(diagnostic["rtol"]) is float and diagnostic["rtol"] == 1e-5
                 and type(diagnostic["violation_count"]) is int and diagnostic["violation_count"] == 0,
                 "Invalid invariance gate.")
    for name in ("maximum_absolute_difference", "maximum_normalized_difference"):
        runs.require(type(diagnostic[name]) is float and math.isfinite(diagnostic[name]) and diagnostic[name] >= 0,
                     "Invalid invariance diagnostic.")
    runs.require(diagnostic["maximum_normalized_difference"] <= 1, "Invariance violation.")


def validate_artifact(record: dict, run: dict) -> None:
    """Reject missing/unknown fields and recompute every semantic identity."""
    schema = record.get("schema_version")
    if schema == "downstream_resolved_run.v1":
        runs.validate_run(record)
        runs.require(record == run, "Resolved run differs.")
        return
    if schema == "downstream_execution_attempt.v1":
        runs.validate_attempt(record)
        runs.require(record["run_id"] == run["run_id"], "Wrong attempt run.")
        environment = record["environment"]
        runs.keys(environment, ("compatibility", "validation_replays", "validation_reuses"), "Attempt environment")
        checkpoints.validate_environment(environment["compatibility"])
        runs.require(record["resume_compatibility_hash"] == checkpoints.environment_hash(environment["compatibility"]),
                     "Attempt compatibility identity differs.")
        for name in ("validation_replays", "validation_reuses"):
            runs.require(type(environment[name]) is list and len(environment[name]) == len(set(environment[name]))
                         and all(runs.identifier(value, "validation_event_") for value in environment[name]),
                         "Invalid attempt validation replay metadata.")
        return
    runs.require(schema in ARTIFACT_FIELDS, "Unknown result schema.")
    runs.keys(record, ("schema_version", "run_id", *ARTIFACT_FIELDS[schema], "manifest_hash"), "Result")
    runs.require(record["run_id"] == run["run_id"], "Result run differs.")
    content = dict(record)
    del content["manifest_hash"]
    runs.require(record["manifest_hash"] == runs.domain_hash(schema, content), "Result semantic hash differs.")
    for field in ("selected_checkpoint_id", "terminal_recovery_checkpoint_id", "checkpoint_id", "pending_checkpoint_id", "epoch_complete_checkpoint_id"):
        if field in record:
            runs.require(runs.identifier(record[field], "ckpt_"), "Invalid checkpoint identity.")
    for field in ("producing_attempt_id", "attempt_id"):
        if field in record:
            runs.require(runs.identifier(record[field], "attempt_"), "Invalid producing attempt.")
    for field in ("validation_event_id", "event_id"):
        if field in record:
            runs.require(runs.identifier(record[field], "validation_event_"), "Invalid validation identity.")
    if "artifact_fingerprints" in record:
        _check_fingerprints(record["artifact_fingerprints"])
    if "status" in record:
        runs.require(record["status"] == "succeeded", "Result is not successful.")
    if "epoch" in record:
        runs.require(runs.integer(record["epoch"]) and record["epoch"] < 2, "Invalid epoch.")
        event_id = record.get("event_id", record.get("validation_event_id"))
        runs.require(event_id == _event_id(run, record["epoch"]), "Wrong epoch event.")
    if schema == "cnn_rc_validation_event.v1":
        _validate_history_event(record["history_event"], run)
        for name in ("event_id", "epoch", "global_update"):
            runs.require(record[name] == record["history_event"][name], "Event fields disagree.")
    elif schema == "cnn_rc_epoch_metrics.v1":
        training = record["training"]
        runs.keys(training, (*ACCUMULATOR_FIELDS, "mse_example_weighted", "regularization_mean_per_update", "total_loss_mean_per_update"), "Training summary")
        count = run["identity"]["selection"]["actual_logical_example_count"]
        updates = run["identity"]["budget"]["batches_per_epoch"]
        runs.require(type(training["example_count"]) is int and training["example_count"] == count
                     and type(training["update_count"]) is int and training["update_count"] == updates,
                     "Training epoch counts differ.")
        for name in (*ACCUMULATOR_FIELDS[2:], "mse_example_weighted",
                     "regularization_mean_per_update", "total_loss_mean_per_update"):
            runs.require(type(training[name]) is float and math.isfinite(training[name]) and training[name] >= 0,
                         "Invalid training summary.")
        runs.require(training["mse_example_weighted"] == training["squared_error_sum"] / count
                     and training["regularization_mean_per_update"] == training["regularization_sum"] / updates
                     and training["total_loss_mean_per_update"] == training["total_loss_sum"] / updates,
                     "Training summary weighting differs.")
        runs.require(record["global_update"] == (record["epoch"] + 1) * updates, "Epoch update differs.")
    elif schema == "cnn_rc_validation_summary.v1":
        for name in ("selection", "seeds", "data", "software"):
            runs.require(record[name] == run["identity"][name], "Validation summary identity differs.")
        runs.require(record["environment_compatibility_hash"] == checkpoints.environment_hash(record["environment_compatibility"]),
                     "Summary environment differs.")
        event = {"event_id": record["validation_event_id"], "epoch": record["best_epoch"],
                 "global_update": record["best_global_update"], "checkpoint_ref": {"kind": "self"},
                 "metrics": record["metrics"], "rc_diagnostic": record["rc_diagnostic"]}
        _validate_history_event(event, run)
    elif schema == "cnn_rc_durable_inventory.v1":
        _check_fingerprints(record["files"])
    elif schema == "downstream_completion.v1":
        for name, expected in (("global_update", run["identity"]["budget"]["maximum_updates"]),
                               ("consumed_validation_event_count", 2), ("consumed_selection_count", 2)):
            runs.require(type(record[name]) is int and record[name] == expected, "Completion budget differs.")
    elif schema == "cnn_rc_validation_verification.v1":
        runs.require(record["selection_unchanged"] is True
                     and type(record["additional_selection_evaluations"]) is int
                     and record["additional_selection_evaluations"] == 0
                     and runs.hex_digest(record["recorded_event_hash"])
                     and runs.hex_digest(record["source_completion_hash"]), "Invalid verification report.")


def _validate_producer(record: dict, attempt_id: str) -> None:
    runs.require(runs.identifier(attempt_id, "attempt_"), "Invalid publication producer.")
    for field in ("attempt_id", "producing_attempt_id"):
        if field in record:
            runs.require(record[field] == attempt_id, "Publication producer differs from artifact receipt.")


def _bundle_files(records: dict[str, dict], run: dict, attempt_id: str) -> dict[str, bytes]:
    files = {}
    entries = []
    for name, record in sorted(records.items()):
        validate_artifact(record, run)
        _validate_producer(record, attempt_id)
        raw = canonical_json_bytes(record) + b"\n"
        files[name] = raw
        entries.append({**_fingerprint(name, raw), "schema_version": record["schema_version"],
                        "semantic_hash": record["manifest_hash"]})
    envelope = _hash_record({"schema_version": "cnn_rc_result_publication.v1", "run_id": run["run_id"],
                             "producing_attempt_id": attempt_id, "files": entries})
    files["manifest.json"] = canonical_json_bytes(envelope) + b"\n"
    return files


def read_artifacts(path: Path, run: dict) -> dict[str, dict]:
    """Validate a complete immutable result bundle before using any record."""
    descriptor = checkpoints._open_directory(path)
    try:
        envelope = runs.strict_json(runs.read_regular(path / "manifest.json"))
        runs.keys(envelope, ("schema_version", "run_id", "producing_attempt_id", "files", "manifest_hash"), "Publication")
        content = dict(envelope)
        del content["manifest_hash"]
        runs.require(envelope["schema_version"] == "cnn_rc_result_publication.v1"
                     and envelope["run_id"] == run["run_id"]
                     and runs.identifier(envelope["producing_attempt_id"], "attempt_")
                     and envelope["manifest_hash"] == runs.domain_hash(envelope["schema_version"], content),
                     "Result publication identity differs.")
        runs.require(type(envelope["files"]) is list and bool(envelope["files"]), "Empty result bundle.")
        records = {}
        for entry in envelope["files"]:
            runs.keys(entry, ("path", "byte_size", "sha256", "schema_version", "semantic_hash"), "Publication file")
            name = runs.logical_path(entry["path"])
            runs.require("/" not in name and name != "manifest.json" and name not in records, "Invalid bundle path.")
            raw = runs.read_regular(path / name)
            runs.require(_fingerprint(name, raw) == {key: entry[key] for key in ("path", "byte_size", "sha256")},
                         "Result byte fingerprint differs.")
            record = runs.strict_json(raw)
            validate_artifact(record, run)
            _validate_producer(record, envelope["producing_attempt_id"])
            runs.require(record["manifest_hash"] == entry["semantic_hash"]
                         and record["schema_version"] == entry["schema_version"]
                         and raw == canonical_json_bytes(record) + b"\n", "Result schema or canonical bytes differ.")
            records[name] = record
        runs.require(list(records) == sorted(records) and set(os.listdir(descriptor)) == {*records, "manifest.json"},
                     "Result bundle inventory differs.")
        checkpoints._assert_directory(path, descriptor)
        return records
    finally:
        os.close(descriptor)


def _event_id(run: dict, epoch: int) -> str:
    return "validation_event_" + runs.domain_hash("downstream_validation_event.v1", {
        "run_id": run["run_id"], "epoch": epoch,
        "global_update": (epoch + 1) * run["identity"]["budget"]["batches_per_epoch"],
    })


def _record(schema: str, run: dict, **fields) -> dict:
    return _hash_record({"schema_version": schema, "run_id": run["run_id"], **fields})


def _empty_accumulators() -> dict:
    return {"example_count": 0, "update_count": 0, "squared_error_sum": 0.0,
            "regularization_sum": 0.0, "total_loss_sum": 0.0}


def _position(run: dict, training, epoch: int) -> dict:
    ids = [training[index].metadata.logical_example_id for index in range(len(training))]
    order = runs.epoch_order(ids, run["identity"]["seeds"]["derived_seeds"]["training_data_order"], epoch)
    return {"completed_epoch_count": epoch, "current_epoch": epoch, "next_batch_index": 0,
            "global_update": epoch * run["identity"]["budget"]["batches_per_epoch"], "phase": "train",
            "epoch_permutation_hash": runs.permutation_hash(ids, order)}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _attempt(run, environment, stage, output, attempt, device, operation, resume=None, parent=None) -> dict:
    nonce = secrets.token_hex(16)
    observation = {"compatibility": environment, "validation_replays": [], "validation_reuses": []}
    result = {
        "schema_version": "downstream_execution_attempt.v1", "run_id": run["run_id"],
        "attempt_id": runs.attempt_id(run["run_id"], nonce), "nonce": nonce,
        "operation": operation, "parent_attempt_id": parent, "resume_checkpoint_id": resume,
        "physical_roots": {"stage": str(stage), "output": str(output), "attempt": str(attempt)},
        "host": platform.node() or "unknown-host", "device": device,
        "hardware": environment["device_class"],
        "requested_resources": {"tasks": 1, "cpus": 1, "gpus": int(device == "cuda:0"), "memory_bytes": None},
        "observed_resources": {"tasks": 1, "cpus": torch.get_num_threads(), "gpus": environment["device_count"], "memory_bytes": None},
        "slurm": {"job_id": os.environ.get("SLURM_JOB_ID"), "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
                  "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"), "account": os.environ.get("SLURM_JOB_ACCOUNT"),
                  "partition": os.environ.get("SLURM_JOB_PARTITION")},
        "environment": observation,
        "environment_hash": runs.domain_hash("downstream_attempt_environment.v1", observation),
        "resume_compatibility_hash": checkpoints.environment_hash(environment),
        "started_at": _now(), "ended_at": None, "status": "running", "exit_code": None, "failure_reason": None,
    }
    return _rehash_attempt(result)


def _rehash_attempt(receipt: dict) -> dict:
    result = copy.deepcopy(receipt)
    result.pop("manifest_hash", None)
    result["environment_hash"] = runs.domain_hash("downstream_attempt_environment.v1", result["environment"])
    result["manifest_hash"] = runs.domain_hash("downstream_execution_attempt_manifest.v1", result)
    return result


def _terminal(receipt: dict, error: BaseException | None = None) -> dict:
    result = copy.deepcopy(receipt)
    result.update(ended_at=_now(), status="succeeded", exit_code=0, failure_reason=None)
    if error is not None:
        result.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      exit_code=130 if isinstance(error, KeyboardInterrupt) else 1,
                      failure_reason=type(error).__name__ + ": " + str(error))
    return _rehash_attempt(result)


def _verify_inventory(root: Path, records: list[dict]) -> None:
    _check_fingerprints(records)
    for record in records:
        raw = runs.read_regular(root / record["path"])
        runs.require(_fingerprint(record["path"], raw) == record, "Durable artifact fingerprint differs.")


def _inventory(root: Path) -> list[dict]:
    records = []
    for current, directories, files in os.walk(root, followlinks=False):
        directories[:] = sorted(name for name in directories if not name.startswith(".b3-private-"))
        for name in directories:
            runs.require(not (Path(current) / name).is_symlink(), "Symlink in result inventory.")
        for name in sorted(files):
            if name != ".writer.lock":
                path = Path(current) / name
                records.append(_fingerprint(path.relative_to(root).as_posix(), runs.read_regular(path)))
    return sorted(records, key=lambda item: item["path"])


def _load_indexes(root, run, environment, training) -> tuple[list[Path], dict | None]:
    paths = sorted(root.glob("update-*"))
    loaded = {}
    for path in paths:
        payload, envelope = checkpoints.load_checkpoint(path, run=run, environment=environment, training=training)
        loaded[envelope["checkpoint_id"]] = (payload, envelope, path)
    phase_order = {"train": 0, "validation_pending": 1, "epoch_complete": 2}
    paths.sort(key=lambda path: (int(path.name.split("-")[1]), phase_order[path.name.split("-")[2]]))
    positions = [(int(path.name.split("-")[1]), path.name.split("-")[2]) for path in paths]
    runs.require(len(positions) == len(set(positions)), "Duplicate checkpoint trajectory position.")
    previous = None
    for revision, path in enumerate(sorted(root.glob("index-*"))):
        runs.require(path.name == "index-" + str(revision).zfill(8), "Missing index revision.")
        index = runs.strict_json(runs.read_regular(path / "index.json"))
        checkpoints.validate_index(index)
        runs.require(index["run_id"] == run["run_id"] and index["revision"] == revision
                     and index["predecessor_hash"] == (None if previous is None else previous["manifest_hash"]),
                     "Index predecessor differs.")
        runs.require(set(os.listdir(path)) == {"index.json"}, "Index inventory differs.")
        if previous is not None:
            runs.require(index["entries"][:len(previous["entries"])] == previous["entries"]
                         and len(index["entries"]) > len(previous["entries"]), "Index does not extend predecessor.")
        for position, entry in enumerate(index["entries"]):
            runs.require(entry["checkpoint_id"] in loaded and position < len(paths), "Indexed checkpoint missing.")
            payload, envelope, checkpoint_path = loaded[entry["checkpoint_id"]]
            expected = {"checkpoint_id": envelope["checkpoint_id"], "path": checkpoint_path.name,
                        "global_update": payload["position"]["global_update"], "phase": payload["position"]["phase"],
                        "producer_attempt_id": envelope["producer_attempt_id"], "semantic_hash": envelope["semantic_hash"],
                        "file": envelope["file"]}
            runs.require(entry == expected and checkpoint_path == paths[position], "Index checkpoint reference differs.")
        final_payload = loaded[index["latest_recovery_checkpoint_id"]][0]
        state = checkpoints.recovery_state(final_payload)
        selected = state["selection_state"]
        best = None if selected is None else selected["checkpoint_ref"]["checkpoint_id"]
        runs.require(index["selected_best_checkpoint_id"] == best, "Index best selection differs.")
        if best is not None:
            chosen = loaded[best][0]
            runs.require(chosen["position"]["phase"] == "epoch_complete"
                         and chosen["validation_history"][-1]["metrics"]["mean"] == selected["metrics"]
                         and chosen["position"]["global_update"] == selected["global_update"], "Selected checkpoint event differs.")
        previous = index
    return paths, previous


def _training_summary(accumulators: dict) -> dict:
    return {**accumulators,
            "mse_example_weighted": accumulators["squared_error_sum"] / accumulators["example_count"],
            "regularization_mean_per_update": accumulators["regularization_sum"] / accumulators["update_count"],
            "total_loss_mean_per_update": accumulators["total_loss_sum"] / accumulators["update_count"]}


def validate_run_artifacts(root: Path, run: dict, environment: dict, training,
                           *, completed: bool = False) -> tuple[list[Path], dict | None]:
    """Verify the exact result graph, including interrupted publication edges."""
    paths, index = _load_indexes(root, run, environment, training)
    snapshots = {}
    producers = []
    for path in paths:
        payload, envelope = checkpoints.load_checkpoint(path, run=run, environment=environment, training=training)
        snapshots[envelope["checkpoint_id"]] = payload
        producers.append(envelope["producer_attempt_id"])
    records = {}
    for path in sorted(root.iterdir()):
        name = path.name
        if name == ".writer.lock":
            runs.require(stat.S_ISREG(path.lstat().st_mode), "Invalid writer lock.")
        elif name.startswith(".b3-private-"):
            runs.require(stat.S_ISDIR(path.lstat().st_mode), "Invalid private publication directory.")
        elif path in paths or name.startswith("index-"):
            pass  # Each checkpoint and index was checked by the B3a readers.
        elif name == "completion":
            runs.require(completed, "Completed run is immutable.")
        else:
            bundle = read_artifacts(path, run)
            publication = runs.strict_json(runs.read_regular(path / "manifest.json"))
            producers.append(publication["producing_attempt_id"])
            runs.require(set(bundle) == {"record.json"}, "Unexpected result bundle fields.")
            record = bundle["record.json"]
            schema = record["schema_version"]
            if name == "resolved-run":
                runs.require(record == run, "Resolved run artifact differs.")
            elif name in ("epoch-0", "epoch-1"):
                runs.require(schema == "cnn_rc_epoch_metrics.v1" and name == "epoch-" + str(record["epoch"]),
                             "Epoch artifact path differs.")
            elif name in ("validation-started-0", "validation-started-1"):
                runs.require(schema == "cnn_rc_validation_started.v1" and name == "validation-started-" + str(record["epoch"]),
                             "Validation-start artifact path differs.")
            elif name in (_event_id(run, 0), _event_id(run, 1)):
                runs.require(schema == "cnn_rc_validation_event.v1" and name == record["event_id"],
                             "Validation event path differs.")
            else:
                runs.require(schema == "downstream_execution_attempt.v1", "Unexpected result root entry.")
                suffix = "-start" if record["status"] == "running" else "-terminal"
                runs.require(name == record["attempt_id"] + suffix, "Attempt artifact path differs.")
            records[name] = record
    runs.require(records.get("resolved-run") == run, "Missing resolved run artifact.")
    for producer in producers:
        runs.require(producer + "-start" in records, "Artifact producing attempt has no start receipt.")
    for name, receipt in records.items():
        if receipt["schema_version"] == "downstream_execution_attempt.v1":
            runs.require(receipt["environment"]["compatibility"] == environment, "Attempt environment differs.")
            if name.endswith("-terminal"):
                start = records.get(receipt["attempt_id"] + "-start")
                runs.require(start is not None and start["started_at"] == receipt["started_at"]
                             and start["physical_roots"] == receipt["physical_roots"], "Attempt start/terminal differs.")
    for epoch in range(2):
        started = records.get("validation-started-" + str(epoch))
        event = records.get(_event_id(run, epoch))
        epoch_metrics = records.get("epoch-" + str(epoch))
        if started is not None:
            pending = snapshots.get(started["pending_checkpoint_id"])
            runs.require(pending is not None and pending["position"]["phase"] == "validation_pending"
                         and pending["position"]["current_epoch"] == epoch,
                         "Validation start references wrong frozen checkpoint.")
            runs.require(started["attempt_id"] + "-start" in records, "Validation producer attempt missing.")
        if event is not None:
            runs.require(started is not None and event["pending_checkpoint_id"] == started["pending_checkpoint_id"],
                         "Validation event frozen checkpoint differs.")
            expected = copy.deepcopy(snapshots[event["pending_checkpoint_id"]])
            expected["position"].update(phase="epoch_complete", completed_epoch_count=epoch + 1)
            expected["validation_history"].append(event["history_event"])
            expected["selection_state"] = select_best(expected["validation_history"])
            expected["consumed_validation_event_count"] = epoch + 1
            expected["consumed_selection_count"] = epoch + 1
            base = dict(expected)
            del base["tensor_inventory"]
            expected["tensor_inventory"] = checkpoints.semantic_state(base)[2]
            checkpoints.validate_checkpoint(expected, run, environment, training)
            runs.require(event["epoch_complete_checkpoint_id"] == "ckpt_" + checkpoints.state_fingerprint(expected),
                         "Validation event does not bind the frozen checkpoint transition.")
            if completed:
                runs.require(event["epoch_complete_checkpoint_id"] in snapshots, "Completed validation checkpoint missing.")
        if epoch_metrics is not None:
            runs.require(event is not None and epoch_metrics["pending_checkpoint_id"] == event["pending_checkpoint_id"]
                         and epoch_metrics["epoch_complete_checkpoint_id"] == event["epoch_complete_checkpoint_id"],
                         "Epoch metric checkpoint references differ.")
            snapshot = snapshots.get(event["epoch_complete_checkpoint_id"])
            runs.require(snapshot is not None and epoch_metrics["training"] == _training_summary(snapshot["training_accumulators"]),
                         "Epoch metrics differ from checkpoint accumulators.")
        if completed:
            runs.require(event is not None and epoch_metrics is not None, "Completed epoch artifacts missing.")
    for snapshot in snapshots.values():
        for entry in snapshot["validation_history"]:
            event = records.get(entry["event_id"])
            normalized = copy.deepcopy(entry)
            normalized["checkpoint_ref"] = {"kind": "self"}
            runs.require(event is not None and event["history_event"] == normalized, "Checkpoint history differs from committed event.")
    if completed:
        bundle = read_artifacts(root / "completion", run)
        _validate_completion_records(bundle, root, run, environment, snapshots, index, records)
    return paths, index


def _validate_completion_records(bundle, root, run, environment, snapshots, index, records):
    runs.require(set(bundle) == {"completion.json", "validation-summary.json", "inventory.json", "attempt-terminal.json"},
                 "Completion bundle fields differ.")
    completion = bundle["completion.json"]
    summary = bundle["validation-summary.json"]
    inventory = bundle["inventory.json"]
    receipt = bundle["attempt-terminal.json"]
    for record, schema in ((completion, "downstream_completion.v1"), (summary, "cnn_rc_validation_summary.v1"),
                           (inventory, "cnn_rc_durable_inventory.v1"), (receipt, "downstream_execution_attempt.v1")):
        runs.require(record["schema_version"] == schema, "Completion artifact schema differs.")
        validate_artifact(record, run)
    runs.require(index is not None and len(index["entries"]) == len(snapshots), "Incomplete final index.")
    runs.require(completion["selected_checkpoint_id"] == index["selected_best_checkpoint_id"]
                 and completion["terminal_recovery_checkpoint_id"] == index["latest_recovery_checkpoint_id"],
                 "Completion checkpoint references differ from index.")
    selected = snapshots[completion["selected_checkpoint_id"]]
    terminal = snapshots[completion["terminal_recovery_checkpoint_id"]]
    chosen = selected["validation_history"][-1]
    runs.require(terminal["position"]["phase"] == "epoch_complete" and terminal["position"]["completed_epoch_count"] == 2
                 and terminal["position"]["global_update"] == completion["global_update"], "Terminal checkpoint is incomplete.")
    runs.require(summary["selected_checkpoint_id"] == completion["selected_checkpoint_id"]
                 and summary["validation_event_id"] == completion["validation_event_id"] == chosen["event_id"]
                 and summary["best_epoch"] == chosen["epoch"] and summary["best_global_update"] == chosen["global_update"]
                 and summary["metrics"] == chosen["metrics"] and summary["rc_diagnostic"] == chosen["rc_diagnostic"],
                 "Completion validation summary differs from selected event.")
    start = records.get(receipt["attempt_id"] + "-start")
    runs.require(receipt["status"] == "succeeded" and start is not None
                 and start["started_at"] == receipt["started_at"]
                 and summary["producing_attempt_id"] == receipt["attempt_id"]
                 and summary["environment_compatibility"] == receipt["environment"]["compatibility"] == environment,
                 "Completion attempt or environment differs.")
    expected_files = [item for item in _inventory(root) if not item["path"].startswith("completion/")]
    runs.require(inventory["files"] == summary["artifact_fingerprints"] == expected_files,
                 "Durable inventory does not equal complete accepted artifacts.")
    _verify_inventory(root, inventory["files"])
    expected = sorted((_fingerprint(name, canonical_json_bytes(record) + b"\n") for name, record in bundle.items()
                       if name != "completion.json"), key=lambda item: item["path"])
    runs.require(completion["artifact_fingerprints"] == expected, "Completion artifact fingerprints differ.")


class _TrainingRun:
    """Private state machine; tests inject interruption at real method boundaries."""

    def __init__(self, run, training, validation, environment, writer, checkout, receipt, attempt_directory):
        self.run = run
        self.training = training
        self.validation = validation
        self.environment = environment
        self.writer = writer
        self.checkout = checkout
        self.receipt = receipt
        self.attempt_directory = attempt_directory
        self.paths = []
        self.index = None
        self.safe = False
        self.model = None
        self.optimizer = None
        self.state = None
        self.pending_path = None

    def publish(self, name, record):
        self.writer.check()
        runs.verify_run_sources(self.run, self.checkout)
        path = self.writer.root / name
        checkpoints.publish_bundle(path, _bundle_files({"record.json": record}, self.run, self.receipt["attempt_id"]))
        runs.require(read_artifacts(path, self.run) == {"record.json": record}, "Result readback differs.")
        return path

    def capture(self):
        runs.require(self.safe, "No checkpoint-safe boundary after failed training forward.")
        return checkpoints.capture_checkpoint(
            run=self.run, model=self.model, optimizer=self.optimizer, training=self.training,
            environment=self.environment, position=self.state["position"],
            training_accumulators=self.state["training_accumulators"],
            validation_history=self.state["validation_history"], selection_state=self.state["selection_state"],
        )

    def index_checkpoints(self):
        if self.paths and (self.index is None or len(self.paths) > len(self.index["entries"])):
            _, self.index = checkpoints.publish_index(
                writer=self.writer, checkpoint_paths=self.paths, run=self.run, environment=self.environment,
                training=self.training, checkout_root=self.checkout, previous=self.index,
            )

    def save_checkpoint(self):
        payload = self.capture()
        identity = "ckpt_" + checkpoints.state_fingerprint(payload)
        if self.paths:
            previous, envelope = checkpoints.load_checkpoint(self.paths[-1], run=self.run,
                                                              environment=self.environment, training=self.training)
            if envelope["checkpoint_id"] == identity:
                self.index_checkpoints()
                return self.paths[-1], previous
        path = checkpoints.publish_checkpoint(payload, writer=self.writer, attempt_id=self.receipt["attempt_id"],
                                              run=self.run, environment=self.environment, training=self.training,
                                              checkout_root=self.checkout)
        self.paths.append(path)
        self.index_checkpoints()
        return path, payload

    def initialize(self, resume):
        if resume is None:
            checkpoints.seed_runtime(self.run["identity"]["seeds"], cuda=self.environment["backend"] == "cuda")
            self.model = models.CNNRC(seed=self.run["identity"]["seeds"]["derived_seeds"]["model_initialization"])
            self.model.to(self.receipt["device"])
            self.optimizer = checkpoints.make_optimizer(self.model, self.run["identity"]["configuration"]["resolved_candidate"]["optimizer"])
            self.state = {"position": _position(self.run, self.training, 0), "training_accumulators": _empty_accumulators(),
                          "validation_history": [], "selection_state": None,
                          "consumed_validation_event_count": 0, "consumed_selection_count": 0,
                          "early_stopping_state": {"enabled": False}}
        else:
            self.paths, self.index = _load_indexes(self.writer.root, self.run, self.environment, self.training)
            runs.require(self.paths and resume == self.paths[-1], "Resume must name the latest recovery checkpoint.")
            payload, _ = checkpoints.load_checkpoint(resume, run=self.run, environment=self.environment, training=self.training)
            self.model, self.optimizer = checkpoints.restore_checkpoint(
                payload, run=self.run, environment=self.environment, training=self.training,
                writer=self.writer, checkout_root=self.checkout,
            )
            self.state = checkpoints.recovery_state(payload)
            self.index_checkpoints()
        self.safe = True

    def step(self):
        position = self.state["position"]
        ids = [self.training[index].metadata.logical_example_id for index in range(len(self.training))]
        order = runs.epoch_order(ids, self.run["identity"]["seeds"]["derived_seeds"]["training_data_order"], position["current_epoch"])
        runs.require(runs.permutation_hash(ids, order) == position["epoch_permutation_hash"], "Epoch permutation differs.")
        size = self.run["identity"]["configuration"]["resolved_candidate"]["batch_size"]
        indices = runs.remaining_batches(order, size, position["next_batch_index"])[0]
        batch = datasets.collate_exd_hox([self.training[index] for index in indices])
        self.safe = False
        increments = training_step(self.model, self.optimizer, batch, self.run, position["current_epoch"])
        for name in ACCUMULATOR_FIELDS:
            self.state["training_accumulators"][name] += increments[name]
        position["next_batch_index"] += 1
        position["global_update"] += 1
        if position["next_batch_index"] == self.run["identity"]["budget"]["batches_per_epoch"]:
            position["phase"] = "validation_pending"
        self.safe = True

    def commit_validation(self, result):
        epoch = self.state["position"]["current_epoch"]
        pending, pending_envelope = checkpoints.load_checkpoint(self.pending_path, run=self.run,
                                                                environment=self.environment, training=self.training)
        event = {"event_id": _event_id(self.run, epoch), "epoch": epoch,
                 "global_update": self.state["position"]["global_update"], "checkpoint_ref": {"kind": "self"}, **result}
        _validate_history_event(event, self.run)
        # Build the exact next payload before publication; it still uses B3a's
        # self-reference encoding, avoiding a circular checkpoint/event hash.
        next_state = checkpoints.recovery_state(pending)
        next_state["position"].update(phase="epoch_complete", completed_epoch_count=epoch + 1)
        next_state["validation_history"].append(event)
        next_state["selection_state"] = select_best(next_state["validation_history"])
        next_state["consumed_validation_event_count"] = epoch + 1
        next_state["consumed_selection_count"] = epoch + 1
        previous_state = self.state
        self.state = next_state
        try:
            complete = self.capture()
        finally:
            self.state = previous_state
        record = _record("cnn_rc_validation_event.v1", self.run, event_id=event["event_id"], epoch=epoch,
                         global_update=event["global_update"], pending_checkpoint_id=pending_envelope["checkpoint_id"],
                         epoch_complete_checkpoint_id="ckpt_" + checkpoints.state_fingerprint(complete), history_event=event)
        self.publish(event["event_id"], record)
        return record

    def validate_epoch(self):
        self.pending_path, pending = self.save_checkpoint()
        epoch = self.state["position"]["current_epoch"]
        event_id = _event_id(self.run, epoch)
        path = self.writer.root / event_id
        pending_id = "ckpt_" + checkpoints.state_fingerprint(pending)
        if os.path.lexists(path):
            record = read_artifacts(path, self.run)["record.json"]
            self.receipt["environment"]["validation_reuses"].append(event_id)
        else:
            started_path = self.writer.root / ("validation-started-" + str(epoch))
            if os.path.lexists(started_path):
                started = read_artifacts(started_path, self.run)["record.json"]
                runs.require(started["pending_checkpoint_id"] == pending_id, "Validation replay checkpoint differs.")
                self.receipt["environment"]["validation_replays"].append(event_id)
            else:
                self.publish(started_path.name, _record("cnn_rc_validation_started.v1", self.run,
                             event_id=event_id, epoch=epoch, pending_checkpoint_id=pending_id,
                             attempt_id=self.receipt["attempt_id"]))
            # Frozen checkpoint is loaded through B3a, including optimizer/RNG;
            # validation cannot alter any of those states.
            self.model, self.optimizer = checkpoints.restore_checkpoint(
                pending, run=self.run, environment=self.environment, training=self.training,
                writer=self.writer, checkout_root=self.checkout,
            )
            result = evaluate_validation(self.model, self.optimizer, self.validation, self.run)
            record = self.commit_validation(result)
        runs.require(record["event_id"] == event_id and record["pending_checkpoint_id"] == pending_id,
                     "Committed validation checkpoint differs.")
        self.state["validation_history"].append(record["history_event"])
        self.state["selection_state"] = select_best(self.state["validation_history"])
        self.state["position"].update(phase="epoch_complete", completed_epoch_count=epoch + 1)
        self.state["consumed_validation_event_count"] = epoch + 1
        self.state["consumed_selection_count"] = epoch + 1
        self.safe = True
        complete_path, complete = self.save_checkpoint()
        runs.require(record["epoch_complete_checkpoint_id"] == "ckpt_" + checkpoints.state_fingerprint(complete),
                     "Committed validation does not match epoch-complete checkpoint.")
        self.state = checkpoints.recovery_state(complete)
        self.publish_epoch(record)

    def publish_epoch(self, event):
        accumulators = self.state["training_accumulators"]
        summary = _training_summary(accumulators)
        record = _record("cnn_rc_epoch_metrics.v1", self.run, epoch=event["epoch"], global_update=event["global_update"],
                         training=summary, validation_event_id=event["event_id"], pending_checkpoint_id=event["pending_checkpoint_id"],
                         epoch_complete_checkpoint_id=event["epoch_complete_checkpoint_id"])
        path = self.writer.root / ("epoch-" + str(event["epoch"]))
        if os.path.lexists(path):
            runs.require(read_artifacts(path, self.run)["record.json"] == record, "Epoch result differs.")
        else:
            self.publish(path.name, record)

    def finish(self):
        paths, index = validate_run_artifacts(self.writer.root, self.run, self.environment, self.training)
        runs.require(index is not None and len(index["entries"]) == len(paths), "Incomplete terminal index.")
        selected_id = index["selected_best_checkpoint_id"]
        runs.require(selected_id is not None, "No eligible selected checkpoint.")
        selected_path = self.writer.root / next(entry["path"] for entry in index["entries"] if entry["checkpoint_id"] == selected_id)
        selected, selected_envelope = checkpoints.load_checkpoint(selected_path, run=self.run, environment=self.environment, training=self.training)
        terminal, terminal_envelope = checkpoints.load_checkpoint(paths[-1], run=self.run, environment=self.environment, training=self.training)
        runs.require(terminal["position"]["phase"] == "epoch_complete" and terminal["position"]["completed_epoch_count"] == 2
                     and terminal["consumed_validation_event_count"] == terminal["consumed_selection_count"] == 2,
                     "Training has not completed its exact budget.")
        chosen = selected["validation_history"][-1]
        for event in terminal["validation_history"]:
            recorded = read_artifacts(self.writer.root / event["event_id"], self.run)["record.json"]
            normalized = copy.deepcopy(event)
            normalized["checkpoint_ref"] = {"kind": "self"}
            runs.require(recorded["history_event"] == normalized, "Committed validation history differs.")
            read_artifacts(self.writer.root / ("epoch-" + str(event["epoch"])), self.run)
        event = read_artifacts(self.writer.root / chosen["event_id"], self.run)["record.json"]
        runs.require(event["epoch_complete_checkpoint_id"] == selected_id, "Selected validation reference differs.")
        inventory = _inventory(self.writer.root)
        _verify_inventory(self.writer.root, inventory)
        summary = _record("cnn_rc_validation_summary.v1", self.run,
                          selected_checkpoint_id=selected_id, selection=self.run["identity"]["selection"],
                          seeds=self.run["identity"]["seeds"], data=self.run["identity"]["data"], software=self.run["identity"]["software"],
                          environment_compatibility=self.environment, environment_compatibility_hash=checkpoints.environment_hash(self.environment),
                          producing_attempt_id=self.receipt["attempt_id"], best_epoch=chosen["epoch"], best_global_update=chosen["global_update"],
                          validation_event_id=chosen["event_id"], metrics=chosen["metrics"], rc_diagnostic=chosen["rc_diagnostic"],
                          artifact_fingerprints=inventory, status="succeeded")
        records = {"validation-summary.json": summary, "attempt-terminal.json": _terminal(self.receipt),
                   "inventory.json": _record("cnn_rc_durable_inventory.v1", self.run, files=inventory)}
        fingerprints = sorted((_fingerprint(name, canonical_json_bytes(record) + b"\n") for name, record in records.items()),
                              key=lambda item: item["path"])
        completion = _record("downstream_completion.v1", self.run, selected_checkpoint_id=selected_id,
                             terminal_recovery_checkpoint_id=terminal_envelope["checkpoint_id"], validation_event_id=chosen["event_id"],
                             global_update=terminal["position"]["global_update"], consumed_validation_event_count=2,
                             consumed_selection_count=2, artifact_fingerprints=fingerprints, status="succeeded")
        records["completion.json"] = completion
        snapshots = {}
        for path in paths:
            payload, envelope = checkpoints.load_checkpoint(path, run=self.run, environment=self.environment, training=self.training)
            snapshots[envelope["checkpoint_id"]] = payload
        starts = {self.receipt["attempt_id"] + "-start": read_artifacts(
            self.writer.root / (self.receipt["attempt_id"] + "-start"), self.run)["record.json"]}
        _validate_completion_records(records, self.writer.root, self.run, self.environment, snapshots, index, starts)
        files = _bundle_files(records, self.run, self.receipt["attempt_id"])
        self.writer.check()
        runs.verify_run_sources(self.run, self.checkout)
        _verify_inventory(self.writer.root, inventory)
        checkpoints.publish_bundle(self.writer.root / "completion", files)
        runs.require(read_artifacts(self.writer.root / "completion", self.run) == records, "Completion readback differs.")
        return completion

    def execute(self):
        while self.state["position"]["completed_epoch_count"] < 2:
            position = self.state["position"]
            if position["phase"] == "epoch_complete":
                event = read_artifacts(self.writer.root / _event_id(self.run, position["current_epoch"]), self.run)["record.json"]
                self.publish_epoch(event)
                self.state["position"] = _position(self.run, self.training, position["current_epoch"] + 1)
                self.state["training_accumulators"] = _empty_accumulators()
            elif position["phase"] == "train":
                self.step()
                if self.state["position"]["phase"] == "train" and self.state["position"]["global_update"] % 100 == 0:
                    self.save_checkpoint()
            else:
                self.validate_epoch()
        event = read_artifacts(self.writer.root / _event_id(self.run, 1), self.run)["record.json"]
        self.publish_epoch(event)
        return self.finish()


def train_cnn_rc(*, stage_root: Path, expected_stage_id: str, config: Path, tf: str,
                 level_id: str, downstream_seed: int, candidate_id: str, output_root: Path,
                 attempt_root: Path, expected_software_commit: str, device: str,
                 resume_checkpoint: Path | None = None) -> dict:
    """Resolve and execute exactly one run, or resume its exact recovery state."""
    checkout, software = _verify_sources(expected_software_commit)
    runs.require(Path(os.path.abspath(config)) == checkout / CONFIG_PATH, "Only the tracked CNN-RC configuration is accepted.")
    stage = _absolute_directory(stage_root)
    output = _absolute_directory(output_root)
    attempt = _absolute_directory(attempt_root)
    _separate(stage, output, attempt)
    run = runs.resolve_run(config_path=Path(config), stage_root=stage, transcription_factor=tf, level_id=level_id,
                           expected_stage_id=expected_stage_id, downstream_seed=downstream_seed, candidate_id=candidate_id,
                           checkout_root=checkout, expected_software_commit=expected_software_commit, source_paths=SOURCE_PATHS)
    runs.require(run["identity"]["software"] == software, "Runtime code changed.")
    data = datasets.open_public_tf_data(stage, transcription_factor=tf, expected_stage_id=expected_stage_id)
    training = data.dataset("training", level_id=level_id)
    validation = data.dataset("validation")
    environment = configure_runtime(device)
    root = output / run["run_id"]
    resume = None if resume_checkpoint is None else Path(os.path.abspath(resume_checkpoint))
    parent = None
    resume_id = None
    if resume is None:
        runs.require(not os.path.lexists(root), "Existing run requires explicit recovery; completed runs are immutable.")
    else:
        runs.require(resume.parent == root, "Resume checkpoint outside the scientific run root.")
        runs.require(not os.path.lexists(root / "completion"), "Completed run cannot be resumed.")
        runs.require(read_artifacts(root / "resolved-run", run)["record.json"] == run, "Resolved run differs.")
        _, envelope = checkpoints.load_checkpoint(resume, run=run, environment=environment, training=training)
        validate_run_artifacts(root, run, environment, training)
        parent = envelope["producer_attempt_id"]
        resume_id = envelope["checkpoint_id"]
    _mkdir(root)
    _mkdir(attempt)
    receipt = _attempt(run, environment, stage, output, attempt, device, "training", resume_id, parent)
    attempt_directory = attempt / receipt["attempt_id"]
    _mkdir(attempt_directory)
    with checkpoints.run_writer(root, run["run_id"]) as writer:
        session = _TrainingRun(run, training, validation, environment, writer, checkout, receipt, attempt_directory)
        if resume is None:
            session.publish("resolved-run", run)
        start_files = _bundle_files({"record.json": receipt}, run, receipt["attempt_id"])
        checkpoints.publish_bundle(attempt_directory / "start", start_files)
        session.publish(receipt["attempt_id"] + "-start", receipt)
        try:
            session.initialize(resume)
            return session.execute()
        except checkpoints.PublishedDurabilityError:
            # A final name may already exist. Never retry or reinterpret it.
            raise
        except BaseException as error:
            # Only an external interruption at a known safe training boundary
            # can create a recovery snapshot. Numerical/forward failures abort.
            if isinstance(error, KeyboardInterrupt) and session.safe and session.state is not None:
                if session.state["position"]["phase"] == "train":
                    session.save_checkpoint()
            if not os.path.lexists(root / "completion"):
                terminal = _terminal(receipt, error)
                session.publish(receipt["attempt_id"] + "-terminal", terminal)
                checkpoints.publish_bundle(attempt_directory / "terminal", _bundle_files({"record.json": terminal}, run, receipt["attempt_id"]))
            raise


def verify_cnn_rc(*, stage_root: Path, expected_stage_id: str, checkpoint: Path,
                  output_root: Path, attempt_root: Path, expected_software_commit: str,
                  device: str) -> dict:
    """Verify recorded fixed validation; never run another selection evaluation."""
    checkout, software = _verify_sources(expected_software_commit)
    stage = _absolute_directory(stage_root)
    output = _absolute_directory(output_root)
    attempt = _absolute_directory(attempt_root)
    _separate(stage, output, attempt)
    path = Path(os.path.abspath(checkpoint))
    source_root = _absolute_directory(path.parent)
    _separate(source_root, stage, output, attempt)
    raw_run = runs.strict_json(runs.read_regular(source_root / "resolved-run" / "record.json"))
    runs.validate_run(raw_run)
    run = read_artifacts(source_root / "resolved-run", raw_run)["record.json"]
    runs.require(source_root.name == run["run_id"] and run["identity"]["software"] == software, "Checkpoint source identity differs.")
    selection = run["identity"]["selection"]
    runs.require(run["identity"]["data"]["stage_id"] == expected_stage_id, "Expected stage differs from checkpoint.")
    data = datasets.open_public_tf_data(stage, transcription_factor=selection["transcription_factor"], expected_stage_id=expected_stage_id)
    training = data.dataset("training", level_id=selection["requested_level_id"])
    validation = data.dataset("validation")
    runs.require(runs.membership_hash(validation) == selection["validation_membership_hash"], "Fixed validation differs.")
    environment = configure_runtime(device)
    payload, envelope = checkpoints.load_checkpoint(path, run=run, environment=environment, training=training)
    runs.require(payload["position"]["phase"] == "epoch_complete", "Checkpoint has no committed validation event.")
    paths, index = validate_run_artifacts(source_root, run, environment, training, completed=True)
    runs.require(index is not None and len(paths) == len(index["entries"]), "Incomplete checkpoint index.")
    completed = read_artifacts(source_root / "completion", run)
    completion = completed["completion.json"]
    _verify_inventory(source_root, completed["inventory.json"]["files"])
    _verify_inventory(source_root / "completion", completion["artifact_fingerprints"])
    event_id = _event_id(run, payload["position"]["current_epoch"])
    event = read_artifacts(source_root / event_id, run)["record.json"]
    runs.require(event["epoch_complete_checkpoint_id"] == envelope["checkpoint_id"]
                 and event["history_event"] == payload["validation_history"][-1], "Recorded checkpoint event differs.")
    before = _inventory(source_root)
    receipt = _attempt(run, environment, stage, output, attempt, device, "validation", envelope["checkpoint_id"], envelope["producer_attempt_id"])
    root = output / run["run_id"]
    runs.require(not os.path.lexists(root), "Verification output already exists.")
    _mkdir(root)
    _mkdir(attempt / receipt["attempt_id"])
    with checkpoints.run_writer(root, run["run_id"]) as writer:
        checkpoints.publish_bundle(attempt / receipt["attempt_id"] / "start", _bundle_files({"record.json": receipt}, run, receipt["attempt_id"]))
        report = _record("cnn_rc_validation_verification.v1", run, checkpoint_id=envelope["checkpoint_id"],
                         validation_event_id=event_id, recorded_event_hash=event["manifest_hash"], source_completion_hash=completion["manifest_hash"],
                         producing_attempt_id=receipt["attempt_id"], selection_unchanged=True, additional_selection_evaluations=0,
                         artifact_fingerprints=before, status="succeeded")
        runs.require(_inventory(source_root) == before, "Training results changed during verification.")
        writer.check()
        runs.verify_run_sources(run, checkout)
        records = {"verification.json": report, "attempt-start.json": receipt, "attempt-terminal.json": _terminal(receipt)}
        checkpoints.publish_bundle(root / "completion", _bundle_files(records, run, receipt["attempt_id"]))
        runs.require(read_artifacts(root / "completion", run) == records, "Verification publication readback differs.")
        return report
