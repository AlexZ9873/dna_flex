"""Strict B3a recovery state and immutable publication, without model execution.

Only primitive values and dense tensors enter the weights-only archive. Tensor
fingerprints describe logical values, not torch.save container bytes. Context
managers below are necessary to release locks/descriptors on every failure.
"""

from __future__ import annotations

from contextlib import contextmanager
from collections import OrderedDict
import copy
import ctypes
import errno
import fcntl
import hashlib
import io
import math
import os
from pathlib import Path
import platform
import random
import stat
import struct
import sys
import tempfile
from typing import Any, Iterator

import numpy as np
import torch

from src.cnn_rc import CNNRC, architecture_settings
from src.downstream_fingerprints import canonical_json_bytes
from src.downstream_run import (
    RunContractError, domain_hash, epoch_order, hex_digest, identifier, integer,
    keys, logical_path, membership_hash, permutation_hash, read_regular, require,
    seed_record, strict_json, validate_run, verify_run_sources,
)


SCHEMA = "downstream_checkpoint.v1"


def make_optimizer(model: CNNRC, definition: dict[str, Any]) -> torch.optim.Adam:
    """Construct the pinned optimizer without taking an optimization step."""
    return torch.optim.Adam(
        model.parameters(), lr=definition["learning_rate"], betas=tuple(definition["betas"]),
        eps=definition["epsilon"], weight_decay=definition["weight_decay"],
        amsgrad=definition["amsgrad"], foreach=definition["foreach"], fused=definition["fused"],
        maximize=definition["maximize"], capturable=definition["capturable"],
        differentiable=definition["differentiable"],
    )


def capture_rng() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    cuda_state = None
    if torch.cuda.is_initialized():
        cuda_state = [state.clone() for state in torch.cuda.get_rng_state_all()]
    return {
        "python": random.getstate(),
        "numpy": {"algorithm": numpy_state[0], "keys": torch.tensor(numpy_state[1].astype(np.int64)),
                  "position": numpy_state[2], "has_gauss": numpy_state[3], "cached_gaussian": numpy_state[4]},
        "torch_cpu": torch.get_rng_state().clone(), "torch_cuda": cuda_state,
        "component_generators": {},
    }


def seed_runtime(record: dict[str, Any], *, cuda: bool = False) -> dict[str, Any]:
    """Explicit initialization only; resume restores states instead."""
    require(record == seed_record(record["parent_seed"]), "Invalid seed record.")
    seeds = record["derived_seeds"]
    random.seed(seeds["python_runtime"])
    numpy_seed = seeds["numpy_runtime"] % 2**32
    np.random.seed(numpy_seed)
    torch.manual_seed(seeds["torch_runtime"])
    if cuda:
        torch.cuda.manual_seed_all(seeds["torch_runtime"])
    return {"numpy_legacy_seed_mod_2_32": numpy_seed}


def _rng_tensor(value: Any) -> None:
    require(type(value) is torch.Tensor and value.device.type == "cpu" and value.dtype == torch.uint8
            and value.ndim == 1 and value.numel() > 0, "Malformed RNG tensor.")


def validate_rng(value: Any) -> None:
    keys(value, ("python", "numpy", "torch_cpu", "torch_cuda", "component_generators"), "RNG")
    python_state = value["python"]
    require(type(python_state) is tuple and len(python_state) == 3, "Malformed Python RNG.")
    require(type(python_state[0]) is int and python_state[0] == 3
            and type(python_state[1]) is tuple and len(python_state[1]) == 625,
            "Malformed Python RNG state.")
    require(all(integer(item) and item < 2**32 for item in python_state[1][:-1])
            and integer(python_state[1][-1]) and python_state[1][-1] <= 624, "Invalid Python MT state.")
    require(python_state[2] is None or (type(python_state[2]) is float and math.isfinite(python_state[2])), "Invalid Python Gaussian cache.")
    random.Random().setstate(python_state)
    numpy_state = value["numpy"]
    keys(numpy_state, ("algorithm", "keys", "position", "has_gauss", "cached_gaussian"), "NumPy RNG")
    require(numpy_state["algorithm"] == "MT19937", "Unknown NumPy RNG.")
    array = numpy_state["keys"]
    require(type(array) is torch.Tensor and array.dtype == torch.int64 and array.device.type == "cpu"
            and tuple(array.shape) == (624,) and bool(((array >= 0) & (array < 2**32)).all()), "Invalid NumPy keys.")
    require(integer(numpy_state["position"]) and numpy_state["position"] <= 624
            and type(numpy_state["has_gauss"]) is int and numpy_state["has_gauss"] in (0, 1)
            and type(numpy_state["cached_gaussian"]) is float and math.isfinite(numpy_state["cached_gaussian"]), "Invalid NumPy position/cache.")
    _rng_tensor(value["torch_cpu"])
    torch.Generator(device="cpu").set_state(value["torch_cpu"])
    require(value["torch_cuda"] is None or (type(value["torch_cuda"]) is list and bool(value["torch_cuda"])), "Malformed CUDA RNG list.")
    if value["torch_cuda"] is not None:
        for tensor in value["torch_cuda"]:
            _rng_tensor(tensor)
    require(value["component_generators"] == {} and type(value["component_generators"]) is dict,
            "v1 has no mutable component generators.")


def restore_rng(value: dict[str, Any]) -> None:
    validate_rng(value)
    if value["torch_cuda"] is not None:
        require(torch.cuda.is_initialized() and len(value["torch_cuda"]) == torch.cuda.device_count(), "CUDA RNG environment differs.")
    random.setstate(value["python"])
    numpy_state = value["numpy"]
    np.random.set_state((numpy_state["algorithm"], numpy_state["keys"].numpy().astype(np.uint32),
                         numpy_state["position"], numpy_state["has_gauss"], numpy_state["cached_gaussian"]))
    torch.set_rng_state(value["torch_cpu"])
    if value["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(value["torch_cuda"])


def _cuda_driver_version() -> str:
    """Bind both CUDA API compatibility and the actual NVIDIA driver release."""
    library = ctypes.CDLL("libcuda.so.1")
    function = library.cuDriverGetVersion
    function.argtypes = [ctypes.POINTER(ctypes.c_int)]
    function.restype = ctypes.c_int
    version = ctypes.c_int()
    require(function(ctypes.byref(version)) == 0 and version.value > 0, "CUDA driver identity unavailable.")
    management = ctypes.CDLL("libnvidia-ml.so.1")
    initialize = management.nvmlInit_v2
    initialize.argtypes = []
    initialize.restype = ctypes.c_int
    shutdown = management.nvmlShutdown
    shutdown.argtypes = []
    shutdown.restype = ctypes.c_int
    query = management.nvmlSystemGetDriverVersion
    query.argtypes = [ctypes.c_char_p, ctypes.c_uint]
    query.restype = ctypes.c_int
    require(initialize() == 0, "NVIDIA driver release unavailable.")
    try:
        release = ctypes.create_string_buffer(256)
        require(query(release, len(release)) == 0 and bool(release.value), "NVIDIA driver release unavailable.")
        return canonical_json_bytes({"cuda_driver_api": version.value, "nvidia_driver_release": release.value.decode("ascii")}).decode("ascii")
    finally:
        require(shutdown() == 0, "NVIDIA management shutdown failed.")


def capture_environment() -> dict[str, Any]:
    """Observe compatibility facts without initializing CUDA or invoking jobs."""
    backend = "cuda" if torch.cuda.is_initialized() else "cpu"
    return {
        "schema_version": "downstream_environment_compatibility.v1",
        "python": platform.python_version(), "numpy": np.__version__, "torch": str(torch.__version__),
        "torch_build": torch.__config__.show(), "numpy_build": str(np.__config__.CONFIG),
        "os": platform.system(), "machine": platform.machine(), "processor": platform.processor(),
        "byteorder": sys.byteorder, "backend": backend,
        "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "driver": None if backend == "cpu" else _cuda_driver_version(),
        "device_class": (platform.machine() + ":" + torch.backends.cpu.get_cpu_capability()) if backend == "cpu" else torch.cuda.get_device_name(0),
        "device_count": 0 if backend == "cpu" else torch.cuda.device_count(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


ENVIRONMENT_FIELDS = (
    "schema_version", "python", "numpy", "torch", "torch_build", "numpy_build", "os", "machine", "processor",
    "byteorder", "backend", "threads", "interop_threads", "cuda", "cudnn", "driver", "device_class", "device_count",
    "deterministic_algorithms", "deterministic_warn_only", "cudnn_benchmark", "cudnn_deterministic",
    "cuda_matmul_tf32", "cudnn_tf32", "float32_matmul_precision", "cublas_workspace_config",
)


def validate_environment(value: Any) -> None:
    keys(value, ENVIRONMENT_FIELDS, "Environment")
    require(value["schema_version"] == "downstream_environment_compatibility.v1", "Environment schema differs.")
    for name in ("python", "numpy", "torch", "torch_build", "numpy_build", "os", "machine", "device_class"):
        require(type(value[name]) is str and bool(value[name]), "Missing environment version.")
    require(type(value["processor"]) is str and value["byteorder"] in ("little", "big"), "Invalid platform.")
    require(value["backend"] in ("cpu", "cuda") and integer(value["threads"], 1)
            and integer(value["interop_threads"], 1) and integer(value["device_count"]), "Invalid execution backend.")
    for name in ("cuda", "driver", "cublas_workspace_config"):
        require(value[name] is None or (type(value[name]) is str and bool(value[name])), "Invalid optional environment field.")
    require(value["cudnn"] is None or integer(value["cudnn"], 1), "Invalid cuDNN version.")
    for name in ("deterministic_algorithms", "deterministic_warn_only", "cudnn_benchmark", "cudnn_deterministic", "cuda_matmul_tf32", "cudnn_tf32"):
        require(type(value[name]) is bool, "Invalid deterministic setting.")
    require(value["deterministic_algorithms"] and not value["deterministic_warn_only"]
            and not value["cudnn_benchmark"] and value["cudnn_deterministic"]
            and not value["cuda_matmul_tf32"] and not value["cudnn_tf32"]
            and value["float32_matmul_precision"] == "highest", "Numerical policy differs.")
    if value["backend"] == "cuda":
        require(value["device_count"] == 1 and value["cuda"] is not None and value["driver"] is not None
                and value["cublas_workspace_config"] == ":4096:8", "Incomplete single-GPU compatibility evidence.")
    else:
        require(value["device_count"] == 0 and value["driver"] is None, "CPU environment contains active CUDA state.")


def environment_hash(value: dict[str, Any]) -> str:
    validate_environment(value)
    return domain_hash("downstream_environment_compatibility.v1", value)


def _tensor_record(tensor: torch.Tensor, path: str) -> dict[str, Any]:
    require(type(tensor) is torch.Tensor and tensor.layout == torch.strided and not tensor.is_quantized,
            "Unsupported tensor representation.")
    require(tensor.dtype in (torch.float32, torch.float64, torch.int64, torch.uint8), "Unsupported tensor dtype.")
    cpu = tensor.detach().cpu().contiguous()
    if cpu.is_floating_point():
        require(bool(torch.isfinite(cpu).all()), "Nonfinite state tensor.")
    array = cpu.numpy()
    little = array.astype(array.dtype.newbyteorder("<"), copy=False)
    return {"path": path, "dtype": str(cpu.dtype), "shape": list(cpu.shape),
            "sha256": hashlib.sha256(little.tobytes(order="C")).hexdigest()}


def _encode(value: Any, path: str, tensors: dict, inventory: list) -> dict:
    if type(value) is torch.Tensor:
        record = _tensor_record(value, path)
        tensors[path] = value.detach().cpu().contiguous().clone()
        inventory.append(record)
        return {"type": "tensor", "record": record}
    if type(value) in (dict, OrderedDict):
        require(all(type(key) in (str, int) for key in value), "Invalid state mapping key.")
        entries = []
        ordered = sorted(value, key=lambda key: (type(key).__name__, str(key)))
        for index, key in enumerate(ordered):
            component = type(key).__name__ + ":" + str(key).replace("~", "~0").replace("/", "~1")
            entries.append([_encode(key, path + "/key/" + str(index), tensors, inventory),
                            _encode(value[key], path + "/" + component, tensors, inventory)])
        return {"type": "dict", "entries": entries}
    if type(value) in (tuple, list):
        items = []
        for index, item in enumerate(value):
            items.append(_encode(item, path + "/" + str(index), tensors, inventory))
        return {"type": type(value).__name__, "items": items}
    require(type(value) in (str, bool, int, float, type(None)), "Unsupported checkpoint object.")
    if type(value) is float:
        require(math.isfinite(value), "Nonfinite state scalar.")
        return {"type": "float", "bits": struct.pack(">d", value).hex()}
    return {"type": type(value).__name__, "value": value}


def _decode(node: Any, tensors: dict, used: set) -> Any:
    require(type(node) is dict and "type" in node, "Malformed state node.")
    kind = node["type"]
    if kind == "tensor":
        keys(node, ("type", "record"), "Tensor node")
        record = node["record"]
        keys(record, ("path", "dtype", "shape", "sha256"), "Tensor record")
        path = record["path"]
        require(type(path) is str and path in tensors and path not in used, "Missing or repeated tensor.")
        require(_tensor_record(tensors[path], path) == record, "Tensor fingerprint differs.")
        used.add(path)
        return tensors[path].clone()
    if kind == "dict":
        keys(node, ("type", "entries"), "Mapping node")
        require(type(node["entries"]) is list, "Invalid mapping entries.")
        result = {}
        for pair in node["entries"]:
            require(type(pair) is list and len(pair) == 2, "Invalid mapping pair.")
            key = _decode(pair[0], tensors, used)
            require(type(key) in (str, int) and key not in result, "Invalid or duplicate state key.")
            result[key] = _decode(pair[1], tensors, used)
        return result
    if kind in ("list", "tuple"):
        keys(node, ("type", "items"), "Sequence node")
        require(type(node["items"]) is list, "Invalid state sequence.")
        items = [_decode(item, tensors, used) for item in node["items"]]
        return tuple(items) if kind == "tuple" else items
    if kind == "float":
        keys(node, ("type", "bits"), "Float node")
        require(hex_digest(node["bits"], 16), "Invalid float bits.")
        value = struct.unpack(">d", bytes.fromhex(node["bits"]))[0]
        require(math.isfinite(value), "Nonfinite scalar.")
        return value
    types = {"str": str, "int": int, "bool": bool, "NoneType": type(None)}
    keys(node, ("type", "value"), "Scalar node")
    require(kind in types and type(node["value"]) is types[kind], "Invalid scalar type.")
    return node["value"]


def semantic_state(value: Any) -> tuple[dict, dict, list]:
    tensors = {}
    inventory = []
    tree = _encode(value, "state", tensors, inventory)
    return tree, tensors, inventory


def state_fingerprint(value: Any) -> str:
    tree, _, _ = semantic_state(value)
    return domain_hash("downstream_checkpoint_semantics.v1", tree)


PAYLOAD_FIELDS = (
    "schema_version", "run_id", "resolved_run_manifest", "resolved_run_hash", "software_identity",
    "model_contract", "model_architecture_metadata", "model_state", "optimizer_definition",
    "optimizer_parameter_names_in_order", "optimizer_state", "scheduler_state", "position",
    "orientation_policy", "seed_record", "rng", "training_accumulators", "validation_history",
    "consumed_validation_event_count", "consumed_selection_count", "selection_state", "early_stopping_state",
    "data_identity", "environment_compatibility", "tensor_inventory",
)


def _tensor_schema(actual: Any, expected: torch.Tensor) -> None:
    require(type(actual) is torch.Tensor and actual.dtype == expected.dtype
            and tuple(actual.shape) == tuple(expected.shape) and actual.device.type == "cpu", "Tensor name/shape/dtype/device differs.")
    _tensor_record(actual, "checked")


def validate_metrics(metrics: Any, run: dict) -> None:
    keys(metrics, ("sample_count", "unique_rc_group_count", "mse", "rmse", "r2", "pearson", "spearman", "undefined_reasons"), "Metrics")
    selection = run["identity"]["selection"]
    require(type(metrics["sample_count"]) is int and metrics["sample_count"] == selection["validation_row_count"]
            and type(metrics["unique_rc_group_count"]) is int and metrics["unique_rc_group_count"] == selection["validation_rc_group_count"], "Metric counts differ.")
    reasons = metrics["undefined_reasons"]
    require(type(reasons) is dict and set(reasons).issubset(("r2", "pearson", "spearman")), "Invalid undefined reasons.")
    for name in ("mse", "rmse", "r2", "pearson", "spearman"):
        value = metrics[name]
        if value is None:
            require(name in reasons and reasons[name] in ("insufficient_samples", "constant_targets", "constant_predictions"), "Missing undefined reason.")
        else:
            require(type(value) is float and math.isfinite(value) and name not in reasons, "Invalid metric value.")
    require(metrics["mse"] is not None and metrics["rmse"] is not None
            and 0 <= metrics["mse"] <= 1 and 0 <= metrics["rmse"] <= 1
            and metrics["rmse"] == math.sqrt(metrics["mse"]), "Invalid error metrics.")
    require(metrics["r2"] is not None and metrics["r2"] <= 1, "Undefined selection R2.")
    for name in ("pearson", "spearman"):
        require(metrics[name] is None or abs(metrics[name]) <= 1 + 1e-14, "Correlation out of range.")


def _validate_history(payload: dict, run: dict) -> None:
    history = payload["validation_history"]
    require(type(history) is list, "Invalid history.")
    position = payload["position"]
    count = position["current_epoch"] + int(position["phase"] == "epoch_complete")
    require(len(history) == count and type(payload["consumed_validation_event_count"]) is int
            and type(payload["consumed_selection_count"]) is int
            and payload["consumed_validation_event_count"] == count == payload["consumed_selection_count"], "Validation budget/history differs.")
    best = None
    for index, event in enumerate(history):
        keys(event, ("event_id", "epoch", "global_update", "checkpoint_ref", "metrics", "rc_diagnostic"), "Validation event")
        require(type(event["epoch"]) is int and event["epoch"] == index
                and type(event["global_update"]) is int and event["global_update"] == (index + 1) * run["identity"]["budget"]["batches_per_epoch"], "Validation event position differs.")
        expected_event = "validation_event_" + domain_hash("downstream_validation_event.v1", {"run_id": run["run_id"], "epoch": index, "global_update": event["global_update"]})
        require(event["event_id"] == expected_event, "Validation event ID differs.")
        reference = event["checkpoint_ref"]
        require(reference == {"kind": "self"} or (type(reference) is dict and set(reference) == {"kind", "checkpoint_id"}
                and reference["kind"] == "checkpoint" and identifier(reference["checkpoint_id"], "ckpt_")), "Invalid checkpoint reference.")
        if reference == {"kind": "self"}:
            require(position["phase"] == "epoch_complete" and index == len(history) - 1, "Invalid self checkpoint reference.")
        keys(event["metrics"], ("forward", "reverse_complement", "mean"), "Orientation metrics")
        for metrics in event["metrics"].values():
            validate_metrics(metrics, run)
        diagnostic = event["rc_diagnostic"]
        keys(diagnostic, ("atol", "rtol", "maximum_absolute_difference", "maximum_normalized_difference", "violation_count"), "RC diagnostic")
        require(type(diagnostic["atol"]) is float and diagnostic["atol"] == 1e-6
                and type(diagnostic["rtol"]) is float and diagnostic["rtol"] == 1e-5
                and type(diagnostic["violation_count"]) is int and diagnostic["violation_count"] == 0, "Failed invariance diagnostic.")
        for name in ("maximum_absolute_difference", "maximum_normalized_difference"):
            require(type(diagnostic[name]) is float and math.isfinite(diagnostic[name]) and diagnostic[name] >= 0, "Invalid RC difference.")
        require(diagnostic["maximum_normalized_difference"] <= 1, "RC tolerance exceeded.")
        score = event["metrics"]["mean"]
        if best is None or (score["r2"], -score["rmse"], -event["global_update"], -index) > best[0]:
            best = ((score["r2"], -score["rmse"], -event["global_update"], -index), event)
    expected = None
    if best is not None:
        event = best[1]
        expected = {"epoch": event["epoch"], "global_update": event["global_update"],
                    "metrics": event["metrics"]["mean"], "checkpoint_ref": event["checkpoint_ref"]}
    require(payload["selection_state"] == expected, "Best-selection state differs from history.")
    require(payload["early_stopping_state"] == {"enabled": False}, "Unexpected early stopping.")


def validate_checkpoint(payload: Any, run: dict, environment: dict, training) -> None:
    """Validate semantics and compatibility before mutating any live state."""
    validate_run(run)
    validate_environment(environment)
    keys(payload, PAYLOAD_FIELDS, "Checkpoint")
    identity = run["identity"]
    require(payload["schema_version"] == SCHEMA and payload["run_id"] == run["run_id"], "Wrong checkpoint run/schema.")
    require(payload["resolved_run_manifest"] == run and payload["resolved_run_hash"] == run["manifest_hash"], "Wrong resolved run.")
    for name, expected in (("software_identity", identity["software"]), ("model_contract", identity["model"]["contract_id"]),
                           ("model_architecture_metadata", architecture_settings()), ("seed_record", identity["seeds"]),
                           ("orientation_policy", identity["configuration"]["resolved_contract"]["orientation"]),
                           ("data_identity", {"data": identity["data"], "selection": identity["selection"]}),
                           ("optimizer_definition", identity["configuration"]["resolved_candidate"]["optimizer"]),
                           ("environment_compatibility", environment)):
        require(canonical_json_bytes(payload[name]) == canonical_json_bytes(expected), "Checkpoint compatibility differs: " + name)
    require(payload["scheduler_state"] is None, "Unexpected scheduler.")
    selection = identity["selection"]
    require(membership_hash(training) == selection["training_membership_hash"] and len(training) == selection["actual_logical_example_count"], "Resume membership differs.")
    metadata = training[0].metadata
    require(metadata.stage_id == identity["data"]["stage_id"] and metadata.transcription_factor == selection["transcription_factor"]
            and metadata.level_id == selection["requested_level_id"] and metadata.canonical_level_id == selection["canonical_level_id"], "Resume dataset selection differs.")
    position = payload["position"]
    keys(position, ("completed_epoch_count", "current_epoch", "next_batch_index", "global_update", "phase", "epoch_permutation_hash"), "Position")
    for name in ("completed_epoch_count", "current_epoch", "next_batch_index", "global_update"):
        require(integer(position[name]), "Invalid training position.")
    epoch = position["current_epoch"]
    batches = identity["budget"]["batches_per_epoch"]
    require(epoch < identity["budget"]["maximum_epochs"] and position["next_batch_index"] <= batches, "Position exceeds budget.")
    require(position["global_update"] == epoch * batches + position["next_batch_index"], "Global update differs from position.")
    require(position["phase"] in ("train", "validation_pending", "epoch_complete"), "Invalid checkpoint phase.")
    if position["phase"] == "train":
        require(position["next_batch_index"] < batches and position["completed_epoch_count"] == epoch, "Invalid train boundary.")
    else:
        require(position["next_batch_index"] == batches
                and position["completed_epoch_count"] == epoch + int(position["phase"] == "epoch_complete"), "Invalid epoch boundary.")
    ids = [training[index].metadata.logical_example_id for index in range(len(training))]
    order = epoch_order(ids, identity["seeds"]["derived_seeds"]["training_data_order"], epoch)
    require(position["epoch_permutation_hash"] == permutation_hash(ids, order), "Epoch permutation differs.")
    accumulators = payload["training_accumulators"]
    keys(accumulators, ("example_count", "update_count", "squared_error_sum", "regularization_sum", "total_loss_sum"), "Accumulators")
    expected_count = min(len(training), position["next_batch_index"] * identity["configuration"]["resolved_candidate"]["batch_size"])
    require(type(accumulators["example_count"]) is int and accumulators["example_count"] == expected_count
            and type(accumulators["update_count"]) is int and accumulators["update_count"] == position["next_batch_index"], "Accumulator counts differ.")
    for name in ("squared_error_sum", "regularization_sum", "total_loss_sum"):
        require(type(accumulators[name]) is float and math.isfinite(accumulators[name]) and accumulators[name] >= 0,
                "Invalid accumulator value.")
        if expected_count == 0:
            require(accumulators[name] == 0.0, "Nonzero initial accumulator.")
    _validate_history(payload, run)
    validate_rng(payload["rng"])
    require((payload["rng"]["torch_cuda"] is None) == (environment["backend"] == "cpu"), "CUDA RNG/backend mismatch.")
    if environment["backend"] == "cuda":
        require(len(payload["rng"]["torch_cuda"]) == environment["device_count"], "CUDA RNG count differs.")
    reference = CNNRC(seed=identity["seeds"]["derived_seeds"]["model_initialization"])
    reference_state = reference.state_dict()
    keys(payload["model_state"], tuple(reference_state), "Model state")
    for name, expected in reference_state.items():
        actual = payload["model_state"][name]
        if type(expected) is torch.Tensor:
            _tensor_schema(actual, expected)
        else:
            require(canonical_json_bytes(actual) == canonical_json_bytes(expected), "Architecture extra state differs.")
    require(bool((payload["model_state"]["running_var"] >= 0).all()), "Negative BN variance.")
    names = [name for name, _ in reference.named_parameters()]
    require(payload["optimizer_parameter_names_in_order"] == names, "Optimizer parameter names/order differ.")
    optimizer = make_optimizer(reference, payload["optimizer_definition"])
    optimizer_state = payload["optimizer_state"]
    keys(optimizer_state, ("state", "param_groups"), "Optimizer state")
    require(state_fingerprint(optimizer_state["param_groups"]) == state_fingerprint(optimizer.state_dict()["param_groups"]), "Optimizer groups differ.")
    moments = optimizer_state["state"]
    require(type(moments) is dict, "Invalid optimizer moments.")
    if position["global_update"] == 0:
        require(moments == {}, "Unexpected optimizer state before updates.")
    else:
        require(set(moments) == set(range(len(names))) and all(type(index) is int for index in moments), "Missing optimizer moments.")
        parameters = dict(reference.named_parameters())
        for index, name in enumerate(names):
            item = moments[index]
            keys(item, ("step", "exp_avg", "exp_avg_sq"), "Adam moments")
            _tensor_schema(item["step"], torch.tensor(0.0, dtype=torch.float32))
            require(item["step"].item() == position["global_update"], "Adam step differs.")
            _tensor_schema(item["exp_avg"], parameters[name].detach())
            _tensor_schema(item["exp_avg_sq"], parameters[name].detach())
            require(bool((item["exp_avg_sq"] >= 0).all()), "Negative Adam second moment.")
    base = dict(payload)
    del base["tensor_inventory"]
    _, _, inventory = semantic_state(base)
    require(payload["tensor_inventory"] == inventory, "Tensor inventory differs.")


def capture_checkpoint(
    *, run: dict, model: CNNRC, optimizer: torch.optim.Adam, training, environment: dict,
    position: dict, training_accumulators: dict, validation_history: list,
    selection_state: dict | None,
) -> dict[str, Any]:
    """Snapshot a caller-supplied safe boundary; does not run a model or step."""
    require(type(model) is CNNRC and type(optimizer) is torch.optim.Adam, "Unsupported state objects.")
    require(all(parameter.grad is None for parameter in model.parameters()), "Checkpoint requires cleared gradients.")
    actual_parameters = []
    for group in optimizer.param_groups:
        actual_parameters.extend(group["params"])
    expected_parameters = list(model.parameters())
    require(len(actual_parameters) == len(expected_parameters)
            and all(actual is expected for actual, expected in zip(actual_parameters, expected_parameters)),
            "Optimizer does not own the model parameters in canonical order.")
    identity = run["identity"]
    payload = {
        "schema_version": SCHEMA, "run_id": run["run_id"], "resolved_run_manifest": copy.deepcopy(run),
        "resolved_run_hash": run["manifest_hash"], "software_identity": copy.deepcopy(identity["software"]),
        "model_contract": model.contract_id, "model_architecture_metadata": model.get_extra_state()["architecture_settings"],
        "model_state": copy.deepcopy(model.state_dict()),
        "optimizer_definition": copy.deepcopy(identity["configuration"]["resolved_candidate"]["optimizer"]),
        "optimizer_parameter_names_in_order": [name for name, _ in model.named_parameters()],
        "optimizer_state": copy.deepcopy(optimizer.state_dict()), "scheduler_state": None,
        "position": copy.deepcopy(position), "orientation_policy": copy.deepcopy(identity["configuration"]["resolved_contract"]["orientation"]),
        "seed_record": copy.deepcopy(identity["seeds"]), "rng": capture_rng(),
        "training_accumulators": copy.deepcopy(training_accumulators), "validation_history": copy.deepcopy(validation_history),
        "consumed_validation_event_count": len(validation_history), "consumed_selection_count": len(validation_history),
        "selection_state": copy.deepcopy(selection_state), "early_stopping_state": {"enabled": False},
        "data_identity": {"data": copy.deepcopy(identity["data"]), "selection": copy.deepcopy(identity["selection"])},
        "environment_compatibility": copy.deepcopy(environment),
    }
    # Normalize every state tensor to CPU without changing semantic values.
    tree, tensors, inventory = semantic_state(payload)
    payload = _decode(tree, tensors, set())
    payload["tensor_inventory"] = inventory
    validate_checkpoint(payload, run, environment, training)
    return payload


def restore_checkpoint(payload: dict, *, run: dict, environment: dict, training, writer: RunWriter, checkout_root: Path):
    """Return fresh loaded objects and restore RNG last; no training forward."""
    writer.check()
    require(writer.run_id == run["run_id"], "Wrong resume writer.")
    verify_run_sources(run, checkout_root)
    require(capture_environment() == environment, "Actual resume environment differs.")
    validate_checkpoint(payload, run, environment, training)
    model = CNNRC(seed=run["identity"]["seeds"]["derived_seeds"]["model_initialization"])
    model.load_state_dict(payload["model_state"], strict=True)
    if environment["backend"] == "cuda":
        model.to("cuda:0")
    optimizer = make_optimizer(model, payload["optimizer_definition"])
    optimizer.load_state_dict(payload["optimizer_state"])
    restore_rng(payload["rng"])
    return model, optimizer


def recovery_state(payload: dict) -> dict[str, Any]:
    """Copy continuation fields and resolve self references to the loaded ID.

    Call after load_checkpoint validation. B3b advances the documented phase
    and batch cursor; this helper never advances training or consumes budgets.
    The original checkpoint remains byte/semantically unchanged.
    """
    keys(payload, PAYLOAD_FIELDS, "Checkpoint")
    checkpoint_id = "ckpt_" + state_fingerprint(payload)
    fields = ("position", "training_accumulators", "validation_history", "consumed_validation_event_count",
              "consumed_selection_count", "selection_state", "early_stopping_state")
    result = {name: copy.deepcopy(payload[name]) for name in fields}
    for event in result["validation_history"]:
        if event["checkpoint_ref"] == {"kind": "self"}:
            event["checkpoint_ref"] = {"kind": "checkpoint", "checkpoint_id": checkpoint_id}
    if result["selection_state"] is not None and result["selection_state"]["checkpoint_ref"] == {"kind": "self"}:
        result["selection_state"]["checkpoint_ref"] = {"kind": "checkpoint", "checkpoint_id": checkpoint_id}
    return result


class PublishedDurabilityError(OSError):
    """Final name exists; preserve it and do not automatically republish."""

    def __init__(self, destination: Path):
        self.destination = destination
        super().__init__("Published, durability unconfirmed; preserve and do not retry: " + str(destination))


def _open_directory(path: Path) -> int:
    absolute = Path(os.path.abspath(path))
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in absolute.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _assert_directory(path: Path, descriptor: int) -> None:
    current = _open_directory(path)
    try:
        expected = os.fstat(descriptor)
        observed = os.fstat(current)
        require((expected.st_dev, expected.st_ino) == (observed.st_dev, observed.st_ino), "Destination parent replaced.")
    finally:
        os.close(current)


def _exclusive_rename(descriptor: int, source: str, destination: str) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    function = None
    flag = 0
    if sys.platform == "darwin":
        function = getattr(library, "renameatx_np", None)
        flag = 4
    elif sys.platform == "linux":
        function = getattr(library, "renameat2", None)
        flag = 1
    require(function is not None, "Exclusive atomic publication unsupported.")
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    if function(descriptor, os.fsencode(source), descriptor, os.fsencode(destination), flag) != 0:
        code = ctypes.get_errno()
        if code in (errno.ENOSYS, errno.ENOTSUP, errno.EINVAL):
            raise RunContractError("Filesystem lacks exclusive atomic publication.")
        raise OSError(code, os.strerror(code), destination)


def _write_file(descriptor: int, name: str, data: bytes) -> None:
    handle = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=descriptor)
    try:
        offset = 0
        while offset < len(data):
            written = os.write(handle, data[offset:])
            require(written > 0, "Short write.")
            offset += written
        os.fchmod(handle, 0o444)
        os.fsync(handle)
    finally:
        os.close(handle)


def publish_bundle(destination: Path, files: dict[str, bytes]) -> None:
    """Publish flat verified files atomically, retaining failed private writes.

    Retention avoids deleting any artifact without approval. Failed private
    directories never match a final checkpoint/index name and cannot be read
    by the checkpoint loader. No existing output is modified.
    """
    require(type(files) is dict and bool(files), "Empty publication.")
    for name, content in files.items():
        require(logical_path(name) == name and "/" not in name and type(content) is bytes, "Invalid bundle file.")
    destination = Path(os.path.abspath(destination))
    parent = _open_directory(destination.parent)
    temporary = None
    descriptor = None
    try:
        temporary = Path(tempfile.mkdtemp(prefix=".b3-private-", dir=destination.parent))
        _assert_directory(destination.parent, parent)
        descriptor = os.open(temporary.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        for name, content in sorted(files.items()):
            _write_file(descriptor, name, content)
            require(read_regular(temporary / name) == content, "Publication readback differs.")
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o555)
        os.fsync(descriptor)
        _assert_directory(destination.parent, parent)
        _exclusive_rename(parent, temporary.name, destination.name)
        try:
            os.fsync(parent)
        except OSError as error:
            raise PublishedDurabilityError(destination) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent)


class RunWriter:
    """Descriptor-backed capability valid only inside run_writer()."""

    def __init__(self, root: Path, run_id: str, descriptor: int, lock: int):
        self.root = root
        self.run_id = run_id
        self.descriptor = descriptor
        self.lock = lock
        self.active = True

    def check(self) -> None:
        require(self.active, "Inactive run writer.")
        _assert_directory(self.root, self.descriptor)
        original = os.fstat(self.lock)
        current = os.stat(".writer.lock", dir_fd=self.descriptor, follow_symlinks=False)
        require((original.st_dev, original.st_ino) == (current.st_dev, current.st_ino), "Writer lock replaced.")
        require(not os.path.lexists(self.root / "completion"), "Finalized run cannot be changed or resumed.")


@contextmanager
def run_writer(root: Path, run_id: str) -> Iterator[RunWriter]:
    """One process owns this scientific run; lock files are never removed."""
    require(identifier(run_id, "run_") and Path(root).name == run_id, "Writer root must be named by run ID.")
    root = Path(os.path.abspath(root))
    descriptor = _open_directory(root)
    lock = None
    writer = None
    try:
        lock = os.open(".writer.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600, dir_fd=descriptor)
        require(stat.S_ISREG(os.fstat(lock).st_mode), "Writer lock is not regular.")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        writer = RunWriter(root, run_id, descriptor, lock)
        writer.check()
        yield writer
    finally:
        if writer is not None:
            writer.active = False
        if lock is not None:
            os.close(lock)
        os.close(descriptor)


def publish_checkpoint(payload: dict, *, writer: RunWriter, attempt_id: str, run: dict, environment: dict, training, checkout_root: Path) -> Path:
    writer.check()
    require(writer.run_id == run["run_id"] and identifier(attempt_id, "attempt_"), "Publication identity differs.")
    verify_run_sources(run, checkout_root)
    validate_checkpoint(payload, run, environment, training)
    tree, tensors, _ = semantic_state(payload)
    checkpoint_id = "ckpt_" + domain_hash("downstream_checkpoint_semantics.v1", tree)
    output = io.BytesIO()
    torch.save({"tree": tree, "tensors": tensors}, output)
    raw = output.getvalue()
    roundtrip = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    roundtrip_used = set()
    decoded = _decode(roundtrip["tree"], roundtrip["tensors"], roundtrip_used)
    require(roundtrip_used == set(roundtrip["tensors"]) and state_fingerprint(decoded) == checkpoint_id[5:],
            "Serialized checkpoint semantic readback differs.")
    envelope = {
        "schema_version": "downstream_checkpoint_publication.v1", "run_id": run["run_id"],
        "checkpoint_id": checkpoint_id, "producer_attempt_id": attempt_id,
        "semantic_hash": checkpoint_id[5:], "file": {"path": "state.pt", "byte_size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()},
    }
    envelope["manifest_hash"] = domain_hash("downstream_checkpoint_publication.v1", envelope)
    position = payload["position"]
    name = "update-" + str(position["global_update"]).zfill(12) + "-" + position["phase"] + "-" + checkpoint_id
    destination = writer.root / name
    verify_run_sources(run, checkout_root)
    writer.check()
    publish_bundle(destination, {"state.pt": raw, "manifest.json": canonical_json_bytes(envelope) + b"\n"})
    return destination


def load_checkpoint(path: Path, *, run: dict, environment: dict, training) -> tuple[dict, dict]:
    require(not Path(path).name.startswith("."), "Private output is not a checkpoint.")
    descriptor = _open_directory(path)
    try:
        require(set(os.listdir(descriptor)) == {"state.pt", "manifest.json"}, "Checkpoint inventory differs.")
        envelope = strict_json(read_regular(path / "manifest.json"))
        keys(envelope, ("schema_version", "run_id", "checkpoint_id", "producer_attempt_id", "semantic_hash", "file", "manifest_hash"), "Checkpoint envelope")
        content = dict(envelope)
        del content["manifest_hash"]
        require(envelope["manifest_hash"] == domain_hash("downstream_checkpoint_publication.v1", content), "Publication hash differs.")
        require(envelope["schema_version"] == "downstream_checkpoint_publication.v1"
                and envelope["run_id"] == run["run_id"] and identifier(envelope["producer_attempt_id"], "attempt_")
                and envelope["checkpoint_id"] == "ckpt_" + envelope["semantic_hash"] and hex_digest(envelope["semantic_hash"]), "Invalid publication identity.")
        record = envelope["file"]
        keys(record, ("path", "byte_size", "sha256"), "Checkpoint file")
        require(record["path"] == "state.pt" and integer(record["byte_size"], 1) and hex_digest(record["sha256"]), "Invalid checkpoint file record.")
        raw = read_regular(path / "state.pt")
        require(len(raw) == record["byte_size"] and hashlib.sha256(raw).hexdigest() == record["sha256"], "Raw checkpoint hash differs.")
        archive = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        keys(archive, ("tree", "tensors"), "State archive")
        require(type(archive["tensors"]) is dict, "Malformed tensor archive.")
        used = set()
        payload = _decode(archive["tree"], archive["tensors"], used)
        require(used == set(archive["tensors"]), "Unreferenced state tensors.")
        require(state_fingerprint(payload) == envelope["semantic_hash"], "Checkpoint semantic hash differs.")
        require(canonical_json_bytes(semantic_state(payload)[0]) == canonical_json_bytes(archive["tree"]), "Noncanonical state encoding.")
        validate_checkpoint(payload, run, environment, training)
        expected_name = "update-" + str(payload["position"]["global_update"]).zfill(12) + "-" + payload["position"]["phase"] + "-" + envelope["checkpoint_id"]
        require(path.name == expected_name, "Checkpoint filename identity differs.")
        _assert_directory(path, descriptor)
        return payload, envelope
    finally:
        os.close(descriptor)


def publish_index(*, writer: RunWriter, checkpoint_paths: list[Path], run: dict, environment: dict, training,
                  checkout_root: Path, previous: dict | None = None) -> tuple[Path, dict]:
    """An immutable cumulative index; latest/best are explicit validated IDs."""
    writer.check()
    verify_run_sources(run, checkout_root)
    require(writer.run_id == run["run_id"] and bool(checkpoint_paths), "Invalid index owner or empty index.")
    entries = []
    latest = None
    best = None
    for path in checkpoint_paths:
        require(path.parent == writer.root, "Index checkpoint outside run root.")
        payload, envelope = load_checkpoint(path, run=run, environment=environment, training=training)
        position = payload["position"]
        entries.append({"checkpoint_id": envelope["checkpoint_id"], "path": path.name,
                        "global_update": position["global_update"], "phase": position["phase"],
                        "producer_attempt_id": envelope["producer_attempt_id"], "semantic_hash": envelope["semantic_hash"],
                        "file": envelope["file"]})
        latest = envelope["checkpoint_id"]
        selection = payload["selection_state"]
        if selection is not None:
            reference = selection["checkpoint_ref"]
            best = latest if reference["kind"] == "self" else reference["checkpoint_id"]
    ids = [entry["checkpoint_id"] for entry in entries]
    require(len(set(ids)) == len(ids), "Duplicate index checkpoint.")
    phases = {"train": 0, "validation_pending": 1, "epoch_complete": 2}
    positions = [(entry["global_update"], phases[entry["phase"]]) for entry in entries]
    require(positions == sorted(set(positions)), "Index order or checkpoint position differs.")
    require(best is None or best in ids, "Selected checkpoint absent from index.")
    if best is not None:
        selected_path = checkpoint_paths[ids.index(best)]
        selected_payload, _ = load_checkpoint(selected_path, run=run, environment=environment, training=training)
        require(selected_payload["position"]["phase"] == "epoch_complete"
                and selected_payload["position"]["global_update"] == selection["global_update"]
                and selected_payload["validation_history"][-1]["metrics"]["mean"] == selection["metrics"],
                "Selected checkpoint does not carry the selected validation event.")
    revision = 0
    predecessor = None
    if previous is not None:
        validate_index(previous)
        require(previous["run_id"] == run["run_id"] and entries[:len(previous["entries"])] == previous["entries"]
                and len(entries) > len(previous["entries"]), "Index must extend predecessor.")
        revision = previous["revision"] + 1
        predecessor = previous["manifest_hash"]
        previous_path = writer.root / ("index-" + str(previous["revision"]).zfill(8)) / "index.json"
        require(strict_json(read_regular(previous_path)) == previous, "Unpublished predecessor index.")
    else:
        require(not os.path.lexists(writer.root / "index-00000000"), "Initial index already exists.")
    index = {"schema_version": "downstream_checkpoint_index.v1", "run_id": run["run_id"], "revision": revision,
             "predecessor_hash": predecessor, "entries": entries, "latest_recovery_checkpoint_id": latest,
             "selected_best_checkpoint_id": best}
    index["manifest_hash"] = domain_hash("downstream_checkpoint_index.v1", index)
    validate_index(index)
    path = writer.root / ("index-" + str(revision).zfill(8))
    verify_run_sources(run, checkout_root)
    writer.check()
    publish_bundle(path, {"index.json": canonical_json_bytes(index) + b"\n"})
    return path, index


def validate_index(index: dict) -> None:
    keys(index, ("schema_version", "run_id", "revision", "predecessor_hash", "entries",
                 "latest_recovery_checkpoint_id", "selected_best_checkpoint_id", "manifest_hash"), "Checkpoint index")
    require(index["schema_version"] == "downstream_checkpoint_index.v1" and identifier(index["run_id"], "run_")
            and integer(index["revision"]), "Invalid index identity.")
    require((index["predecessor_hash"] is None and index["revision"] == 0)
            or (hex_digest(index["predecessor_hash"]) and index["revision"] > 0), "Invalid index predecessor.")
    require(type(index["entries"]) is list and bool(index["entries"]), "Empty checkpoint index.")
    ids = []
    for entry in index["entries"]:
        keys(entry, ("checkpoint_id", "path", "global_update", "phase", "producer_attempt_id", "semantic_hash", "file"), "Index entry")
        require(identifier(entry["checkpoint_id"], "ckpt_") and entry["semantic_hash"] == entry["checkpoint_id"][5:]
                and identifier(entry["producer_attempt_id"], "attempt_") and integer(entry["global_update"])
                and entry["phase"] in ("train", "validation_pending", "epoch_complete"), "Invalid index entry identity.")
        require(logical_path(entry["path"]) == entry["path"] and "/" not in entry["path"], "Invalid index path.")
        expected_name = "update-" + str(entry["global_update"]).zfill(12) + "-" + entry["phase"] + "-" + entry["checkpoint_id"]
        require(entry["path"] == expected_name, "Index checkpoint filename differs.")
        keys(entry["file"], ("path", "byte_size", "sha256"), "Index file fingerprint")
        require(entry["file"]["path"] == "state.pt" and integer(entry["file"]["byte_size"], 1)
                and hex_digest(entry["file"]["sha256"]), "Invalid index file fingerprint.")
        ids.append(entry["checkpoint_id"])
    phase_order = {"train": 0, "validation_pending": 1, "epoch_complete": 2}
    positions = [(entry["global_update"], phase_order[entry["phase"]]) for entry in index["entries"]]
    require(positions == sorted(set(positions)), "Index checkpoint order differs.")
    require(len(set(ids)) == len(ids) and index["latest_recovery_checkpoint_id"] == ids[-1]
            and (index["selected_best_checkpoint_id"] is None or index["selected_best_checkpoint_id"] in ids), "Invalid index references.")
    content = dict(index)
    del content["manifest_hash"]
    require(index["manifest_hash"] == domain_hash("downstream_checkpoint_index.v1", content), "Index hash differs.")
