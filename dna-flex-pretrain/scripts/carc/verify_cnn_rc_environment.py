"""Offline B4a inventory, fixed-tensor verification, checking and acceptance.

Run with ``python -B -m scripts.carc.verify_cnn_rc_environment SUBCOMMAND``.
Every public operation requires the installed pinned prefix and live Slurm
evidence. The in-memory numerical helpers produce no acceptance manifests.
Finalization reads existing evidence; it never substitutes or reruns a probe.
No biological dataset or training/validation entry point is opened here.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import copy
import hashlib
import math
import os
import platform
import random
from pathlib import Path
import re
import sys
import struct
import tempfile
from typing import Any

from scripts.carc import cnn_rc_environment as env


VERIFICATION_SCHEMA = "carc_cnn_rc_environment_verification.v1"
RESOLVED_SCHEMA = "carc_cnn_rc_resolved_environment.v1"
FIXTURE_ID = "cnn_rc_environment_fixed_tensor.v1"
FIXTURE_SEED = 43001
INPUT_SHA256 = "c6e18bb3c941a7c8f2505af73bdf2de022cf8cd12a4d1472c26dec29ccf47b4d"
TARGET_SHA256 = "806f954b703da622d5547ae29471b4baaa6a6e33f98fced664118c588cddec04"
VERIFICATION_FIELDS = (
    "schema_version", "verification_id", "mode", "status", "environment_id",
    "inventory_sha256", "software", "fixture", "observations", "checks",
    "failure", "execution", "manifest_hash",
)
RESOLVED_FIELDS = (
    "schema_version", "status", "environment_id", "intent_sha256", "inventory",
    "cpu_verification", "p100_verification", "acceptance_policy", "manifest_hash",
)
BASE_CHECKS = (
    "numpy_config_api", "b3_environment_api", "fixed_tensor_finite",
    "same_device_repeat_exact", "forward_rc_invariance", "accepted_adam_update",
    "controlled_serialization_round_trip",
)
ACCEPTANCE_POLICY = {
    "identifier": "carc_cnn_rc_cu126_p100.v1",
    "required_modes": ["cpu", "p100"],
    "same_inventory_bytes": True,
    "same_installed_prefix": True,
    "live_slurm_evidence": True,
    "fixture": FIXTURE_ID,
    "verification_scope": "fixed_tensor_only_in_actual_installed_environment",
    "biological_smoke_accepted": False,
    "cpu_gpu_bitwise_trajectory_equivalence": False,
    "b4b_regression_gate_required": True,
}


def _keys(value: Any, names, label: str) -> None:
    env.require(type(value) is dict and set(value) == set(names), label + " fields differ.")


def _hash(value: Any) -> bool:
    return type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None


# B4a-local compatibility facts retain the existing observation field schema.
# They establish neither scientific checkpoint identities nor B3a/B3b acceptance.
# The accepted 202-test CARC integration gate remains a separate B4b requirement.
def _nonnegative_int(value: Any, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


_ENVIRONMENT_FIELDS = (
    "schema_version", "python", "numpy", "torch", "torch_build", "numpy_build", "os", "machine", "processor",
    "byteorder", "backend", "threads", "interop_threads", "cuda", "cudnn", "driver", "device_class", "device_count",
    "deterministic_algorithms", "deterministic_warn_only", "cudnn_benchmark", "cudnn_deterministic",
    "cuda_matmul_tf32", "cudnn_tf32", "float32_matmul_precision", "cublas_workspace_config",
)


def _capture_rng() -> dict[str, Any]:
    """Capture controlled Python, NumPy and applicable Torch RNG state for B4a."""
    import numpy as np
    import torch

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


def _rng_tensor(value: Any) -> None:
    import torch

    env.require(type(value) is torch.Tensor and value.device.type == "cpu" and value.dtype == torch.uint8
            and value.ndim == 1 and value.numel() > 0, "Malformed RNG tensor.")


def _validate_rng(value: Any) -> None:
    import torch

    _keys(value, ("python", "numpy", "torch_cpu", "torch_cuda", "component_generators"), "RNG")
    python_state = value["python"]
    env.require(type(python_state) is tuple and len(python_state) == 3, "Malformed Python RNG.")
    env.require(type(python_state[0]) is int and python_state[0] == 3
            and type(python_state[1]) is tuple and len(python_state[1]) == 625,
            "Malformed Python RNG state.")
    env.require(all(_nonnegative_int(item) and item < 2**32 for item in python_state[1][:-1])
            and _nonnegative_int(python_state[1][-1]) and python_state[1][-1] <= 624, "Invalid Python MT state.")
    env.require(python_state[2] is None or (type(python_state[2]) is float and math.isfinite(python_state[2])), "Invalid Python Gaussian cache.")
    random.Random().setstate(python_state)
    numpy_state = value["numpy"]
    _keys(numpy_state, ("algorithm", "keys", "position", "has_gauss", "cached_gaussian"), "NumPy RNG")
    env.require(numpy_state["algorithm"] == "MT19937", "Unknown NumPy RNG.")
    array = numpy_state["keys"]
    env.require(type(array) is torch.Tensor and array.dtype == torch.int64 and array.device.type == "cpu"
            and tuple(array.shape) == (624,) and bool(((array >= 0) & (array < 2**32)).all()), "Invalid NumPy keys.")
    env.require(_nonnegative_int(numpy_state["position"]) and numpy_state["position"] <= 624
            and type(numpy_state["has_gauss"]) is int and numpy_state["has_gauss"] in (0, 1)
            and type(numpy_state["cached_gaussian"]) is float and math.isfinite(numpy_state["cached_gaussian"]), "Invalid NumPy position/cache.")
    _rng_tensor(value["torch_cpu"])
    torch.Generator(device="cpu").set_state(value["torch_cpu"])
    env.require(value["torch_cuda"] is None or (type(value["torch_cuda"]) is list and bool(value["torch_cuda"])), "Malformed CUDA RNG list.")
    if value["torch_cuda"] is not None:
        for tensor in value["torch_cuda"]:
            _rng_tensor(tensor)
    env.require(value["component_generators"] == {} and type(value["component_generators"]) is dict,
            "v1 has no mutable component generators.")


def _restore_rng(value: dict[str, Any]) -> None:
    """Restore only the validated controlled B4a RNG state."""
    import numpy as np
    import torch

    _validate_rng(value)
    if value["torch_cuda"] is not None:
        env.require(torch.cuda.is_initialized() and len(value["torch_cuda"]) == torch.cuda.device_count(), "CUDA RNG environment differs.")
    random.setstate(value["python"])
    numpy_state = value["numpy"]
    np.random.set_state((numpy_state["algorithm"], numpy_state["keys"].numpy().astype(np.uint32),
                         numpy_state["position"], numpy_state["has_gauss"], numpy_state["cached_gaussian"]))
    torch.set_rng_state(value["torch_cpu"])
    if value["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(value["torch_cuda"])


def _cuda_driver_version() -> str:
    """Bind both CUDA API compatibility and the actual NVIDIA driver release."""
    import ctypes

    library = ctypes.CDLL("libcuda.so.1")
    function = library.cuDriverGetVersion
    function.argtypes = [ctypes.POINTER(ctypes.c_int)]
    function.restype = ctypes.c_int
    version = ctypes.c_int()
    env.require(function(ctypes.byref(version)) == 0 and version.value > 0, "CUDA driver identity unavailable.")
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
    env.require(initialize() == 0, "NVIDIA driver release unavailable.")
    try:
        release = ctypes.create_string_buffer(256)
        env.require(query(release, len(release)) == 0 and bool(release.value), "NVIDIA driver release unavailable.")
        return env.canonical_bytes({"cuda_driver_api": version.value, "nvidia_driver_release": release.value.decode("ascii")}).decode("ascii")
    finally:
        env.require(shutdown() == 0, "NVIDIA management shutdown failed.")


def _capture_environment() -> dict[str, Any]:
    """Observe compatibility facts without initializing CUDA or invoking jobs."""
    import numpy as np
    import torch

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


def _validate_environment(value: Any) -> None:
    _keys(value, _ENVIRONMENT_FIELDS, "Environment")
    env.require(value["schema_version"] == "downstream_environment_compatibility.v1", "Environment schema differs.")
    for name in ("python", "numpy", "torch", "torch_build", "numpy_build", "os", "machine", "device_class"):
        env.require(type(value[name]) is str and bool(value[name]), "Missing environment version.")
    env.require(type(value["processor"]) is str and value["byteorder"] in ("little", "big"), "Invalid platform.")
    env.require(value["backend"] in ("cpu", "cuda") and _nonnegative_int(value["threads"], 1)
            and _nonnegative_int(value["interop_threads"], 1) and _nonnegative_int(value["device_count"]), "Invalid execution backend.")
    for name in ("cuda", "driver", "cublas_workspace_config"):
        env.require(value[name] is None or (type(value[name]) is str and bool(value[name])), "Invalid optional environment field.")
    env.require(value["cudnn"] is None or _nonnegative_int(value["cudnn"], 1), "Invalid cuDNN version.")
    for name in ("deterministic_algorithms", "deterministic_warn_only", "cudnn_benchmark", "cudnn_deterministic", "cuda_matmul_tf32", "cudnn_tf32"):
        env.require(type(value[name]) is bool, "Invalid deterministic setting.")
    env.require(value["deterministic_algorithms"] and not value["deterministic_warn_only"]
            and not value["cudnn_benchmark"] and value["cudnn_deterministic"]
            and not value["cuda_matmul_tf32"] and not value["cudnn_tf32"]
            and value["float32_matmul_precision"] == "highest", "Numerical policy differs.")
    if value["backend"] == "cuda":
        env.require(value["device_count"] == 1 and value["cuda"] is not None and value["driver"] is not None
                and value["cublas_workspace_config"] == ":4096:8", "Incomplete single-GPU compatibility evidence.")
    else:
        env.require(value["device_count"] == 0 and value["driver"] is None, "CPU environment contains active CUDA state.")


def _configure_runtime(device: str) -> dict:
    """Fix one-device float32 numerical settings before any model execution."""
    import torch

    env.require(device in ("cpu", "cuda:0"), "Only cpu or cuda:0 is supported.")
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
        env.require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") in (None, ":4096:8"),
                     "Incompatible CUBLAS_WORKSPACE_CONFIG.")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        env.require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
                     "Exactly one visible CUDA device is required.")
        torch.cuda.init()
    else:
        env.require(not torch.cuda.is_initialized(), "CPU execution has active CUDA state.")
    environment = _capture_environment()
    _validate_environment(environment)
    return environment


def _tensor_record(tensor) -> dict:
    import torch

    env.require(type(tensor) is torch.Tensor and tensor.layout == torch.strided
                and not tensor.is_quantized and tensor.dtype in (torch.float32, torch.int64, torch.uint8),
                "Expected a dense tensor.")
    cpu = tensor.detach().cpu().contiguous()
    env.require(bool(torch.isfinite(cpu).all()), "Nonfinite verification tensor.")
    array = cpu.numpy()
    raw = array.astype(array.dtype.newbyteorder("<"), copy=False).tobytes(order="C")
    return {"dtype": str(cpu.dtype), "shape": list(cpu.shape),
            "sha256": hashlib.sha256(raw).hexdigest()}


def _state_tree(value: Any, path: str = "$") -> dict:
    """Encode only controlled B4a tensor/primitive values, never arbitrary objects."""
    import torch

    if type(value) is torch.Tensor:
        return {"type": "tensor", "record": {"path": path, **_tensor_record(value)}}
    if type(value) in (dict, OrderedDict):
        env.require(all(type(key) in (str, int) for key in value), "Unsupported B4a mapping key.")
        entries = []
        for key in sorted(value, key=lambda item: (type(item).__name__, str(item))):
            component = type(key).__name__ + ":" + str(key).replace("~", "~0").replace("/", "~1")
            entries.append([_state_tree(key, path + "/key/" + component),
                            _state_tree(value[key], path + "/" + component)])
        return {"type": "dict", "entries": entries}
    if type(value) in (tuple, list):
        return {"type": type(value).__name__, "items": [
            _state_tree(item, path + "/" + str(index)) for index, item in enumerate(value)]}
    env.require(type(value) in (str, bool, int, float, type(None)), "Unsupported B4a state object.")
    if type(value) is float:
        env.require(math.isfinite(value), "Nonfinite B4a state scalar.")
        return {"type": "float", "bits": struct.pack(">d", value).hex()}
    return {"type": type(value).__name__, "value": value}


def _state_fingerprint(value: Any) -> str:
    """B4a-only identity, deliberately separate from scientific checkpoint IDs."""
    return env.digest({"schema_version": "carc_cnn_rc_environment_state.v1", "state": _state_tree(value)})


def _controlled_state_copy(value: Any) -> Any:
    """Copy validated value contents into plain safe containers before torch.save."""
    import torch

    if type(value) is torch.Tensor:
        return value.detach().cpu().contiguous().clone()
    if type(value) in (dict, OrderedDict):
        return {key: _controlled_state_copy(item) for key, item in value.items()}
    if type(value) in (list, tuple):
        items = [_controlled_state_copy(item) for item in value]
        return tuple(items) if type(value) is tuple else items
    env.require(type(value) in (str, bool, int, float, type(None)), "Unsupported B4a serialization object.")
    return value


def _serialization_probe(state: Any) -> Any:
    """Temporary controlled environment round trip; no scientific checkpoint or index.

    Scientific B3a/B3b serialization integration remains the separate accepted
    202-test gate in the real CARC environment. No unsafe loading fallback exists.
    """
    import torch

    expected = _state_fingerprint(state)
    controlled = _controlled_state_copy(state)
    env.require(_state_fingerprint(controlled) == expected, "Controlled state copy differs.")
    with tempfile.TemporaryDirectory(prefix="b4a-tensor-roundtrip-") as directory:
        path = Path(directory) / "controlled-state.pt"
        torch.save(controlled, path)
        restored = torch.load(path, map_location="cpu", weights_only=True)
    env.require(_state_fingerprint(restored) == expected, "Controlled serialization round trip differs.")
    return restored


def _fixed_optimizer_spec() -> dict:
    """Required smoke Adam values, used for comparison without inserting defaults."""
    return {"name": "Adam", "learning_rate": 5e-5, "betas": [0.9, 0.999], "epsilon": 1e-8,
            "weight_decay": 0.0, "amsgrad": False, "foreach": False, "fused": False,
            "maximize": False, "capturable": False, "differentiable": False}


def _fixed_candidate() -> dict:
    """Verify the accepted inert JSON candidate; never import its training stack."""
    path = env.PROJECT_ROOT / "configs/exd_hox_cnn_rc_v1.json"
    config = env.strict_json(env.read_regular(path))
    env.require(config.get("schema_version") == "exd_hox_cnn_rc_training_config.v1"
                and config.get("model_contract") == "cnn_rc_wang100_kundaje122_population_bn.v1"
                and config.get("model_family") == "cnn_rc", "Fixed candidate config identity differs.")
    _keys(config.get("candidates"), ("smoke_adam_v1",), "Fixed candidate names")
    candidate = config["candidates"]["smoke_adam_v1"]
    env.require(type(candidate) is dict and type(candidate.get("batch_size")) is int
                and candidate["batch_size"] == 128
                and candidate.get("loss") == "batch_mean_mse_plus_independent_convolution_penalty.v1",
                "Fixed candidate batch or loss differs.")
    expected = {"optimizer": _fixed_optimizer_spec(), "regularization": {"l1": 5e-6, "l2": 1e-5}}
    for key, value in expected.items():
        env._same_shape(candidate.get(key), value)
        env.require(candidate[key] == value, "Fixed candidate " + key + " differs.")
    return candidate


def _make_optimizer(model, definition: dict):
    """One fixed synthetic Adam optimizer; no scheduler, epochs or recovery engine."""
    import torch

    expected = _fixed_optimizer_spec()
    env._same_shape(definition, expected)
    env.require(definition == expected, "Fixed Adam definition differs.")
    return torch.optim.Adam(
        model.parameters(), lr=definition["learning_rate"], betas=tuple(definition["betas"]),
        eps=definition["epsilon"], weight_decay=definition["weight_decay"],
        amsgrad=definition["amsgrad"], foreach=definition["foreach"], fused=definition["fused"],
        maximize=definition["maximize"], capturable=definition["capturable"],
        differentiable=definition["differentiable"],
    )


def fixed_fixture():
    """Construct the sole CPU model and exact unaugmented synthetic tensors."""
    import torch
    from src.cnn_rc import CNNRC

    inputs = torch.zeros((128, 14, 4), dtype=torch.float32, device="cpu")
    targets = torch.empty((128, 1), dtype=torch.float32, device="cpu")
    for row in range(128):
        for position in range(14):
            source = FIXTURE_ID + "\0" + str(row) + "\0" + str(position)
            channel = hashlib.sha256(source.encode("utf-8")).digest()[0] % 4
            inputs[row, position, channel] = 1
        targets[row, 0] = (((37 * row) % 127) + 0.5) / 128
    model = CNNRC(seed=FIXTURE_SEED)
    metadata = {
        "identifier": FIXTURE_ID, "model_seed": FIXTURE_SEED,
        "input": _tensor_record(inputs), "targets": _tensor_record(targets),
        "initial_state_sha256": _state_fingerprint(model.state_dict()),
    }
    return model, inputs, targets, metadata


def _tolerance(name: str, exact: bool = False) -> tuple[float, float]:
    if exact:
        return 0.0, 0.0
    if name.startswith("loss.") or name.endswith(".output"):
        return 1e-6, 1e-5
    return 1e-5, 1e-4


def compare_tensors(reference: dict, observed: dict, *, exact: bool = False) -> list[dict]:
    """Return every tensor's errors; never widen a tolerance after failure."""
    import torch

    env.require(type(reference) is dict and type(observed) is dict
                and bool(reference) and reference.keys() == observed.keys(),
                "Comparison tensor names differ or are empty.")
    records = []
    for name in sorted(reference):
        left, right = reference[name], observed[name]
        env.require(type(name) is str and type(left) is torch.Tensor
                    and type(right) is torch.Tensor, "Invalid comparison tensor.")
        env.require(left.shape == right.shape and left.dtype == right.dtype,
                    "Comparison shape or dtype differs: " + name)
        left_record, right_record = _tensor_record(left), _tensor_record(right)
        a, b = left.detach().cpu().double(), right.detach().cpu().double()
        delta = (a - b).abs()
        atol, rtol = _tolerance(name, exact)
        mismatches = int((delta > atol + rtol * a.abs()).sum().item())
        if exact:
            left_bytes = left.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
            right_bytes = right.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
            changed = (left_bytes != right_bytes).reshape(left.numel(), left.element_size())
            mismatches = int(changed.any(dim=1).sum().item())
        # This finite relative diagnostic uses atol as the zero-reference floor.
        floor = atol if atol > 0 else sys.float_info.min
        relative = delta / a.abs().clamp_min(floor)
        max_relative = float(relative.max().item()) if relative.numel() else 0.0
        if not math.isfinite(max_relative):
            max_relative = sys.float_info.max
        records.append({
            "name": name, "reference": left_record, "observed": right_record,
            "atol": atol, "rtol": rtol,
            "max_absolute_error": float(delta.max().item()) if delta.numel() else 0.0,
            "max_relative_error": max_relative,
            "mismatched_elements": mismatches,
            "passed": mismatches == 0,
        })
    return records


def _state_tensors(value, path: str, destination: dict) -> None:
    import torch

    if type(value) is torch.Tensor:
        _tensor_record(value)
        destination[path] = value.detach().cpu().clone()
    elif type(value) in (dict, OrderedDict):
        for key, item in value.items():
            _state_tensors(item, path + "." + str(key), destination)
    elif type(value) in (tuple, list):
        for index, item in enumerate(value):
            _state_tensors(item, path + "." + str(index), destination)
    elif type(value) is float:
        env.require(math.isfinite(value), "Nonfinite model or optimizer metadata.")
    else:
        env.require(type(value) in (str, int, bool, type(None)), "Unsupported state value.")


def _trajectory(template, inputs, targets, device: str, initial_rng: dict) -> dict:
    import torch
    from src.cnn_rc import reverse_complement

    _restore_rng(initial_rng)
    model = copy.deepcopy(template).to(device=device, dtype=torch.float32)
    x, y = inputs.clone().to(device), targets.clone().to(device)
    candidate = _fixed_candidate()
    optimizer = _make_optimizer(model, candidate["optimizer"])
    env.require(all(parameter.grad is None for parameter in model.parameters()),
                "Fixed trajectory must start without accumulated gradients.")
    initial = _serialization_probe({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "gradients": {name: parameter.grad for name, parameter in model.named_parameters()},
        "rng": initial_rng,
    })
    model.load_state_dict(initial["model"], strict=True)
    optimizer.load_state_dict(initial["optimizer"])
    for name, parameter in model.named_parameters():
        parameter.grad = initial["gradients"][name]
    _restore_rng(initial["rng"])
    tensors, reverse_outputs = {}, {}
    model.eval()
    with torch.no_grad():
        initial_eval = model.forward_intermediates(x)
        reverse_outputs["eval.output"] = model(reverse_complement(x)).detach().cpu().clone()
    for name, tensor in initial_eval.items():
        tensors["eval." + name] = tensor.detach().cpu().clone()
    # Training forward/RC must start with identical BN buffers.
    reverse_model = copy.deepcopy(model)
    reverse_model.train()
    with torch.no_grad():
        reverse_outputs["train.output"] = reverse_model(reverse_complement(x)).detach().cpu().clone()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    stages = model.forward_intermediates(x)
    for name, tensor in stages.items():
        _tensor_record(tensor)
        tensors["train." + name] = tensor.detach().cpu().clone()
    mse = (stages["output"] - y).square().mean()
    penalty = model.convolution_kernel_penalty(**candidate["regularization"])
    total = mse + penalty
    for name, tensor in (("mse", mse), ("penalty", penalty), ("total", total)):
        _tensor_record(tensor)
        tensors["loss." + name] = tensor.detach().cpu().clone()
    total.backward()
    for name, parameter in model.named_parameters():
        env.require(parameter.grad is not None, "Missing gradient: " + name)
        _tensor_record(parameter.grad)
        tensors["gradient." + name] = parameter.grad.detach().cpu().clone()
    optimizer.step()
    env.require(len(optimizer.state) == len(list(model.parameters())), "Incomplete Adam state.")
    for state in optimizer.state.values():
        env.require(float(state["step"].item()) == 1.0, "Expected exactly one Adam update.")
    _state_tensors(model.state_dict(), "state", tensors)
    _state_tensors(optimizer.state_dict(), "optimizer", tensors)
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    before_eval = _state_fingerprint(model.state_dict())
    with torch.no_grad():
        stages_after = model.forward_intermediates(x)
        reverse_outputs["eval_after.output"] = model(reverse_complement(x)).detach().cpu().clone()
    for name, tensor in stages_after.items():
        tensors["eval_after." + name] = tensor.detach().cpu().clone()
    env.require(_state_fingerprint(model.state_dict()) == before_eval,
                "Evaluation changed model or BN state.")
    rc_reference = {}
    for name in reverse_outputs:
        rc_reference[name] = tensors[name]
    if device == "cuda:0":
        torch.cuda.synchronize(0)
    final_rng = _capture_rng()
    for tensor in tensors.values():
        _tensor_record(tensor)
    final_state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": final_rng}
    _serialization_probe({"state": final_state, "tensors": tensors})
    return {
        "tensors": tensors,
        "rc": compare_tensors(rc_reference, reverse_outputs),
        "state_sha256": _state_fingerprint(final_state),
    }


def require_p100_device() -> dict:
    """Require native CUDA/P100 facts; this helper cannot publish evidence."""
    import torch

    env.require(str(torch.__version__) == "2.14.0+cu126", "Exact torch 2.14.0+cu126 is required.")
    env.require(torch.version.cuda == "12.6", "Exact torch CUDA runtime 12.6 is required.")
    env.require(torch.cuda.is_available(), "CUDA is unavailable; fallback is prohibited.")
    env.require(torch.cuda.device_count() == 1, "Exactly one visible GPU is required.")
    name = torch.cuda.get_device_name(0)
    capability = list(torch.cuda.get_device_capability(0))
    architectures = torch.cuda.get_arch_list()
    env.require(name == "Tesla P100-PCIE-16GB", "Required Tesla P100-PCIE-16GB is absent.")
    env.require(capability == [6, 0], "Required compute capability 6.0 is absent.")
    env.require("sm_60" in architectures, "Installed torch lacks sm_60 kernels.")
    torch.cuda.set_device(0)
    torch.cuda.init()
    properties = torch.cuda.get_device_properties(0)
    driver = _cuda_driver_version()
    cudnn = torch.backends.cudnn.version()
    env.require(type(cudnn) is int and cudnn > 0, "Actual cuDNN version unavailable.")
    uuid = str(getattr(properties, "uuid", ""))
    env.require(bool(uuid), "GPU UUID unavailable.")
    return {
        "name": name, "capability": capability, "device_count": 1, "index": 0,
        "driver": driver, "cudnn_version": cudnn, "torch_version": str(torch.__version__),
        "cuda_runtime": torch.version.cuda, "architecture_list": architectures,
        "uuid": uuid, "total_memory_bytes": int(properties.total_memory),
    }


def run_fixed_probe(device: str = "cpu") -> dict:
    """Exercise fixed kernels in memory; a Local result is never acceptance."""
    import numpy as np
    import torch

    env.require(device in ("cpu", "cuda:0"), "Only cpu or cuda:0 is supported.")
    env.require(not torch.cuda.is_initialized(), "Probe must begin without active CUDA state.")
    env.require(type(getattr(np.__config__, "CONFIG", None)) is dict
                and bool(np.__config__.CONFIG), "np.__config__.CONFIG is unavailable.")
    original_rng = _capture_rng()
    try:
        cpu_environment = _configure_runtime("cpu")
        _validate_environment(cpu_environment)
        template, inputs, targets, fixture = fixed_fixture()
        cpu_rng = _capture_rng()
        first = _trajectory(template, inputs, targets, "cpu", cpu_rng)
        second = _trajectory(template, inputs, targets, "cpu", cpu_rng)
        repeated_cpu = compare_tensors(first["tensors"], second["tensors"], exact=True)
        comparisons = {"repeat_cpu": repeated_cpu, "rc_cpu": first["rc"],
                       "repeat_p100": [], "rc_p100": [], "cpu_p100": []}
        states = {"cpu_first": first["state_sha256"], "cpu_repeat": second["state_sha256"],
                  "p100_first": None, "p100_repeat": None}
        checks = {
            "numpy_config_api": True, "b3_environment_api": True,
            "fixed_tensor_finite": True,
            "same_device_repeat_exact": all(item["passed"] for item in repeated_cpu)
            and first["state_sha256"] == second["state_sha256"],
            "forward_rc_invariance": all(item["passed"] for item in first["rc"]),
            "accepted_adam_update": True,
            "controlled_serialization_round_trip": True,
        }
        gpu, actual_environment = None, cpu_environment
        memory = {"peak_allocated_bytes": 0, "peak_reserved_bytes": 0}
        if device == "cuda:0":
            gpu = require_p100_device()
            actual_environment = _configure_runtime("cuda:0")
            _validate_environment(actual_environment)
            torch.cuda.synchronize(0)
            torch.cuda.reset_peak_memory_stats(0)
            gpu_rng = _capture_rng()
            gpu_first = _trajectory(template, inputs, targets, device, gpu_rng)
            gpu_repeat = _trajectory(template, inputs, targets, device, gpu_rng)
            comparisons["repeat_p100"] = compare_tensors(
                gpu_first["tensors"], gpu_repeat["tensors"], exact=True)
            comparisons["rc_p100"] = gpu_first["rc"]
            comparisons["cpu_p100"] = compare_tensors(first["tensors"], gpu_first["tensors"])
            states["p100_first"], states["p100_repeat"] = (
                gpu_first["state_sha256"], gpu_repeat["state_sha256"])
            checks["same_device_repeat_exact"] = checks["same_device_repeat_exact"] and all(
                item["passed"] for item in comparisons["repeat_p100"])
            checks["same_device_repeat_exact"] = checks["same_device_repeat_exact"] and (
                gpu_first["state_sha256"] == gpu_repeat["state_sha256"])
            checks["forward_rc_invariance"] = checks["forward_rc_invariance"] and all(
                item["passed"] for item in comparisons["rc_p100"])
            checks["cpu_gpu_tolerances"] = all(item["passed"] for item in comparisons["cpu_p100"])
            checks["device_contract"] = True
            torch.cuda.synchronize(0)
            memory = {"peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
                      "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0))}
        else:
            checks["no_cuda_initialization"] = not torch.cuda.is_initialized()
        return {"fixture": fixture, "checks": checks, "observations": {
            "b3_environment": actual_environment,
            "numpy_config_sha256": env.digest(np.__config__.CONFIG),
            "gpu": gpu, "comparisons": comparisons, "state_sha256": states,
            "memory": memory,
        }}
    finally:
        _restore_rng(original_rng)


def _validate_tensor_record(value: Any) -> None:
    _keys(value, ("dtype", "shape", "sha256"), "Tensor fingerprint")
    env.require(value["dtype"] in ("torch.float32", "torch.int64", "torch.uint8")
                and type(value["shape"]) is list
                and all(type(item) is int and item >= 0 for item in value["shape"])
                and _hash(value["sha256"]), "Malformed tensor fingerprint.")


def _validate_fixture(value: Any) -> None:
    _keys(value, ("identifier", "model_seed", "input", "targets", "initial_state_sha256"), "Fixture")
    env.require(value["identifier"] == FIXTURE_ID and type(value["model_seed"]) is int
                and value["model_seed"] == FIXTURE_SEED, "Fixed fixture differs.")
    for name, shape in (("input", [128, 14, 4]), ("targets", [128, 1])):
        _validate_tensor_record(value[name])
        env.require(value[name]["shape"] == shape and value[name]["dtype"] == "torch.float32",
                    "Fixed tensor shape or dtype differs.")
    env.require(_hash(value["initial_state_sha256"]), "Missing initial model fingerprint.")
    env.require(value["input"]["sha256"] == INPUT_SHA256 and value["targets"]["sha256"] == TARGET_SHA256,
                "Fixed tensor contents differ.")


def _comparison_shapes() -> dict:
    stages = {
        "input": [128, 14, 4], "convolution": [128, 200, 14],
        "first_relu": [128, 200, 14], "batch_normalization": [128, 200, 14],
        "second_relu": [128, 200, 14], "pooled": [128, 200, 7],
        "weighted_sum": [128, 200], "dense": [128, 512], "dense_relu": [128, 512],
        "logit": [128, 1], "output": [128, 1],
    }
    parameters = {"W": [100, 4, 11], "A": [7, 100], "V": [100, 512], "q": [512, 1],
                  "b": [100], "d": [512], "e": [1], "gamma": [100], "beta": [100]}
    shapes = {}
    for mode in ("eval", "train", "eval_after"):
        for name, shape in stages.items():
            shapes[mode + "." + name] = shape
    for name, shape in parameters.items():
        shapes["gradient." + name] = shape
        shapes["state." + name] = shape
    shapes.update({"state.running_mean": [100], "state.running_var": [100],
                   "loss.mse": [], "loss.penalty": [], "loss.total": []})
    for index, shape in enumerate(parameters.values()):
        shapes["optimizer.state." + str(index) + ".step"] = []
        for moment in ("exp_avg", "exp_avg_sq"):
            shapes["optimizer.state." + str(index) + "." + moment] = shape
    return shapes


def _comparison_names() -> list[str]:
    return sorted(_comparison_shapes())


def _validate_comparisons(records: Any, *, exact: bool) -> None:
    env.require(type(records) is list and bool(records), "Missing tensor comparisons.")
    names = []
    for record in records:
        _keys(record, ("name", "reference", "observed", "atol", "rtol", "max_absolute_error",
                       "max_relative_error", "mismatched_elements", "passed"), "Comparison")
        name = record["name"]
        env.require(type(name) is str and bool(name), "Invalid comparison name.")
        names.append(name)
        _validate_tensor_record(record["reference"])
        _validate_tensor_record(record["observed"])
        env.require(record["reference"]["dtype"] == record["observed"]["dtype"]
                    and record["reference"]["shape"] == record["observed"]["shape"],
                    "Recorded comparison tensor contracts differ.")
        expected = _tolerance(name, exact)
        for key, number in zip(("atol", "rtol"), expected):
            env.require(type(record[key]) is float and record[key] == number, "Comparison tolerance differs.")
        for key in ("max_absolute_error", "max_relative_error"):
            env.require(type(record[key]) is float and math.isfinite(record[key]) and record[key] >= 0,
                        "Invalid recorded numerical error.")
        env.require(type(record["mismatched_elements"]) is int and record["mismatched_elements"] >= 0
                    and type(record["passed"]) is bool
                    and record["passed"] == (record["mismatched_elements"] == 0),
                    "Invalid comparison result.")
        if exact and record["passed"]:
            env.require(record["reference"] == record["observed"]
                        and record["max_absolute_error"] == 0.0, "Exact comparison fingerprints differ.")
    env.require(names == sorted(set(names)), "Comparison names must be sorted and unique.")


def _validate_gpu(value: Any) -> None:
    _keys(value, ("name", "capability", "device_count", "index", "driver", "cudnn_version",
                  "torch_version", "cuda_runtime", "architecture_list", "uuid", "total_memory_bytes"), "GPU")
    env.require(value["name"] == "Tesla P100-PCIE-16GB"
                and type(value["capability"]) is list and value["capability"] == [6, 0]
                and all(type(item) is int for item in value["capability"])
                and type(value["device_count"]) is int and value["device_count"] == 1
                and type(value["index"]) is int and value["index"] == 0,
                "P100 device contract differs.")
    env.require(value["torch_version"] == "2.14.0+cu126" and value["cuda_runtime"] == "12.6",
                "P100 runtime differs.")
    env.require(type(value["architecture_list"]) is list and "sm_60" in value["architecture_list"]
                and all(type(item) is str and bool(item) for item in value["architecture_list"]),
                "Missing supported architecture evidence.")
    for name in ("cudnn_version", "total_memory_bytes"):
        env.require(type(value[name]) is int and value[name] > 0, "Missing GPU runtime fact.")
    env.require(type(value["uuid"]) is str and bool(value["uuid"]), "Missing GPU UUID.")
    env.require(type(value["driver"]) is str, "Missing actual driver.")
    driver = env.strict_json(value["driver"].encode("utf-8"))
    _keys(driver, ("cuda_driver_api", "nvidia_driver_release"), "Driver")
    env.require(type(driver["cuda_driver_api"]) is int and driver["cuda_driver_api"] > 0
                and type(driver["nvidia_driver_release"]) is str and bool(driver["nvidia_driver_release"]),
                "Missing actual NVIDIA driver release.")


def _validate_observations(value: Any, mode: str, *, intended: bool = True) -> None:

    _keys(value, ("b3_environment", "numpy_config_sha256", "gpu", "comparisons", "state_sha256", "memory"),
          "Observations")
    _validate_environment(value["b3_environment"])
    compatibility = value["b3_environment"]
    if intended:
        env.require(compatibility["python"] == "3.11.14" and compatibility["numpy"] == "2.2.6"
                and compatibility["torch"] == "2.14.0+cu126" and compatibility["cuda"] == "12.6"
                and compatibility["os"] == "Linux" and compatibility["machine"] == "x86_64"
                and compatibility["threads"] == compatibility["interop_threads"] == 1
                and compatibility["cublas_workspace_config"] == ":4096:8",
                    "Verified runtime does not match intended environment.")
    env.require(_hash(value["numpy_config_sha256"]), "Missing NumPy CONFIG evidence.")
    comparisons = value["comparisons"]
    _keys(comparisons, ("repeat_cpu", "rc_cpu", "repeat_p100", "rc_p100", "cpu_p100"), "Comparisons")
    _validate_comparisons(comparisons["repeat_cpu"], exact=True)
    _validate_comparisons(comparisons["rc_cpu"], exact=False)
    _keys(value["state_sha256"], ("cpu_first", "cpu_repeat", "p100_first", "p100_repeat"), "State hashes")
    env.require(_hash(value["state_sha256"]["cpu_first"])
                and _hash(value["state_sha256"]["cpu_repeat"]), "Missing repeated CPU state hashes.")
    _keys(value["memory"], ("peak_allocated_bytes", "peak_reserved_bytes"), "Verifier memory")
    env.require(all(type(item) is int and item >= 0 for item in value["memory"].values()),
                "Invalid verifier memory measurement.")
    if mode == "cpu":
        env.require(compatibility["backend"] == "cpu" and value["gpu"] is None
                    and comparisons["repeat_p100"] == [] and comparisons["rc_p100"] == []
                    and comparisons["cpu_p100"] == []
                    and value["state_sha256"]["p100_first"] is None
                    and value["state_sha256"]["p100_repeat"] is None
                    and all(item == 0 for item in value["memory"].values()), "CPU evidence contains GPU execution.")
    else:
        env.require(compatibility["backend"] == "cuda", "P100 evidence lacks active CUDA execution.")
        _validate_gpu(value["gpu"])
        env.require(value["gpu"]["driver"] == compatibility["driver"]
                    and value["gpu"]["cudnn_version"] == compatibility["cudnn"], "GPU observations disagree.")
        _validate_comparisons(comparisons["repeat_p100"], exact=True)
        _validate_comparisons(comparisons["rc_p100"], exact=False)
        _validate_comparisons(comparisons["cpu_p100"], exact=False)
        env.require(_hash(value["state_sha256"]["p100_first"])
                    and _hash(value["state_sha256"]["p100_repeat"]), "Missing repeated P100 state hashes.")
        env.require(value["memory"]["peak_reserved_bytes"] >= value["memory"]["peak_allocated_bytes"] > 0,
                    "Missing measured verifier memory.")
    for name, records in comparisons.items():
        if records:
            expected_names = (["eval.output", "eval_after.output", "train.output"]
                              if name.startswith("rc_") else _comparison_names())
            env.require([item["name"] for item in records] == expected_names,
                        "Incomplete fixed-model tensor comparison coverage.")
            shapes = _comparison_shapes()
            for item in records:
                env.require(item["reference"]["shape"] == shapes[item["name"]]
                            and item["reference"]["dtype"] == "torch.float32",
                            "Fixed-model comparison shape or FP32 dtype differs.")
    cpu_tensors = {item["name"]: item["reference"] for item in comparisons["repeat_cpu"]}
    gpu_tensors = {item["name"]: item["reference"] for item in comparisons["repeat_p100"]}
    for group, tensors in (("rc_cpu", cpu_tensors), ("rc_p100", gpu_tensors)):
        for item in comparisons[group]:
            env.require(item["reference"] == tensors[item["name"]], "RC comparison refers to another trajectory.")
    for item in comparisons["cpu_p100"]:
        env.require(item["reference"] == cpu_tensors[item["name"]]
                    and item["observed"] == gpu_tensors[item["name"]],
                    "CPU/P100 comparison refers to another trajectory.")


def _verification_id(record: dict) -> str:
    identity = {key: value for key, value in record.items() if key not in ("manifest_hash", "verification_id")}
    return "verification_" + env.digest(identity)


def validate_verification(record: Any) -> None:

    _keys(record, VERIFICATION_FIELDS, "Verification")
    env.validate_seal(record)
    env.require(record["schema_version"] == VERIFICATION_SCHEMA
                and record["mode"] in ("cpu", "p100", "check")
                and record["status"] in ("successful", "failed")
                and record["verification_id"] == _verification_id(record), "Invalid verification identity.")
    env.require(type(record["checks"]) is dict
                and all(type(value) is bool for value in record["checks"].values()), "Malformed checks.")
    if record["status"] == "failed":
        _keys(record["failure"], ("type", "message", "stage"), "Failure")
        env.require(all(type(value) is str and bool(value) for value in record["failure"].values()),
                    "Malformed failure diagnostics.")
        # A failed preflight may have no trustworthy software/install/device facts.
        for field in ("environment_id", "inventory_sha256"):
            env.require(record[field] is None or type(record[field]) is str, "Invalid failed identity.")
        if record["software"] is not None:
            env._b4a_validate_software(record["software"])
        if record["execution"] is not None:
            env.validate_execution(record["execution"], "p100" if record["mode"] == "p100" else "cpu")
        if record["fixture"] is not None:
            _validate_fixture(record["fixture"])
        if record["observations"] is not None:
            env.require(record["mode"] != "check", "Failed check must not claim completed observations.")
            _validate_observations(record["observations"], record["mode"], intended=False)
        allowed = set(BASE_CHECKS + ("no_cuda_initialization", "cpu_gpu_tolerances", "device_contract"))
        if record["mode"] == "check":
            allowed = {"accepted_environment_matches", "device_matches"}
        env.require(set(record["checks"]).issubset(allowed), "Unknown failure checks.")
    else:
        env.require(record["failure"] is None and _hash(record["inventory_sha256"])
                    and type(record["environment_id"]) is str
                    and record["environment_id"].startswith("env_")
                    and _hash(record["environment_id"][4:]), "Missing successful evidence identity.")
        env._b4a_validate_software(record["software"])
        if record["mode"] == "check":
            _keys(record["checks"], ("accepted_environment_matches", "device_matches"), "Check checks")
            _keys(record["observations"], ("resolved_environment_sha256", "b3_environment", "gpu"), "Check observations")
            env.require(_hash(record["observations"]["resolved_environment_sha256"]), "Missing accepted manifest hash.")
            _validate_environment(record["observations"]["b3_environment"])
            env.require(record["fixture"] is None, "Check cannot rerun the fixture.")
            mode = "cpu" if record["observations"]["gpu"] is None else "p100"
            if mode == "p100":
                _validate_gpu(record["observations"]["gpu"])
            env.validate_execution(record["execution"], mode)
        else:
            mode = record["mode"]
            env.validate_execution(record["execution"], mode)
            _validate_fixture(record["fixture"])
            _validate_observations(record["observations"], mode)
            expected_checks = BASE_CHECKS + (("no_cuda_initialization",) if mode == "cpu"
                                            else ("cpu_gpu_tolerances", "device_contract"))
            _keys(record["checks"], expected_checks, "Verification checks")
            observations = record["observations"]
            for group in ("repeat_cpu", "repeat_p100"):
                for item in observations["comparisons"][group]:
                    if item["name"].endswith(".input"):
                        env.require(item["reference"] == record["fixture"]["input"],
                                    "Model trajectory input differs from the fixed fixture.")
            env.require(observations["state_sha256"]["cpu_first"] == observations["state_sha256"]["cpu_repeat"],
                        "CPU repetition is not exact.")
            if mode == "p100":
                env.require(observations["state_sha256"]["p100_first"] == observations["state_sha256"]["p100_repeat"],
                            "P100 repetition is not exact.")
            for comparisons in observations["comparisons"].values():
                env.require(all(item["passed"] for item in comparisons), "Failed tensor comparison in successful evidence.")
        env.require(all(record["checks"].values()), "Unsuccessful checks cannot establish acceptance.")


def _finish_verification(record: dict, output: Path) -> dict:
    record["verification_id"] = _verification_id(record)
    record = env.seal(record)
    validate_verification(record)
    env.publish(output, record)
    return record


def _installation(spec: Path, prefix: Path, inventory: Path, expected_environment_id: str,
                  acquisition_lock: Path, exports_root: Path, expected_software_commit: str,
                  mode: str) -> tuple[dict, dict, dict]:
    intent = env.load_intent(spec)
    record = env.read_manifest(inventory)
    env.validate_inventory(record)
    env.require(record["environment_id"] == expected_environment_id, "Expected environment identity differs.")
    context = env.execution_context(prefix, mode)
    env.revalidate_inventory(record, intent, prefix, exports_root, acquisition_lock, mode=mode)
    for key in ("prefix", "python_executable", "python_version"):
        env.require(context[key] == record["execution"][key], "Verification is from a different installation.")
    software = env.verify_software(expected_software_commit)
    return record, context, software


def _empty_verification(mode: str) -> dict:
    return {"schema_version": VERIFICATION_SCHEMA, "verification_id": None, "mode": mode,
            "status": "failed", "environment_id": None, "inventory_sha256": None,
            "software": None, "fixture": None, "observations": None, "checks": {},
            "failure": None, "execution": None}


def verify_environment(*, mode: str, spec: Path, prefix: Path, inventory: Path,
                       expected_environment_id: str, acquisition_lock: Path, exports_root: Path,
                       expected_software_commit: str, output: Path) -> dict:
    """Publish real-install verification or explicit failed diagnostics only."""
    env.require(mode in ("cpu", "p100"), "Only CPU/P100 verification is supported.")
    _fresh_output(output, (spec, inventory, acquisition_lock))
    record, stage = _empty_verification(mode), "installed_environment_preflight"
    try:
        installed, context, software = _installation(
            spec, prefix, inventory, expected_environment_id, acquisition_lock,
            exports_root, expected_software_commit, mode)
        record.update(environment_id=installed["environment_id"],
                      inventory_sha256=env.file_reference(inventory)["sha256"],
                      software=software, execution=context)
        stage = "fixed_tensor_verification"
        report = run_fixed_probe("cpu" if mode == "cpu" else "cuda:0")
        record.update(report)
        env.require(all(report["checks"].values()), "Fixed-tensor numerical acceptance failed.")
        # Detect package/export or source changes across execution before success.
        stage = "post_verification_identity"
        again = _installation(spec, prefix, inventory, expected_environment_id,
                              acquisition_lock, exports_root, expected_software_commit, mode)
        env.require(again[0] == installed and again[2] == software, "Environment changed during verification.")
        record["status"] = "successful"
    except (ValueError, OSError, RuntimeError, ImportError, AttributeError) as error:
        record["failure"] = {"type": type(error).__name__, "message": str(error) or type(error).__name__, "stage": stage}
    return _finish_verification(record, output)


def _fresh_output(output: Path, inputs=()) -> None:
    env.require(type(output) is Path or isinstance(output, Path), "Output must be a path.")
    env.require(output.is_absolute(), "Output must be absolute.")
    env._physical(output)
    env._physical(output.parent, directory=True)
    env.require(not output.exists() and not output.is_symlink(), "Output already exists; overwrite is prohibited.")
    env.require(output.parent.is_dir(), "Output parent must already exist.")
    for item in inputs:
        env.require(output != Path(item).absolute(), "Output aliases an input.")


def inventory_environment(*, spec: Path, prefix: Path, acquisition_lock: Path,
                          exports_root: Path, expected_software_commit: str, output: Path) -> dict:
    """Record actual installed facts and pre-existing acquisition/export bytes."""
    _fresh_output(output, (spec, acquisition_lock))
    intent = env.load_intent(spec)
    env.execution_context(prefix, "cpu")
    env.verify_software(expected_software_commit)
    env.collect_exports(prefix, exports_root)
    record = env.build_inventory(intent, prefix, acquisition_lock, exports_root)
    env.validate_inventory(record)
    env.publish(output, record)
    return record


def validate_resolved(record: Any) -> None:
    _keys(record, RESOLVED_FIELDS, "Resolved environment")
    env.validate_seal(record)
    env.require(record["schema_version"] == RESOLVED_SCHEMA and record["status"] == "accepted"
                and type(record["environment_id"]) is str and record["environment_id"].startswith("env_")
                and _hash(record["environment_id"][4:]) and _hash(record["intent_sha256"]),
                "Malformed resolved identity.")
    for key in ("inventory", "cpu_verification", "p100_verification"):
        env.validate_reference(record[key])
    # Canonical comparison also distinguishes booleans from integer impostors.
    env.require(env.canonical_bytes(record["acceptance_policy"]) == env.canonical_bytes(ACCEPTANCE_POLICY),
                "Acceptance policy differs.")


def _evidence_pair(inventory_record: dict, inventory: Path, cpu_verification: Path,
                   p100_verification: Path, expected_software_commit: str) -> tuple[dict, dict]:
    cpu, p100 = env.read_manifest(cpu_verification), env.read_manifest(p100_verification)
    inventory_hash = env.file_reference(inventory)["sha256"]
    env.require(inventory_hash == _manifest_file_hash(inventory_record),
                "Inventory bytes changed while loading verification evidence.")
    for mode, record in (("cpu", cpu), ("p100", p100)):
        validate_verification(record)
        env.require(record["mode"] == mode and record["status"] == "successful",
                    "Finalization requires existing successful real CPU and P100 evidence.")
        env.require(record["environment_id"] == inventory_record["environment_id"]
                    and record["inventory_sha256"] == inventory_hash,
                    "Verification does not bind the exact installed inventory.")
        for key in ("prefix", "python_executable", "python_version"):
            env.require(record["execution"][key] == inventory_record["execution"][key],
                        "CPU/P100 verification belongs to a different installation.")
        env.require(record["software"]["runtime_commit"] == expected_software_commit,
                    "Verification producer commit differs.")
    env.require(cpu["software"] == p100["software"] and cpu["fixture"] == p100["fixture"],
                "CPU/P100 source or fixture evidence differs.")
    return cpu, p100


def _manifest_file_hash(record: dict) -> str:
    return hashlib.sha256(env.canonical_bytes(record) + b"\n").hexdigest()


def finalize_environment(*, spec: Path, prefix: Path, inventory: Path, expected_environment_id: str,
                         acquisition_lock: Path, exports_root: Path, expected_software_commit: str,
                         cpu_verification: Path, p100_verification: Path, output: Path) -> dict:
    """Accept immutable successful evidence from this actual installed prefix."""
    _fresh_output(output, (spec, inventory, acquisition_lock, cpu_verification, p100_verification))
    references = {}
    for name, path in (("inventory", inventory), ("cpu_verification", cpu_verification),
                       ("p100_verification", p100_verification)):
        references[name] = _evidence_reference(path, output.parent)
    installed, _, software = _installation(spec, prefix, inventory, expected_environment_id,
                                          acquisition_lock, exports_root, expected_software_commit, "cpu")
    cpu, p100 = _evidence_pair(installed, inventory, cpu_verification, p100_verification, expected_software_commit)
    env.require(cpu["software"] == software == p100["software"], "Executing finalizer source differs.")
    for name, value in (("inventory", installed), ("cpu_verification", cpu), ("p100_verification", p100)):
        env.require(references[name]["sha256"] == _manifest_file_hash(value),
                    "Acceptance evidence changed during finalization.")
    record = env.seal({
        "schema_version": RESOLVED_SCHEMA, "status": "accepted",
        "environment_id": installed["environment_id"], "intent_sha256": installed["intent_sha256"],
        **references,
        "acceptance_policy": copy.deepcopy(ACCEPTANCE_POLICY),
    })
    validate_resolved(record)
    env.publish(output, record)
    return record


def _reference_from_sibling(root: Path, reference: dict) -> Path:
    env.validate_reference(reference)
    path = root / reference["logical_path"]
    env.require(env.file_reference(path, reference["logical_path"]) == reference, "Acceptance evidence bytes differ.")
    return path


def _evidence_reference(path: Path, root: Path) -> dict:
    env.require(path.is_absolute() and root in path.parents,
                "Resolved evidence must be below the final manifest directory.")
    return env.file_reference(path, path.relative_to(root).as_posix())


def check_environment(*, spec: Path, prefix: Path, inventory: Path, expected_environment_id: str,
                      acquisition_lock: Path, exports_root: Path, expected_software_commit: str,
                      environment: Path, device: str, output: Path) -> dict:
    """Compare a real job with existing acceptance; execute no model kernels."""

    env.require(device in ("cpu", "cuda:0"), "Only cpu or cuda:0 is supported.")
    _fresh_output(output, (spec, inventory, acquisition_lock, environment))
    record, stage = _empty_verification("check"), "accepted_environment_preflight"
    try:
        mode = "cpu" if device == "cpu" else "p100"
        installed, context, software = _installation(spec, prefix, inventory, expected_environment_id,
                                                    acquisition_lock, exports_root, expected_software_commit, mode)
        resolved = env.read_manifest(environment)
        validate_resolved(resolved)
        env.require(resolved["environment_id"] == installed["environment_id"]
                    and resolved["intent_sha256"] == installed["intent_sha256"]
                    and resolved["inventory"] == _evidence_reference(inventory, environment.parent), "Accepted inventory differs.")
        cpu_path = _reference_from_sibling(environment.parent, resolved["cpu_verification"])
        p100_path = _reference_from_sibling(environment.parent, resolved["p100_verification"])
        cpu, p100 = _evidence_pair(installed, inventory, cpu_path, p100_path, expected_software_commit)
        env.require(resolved["cpu_verification"]["sha256"] == _manifest_file_hash(cpu)
                    and resolved["p100_verification"]["sha256"] == _manifest_file_hash(p100),
                    "Verification evidence changed while checking acceptance.")
        env.require(software == cpu["software"] == p100["software"], "Accepted verifier software differs.")
        stage = "job_device_comparison"
        import torch
        env.require(not torch.cuda.is_initialized(), "Check must begin without active CUDA state.")
        gpu = None if device == "cpu" else require_p100_device()
        actual = _configure_runtime(device)
        expected = cpu if device == "cpu" else p100
        env.require(actual == expected["observations"]["b3_environment"], "Job B3 compatibility differs from accepted evidence.")
        if gpu is not None:
            # Physical GPU UUID is attempt evidence, not software compatibility.
            accepted_gpu = expected["observations"]["gpu"]
            for key in gpu:
                if key != "uuid":
                    env.require(gpu[key] == accepted_gpu[key], "Job GPU differs: " + key)
        stage = "post_check_identity"
        again = _installation(spec, prefix, inventory, expected_environment_id,
                              acquisition_lock, exports_root, expected_software_commit, mode)
        env.require(again[0] == installed and again[2] == software, "Environment changed during check.")
        env.require(env.read_manifest(environment) == resolved,
                    "Resolved acceptance changed during check.")
        record.update(status="successful", environment_id=installed["environment_id"],
                      inventory_sha256=env.file_reference(inventory)["sha256"], software=software,
                      observations={"resolved_environment_sha256": env.file_reference(environment)["sha256"],
                                    "b3_environment": actual, "gpu": gpu},
                      execution=context, checks={"accepted_environment_matches": True, "device_matches": True})
    except (ValueError, OSError, RuntimeError, ImportError, AttributeError) as error:
        record["failure"] = {"type": type(error).__name__, "message": str(error) or type(error).__name__, "stage": stage}
    return _finish_verification(record, output)


class _Once(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, self.dest, None) is not None:
            parser.error("Repeated option: " + str(option_string))
        setattr(namespace, self.dest, values)


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    for operation in ("inventory", "cpu", "p100", "check", "finalize"):
        child = subparsers.add_parser(operation, allow_abbrev=False)
        for name in ("spec", "prefix", "acquisition-lock", "exports-root", "output"):
            child.add_argument("--" + name, type=Path, required=True, action=_Once)
        child.add_argument("--expected-software-commit", required=True, action=_Once)
        if operation != "inventory":
            child.add_argument("--inventory", type=Path, required=True, action=_Once)
            child.add_argument("--expected-environment-id", required=True, action=_Once)
        if operation == "check":
            child.add_argument("--environment", type=Path, required=True, action=_Once)
            child.add_argument("--device", choices=("cpu", "cuda:0"), required=True, action=_Once)
        if operation == "finalize":
            child.add_argument("--cpu-verification", type=Path, required=True, action=_Once)
            child.add_argument("--p100-verification", type=Path, required=True, action=_Once)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = vars(argument_parser().parse_args(argv))
    operation = arguments.pop("operation")
    try:
        if operation == "inventory":
            record = inventory_environment(**arguments)
        elif operation in ("cpu", "p100"):
            record = verify_environment(mode=operation, **arguments)
        elif operation == "check":
            record = check_environment(**arguments)
        else:
            record = finalize_environment(**arguments)
    except env.PublishedDurabilityError as error:
        print(env.canonical_bytes({"status": "published_but_durability_unconfirmed", "error": str(error)}).decode(),
              file=sys.stderr)
        return 2
    except (ValueError, OSError, RuntimeError, ImportError, AttributeError) as error:
        print(env.canonical_bytes({"status": "failed", "error": str(error)}).decode(), file=sys.stderr)
        return 1
    print(env.canonical_bytes(record).decode())
    return 1 if record.get("status") == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
