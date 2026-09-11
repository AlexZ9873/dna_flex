"""B4a intended binary contract, acquisition, and installed environment facts.

Only ``acquire_environment`` downloads or installs anything. Inventory and
verification are read-only with respect to installed packages and caches.
No function here accepts an environment or manufactures verification evidence.
"""

from __future__ import annotations

import argparse
import base64
import copy
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import site
import socket
import secrets
import stat
import subprocess
import sys
from typing import Any
from urllib.parse import unquote, urlparse
import urllib.request
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTENT_SCHEMA = "carc_cnn_rc_environment_intent.v1"
INVENTORY_SCHEMA = "carc_cnn_rc_environment_inventory.v1"
LOCK_SCHEMA = "carc_cnn_rc_environment_acquisition.v1"
POLICY = "carc_cnn_rc_cu126_p100.v1"
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
PACKAGE_FIELDS = ("manager", "name", "version", "build", "subdir", "filename", "origin_url", "sha256", "upstream_digest", "installed_metadata_sha256")
ACQUISITION_FIELDS = PACKAGE_FIELDS[:-1] + ("byte_size", "platform_tags", "cache_path")
SOURCE_PATHS = (
    "configs/carc_exd_hox_training_staging_v1.json", "configs/exd_hox_cnn_rc_v1.json",
    "environments/carc_cnn_rc_v1.json", "environments/carc_cnn_rc_v1_requirements.txt",
    "scripts/carc/cnn_rc_environment.py", "scripts/carc/create_cnn_rc_environment.sh",
    "scripts/carc/verify_cnn_rc_environment.py", "scripts/downstream/train_cnn_rc.py",
    "scripts/downstream/validate_cnn_rc.py", "src/__init__.py", "src/cnn_rc.py",
    "src/cnn_rc_training.py", "src/downstream_checkpoint.py", "src/downstream_fingerprints.py",
    "src/downstream_metrics.py", "src/downstream_run.py", "src/exd_hox_dataset.py",
)


class EnvironmentError(ValueError):
    """The intended environment, installation facts, or provenance differs."""


class PublishedDurabilityError(EnvironmentError):
    """Evidence was published but directory durability could not be confirmed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EnvironmentError(message)


def keys(value: Any, expected: Any, name: str) -> None:
    require(type(value) is dict and set(value) == set(expected), name + " fields differ.")


def hex_digest(value: Any, length: int = 64) -> bool:
    return type(value) is str and re.fullmatch("[0-9a-f]{" + str(length) + "}", value) is not None


def _plain(value: Any) -> None:
    if type(value) is dict:
        for key, item in value.items():
            require(type(key) is str, "JSON keys must be strings.")
            _plain(item)
    elif type(value) is list:
        for item in value:
            _plain(item)
    else:
        require(type(value) in (str, int, float, bool, type(None)), "Unsupported JSON type.")
        if type(value) is float:
            require(math.isfinite(value), "Nonfinite JSON value.")


def canonical_bytes(value: Any) -> bytes:
    _plain(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def strict_json(data: bytes) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key: " + key)
            result[key] = value
        return result

    def invalid(value):
        raise EnvironmentError("Nonfinite JSON constant: " + value)

    try:
        result = json.loads(data.decode("utf-8"), object_pairs_hook=pairs, parse_constant=invalid)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise EnvironmentError("Malformed JSON.") from error
    require(type(result) is dict, "JSON root must be an object.")
    _plain(result)
    return result


def seal(value: dict) -> dict:
    require(type(value) is dict and "manifest_hash" not in value, "Manifest already sealed.")
    result = copy.deepcopy(value)
    result["manifest_hash"] = digest(value)
    return result


def validate_seal(value: dict) -> None:
    require(type(value) is dict and hex_digest(value.get("manifest_hash")), "Missing manifest hash.")
    content = dict(value)
    del content["manifest_hash"]
    require(value["manifest_hash"] == digest(content), "Manifest hash differs.")


def _physical(path: Path, *, directory: bool = False) -> Path:
    path = Path(path)
    require(path.is_absolute() and path != Path("/"), "An absolute non-root path is required.")
    require(".." not in path.parts, "Parent traversal is forbidden.")
    for ancestor in reversed((path, *path.parents)):
        if os.path.lexists(ancestor):
            require(not ancestor.is_symlink(), "Symlink paths are forbidden.")
    if directory:
        require(path.is_dir(), "Directory does not exist.")
    return path


def _open_parent(path: Path) -> int:
    """Anchor every ancestor, so a symlink replacement cannot redirect IO."""
    _physical(path)
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in path.parts[1:-1]:
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        # This is descriptor cleanup only; preserve every original failure.
        os.close(descriptor)
        raise


def read_regular(path: Path) -> bytes:
    path = _physical(Path(path))
    parent = _open_parent(path)
    try:
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        try:
            before = os.fstat(descriptor)
            require(stat.S_ISREG(before.st_mode), "Expected a regular file.")
            chunks = []
            block = os.read(descriptor, 1024 * 1024)
            while block:
                chunks.append(block)
                block = os.read(descriptor, 1024 * 1024)
            after = os.fstat(descriptor)
            current = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
            fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
            require(all(getattr(before, field) == getattr(after, field) == getattr(current, field) for field in fields), "File changed during read.")
            return b"".join(chunks)
        finally:
            os.close(descriptor)
    finally:
        os.close(parent)


def _require_published_name(path: Path) -> None:
    require(re.fullmatch(r"\..+\.partial-[0-9a-f]{32}", path.name, flags=re.DOTALL) is None,
            "Private partial evidence is not a published artifact.")


def _verify_completed_partial(parent: int, partial: str, completed: os.stat_result, data: bytes) -> None:
    """Revalidate the exact private inode through its anchored, no-follow name."""
    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(partial, flags, dir_fd=parent)
    try:
        fields = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode) and stat.S_IMODE(before.st_mode) == 0o400
                and before.st_nlink == 1 and before.st_size == len(data),
                "Completed partial mode, type, link count, or size differs.")
        require(all(getattr(before, field) == getattr(completed, field) for field in fields),
                "Completed partial inode changed before validation.")
        chunks = []
        block = os.read(descriptor, 1024 * 1024)
        while block:
            chunks.append(block)
            block = os.read(descriptor, 1024 * 1024)
        content = b"".join(chunks)
        after = os.fstat(descriptor)
        current = os.stat(partial, dir_fd=parent, follow_symlinks=False)
        require(all(getattr(before, field) == getattr(after, field) == getattr(current, field)
                    for field in fields), "Completed partial changed during validation.")
        require(content == data and hashlib.sha256(content).digest() == hashlib.sha256(data).digest(),
                "Completed partial bytes or SHA-256 differ.")
    finally:
        os.close(descriptor)


def _exclusive_bytes(path: Path, data: bytes) -> None:
    """Preserve every partial file and never silently retry publication."""
    path = _physical(Path(path))
    _require_published_name(path)
    require(type(data) is bytes, "Artifact content must be complete bytes.")
    _physical(path.parent, directory=True)
    parent = _open_parent(path)
    partial = "." + path.name + ".partial-" + secrets.token_hex(16)
    published = False
    try:
        require(not os.path.lexists(path), "Evidence output already exists; preserve it.")
        flags = (os.O_WRONLY | os.O_CREAT | os.O_EXCL
                 | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0))
        descriptor = os.open(partial, flags, 0o600, dir_fd=parent)
        try:
            # Creation is private even with umask 0; restore owner bits removed
            # by a restrictive umask before exposing any artifact content.
            os.fchmod(descriptor, 0o600)
            created = os.fstat(descriptor)
            current = os.stat(partial, dir_fd=parent, follow_symlinks=False)
            require(stat.S_ISREG(created.st_mode) and stat.S_IMODE(created.st_mode) == 0o600
                    and created.st_nlink == 1 and created.st_size == 0
                    and (current.st_dev, current.st_ino, current.st_mode, current.st_nlink)
                    == (created.st_dev, created.st_ino, created.st_mode, created.st_nlink),
                    "Partial must be a new private regular file.")
            offset = 0
            while offset < len(data):
                written = os.write(descriptor, data[offset:])
                require(type(written) is int and 0 < written <= len(data) - offset,
                        "Short evidence write; partial evidence preserved.")
                offset += written
            # os.write is unbuffered: every byte has left Python userspace.
            # There is no Python output buffer requiring a separate flush.
            written_state = os.fstat(descriptor)
            require((written_state.st_dev, written_state.st_ino, written_state.st_mode, written_state.st_nlink)
                    == (created.st_dev, created.st_ino, created.st_mode, created.st_nlink)
                    and written_state.st_size == len(data), "Incomplete or changed private partial.")
            os.fchmod(descriptor, 0o400)
            os.fsync(descriptor)
            completed = os.fstat(descriptor)
            require(stat.S_IMODE(completed.st_mode) == 0o400, "Completed partial must be owner-read-only.")
        finally:
            os.close(descriptor)
        _verify_completed_partial(parent, partial, completed, data)
        try:
            os.link(partial, path.name, src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)
            published = True
            os.unlink(partial, dir_fd=parent)
            os.fsync(parent)
        except OSError as error:
            if published:
                raise PublishedDurabilityError("Published evidence has unconfirmed directory durability: " + str(path) + ". Preserve it; do not automatically retry.") from error
            raise
    finally:
        os.close(parent)


def publish(path: Path, value: dict) -> None:
    validate_seal(value)
    _exclusive_bytes(path, canonical_bytes(value) + b"\n")


def read_manifest(path: Path) -> dict:
    path = Path(path)
    _require_published_name(path)
    data = read_regular(path)
    result = strict_json(data)
    require(data == canonical_bytes(result) + b"\n", "Evidence must use canonical JSON and one final newline.")
    validate_seal(result)
    return result


def logical_path(value: Any) -> str:
    require(type(value) is str and bool(value) and not value.startswith("/"), "Invalid logical path.")
    require("\\" not in value and all(part not in ("", ".", "..") for part in value.split("/")), "Escaping logical path.")
    require(all(ord(character) >= 32 for character in value), "Control character in logical path.")
    return value


def file_reference(path: Path, logical_path: str | None = None) -> dict:
    data = read_regular(Path(path))
    name = Path(path).name if logical_path is None else logical_path
    result = {"logical_path": name, "byte_size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    validate_reference(result)
    return result


def validate_reference(value: Any) -> None:
    keys(value, ("logical_path", "byte_size", "sha256"), "File reference")
    logical_path(value["logical_path"])
    require(type(value["byte_size"]) is int and value["byte_size"] >= 0 and hex_digest(value["sha256"]), "Invalid file fingerprint.")


def default_intent() -> dict:
    """Advertised candidate hashes are intended pins, not acquired facts."""
    return {
        "schema_version": INTENT_SCHEMA,
        "environment_policy_identifier": POLICY,
        "platform": {"system": "Linux", "machine": "x86_64", "implementation": "CPython", "python_abi": "cp311", "minimum_glibc": "2.28"},
        "bootstrap": {"module": "conda/25.11.0", "channel": "conda-forge", "channel_priority": "strict", "python": "3.11.14", "pip": "25.2"},
        "direct_dependencies": [
            {"name": "numpy", "version": "2.2.6", "source": "pypi", "advertised_sha256": "ba10f8411898fc418a521833e014a77d3ca01c15b0c6cdcce6a0d2897e6dbbdf"},
            {"name": "PyYAML", "version": "6.0.3", "source": "pypi", "advertised_sha256": "b8bb0864c5a28024fac8a632c443c87c5aa6f215c0b126c449ae1a150412f31d"},
            {"name": "torch", "version": "2.14.0+cu126", "source": "torch_cu126", "advertised_sha256": "898b03fc60e642f1b28f59da9f647022a8ad339c1799693b1385fdcb8889f6ba"},
        ],
        "sources": {"conda": "https://conda.anaconda.org/conda-forge", "pypi": "https://pypi.org/simple", "torch_cu126": "https://download.pytorch.org/whl/cu126"},
        "installation": {"binary_only": True, "complete_closure": True, "offline_hash_locked_install": True, "fresh_prefix": True, "preserve_failed_environment_and_cache": True, "allow_system_install": False, "allow_user_install": False, "allow_extra_index": False, "allow_system_cuda_module": False},
        "numerics": {"precision": "float32", "amp": False, "compile": False, "deterministic_algorithms": True, "deterministic_warn_only": False, "cudnn_benchmark": False, "cudnn_deterministic": True, "allow_tf32": False, "float32_matmul_precision": "highest", "cublas_workspace_config": ":4096:8", "torch_intraop_threads": 1, "torch_interop_threads": 1, "omp_threads": 1, "mkl_threads": 1, "openblas_threads": 1, "numexpr_threads": 1},
        "gpu": {"device": "cuda:0", "visible_count": 1, "name": "Tesla P100-PCIE-16GB", "capability": [6, 0], "required_architecture": "sm_60", "torch_cuda_runtime": "12.6"},
        "verification": {"fixture": "cnn_rc_environment_fixed_tensor.v1", "seed": 43001, "input_shape": [128, 14, 4], "target_shape": [128, 1], "repetitions": 2, "output_atol": 1e-6, "output_rtol": 1e-5, "state_atol": 1e-5, "state_rtol": 1e-4, "same_device_exact": True, "real_slurm_required": True, "accepted_cpu_and_p100_required": True, "synthetic_or_mock_acceptance": False},
    }


def _same_shape(value: Any, template: Any) -> None:
    require(type(value) is type(template), "Contract field type differs.")
    if type(template) is dict:
        keys(value, template, "Contract")
        for key in template:
            _same_shape(value[key], template[key])
    elif type(template) is list:
        require(len(value) == len(template), "Contract list length differs.")
        for item, expected in zip(value, template):
            _same_shape(item, expected)


def validate_intent(value: Any) -> None:
    expected = default_intent()
    _plain(value)
    _same_shape(value, expected)
    require(value == expected, "Unsupported v1 environment intent; exact pins and policy are required.")


def load_intent(path: Path) -> dict:
    result = strict_json(read_regular(Path(path)))
    validate_intent(result)
    return result


def _run(command: list[str], *, env: dict | None = None) -> bytes:
    result = subprocess.run(command, capture_output=True, check=False, env=env)
    require(result.returncode == 0, "Command failed: " + " ".join(command) + "\nstdout:\n" + result.stdout[-16000:].decode("utf-8", "replace") + "\nstderr:\n" + result.stderr[-16000:].decode("utf-8", "replace"))
    return result.stdout


def _normalized_name(value: str) -> str:
    return re.sub("[-_.]+", "-", value).lower()


def platform_facts() -> dict:
    name, version = platform.libc_ver()
    require(platform.system() == "Linux" and platform.machine() == "x86_64", "Linux x86_64 is required.")
    require(name == "glibc" and re.fullmatch(r"\d+\.\d+(?:\.\d+)?", version) is not None, "glibc version is unavailable.")
    require(tuple(int(part) for part in version.split(".")[:2]) >= (2, 28), "glibc >= 2.28 is required.")
    return {"system": "Linux", "machine": "x86_64", "glibc": version}


def _slurm_context(mode: str) -> dict:
    require(mode in ("cpu", "p100"), "Unsupported execution mode.")
    job = os.environ.get("SLURM_JOB_ID", "")
    require(re.fullmatch(r"[0-9]+", job) is not None, "A real Slurm compute allocation is required.")
    raw = _run(["scontrol", "show", "job", job, "-o"])
    fields = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=([^\s]+)", raw.decode("utf-8")))
    require(fields.get("JobId") == job and fields.get("JobState") == "RUNNING", "Slurm job is not RUNNING.")
    require(fields.get("UserId", "").endswith("(" + str(os.getuid()) + ")"), "Slurm job owner differs.")
    require(fields.get("NumNodes") == "1" and fields.get("NumTasks") == "1", "One Slurm node and task are required.")
    nodes = _run(["scontrol", "show", "hostnames", fields.get("NodeList", "")]).decode().split()
    host = socket.gethostname()
    require(len(nodes) == 1 and host.split(".")[0] == nodes[0].split(".")[0], "Process is outside its allocated compute node.")
    require(re.search(r"login|headnode", host, re.IGNORECASE) is None, "Login nodes are forbidden.")
    cluster = os.environ.get("SLURM_CLUSTER_NAME", "")
    require(bool(cluster) and bool(fields.get("Partition")), "Slurm cluster and partition are required.")
    config = _run(["scontrol", "show", "config"]).decode()
    match = re.search(r"(?m)^\s*ClusterName\s*=\s*(\S+)", config)
    require(match is not None and match.group(1) == cluster, "Slurm controller cluster differs.")
    gpu = fields.get("AllocTRES", "")
    if mode == "cpu":
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and re.search(r"gres/gpu(?:[:=])", gpu) is None, "CPU verification requires a CPU allocation and hidden CUDA.")
    else:
        require(re.search(r"(?:^|,)gres/gpu=1(?:,|$)", gpu) is not None, "Exactly one allocated GPU is required.")
        require(bool(os.environ.get("CUDA_VISIBLE_DEVICES", "")) and "," not in os.environ["CUDA_VISIBLE_DEVICES"], "Exactly one GPU must be visible.")
    return {"job_id": job, "cluster": cluster, "partition": fields["Partition"], "node": nodes[0], "job_record_sha256": hashlib.sha256(raw).hexdigest()}


def _module_guard() -> list[str]:
    modules = os.environ.get("LOADEDMODULES", "").split(":")
    require("conda/25.11.0" in modules, "Explicit conda/25.11.0 module loading is required.")
    require(all(item.lower().split("/")[0] != "cuda" for item in modules), "System CUDA modules are forbidden.")
    return sorted(item for item in modules if item)


def activation_guard(prefix: Path) -> None:
    prefix = _physical(Path(prefix), directory=True)
    require(platform.python_implementation() == "CPython" and platform.python_version() == "3.11.14", "CPython 3.11.14 is required.")
    require(Path(sys.prefix) == prefix and Path(sys.executable).parent == prefix / "bin", "Active Python does not belong to the explicit prefix.")
    require(os.environ.get("CONDA_PREFIX") == str(prefix), "Explicit noninteractive Conda activation is required.")
    require(os.environ.get("PYTHONNOUSERSITE") == "1" and not site.ENABLE_USER_SITE, "User site must be disabled before Python startup.")
    require(os.environ.get("PYTHONDONTWRITEBYTECODE") == "1" and sys.dont_write_bytecode, "Bytecode writes must be disabled.")
    require(not os.environ.get("PYTHONPATH") and not os.environ.get("PYTHONHOME"), "Python path overrides are forbidden.")
    require(all(os.environ.get(name) == "1" for name in THREAD_VARIABLES), "All four numerical thread variables must equal 1.")
    require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "CUBLAS workspace policy differs.")
    require(importlib.metadata.version("pip") == "25.2", "pip 25.2 is required.")
    site_packages = prefix / "lib/python3.11/site-packages"
    for module_name, distribution_name, expected_version in (("numpy", "numpy", "2.2.6"), ("torch", "torch", "2.14.0+cu126"), ("yaml", "PyYAML", "6.0.3")):
        module = importlib.import_module(module_name)
        physical = Path(os.path.abspath(module.__file__))
        require(site_packages in physical.parents, "Imported dependency is outside the pinned prefix: " + module_name)
        require(all(site_packages in Path(os.path.abspath(path)).parents for path in module.__path__), "Imported dependency package path is shadowed.")
        distribution = importlib.metadata.distribution(distribution_name)
        require(distribution.version == expected_version, "Imported dependency version differs.")
        expected_files = {Path(os.path.abspath(distribution.locate_file(item))) for item in distribution.files or ()}
        require(physical in expected_files, "Imported dependency file is not owned by installed metadata.")
        _verify_distribution_record(distribution, prefix)


def execution_context(prefix: Path, mode: str = "cpu") -> dict:
    platform_facts()
    modules = _module_guard()
    activation_guard(prefix)
    slurm = _slurm_context(mode)
    result = {"prefix": str(prefix), "python_executable": sys.executable, "python_version": platform.python_version(), "host": socket.gethostname(), "utc": datetime.now(timezone.utc).isoformat(), "slurm": slurm, "loaded_modules": modules}
    validate_execution(result, mode)
    return result


def validate_execution(value: Any, mode: str = "cpu") -> None:
    keys(value, ("prefix", "python_executable", "python_version", "host", "utc", "slurm", "loaded_modules"), "Execution")
    require(mode in ("cpu", "p100"), "Invalid execution mode.")
    require(type(value["prefix"]) is str and Path(value["prefix"]).is_absolute() and value["prefix"] != "/", "Invalid execution prefix.")
    require(type(value["python_executable"]) is str and Path(value["python_executable"]).parent == Path(value["prefix"]) / "bin", "Invalid executable location.")
    require(value["python_version"] == "3.11.14", "Invalid Python execution version.")
    for key in ("host", "utc"):
        require(type(value[key]) is str and bool(value[key]), "Missing execution fact.")
    require(re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?\+00:00", value["utc"]) is not None, "Invalid UTC execution time.")
    keys(value["slurm"], ("job_id", "cluster", "partition", "node", "job_record_sha256"), "Slurm evidence")
    for key, item in value["slurm"].items():
        require(type(item) is str and bool(item), "Missing Slurm evidence.")
    require(value["slurm"]["job_id"].isdigit() and hex_digest(value["slurm"]["job_record_sha256"]), "Invalid Slurm identity.")
    require(value["host"].split(".")[0] == value["slurm"]["node"].split(".")[0], "Execution node differs.")
    modules = value["loaded_modules"]
    require(type(modules) is list and all(type(item) is str and bool(item) for item in modules), "Invalid module inventory.")
    require(modules == sorted(set(modules)) and "conda/25.11.0" in modules and all(item.lower().split("/")[0] != "cuda" for item in modules), "Module policy differs.")


def _b4a_git(root: Path, *arguments: str, data: bytes | None = None) -> bytes:
    """Read Git evidence for B4a only, without source execution or shell parsing."""
    result = subprocess.run(
        ["git", "--literal-pathspecs", "-C", str(root), *arguments],
        input=data, capture_output=True, check=False,
    )
    require(result.returncode == 0, "Git evidence unavailable or invalid.")
    require(result.stderr == b"", "Unexpected Git diagnostic output.")
    return result.stdout


def _b4a_git_line(value: bytes) -> str:
    require(value.endswith(b"\n") and value.count(b"\n") == 1
            and b"\0" not in value and b"\r" not in value, "Malformed Git text evidence.")
    try:
        return value[:-1].decode("utf-8")
    except UnicodeError as error:
        raise EnvironmentError("Malformed Git text encoding.") from error


def _b4a_git_context(root: Path) -> tuple[Path, str]:
    """Bind Git's top-level and project prefix to the supplied physical root."""
    root = _physical(Path(root), directory=True)
    top = Path(_b4a_git_line(_b4a_git(root, "rev-parse", "--show-toplevel")))
    _physical(top, directory=True)
    require(root == top or top in root.parents, "Git top-level differs from project root.")
    expected_prefix = "" if root == top else root.relative_to(top).as_posix() + "/"
    if expected_prefix:
        logical_path(expected_prefix[:-1])
    prefix = _b4a_git_line(_b4a_git(root, "rev-parse", "--show-prefix"))
    require(prefix == expected_prefix, "Git project prefix differs.")
    return top, prefix


def _b4a_validate_software(software: Any) -> None:
    """Validate the existing source-record envelope, solely for B4a evidence."""
    keys(software, ("runtime_commit", "source_inventory"), "Software")
    require(hex_digest(software["runtime_commit"], 40), "Invalid software commit.")
    inventory = software["source_inventory"]
    require(type(inventory) is list and bool(inventory), "Empty source inventory.")
    paths = []
    for record in inventory:
        keys(record, ("path", "git_blob", "sha256", "byte_size"), "Source record")
        paths.append(logical_path(record["path"]))
        require(hex_digest(record["git_blob"], 40) and hex_digest(record["sha256"])
                and type(record["byte_size"]) is int and record["byte_size"] >= 0,
                "Malformed source fingerprint.")
    require(paths == sorted(set(paths)), "Source paths must be sorted and unique.")


def _b4a_source_record(root: Path, prefix: str, commit: str, path: str) -> tuple[dict, bytes, str]:
    """Read precisely one allowlisted tree entry and its committed blob bytes."""
    tree_path = prefix + logical_path(path)
    entry = _b4a_git(root, "ls-tree", "--full-tree", "-z", commit, "--", tree_path)
    require(entry.endswith(b"\0") and entry.count(b"\0") == 1, "Source is not one tracked entry at commit.")
    fields = entry[:-1].split(b"\t")
    require(len(fields) == 2 and fields[1] == tree_path.encode("utf-8"), "Source tree path differs.")
    metadata = fields[0].split(b" ")
    require(len(metadata) == 3 and metadata[0] in (b"100644", b"100755")
            and metadata[1] == b"blob", "Source must be a regular tracked blob.")
    try:
        blob = metadata[2].decode("ascii")
    except UnicodeError as error:
        raise EnvironmentError("Malformed Git blob identity.") from error
    require(hex_digest(blob, 40), "Malformed Git blob identity.")
    content = _b4a_git(root, "cat-file", "blob", blob)
    require(_b4a_git_line(_b4a_git(root, "hash-object", "--stdin", data=content)) == blob,
            "Committed Git blob identity differs.")
    record = {"path": path, "git_blob": blob, "sha256": hashlib.sha256(content).hexdigest(),
              "byte_size": len(content)}
    return record, content, metadata[0].decode("ascii")


def _b4a_verify_historical_sources(
    root: Path, software: dict, source_paths: tuple[str, ...] | list[str] = SOURCE_PATHS,
) -> None:
    """Verify recorded objects without requiring current HEAD or worktree bytes.

    Missing historical objects fail explicitly; absence is never verification.
    This is B4a infrastructure, not the accepted scientific-run provenance API.
    """
    _b4a_validate_software(software)
    require(type(source_paths) in (tuple, list) and bool(source_paths), "Missing source inventory.")
    for path in source_paths:
        logical_path(path)
    require(len(set(source_paths)) == len(source_paths), "Duplicate source paths.")
    require([record["path"] for record in software["source_inventory"]] == sorted(source_paths),
            "Historical source inventory differs from the explicit allowlist.")
    root = _physical(Path(root), directory=True)
    _, prefix = _b4a_git_context(root)
    require(_b4a_git(root, "cat-file", "-t", software["runtime_commit"]) == b"commit\n",
            "Historical source commit is unavailable or invalid.")
    for record in software["source_inventory"]:
        historical, _, _ = _b4a_source_record(root, prefix, software["runtime_commit"], record["path"])
        require(historical == record, "Historical source blob mismatch.")


def _b4a_verify_runtime_sources(
    root: Path, expected_commit: str, source_paths: tuple[str, ...] | list[str],
    *, executing_paths: dict[str, Path] | None = None,
) -> dict:
    """Bind an explicit B4a allowlist to clean Git state and exact source bytes.

    Source blob and worktree reads use only the supplied allowlist. Git status
    separately checks staged and unstaged tracked state.
    """
    require(hex_digest(expected_commit, 40), "Expected commit must be a full SHA-1.")
    root = _physical(Path(root), directory=True)
    top, prefix = _b4a_git_context(root)
    require(_b4a_git_line(_b4a_git(root, "rev-parse", "--verify", "HEAD^{commit}")) == expected_commit,
            "Wrong runtime HEAD.")
    require(_b4a_git(root, "cat-file", "-t", expected_commit) == b"commit\n", "Invalid source commit.")
    require(not _b4a_git(top, "status", "--porcelain=v1", "--untracked-files=no"), "Dirty tracked state.")
    require(type(source_paths) in (tuple, list) and bool(source_paths), "Missing source inventory.")
    for path in source_paths:
        logical_path(path)
    require(len(set(source_paths)) == len(source_paths), "Duplicate source paths.")
    inventory = []
    for path in sorted(source_paths):
        record, committed, mode = _b4a_source_record(root, prefix, expected_commit, path)
        physical = root / path
        content = read_regular(physical)
        facts = os.stat(physical, follow_symlinks=False)
        require(stat.S_ISREG(facts.st_mode), "Source worktree mode is not regular.")
        actual_mode = "100755" if facts.st_mode & stat.S_IXUSR else "100644"
        require(actual_mode == mode, "Source worktree Git mode differs.")
        require(content == committed, "Executing source bytes differ from commit.")
        inventory.append(record)
    if executing_paths is not None:
        require(type(executing_paths) is dict and set(executing_paths).issubset(source_paths),
                "Executing module absent from source inventory.")
        records = {record["path"]: record for record in inventory}
        for name, physical in executing_paths.items():
            require(Path(os.path.abspath(physical)) == root / name, "Shadowed project import.")
            content = read_regular(Path(physical))
            require(len(content) == records[name]["byte_size"]
                    and hashlib.sha256(content).hexdigest() == records[name]["sha256"],
                    "Executing source bytes changed.")
    require(_b4a_git_context(root) == (top, prefix)
            and _b4a_git_line(_b4a_git(root, "rev-parse", "--verify", "HEAD^{commit}")) == expected_commit
            and not _b4a_git(top, "status", "--porcelain=v1", "--untracked-files=no"), "Checkout changed.")
    result = {"runtime_commit": expected_commit, "source_inventory": inventory}
    _b4a_validate_software(result)
    return result


def verify_software(expected_commit: str, root: Path = PROJECT_ROOT) -> dict:
    """Bind the fixed allowlist to Git, checking already loaded module locations."""
    require(hex_digest(expected_commit, 40), "Expected software commit must be a full lowercase SHA-1.")

    executing = {}
    for name, module in tuple(sys.modules.items()):
        if name == "src" or name.startswith("src.") or name.startswith("scripts.carc."):
            physical = getattr(module, "__file__", None)
            if physical is not None:
                relative = "src/__init__.py" if name == "src" else name.replace(".", "/") + ".py"
                require(relative in SOURCE_PATHS, "Imported project module absent from source inventory.")
                executing[relative] = Path(physical)
    executing["scripts/carc/cnn_rc_environment.py"] = Path(__file__)
    main = sys.modules.get("__main__")
    main_file = getattr(main, "__file__", "")
    if Path(main_file).name == "verify_cnn_rc_environment.py":
        executing["scripts/carc/verify_cnn_rc_environment.py"] = Path(main_file)
    for namespace in ("scripts", "scripts/carc"):
        require(not os.path.lexists(root / namespace / "__init__.py"), "Unexpected package initializer outside source inventory.")
    return _b4a_verify_runtime_sources(root, expected_commit, SOURCE_PATHS, executing_paths=executing)


def _validate_package(record: dict, *, acquisition: bool = False) -> None:
    keys(record, ACQUISITION_FIELDS if acquisition else PACKAGE_FIELDS, "Package")
    require(record["manager"] in ("conda", "pip"), "Unknown package manager.")
    for key in ("name", "version", "build", "subdir", "filename", "origin_url"):
        require(type(record[key]) is str and bool(record[key]), "Missing exact package identity: " + key)
    require(re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", record["name"]) is not None, "Invalid package name.")
    for key in ("build", "subdir"):
        require(re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.+!-]*", record[key]) is not None, "Invalid package identity component.")
    require(re.fullmatch(r"[0-9][A-Za-z0-9_.+!-]*", record["version"]) is not None, "Unpinned package version.")
    require(Path(record["filename"]).name == record["filename"] and not record["filename"].startswith("."), "Invalid package filename.")
    require(hex_digest(record["sha256"]), "Missing archive SHA-256.")
    keys(record["upstream_digest"], ("algorithm", "value"), "Upstream digest")
    algorithm = record["upstream_digest"]["algorithm"]
    require(algorithm in ("sha256", "md5") and hex_digest(record["upstream_digest"]["value"], 64 if algorithm == "sha256" else 32), "Invalid upstream digest.")
    if algorithm == "sha256":
        require(record["upstream_digest"]["value"] == record["sha256"], "Upstream SHA-256 differs from archive SHA-256.")
    parsed = urlparse(record["origin_url"])
    require(parsed.scheme == "https" and not parsed.username and not parsed.password and not parsed.fragment, "Archive origin must be an exact HTTPS URL.")
    require(unquote(Path(parsed.path).name) == record["filename"], "Archive filename differs from origin URL.")
    if record["manager"] == "conda":
        require(parsed.netloc == "conda.anaconda.org" and parsed.path.startswith("/conda-forge/"), "Unexpected Conda origin.")
        require(record["subdir"] in ("linux-64", "noarch") and record["filename"].endswith((".conda", ".tar.bz2")), "Invalid Conda binary archive.")
        stem = record["name"] + "-" + record["version"] + "-" + record["build"]
        require(record["filename"] in (stem + ".conda", stem + ".tar.bz2"), "Conda filename does not bind its name, version and build.")
        require(parsed.path == "/conda-forge/" + record["subdir"] + "/" + record["filename"], "Conda origin subdirectory differs.")
    else:
        require(record["filename"].endswith(".whl") and record["build"] == "wheel", "Only binary wheels are permitted.")
        wheel_identity = record["filename"][:-4].rsplit("-", 3)[0].split("-")
        require(len(wheel_identity) in (2, 3) and _normalized_name(wheel_identity[0]) == _normalized_name(record["name"]) and wheel_identity[1] == record["version"], "Wheel filename does not bind its distribution and version.")
        if _normalized_name(record["name"]) == "torch":
            require(parsed.netloc == "download.pytorch.org" and parsed.path.startswith("/whl/cu126/"), "Torch must come from the official cu126 index.")
        else:
            require(parsed.netloc == "files.pythonhosted.org", "Other wheels must come from PyPI.")
    if acquisition:
        require(type(record["byte_size"]) is int and record["byte_size"] > 0, "Missing archive size.")
        logical_path(record["cache_path"])
        require(record["cache_path"] == record["manager"] + "/" + record["filename"], "Archive cache location differs.")
        require(type(record["platform_tags"]) is list and bool(record["platform_tags"]) and all(type(item) is str and bool(item) for item in record["platform_tags"]), "Missing archive platform tags.")
        if record["manager"] == "conda":
            require(record["platform_tags"] == [record["subdir"]], "Conda platform tags differ.")
        else:
            require(record["platform_tags"] == _wheel_tags(record["filename"]), "Wheel platform tags differ from filename.")
            require(record["subdir"] == record["filename"][:-4].rsplit("-", 1)[1], "Wheel subdir differs from filename.")
            compatible = False
            for tag in record["platform_tags"]:
                python_tag, abi, platform_tag = tag.split("-")
                python_ok = python_tag in ("py3", "cp311")
                if abi == "abi3" and re.fullmatch(r"cp3\d+", python_tag):
                    python_ok = int(python_tag[3:]) <= 11
                platform_ok = platform_tag == "any" or platform_tag == "linux_x86_64" or re.fullmatch(r"manylinux(?:_[0-9]+_[0-9]+|1|2010|2014)_x86_64", platform_tag) is not None
                compatible = compatible or (python_ok and abi in ("none", "cp311", "abi3") and platform_ok)
            require(compatible, "Wheel is incompatible with CPython cp311 Linux x86_64.")
    else:
        require(hex_digest(record["installed_metadata_sha256"]), "Missing installed metadata hash.")


def validate_acquisition_lock(lock: dict, spec: dict) -> None:
    validate_intent(spec)
    keys(lock, ("schema_version", "intent_sha256", "conda", "pip", "manifest_hash"), "Acquisition lock")
    validate_seal(lock)
    require(lock["schema_version"] == LOCK_SCHEMA and lock["intent_sha256"] == digest(spec), "Acquisition intent differs.")
    seen = set()
    for manager in ("conda", "pip"):
        records = lock[manager]
        require(type(records) is list and bool(records), "Complete package closure is required.")
        order = []
        for record in records:
            _validate_package(record, acquisition=True)
            require(record["manager"] == manager, "Package manager differs.")
            name = _normalized_name(record["name"])
            require(name not in seen, "Duplicate or overlapping managed package; replacing Conda bootstrap packages is forbidden.")
            require(manager != "pip" or name not in ("python", "pip"), "Pip may not replace the Conda bootstrap pins.")
            seen.add(name)
            order.append(name)
        require(order == sorted(order), "Package closure must be sorted.")
    bootstrap = {record["name"]: record["version"] for record in lock["conda"]}
    require(bootstrap.get("python") == "3.11.14" and bootstrap.get("pip") == "25.2", "Bootstrap pins differ.")
    wheels = {_normalized_name(record["name"]): record for record in lock["pip"]}
    for direct in spec["direct_dependencies"]:
        record = wheels.get(_normalized_name(direct["name"]))
        require(record is not None and record["version"] == direct["version"] and record["sha256"] == direct["advertised_sha256"], "Intended direct wheel pin differs.")


EXPORT_COMMANDS = {
    "conda-explicit.txt": ["conda", "list", "--explicit"],
    "conda-list.json": ["conda", "list", "--json"],
    "pip-list.json": ["-m", "pip", "--isolated", "--disable-pip-version-check", "list", "--no-index", "--format=json"],
    "pip-inspect.json": ["-m", "pip", "--isolated", "--disable-pip-version-check", "inspect", "--local"],
    "pip-check.txt": ["-m", "pip", "--isolated", "--disable-pip-version-check", "check"],
}


def collect_exports(prefix: Path, exports_root: Path) -> None:
    """Exclusively collect real command output; never invent export contents."""
    execution_context(prefix, "cpu")
    root = _physical(Path(exports_root))
    root.mkdir(mode=0o700)
    for name, arguments in EXPORT_COMMANDS.items():
        command = arguments + ["--prefix", str(prefix)] if arguments[0] == "conda" else [sys.executable, *arguments]
        _exclusive_bytes(root / name, _run(command, env=_read_only_environment()))


def _archive_fingerprint(path: Path, algorithms: tuple[str, ...] = ("sha256",)) -> tuple[int, dict]:
    path = _physical(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), "Archive must be a regular file.")
        checksums = {name: hashlib.new(name) for name in algorithms}
        size = 0
        block = os.read(descriptor, 1024 * 1024)
        while block:
            size += len(block)
            for checksum in checksums.values():
                checksum.update(block)
            block = os.read(descriptor, 1024 * 1024)
        after = os.fstat(descriptor)
        current = path.stat()
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        require(all(getattr(before, field) == getattr(after, field) == getattr(current, field) for field in fields), "Archive changed during read.")
        return size, {name: checksum.hexdigest() for name, checksum in checksums.items()}
    finally:
        os.close(descriptor)


def _verify_archive_cache(lock: dict, cache_root: Path) -> None:
    for manager in ("conda", "pip"):
        records = lock[manager]
        require({path.name for path in (cache_root / manager).iterdir()} == {record["filename"] for record in records}, "Archive cache inventory differs.")
        for record in records:
            size, hashes = _archive_fingerprint(cache_root / record["cache_path"])
            require(size == record["byte_size"] and hashes["sha256"] == record["sha256"], "Acquired archive cache bytes differ.")


def _verify_distribution_record(distribution, prefix: Path) -> None:
    text = distribution.read_text("RECORD")
    require(type(text) is str and bool(text), "Installed wheel RECORD is absent.")
    seen = set()
    for row in csv.reader(io.StringIO(text)):
        require(len(row) == 3 and row[0] not in seen, "Malformed installed wheel RECORD.")
        seen.add(row[0])
        physical = Path(os.path.abspath(distribution.locate_file(row[0])))
        require(prefix in physical.parents, "Wheel RECORD escapes the environment prefix.")
        if row[1]:
            algorithm, encoded = row[1].split("=", 1)
            require(algorithm == "sha256" and row[2].isdigit(), "Wheel RECORD must contain SHA-256 and size.")
            size, hashes = _archive_fingerprint(physical)
            observed = base64.urlsafe_b64encode(bytes.fromhex(hashes["sha256"])).decode().rstrip("=")
            require(observed == encoded and size == int(row[2]), "Installed wheel payload differs from RECORD.")
        else:
            require(row[0].endswith(".dist-info/RECORD") or row[0].endswith(".pyc"), "Unhashed installed wheel payload.")
            require(physical.is_file(), "Installed wheel payload is absent.")


def _verify_wheel_archive(distribution, prefix: Path, archive_path: Path) -> None:
    """Compare installed payloads to the acquired archive's immutable RECORD."""
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), "Duplicate archive member.")
        record_names = [name for name in names if name.endswith(".dist-info/RECORD")]
        require(len(record_names) == 1, "Acquired wheel RECORD is missing or ambiguous.")
        rows = list(csv.reader(io.StringIO(archive.read(record_names[0]).decode("utf-8"))))
        require({row[0] for row in rows if len(row) == 3} == {name for name in names if not name.endswith("/")}, "Wheel RECORD does not cover its payload.")
        for row in rows:
            require(len(row) == 3, "Invalid acquired wheel RECORD.")
            logical_path(row[0])
            if row[0] != record_names[0]:
                require(row[1].startswith("sha256=") and row[2].isdigit(), "Acquired wheel payload hash is missing.")
                relative = row[0]
                script = False
                if ".data/" in relative:
                    unused, relocated = relative.split(".data/", 1)
                    category, relative = relocated.split("/", 1)
                    require(category in ("purelib", "platlib", "scripts", "data"), "Unsupported wheel relocation scheme.")
                    if category == "scripts":
                        physical = prefix / "bin" / relative
                        script = True
                    elif category == "data":
                        physical = prefix / relative
                    else:
                        physical = prefix / "lib/python3.11/site-packages" / relative
                else:
                    physical = Path(distribution.locate_file(relative))
                require(prefix in physical.parents, "Wheel archive payload escapes prefix.")
                if script:
                    original = archive.read(row[0])
                    installed = read_regular(physical)
                    if original.startswith((b"#!python\n", b"#!pythonw\n")):
                        original = b"#!" + str(prefix / "bin/python").encode() + b"\n" + original.split(b"\n", 1)[1]
                    require(installed == original, "Installed wheel script differs from acquired archive.")
                else:
                    size, hashes = _archive_fingerprint(physical)
                    observed = base64.urlsafe_b64encode(bytes.fromhex(hashes["sha256"])).decode().rstrip("=")
                    require(observed == row[1].split("=", 1)[1] and size == int(row[2]), "Installed wheel payload differs from acquired archive.")


def _verify_conda_payload(installed: dict, prefix: Path) -> list[dict]:
    """Check Conda's installed path hashes, including relocated prefix bytes."""
    paths = installed.get("paths_data", {}).get("paths")
    require(type(paths) is list, "Installed Conda path evidence is unavailable.")
    require(bool(paths) or installed.get("files") == [], "Empty Conda payload conflicts with installed file list.")
    seen = set()
    generated = []
    for record in paths:
        require(type(record) is dict and type(record.get("_path")) is str, "Malformed installed Conda path record.")
        relative = logical_path(record["_path"])
        require(relative not in seen, "Duplicate installed Conda path.")
        seen.add(relative)
        physical = prefix / relative
        path_type = record.get("path_type")
        if path_type == "softlink":
            require(physical.is_symlink(), "Installed Conda symlink differs.")
            resolved = physical.resolve(strict=True)
            require(prefix in resolved.parents, "Installed Conda symlink escapes the prefix.")
        elif path_type == "directory":
            _physical(physical, directory=True)
        elif path_type in ("pyc_file", "unix_python_entry_point"):
            # Conda generates these during initial linking. Its archive path
            # metadata supplies no checksum; record observed generated bytes
            # as installation facts outside the semantic environment identity.
            generated.append(file_reference(physical, relative))
        else:
            require(path_type == "hardlink", "Unsupported Conda path type.")
            expected_hash = record.get("sha256_in_prefix")
            if not hex_digest(expected_hash):
                require(not record.get("prefix_placeholder"), "Relocated Conda payload lacks its installed checksum.")
                expected_hash = record.get("sha256")
            require(hex_digest(expected_hash), "Installed Conda payload checksum is missing.")
            unused_size, hashes = _archive_fingerprint(physical)
            require(hashes["sha256"] == expected_hash, "Installed Conda payload differs from package metadata.")
    return generated


def _installed_packages(prefix: Path, lock: dict, cache_root: Path) -> list[dict]:
    _verify_archive_cache(lock, cache_root)
    packages = []
    conda_records = {}
    for path in sorted((prefix / "conda-meta").glob("*.json")):
        data = read_regular(path)
        record = strict_json(data)
        name = record.get("name")
        require(type(name) is str and name not in conda_records, "Invalid installed Conda metadata.")
        conda_records[name] = (record, hashlib.sha256(data).hexdigest())
    require(set(conda_records) == {record["name"] for record in lock["conda"]}, "Installed Conda closure differs.")
    for archived in lock["conda"]:
        installed, metadata_hash = conda_records[archived["name"]]
        require(installed.get("version") == archived["version"] and installed.get("build") == archived["build"] and installed.get("subdir") == archived["subdir"], "Installed Conda identity differs.")
        require(installed.get("sha256") in (None, archived["sha256"]), "Installed Conda archive hash differs.")
        require(installed.get("fn") == archived["filename"] and installed.get("url", "").split("#", 1)[0] == (cache_root / archived["cache_path"]).as_uri(), "Installed Conda archive origin differs.")
        unused_size, archive_hashes = _archive_fingerprint(cache_root / archived["cache_path"], ("md5",))
        require(installed.get("md5") == archive_hashes["md5"], "Installed Conda archive checksum differs.")
        generated = _verify_conda_payload(installed, prefix)
        metadata_hash = digest({"metadata_sha256": metadata_hash, "generated_files": generated})
        record = {key: archived[key] for key in PACKAGE_FIELDS[:-1]}
        record["installed_metadata_sha256"] = metadata_hash
        packages.append(record)
    expected_pip = {_normalized_name(record["name"]): record for record in lock["pip"]}
    conda_python = set()
    for installed, unused in conda_records.values():
        for name in installed.get("files", []):
            if ".dist-info/METADATA" in name:
                metadata_directory = Path(name).parent.name[:-10]
                conda_python.add(_normalized_name(metadata_directory.rsplit("-", 1)[0]))
    seen = set()
    distributions = importlib.metadata.distributions(path=[str(prefix / "lib/python3.11/site-packages")])
    for distribution in distributions:
        name = _normalized_name(distribution.metadata["Name"])
        require(name not in seen, "Duplicate installed Python distribution.")
        seen.add(name)
        if name in expected_pip:
            archived = expected_pip[name]
            require(distribution.version == archived["version"], "Installed wheel version differs.")
            direct_url_text = distribution.read_text("direct_url.json")
            require(type(direct_url_text) is str, "Installed wheel lacks direct archive provenance.")
            direct_url = strict_json(direct_url_text.encode())
            keys(direct_url, ("url", "archive_info"), "Installed wheel origin")
            require(direct_url["url"] == (cache_root / archived["cache_path"]).as_uri(), "Installed wheel archive origin differs.")
            require(direct_url["archive_info"].get("hashes", {}).get("sha256") == archived["sha256"], "Installed wheel archive hash differs.")
            wheel_text = distribution.read_text("WHEEL")
            require(type(wheel_text) is str and sorted(re.findall(r"(?m)^Tag: (\S+)$", wheel_text)) == sorted(archived["platform_tags"]), "Installed wheel platform tags differ.")
            _verify_distribution_record(distribution, prefix)
            _verify_wheel_archive(distribution, prefix, cache_root / archived["cache_path"])
            records = []
            for item in sorted(distribution.files or (), key=str):
                if ".dist-info/" in str(item):
                    physical = Path(distribution.locate_file(item))
                    require(prefix in physical.parents, "Distribution metadata escapes prefix.")
                    records.append(file_reference(physical, str(item)))
            require(bool(records), "Installed wheel metadata is absent.")
            record = {key: archived[key] for key in PACKAGE_FIELDS[:-1]}
            record["installed_metadata_sha256"] = digest(records)
            packages.append(record)
        else:
            require(name in conda_python, "Unlisted installed Python package: " + name)
    require(set(expected_pip).issubset(seen), "Installed wheel closure is incomplete.")
    return sorted(packages, key=lambda item: (item["manager"], _normalized_name(item["name"])))


def semantic_environment(spec: dict, packages: list[dict], observed_platform: dict) -> dict:
    """Exclude installed metadata hashes because metadata may contain prefixes."""
    validate_intent(spec)
    semantic_packages = []
    for record in packages:
        _validate_package(record)
        semantic_packages.append({key: record[key] for key in PACKAGE_FIELDS[:-1]})
    return {"policy": POLICY, "platform": observed_platform, "python": {"implementation": "CPython", "version": "3.11.14", "abi": "cp311"}, "packages": semantic_packages, "module": "conda/25.11.0", "numerics": copy.deepcopy(spec["numerics"])}


def build_inventory(spec: dict, prefix: Path, acquisition_lock_path: Path, exports_root: Path) -> dict:
    validate_intent(spec)
    execution = execution_context(prefix, "cpu")
    lock = read_manifest(acquisition_lock_path)
    validate_acquisition_lock(lock, spec)
    packages = _installed_packages(Path(prefix), lock, Path(acquisition_lock_path).parent)
    exports = []
    root = _physical(Path(exports_root), directory=True)
    require({path.name for path in root.iterdir()} == set(EXPORT_COMMANDS), "Export inventory differs.")
    for name in sorted(EXPORT_COMMANDS):
        reference = file_reference(root / name, name)
        command = EXPORT_COMMANDS[name]
        actual_command = command + ["--prefix", str(prefix)] if command[0] == "conda" else [sys.executable, *command]
        require(hashlib.sha256(_run(actual_command, env=_read_only_environment())).hexdigest() == reference["sha256"], "Installed export differs from recorded export: " + name)
        exports.append(reference)
    semantic = semantic_environment(spec, packages, platform_facts())
    result = seal({"schema_version": INVENTORY_SCHEMA, "intent_sha256": digest(spec), "environment_id": "env_" + digest(semantic), "semantic_environment": semantic, "execution": execution, "packages": packages, "exports": exports, "acquisition_lock": file_reference(acquisition_lock_path)})
    validate_inventory(result)
    return result


def validate_inventory(value: dict) -> None:
    keys(value, ("schema_version", "intent_sha256", "environment_id", "semantic_environment", "execution", "packages", "exports", "acquisition_lock", "manifest_hash"), "Inventory")
    validate_seal(value)
    require(value["schema_version"] == INVENTORY_SCHEMA and value["intent_sha256"] == digest(default_intent()), "Inventory intent differs.")
    validate_execution(value["execution"])
    require(type(value["packages"]) is list and bool(value["packages"]), "Missing package inventory.")
    for record in value["packages"]:
        _validate_package(record)
    conda_versions = {record["name"]: record["version"] for record in value["packages"] if record["manager"] == "conda"}
    require(conda_versions.get("python") == "3.11.14" and conda_versions.get("pip") == "25.2", "Inventory bootstrap pins differ.")
    wheel_packages = {_normalized_name(record["name"]): record for record in value["packages"] if record["manager"] == "pip"}
    for intended in default_intent()["direct_dependencies"]:
        record = wheel_packages.get(_normalized_name(intended["name"]))
        require(record is not None and record["version"] == intended["version"] and record["sha256"] == intended["advertised_sha256"], "Inventory intended wheel pin differs.")
    order = [(record["manager"], _normalized_name(record["name"])) for record in value["packages"]]
    require(order == sorted(set(order)), "Package inventory must be sorted and unique.")
    semantic = value["semantic_environment"]
    keys(semantic, ("policy", "platform", "python", "packages", "module", "numerics"), "Semantic environment")
    keys(semantic["platform"], ("system", "machine", "glibc"), "Installed platform")
    require(semantic["platform"]["system"] == "Linux" and semantic["platform"]["machine"] == "x86_64", "Installed platform differs.")
    glibc = semantic["platform"]["glibc"]
    require(type(glibc) is str and re.fullmatch(r"\d+\.\d+(?:\.\d+)?", glibc) is not None and tuple(int(item) for item in glibc.split(".")[:2]) >= (2, 28), "Invalid installed glibc.")
    expected = semantic_environment(default_intent(), value["packages"], semantic["platform"])
    require(canonical_bytes(semantic) == canonical_bytes(expected), "Semantic environment differs from installation facts.")
    require(value["environment_id"] == "env_" + digest(semantic), "Environment identity differs.")
    require(type(value["exports"]) is list, "Missing exports.")
    for reference in value["exports"]:
        validate_reference(reference)
    require([item["logical_path"] for item in value["exports"]] == sorted(EXPORT_COMMANDS), "Export references differ.")
    validate_reference(value["acquisition_lock"])


def revalidate_inventory(inventory: dict, spec: dict, prefix: Path, exports_root: Path, acquisition_lock_path: Path, mode: str = "cpu") -> None:
    """Recheck installed facts and all existing evidence without writing files."""
    validate_inventory(inventory)
    validate_intent(spec)
    execution_context(prefix, mode)
    require(str(prefix) == inventory["execution"]["prefix"], "Inventory belongs to another installed prefix.")
    lock = read_manifest(acquisition_lock_path)
    validate_acquisition_lock(lock, spec)
    require(file_reference(acquisition_lock_path) == inventory["acquisition_lock"], "Acquisition lock bytes differ.")
    packages = _installed_packages(Path(prefix), lock, Path(acquisition_lock_path).parent)
    require(packages == inventory["packages"], "Installed package facts changed.")
    require(semantic_environment(spec, packages, platform_facts()) == inventory["semantic_environment"], "Installed platform differs.")
    require({path.name for path in Path(exports_root).iterdir()} == set(EXPORT_COMMANDS), "Export inventory changed.")
    for reference in inventory["exports"]:
        require(file_reference(Path(exports_root) / reference["logical_path"]) == reference, "Export bytes changed.")
        command = EXPORT_COMMANDS[reference["logical_path"]]
        actual_command = command + ["--prefix", str(prefix)] if command[0] == "conda" else [sys.executable, *command]
        require(hashlib.sha256(_run(actual_command, env=_read_only_environment())).hexdigest() == reference["sha256"], "Installed export changed.")


def _pip_environment() -> dict:
    environment = dict(os.environ)
    for name in list(environment):
        if name.startswith("PIP_"):
            del environment[name]
    environment["PIP_CONFIG_FILE"] = os.devnull
    environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _read_only_environment() -> dict:
    environment = _pip_environment()
    environment["CONDA_OFFLINE"] = "true"
    environment["CONDA_NOTICES_CORE"] = "false"
    return environment


def _download(url: str, path: Path, expected: dict) -> tuple[int, str]:
    """Acquisition-only HTTPS download with exclusive, preserved partial output."""
    require(urlparse(url).scheme == "https", "Only HTTPS binary downloads are permitted.")
    _physical(path)
    algorithm = expected["algorithm"]
    checksum = hashlib.new(algorithm)
    sha256 = hashlib.sha256()
    size = 0
    # Context managers ensure both network and output handles close on failure.
    with urllib.request.urlopen(url) as response, open(path, "xb") as destination:
        require(urlparse(response.url).scheme == "https", "Insecure archive redirect.")
        block = response.read(1024 * 1024)
        while block:
            destination.write(block)
            checksum.update(block)
            sha256.update(block)
            size += len(block)
            block = response.read(1024 * 1024)
        destination.flush()
        os.fsync(destination.fileno())
    require(size > 0 and checksum.hexdigest() == expected["value"], "Downloaded archive hash differs; preserve failed cache.")
    return size, sha256.hexdigest()


def _wheel_tags(filename: str) -> list[str]:
    require(filename.endswith(".whl"), "Source distributions are forbidden.")
    parts = filename[:-4].rsplit("-", 3)
    require(len(parts) == 4, "Malformed wheel filename.")
    return [python_tag + "-" + abi + "-" + platform_tag for python_tag in parts[1].split(".") for abi in parts[2].split(".") for platform_tag in parts[3].split(".")]


def _conda_record(item: dict) -> dict:
    name = item.get("name")
    version = item.get("version")
    build = item.get("build", item.get("build_string"))
    subdir = item.get("subdir", item.get("platform"))
    filename = item.get("fn", item.get("filename"))
    url = item.get("url")
    require(all(type(value) is str and bool(value) for value in (name, version, build, subdir)), "Conda transaction lacks exact binary identity.")
    if url is None:
        require(type(filename) is str and bool(filename), "Conda transaction lacks filename; no archive-format guessing.")
        url = "https://conda.anaconda.org/conda-forge/" + subdir + "/" + filename
    if filename is None:
        filename = unquote(Path(urlparse(url).path).name)
    algorithm = "sha256" if hex_digest(item.get("sha256")) else "md5"
    expected = item.get(algorithm)
    require(hex_digest(expected, 64 if algorithm == "sha256" else 32), "Conda transaction lacks upstream digest.")
    sha256 = item["sha256"] if hex_digest(item.get("sha256")) else "0" * 64
    return {"manager": "conda", "name": name, "version": version, "build": build, "subdir": subdir, "filename": filename, "origin_url": url, "sha256": sha256, "upstream_digest": {"algorithm": algorithm, "value": expected}, "byte_size": 1, "platform_tags": [subdir], "cache_path": "conda/" + filename}


def _pip_report_record(item: dict, originals: dict[str, dict]) -> dict:
    keys(item["download_info"], ("url", "archive_info"), "Pip archive download")
    information = item["download_info"]
    metadata = item["metadata"]
    name = _normalized_name(metadata["name"])
    if name in originals:
        result = copy.deepcopy(originals[name])
        require(metadata["version"] == result["version"], "Resolved direct wheel version differs.")
        require(urlparse(information["url"]).scheme == "file" and unquote(Path(urlparse(information["url"]).path).name) == result["filename"], "Resolved direct wheel archive differs.")
        require(information["archive_info"].get("hashes", {}).get("sha256") == result["sha256"], "Resolved direct wheel hash differs.")
        return result
    url = information["url"]
    filename = unquote(Path(urlparse(url).path).name)
    archive_hash = information["archive_info"].get("hashes", {}).get("sha256")
    require(hex_digest(archive_hash), "Pip report lacks exact wheel SHA-256.")
    tags = _wheel_tags(filename)
    return {"manager": "pip", "name": metadata["name"], "version": metadata["version"], "build": "wheel", "subdir": filename[:-4].rsplit("-", 1)[1], "filename": filename, "origin_url": url, "sha256": archive_hash, "upstream_digest": {"algorithm": "sha256", "value": archive_hash}, "byte_size": 1, "platform_tags": tags, "cache_path": "pip/" + filename}


def acquire_environment(spec_path: Path, prefix: Path, cache_root: Path, exports_root: Path, expected_commit: str) -> None:
    """Future authorized CPU allocation only; never called by a verifier."""
    spec = load_intent(spec_path)
    platform_facts()
    _module_guard()
    _slurm_context("cpu")
    require(hex_digest(expected_commit, 40), "Expected software commit must be a full SHA-1.")
    # The same strict provenance is standard-library-only before pinned Python exists.
    _b4a_verify_runtime_sources(
        PROJECT_ROOT, expected_commit, SOURCE_PATHS,
        executing_paths={"scripts/carc/cnn_rc_environment.py": Path(__file__)},
    )
    prefix = _physical(Path(prefix))
    cache = _physical(Path(cache_root))
    exports = _physical(Path(exports_root))
    for path in (prefix, cache, exports):
        require(not os.path.lexists(path), "Fresh prefix, cache and export directory are required; preserve existing contents.")
    for first, second in ((prefix, cache), (prefix, exports), (cache, exports)):
        require(first != second and first not in second.parents and second not in first.parents, "Creation paths must be separate.")
    cache.mkdir(mode=0o700)
    (cache / "conda").mkdir()
    (cache / "pip").mkdir()
    environment = _pip_environment()
    environment["CONDA_PKGS_DIRS"] = str(cache / "conda-package-cache")
    environment["CONDA_CHANNEL_PRIORITY"] = "strict"
    transaction_bytes = _run(["conda", "create", "--dry-run", "--json", "--yes", "--no-default-packages", "--prefix", str(prefix), "--override-channels", "--channel", "conda-forge", "--strict-channel-priority", "python=3.11.14", "pip=25.2"], env=environment)
    _exclusive_bytes(cache / "conda-transaction.json", transaction_bytes)
    transaction = strict_json(transaction_bytes)
    require(transaction.get("success") is True and type(transaction.get("actions")) is dict, "Conda could not resolve the intended binaries.")
    links = transaction["actions"].get("LINK")
    require(type(links) is list and bool(links) and not transaction["actions"].get("UNLINK"), "Conda must resolve a complete fresh bootstrap transaction.")
    fetches = transaction["actions"].get("FETCH", [])
    conda_records = []
    explicit = ["@EXPLICIT"]
    for item in links:
        combined = dict(item)
        for fetched in fetches:
            if fetched.get("name") == item.get("name") and fetched.get("version") == item.get("version") and fetched.get("build", fetched.get("build_string")) == item.get("build", item.get("build_string")):
                combined.update(fetched)
        record = _conda_record(combined)
        _validate_package(record, acquisition=True)
        path = cache / record["cache_path"]
        record["byte_size"], record["sha256"] = _download(record["origin_url"], path, record["upstream_digest"])
        conda_records.append(record)
        # Conda explicit files support MD5 fragments; SHA-256 is independently locked.
        unused_size, hashes = _archive_fingerprint(path, ("md5",))
        md5 = hashes["md5"]
        explicit.append(path.as_uri() + "#" + md5)
    conda_records.sort(key=lambda item: _normalized_name(item["name"]))
    _exclusive_bytes(cache / "conda-explicit-install.txt", ("\n".join(explicit) + "\n").encode())
    publish(cache / "conda-bootstrap-lock.json", seal({"schema_version": "carc_cnn_rc_conda_acquisition.v1", "intent_sha256": digest(spec), "packages": conda_records}))
    _run(["conda", "create", "--yes", "--offline", "--no-default-packages", "--prefix", str(prefix), "--file", str(cache / "conda-explicit-install.txt")], env=environment)
    python = str(prefix / "bin/python")
    version = _run([python, "-B", "-c", "import platform, importlib.metadata; print(platform.python_version()); print(importlib.metadata.version('pip'))"], env=environment).decode().splitlines()
    require(version == ["3.11.14", "25.2"], "Installed bootstrap pins differ.")
    originals = {}
    direct_paths = []
    for intended in spec["direct_dependencies"]:
        report_path = cache / (_normalized_name(intended["name"]) + "-candidate-report.json")
        _run([python, "-B", "-m", "pip", "--isolated", "install", "--dry-run", "--ignore-installed", "--no-deps", "--no-cache-dir", "--only-binary=:all:", "--index-url", spec["sources"][intended["source"]], "--report", str(report_path), intended["name"] + "==" + intended["version"]], env=environment)
        report = strict_json(read_regular(report_path))
        require(type(report.get("install")) is list and len(report["install"]) == 1, "Exact intended wheel is unavailable.")
        record = _pip_report_record(report["install"][0], {})
        require(_normalized_name(record["name"]) == _normalized_name(intended["name"]) and record["version"] == intended["version"] and record["sha256"] == intended["advertised_sha256"], "Advertised candidate hash or pin differs; no fallback is allowed.")
        _validate_package(record, acquisition=True)
        path = cache / record["cache_path"]
        record["byte_size"], actual_hash = _download(record["origin_url"], path, record["upstream_digest"])
        require(actual_hash == record["sha256"], "Direct archive hash differs.")
        originals[_normalized_name(record["name"])] = record
        direct_paths.append(str(path))
    full_report = cache / "pip-closure-report.json"
    bootstrap_distributions = _run([python, "-B", "-m", "pip", "--isolated", "--disable-pip-version-check", "list", "--no-index", "--format=json"], env=environment)
    _exclusive_bytes(cache / "bootstrap-python-distributions.json", bootstrap_distributions)
    bootstrap_names = {_normalized_name(item["name"]) for item in strict_json(b'{"packages":' + bootstrap_distributions + b'}')["packages"]}
    _run([python, "-B", "-m", "pip", "--isolated", "install", "--dry-run", "--ignore-installed", "--no-cache-dir", "--only-binary=:all:", "--index-url", spec["sources"]["pypi"], "--report", str(full_report), *direct_paths], env=environment)
    report = strict_json(read_regular(full_report))
    require(type(report.get("install")) is list and bool(report["install"]), "Full wheel closure is unavailable.")
    pip_records = []
    for item in report["install"]:
        record = _pip_report_record(item, originals)
        _validate_package(record, acquisition=True)
        require(_normalized_name(record["name"]) not in bootstrap_names, "Pip closure collides with an installed Conda Python distribution; preserve the environment and stop without replacement.")
        if _normalized_name(record["name"]) not in originals:
            record["byte_size"], actual_hash = _download(record["origin_url"], cache / record["cache_path"], record["upstream_digest"])
            require(actual_hash == record["sha256"], "Transitive wheel archive hash differs.")
        pip_records.append(record)
    pip_records.sort(key=lambda item: _normalized_name(item["name"]))
    lock = seal({"schema_version": LOCK_SCHEMA, "intent_sha256": digest(spec), "conda": conda_records, "pip": pip_records})
    validate_acquisition_lock(lock, spec)
    publish(cache / "acquisition-lock.json", lock)
    requirements = []
    for record in pip_records:
        requirements.append(record["name"] + " @ " + (cache / record["cache_path"]).as_uri() + " --hash=sha256:" + record["sha256"])
    _exclusive_bytes(cache / "resolved-requirements.txt", ("\n".join(requirements) + "\n").encode())
    _run([python, "-B", "-m", "pip", "--isolated", "--disable-pip-version-check", "install", "--no-index", "--no-cache-dir", "--no-deps", "--only-binary=:all:", "--require-hashes", "--requirement", str(cache / "resolved-requirements.txt")], env=environment)
    _run([python, "-B", "-m", "pip", "--isolated", "check"], env=environment)
    # The calling shell activates this new prefix and invokes inventory next.


def _main() -> None:
    parser = argparse.ArgumentParser(description="Future authorized binary acquisition only; never verification.", allow_abbrev=False)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--exports-root", type=Path, required=True)
    parser.add_argument("--expected-software-commit", required=True)
    arguments = parser.parse_args()
    acquire_environment(arguments.spec, arguments.prefix, arguments.cache_root, arguments.exports_root, arguments.expected_software_commit)


if __name__ == "__main__":
    _main()
