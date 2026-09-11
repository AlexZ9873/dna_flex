"""B4a contracts and synthetic kernels; no test produces accepted CARC evidence."""

import ast
import base64
import builtins
import copy
import errno
import hashlib
import inspect
import importlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import textwrap
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from scripts.carc import cnn_rc_environment as environment
from scripts.carc import verify_cnn_rc_environment as verifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTENT_PATH = PROJECT_ROOT / "environments/carc_cnn_rc_v1.json"
REQUIREMENTS_PATH = PROJECT_ROOT / "environments/carc_cnn_rc_v1_requirements.txt"
VERIFIER_MODULE = "scripts.carc.verify_cnn_rc_environment"
FIXTURE_ID = "cnn_rc_environment_fixed_tensor.v1"
FORBIDDEN_MODULES = (
    "src.downstream_run", "src.downstream_checkpoint", "src.cnn_rc_training",
    "src.exd_hox_dataset", "src.exd_hox_splits", "src.sealed_test_access",
)


def forbidden_module(name):
    """Include short and relative aliases without rejecting inert file paths."""
    return type(name) is str and any(
        name.lstrip(".") == module or name.lstrip(".") == module.split(".")[-1]
        or name.lstrip(".").startswith(module + ".") for module in FORBIDDEN_MODULES
    )


def guard_b4a_imports(test):
    """Scope guards to B4a tests; accepted regression tests import independently."""
    original_import = builtins.__import__
    original_dynamic = importlib.import_module

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        requested = [name]
        requested.extend(name + "." + item for item in (fromlist or ()) if type(item) is str)
        if any(forbidden_module(item) for item in requested):
            raise AssertionError("B4a requested a forbidden biological module: " + repr(requested))
        return original_import(name, globals, locals, fromlist, level)

    def guarded_dynamic(name, package=None):
        resolved = importlib.util.resolve_name(name, package) if name.startswith(".") else name
        if forbidden_module(resolved):
            raise AssertionError("B4a requested a forbidden dynamic import: " + resolved)
        return original_dynamic(name, package)

    for guard in (patch.object(builtins, "__import__", side_effect=guarded_import),
                  patch.object(importlib, "import_module", side_effect=guarded_dynamic)):
        guard.start()
        test.addCleanup(guard.stop)


def forbidden_imports(source):
    """Inspect imports and obvious Python subprocess/dynamic import forms."""
    tree = ast.parse(source)
    aliases = {}
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                aliases[item.asname or item.name] = item.name
                if forbidden_module(item.name):
                    violations.append(item.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for item in node.names:
                full_name = module + "." + item.name if module else item.name
                aliases[item.asname or item.name] = full_name
                if forbidden_module(module) or forbidden_module(full_name):
                    violations.append(full_name)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            function = ast.unparse(node.func)
            first = function.split(".")[0]
            function = aliases.get(first, first) + function[len(first):]
            if function in ("__import__", "builtins.__import__", "importlib.import_module"):
                for argument in node.args:
                    if isinstance(argument, ast.Constant) and forbidden_module(argument.value):
                        violations.append(argument.value)
                fromlists = [keyword.value for keyword in node.keywords if keyword.arg == "fromlist"]
                if function.endswith("__import__") and len(node.args) > 3:
                    fromlists.append(node.args[3])
                for fromlist in fromlists:
                    if isinstance(fromlist, (ast.List, ast.Tuple)):
                        for name in fromlist.elts:
                            if isinstance(name, ast.Constant) and forbidden_module(name.value):
                                violations.append(name.value)
            if function.startswith("subprocess."):
                for argument in node.args:
                    if isinstance(argument, (ast.List, ast.Tuple)):
                        for index, component in enumerate(argument.elts[:-1]):
                            following = argument.elts[index + 1]
                            if isinstance(component, ast.Constant) and component.value == "-c" \
                                    and isinstance(following, ast.Constant) and type(following.value) is str:
                                violations.extend(forbidden_imports(following.value))
    return violations


def intended_resolver_pins(raw: bytes) -> list[str]:
    """Test-only format contract; this does not resolve or install packages."""
    active = []
    for line in raw.decode("utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            active.append(line)
    if active != ["numpy==2.2.6", "PyYAML==6.0.3", "torch==2.14.0+cu126"]:
        raise ValueError("Intended resolver input requires exactly three ordered bare pins.")
    return active


def locations(value, prefix=()):
    """Enumerate nested containers and leaves for recursive schema failures."""
    result = [(prefix, value)]
    if type(value) is dict:
        for key, child in value.items():
            result.extend(locations(child, prefix + (key,)))
    elif type(value) is list:
        for index, child in enumerate(value):
            result.extend(locations(child, prefix + (index,)))
    return result


def at_location(value, location):
    for component in location:
        value = value[component]
    return value


def synthetic_lock():
    """Schema fixture only: no archives, installed prefix, or acceptance evidence."""
    spec = environment.default_intent()
    records = {"conda": [], "pip": []}
    for name, version in (("pip", "25.2"), ("python", "3.11.14")):
        filename = name + "-" + version + "-synthetic_0.conda"
        records["conda"].append({
            "manager": "conda", "name": name, "version": version, "build": "synthetic_0",
            "subdir": "linux-64", "filename": filename,
            "origin_url": "https://conda.anaconda.org/conda-forge/linux-64/" + filename,
            "sha256": "a" * 64, "upstream_digest": {"algorithm": "sha256", "value": "a" * 64},
            "byte_size": 17, "platform_tags": ["linux-64"], "cache_path": "conda/" + filename,
        })
    for direct in spec["direct_dependencies"]:
        filename = direct["name"] + "-" + direct["version"] + "-cp311-cp311-manylinux_2_28_x86_64.whl"
        origin = "https://download.pytorch.org/whl/cu126/" if direct["name"] == "torch" else "https://files.pythonhosted.org/packages/"
        records["pip"].append({
            "manager": "pip", "name": direct["name"], "version": direct["version"], "build": "wheel",
            "subdir": "manylinux_2_28_x86_64", "filename": filename, "origin_url": origin + filename,
            "sha256": direct["advertised_sha256"],
            "upstream_digest": {"algorithm": "sha256", "value": direct["advertised_sha256"]},
            "byte_size": 19, "platform_tags": ["cp311-cp311-manylinux_2_28_x86_64"],
            "cache_path": "pip/" + filename,
        })
    return environment.seal({"schema_version": environment.LOCK_SCHEMA,
                             "intent_sha256": environment.digest(spec), **records})


def reseal(value):
    value = copy.deepcopy(value)
    value.pop("manifest_hash", None)
    return environment.seal(value)


def synthetic_inventory():
    """Unverified schema facts never written as accepted runtime evidence."""
    spec = environment.default_intent()
    lock = synthetic_lock()
    packages = []
    for manager in ("conda", "pip"):
        for archived in lock[manager]:
            package = {key: archived[key] for key in environment.PACKAGE_FIELDS[:-1]}
            package["installed_metadata_sha256"] = "b" * 64
            packages.append(package)
    semantic = environment.semantic_environment(spec, packages, {"system": "Linux", "machine": "x86_64", "glibc": "2.28"})
    execution = {
        "prefix": "/synthetic/uninstalled/prefix", "python_executable": "/synthetic/uninstalled/prefix/bin/python",
        "python_version": "3.11.14", "host": "synthetic-compute", "utc": "2026-09-10T00:00:00+00:00",
        "slurm": {"job_id": "123", "cluster": "synthetic", "partition": "synthetic-cpu",
                  "node": "synthetic-compute", "job_record_sha256": "c" * 64},
        "loaded_modules": ["conda/25.11.0"],
    }
    exports = [{"logical_path": name, "byte_size": 1, "sha256": "d" * 64}
               for name in sorted(environment.EXPORT_COMMANDS)]
    return environment.seal({
        "schema_version": environment.INVENTORY_SCHEMA, "intent_sha256": environment.digest(spec),
        "environment_id": "env_" + environment.digest(semantic), "semantic_environment": semantic,
        "execution": execution, "packages": packages, "exports": exports,
        "acquisition_lock": {"logical_path": "acquisition-lock.json", "byte_size": 1, "sha256": "e" * 64},
    })


def synthetic_verification_payloads(inventory):
    """Inert success-status schemas; no observed installation or CARC acceptance."""
    fixture = {
        "identifier": verifier.FIXTURE_ID, "model_seed": verifier.FIXTURE_SEED,
        "input": {"dtype": "torch.float32", "shape": [128, 14, 4], "sha256": verifier.INPUT_SHA256},
        "targets": {"dtype": "torch.float32", "shape": [128, 1], "sha256": verifier.TARGET_SHA256},
        "initial_state_sha256": environment.digest("synthetic-initial-state"),
    }
    software = {"runtime_commit": "2" * 40, "source_inventory": []}
    for path in sorted(environment.SOURCE_PATHS):
        software["source_inventory"].append({
            "path": path, "git_blob": "1" * 40, "byte_size": 1,
            "sha256": environment.digest(["synthetic-source", path]),
        })
    driver = environment.canonical_bytes({
        "cuda_driver_api": 12060, "nvidia_driver_release": "synthetic-unobserved-driver",
    }).decode("utf-8")

    def schema_comparisons(names, exact):
        comparisons = []
        shapes = verifier._comparison_shapes()
        for name in names:
            tensor = {"dtype": "torch.float32", "shape": copy.deepcopy(shapes[name]),
                      "sha256": environment.digest(["synthetic-tensor", name])}
            if name.endswith(".input"):
                tensor = copy.deepcopy(fixture["input"])
            atol, rtol = verifier._tolerance(name, exact)
            comparisons.append({
                "name": name, "reference": copy.deepcopy(tensor), "observed": copy.deepcopy(tensor),
                "atol": atol, "rtol": rtol, "max_absolute_error": 0.0,
                "max_relative_error": 0.0, "mismatched_elements": 0, "passed": True,
            })
        return comparisons

    payloads = {"inventory": inventory}
    for mode in ("cpu", "p100"):
        is_gpu = mode == "p100"
        compatibility = {
            "schema_version": "downstream_environment_compatibility.v1",
            "python": "3.11.14", "numpy": "2.2.6", "torch": "2.14.0+cu126",
            "torch_build": "synthetic-unobserved-build", "numpy_build": "synthetic-unobserved-build",
            "os": "Linux", "machine": "x86_64", "processor": "synthetic", "byteorder": "little",
            "backend": "cuda" if is_gpu else "cpu", "threads": 1, "interop_threads": 1,
            "cuda": "12.6", "cudnn": 90000, "driver": driver if is_gpu else None,
            "device_class": "Tesla P100-PCIE-16GB" if is_gpu else "x86_64",
            "device_count": 1 if is_gpu else 0, "deterministic_algorithms": True,
            "deterministic_warn_only": False, "cudnn_benchmark": False, "cudnn_deterministic": True,
            "cuda_matmul_tf32": False, "cudnn_tf32": False,
            "float32_matmul_precision": "highest", "cublas_workspace_config": ":4096:8",
        }
        gpu = None
        if is_gpu:
            gpu = {
                "name": "Tesla P100-PCIE-16GB", "capability": [6, 0], "device_count": 1, "index": 0,
                "driver": driver, "cudnn_version": 90000, "torch_version": "2.14.0+cu126",
                "cuda_runtime": "12.6", "architecture_list": ["sm_60"],
                "uuid": "synthetic-unobserved-device", "total_memory_bytes": 17179869184,
            }
        groups = {"repeat_cpu": [], "rc_cpu": [], "repeat_p100": [], "rc_p100": [], "cpu_p100": []}
        for group in groups:
            if is_gpu or group in ("repeat_cpu", "rc_cpu"):
                names = (["eval.output", "eval_after.output", "train.output"]
                         if group.startswith("rc_") else verifier._comparison_names())
                groups[group] = schema_comparisons(names, group.startswith("repeat_"))
        states = {"cpu_first": "3" * 64, "cpu_repeat": "3" * 64,
                  "p100_first": "4" * 64 if is_gpu else None,
                  "p100_repeat": "4" * 64 if is_gpu else None}
        check_names = verifier.BASE_CHECKS + (("cpu_gpu_tolerances", "device_contract")
                                              if is_gpu else ("no_cuda_initialization",))
        verification = {
            "schema_version": verifier.VERIFICATION_SCHEMA, "mode": mode, "status": "successful",
            "environment_id": inventory["environment_id"],
            "inventory_sha256": verifier._manifest_file_hash(inventory),
            "software": copy.deepcopy(software), "fixture": copy.deepcopy(fixture),
            "observations": {
                "b3_environment": compatibility, "numpy_config_sha256": environment.digest("synthetic-config"),
                "gpu": gpu, "comparisons": groups, "state_sha256": states,
                "memory": {"peak_allocated_bytes": 1 if is_gpu else 0,
                           "peak_reserved_bytes": 1 if is_gpu else 0},
            },
            "checks": dict.fromkeys(check_names, True), "failure": None,
            "execution": copy.deepcopy(inventory["execution"]),
        }
        verification["verification_id"] = verifier._verification_id(verification)
        payloads[mode + "_verification"] = environment.seal(verification)
    return payloads


class TemporaryCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()


class IntendedContractTests(TemporaryCase):
    def test_exact_approved_top_level_and_candidate_pins(self):
        intent = environment.load_intent(INTENT_PATH)
        parsed = environment.strict_json(INTENT_PATH.read_bytes())
        self.assertIs(type(parsed["verification"]["repetitions"]), int)
        self.assertEqual(parsed["verification"]["repetitions"], 2)
        before = environment.canonical_bytes(parsed)
        environment.validate_intent(parsed)
        self.assertEqual(environment.canonical_bytes(parsed), before)
        self.assertEqual(intent, parsed)
        self.assertEqual(intent, environment.default_intent())
        self.assertEqual(set(intent), {
            "schema_version", "environment_policy_identifier", "platform", "bootstrap",
            "direct_dependencies", "sources", "installation", "numerics", "gpu", "verification",
        })
        self.assertEqual(intent["schema_version"], "carc_cnn_rc_environment_intent.v1")
        self.assertEqual(intent["environment_policy_identifier"], "carc_cnn_rc_cu126_p100.v1")
        serialized = json.dumps(intent)
        for required in (
            "3.11.14", "25.2", "2.2.6", "6.0.3", "2.14.0+cu126", "conda/25.11.0",
            "ba10f8411898fc418a521833e014a77d3ca01c15b0c6cdcce6a0d2897e6dbbdf",
            "b8bb0864c5a28024fac8a632c443c87c5aa6f215c0b126c449ae1a150412f31d",
            "898b03fc60e642f1b28f59da9f647022a8ad339c1799693b1385fdcb8889f6ba",
        ):
            self.assertIn(required, serialized)

        self.assertEqual(intent["direct_dependencies"], [
            {"name": "numpy", "version": "2.2.6", "source": "pypi",
             "advertised_sha256": "ba10f8411898fc418a521833e014a77d3ca01c15b0c6cdcce6a0d2897e6dbbdf"},
            {"name": "PyYAML", "version": "6.0.3", "source": "pypi",
             "advertised_sha256": "b8bb0864c5a28024fac8a632c443c87c5aa6f215c0b126c449ae1a150412f31d"},
            {"name": "torch", "version": "2.14.0+cu126", "source": "torch_cu126",
             "advertised_sha256": "898b03fc60e642f1b28f59da9f647022a8ad339c1799693b1385fdcb8889f6ba"},
        ])
        intended_bytes = INTENT_PATH.read_bytes()
        requirements_bytes = REQUIREMENTS_PATH.read_bytes()
        pins = intended_resolver_pins(requirements_bytes)
        self.assertEqual(pins, ["numpy==2.2.6", "PyYAML==6.0.3", "torch==2.14.0+cu126"])
        self.assertEqual(len(pins), 3)
        self.assertEqual(pins, [item["name"] + "==" + item["version"] for item in intent["direct_dependencies"]])
        comments = [line for line in requirements_bytes.decode("utf-8").splitlines() if line.startswith("#")]
        explanation = "\n".join(comments)
        self.assertIn("Intended direct-dependency resolver input", explanation)
        self.assertIn("not a complete transitive lock", explanation)
        self.assertIn("environments/carc_cnn_rc_v1.json", explanation)
        for prefix in (b"\n# hash identity stays in JSON\n\n", b"# --hash=sha256:comment-only\n"):
            with self.subTest(comment=prefix):
                self.assertEqual(intended_resolver_pins(prefix + requirements_bytes), pins)

        malformed = [
            ("extra", pins + ["extra==1.0"]),
            ("missing", pins[:-1]),
            ("duplicate", pins + [pins[0]]),
            ("reordered", [pins[1], pins[0], pins[2]]),
        ]
        for index, dependency in enumerate(intent["direct_dependencies"]):
            changed = list(pins)
            changed[index] += " --hash=sha256:" + dependency["advertised_sha256"]
            malformed.append(("active hash for " + dependency["name"], changed))
        for option in (
            "--require-hashes", "--index-url https://example.invalid/simple",
            "--extra-index-url https://example.invalid/simple", "--find-links /synthetic/wheels",
            "--trusted-host example.invalid", "-r included.txt", "--requirement included.txt",
            "-c constraints.txt", "--constraint constraints.txt", "-e ./synthetic-package",
            "--editable ./synthetic-package", "--hash=sha256:" + "a" * 64,
        ):
            malformed.append((option, pins + [option]))
        for replacement in (
            "numpy", "numpy>=2.2.6", "numpy~=2.2.6", "numpy==2.2.*",
            "numpy @ https://example.invalid/numpy.whl", "https://example.invalid/numpy.whl",
            "numpy @ file:///synthetic/numpy.whl", "./synthetic/numpy.whl", "/synthetic/numpy.whl",
            "numpy==2.2.6; python_version >= '3.11'", "numpy==2.2.6 \\",
            "numpy==2.2.6; echo synthetic", "numpy==2.2.6 # inline comment", " numpy==2.2.6",
        ):
            malformed.append((replacement, [replacement, *pins[1:]]))
        for label, lines in malformed:
            with self.subTest(requirements=label):
                raw = (explanation + "\n" + "\n".join(lines) + "\n").encode("utf-8")
                with self.assertRaisesRegex(ValueError, "three ordered bare pins"):
                    intended_resolver_pins(raw)
                self.assertEqual(environment.canonical_bytes(intent), before)
        self.assertEqual(INTENT_PATH.read_bytes(), intended_bytes)
        self.assertEqual(REQUIREMENTS_PATH.read_bytes(), requirements_bytes)

    def test_every_nested_dictionary_rejects_unknown_and_missing_keys(self):
        intent = environment.default_intent()
        for location, node in locations(intent):
            if type(node) is dict:
                for mutation in ("unknown", "missing"):
                    with self.subTest(location=location, mutation=mutation):
                        changed = copy.deepcopy(intent)
                        target = at_location(changed, location)
                        if mutation == "unknown":
                            target["unapproved"] = True
                        else:
                            target.pop(next(iter(target)))
                        with self.assertRaises(ValueError):
                            environment.validate_intent(changed)

        missing = copy.deepcopy(intent)
        del missing["verification"]["repetitions"]
        before = environment.canonical_bytes(missing)
        with self.assertRaises(ValueError):
            environment.validate_intent(missing)
        self.assertEqual(environment.canonical_bytes(missing), before)
        path = self.root / "missing-repetitions.json"
        path.write_bytes(before)
        with self.assertRaises(ValueError):
            environment.load_intent(path)
        self.assertEqual(path.read_bytes(), before)
        self.assertNotIn("repetitions", missing["verification"])

    def test_every_leaf_rejects_type_changes_and_boolean_integer_confusion(self):
        intent = environment.default_intent()
        for location, node in locations(intent):
            if location and type(node) not in (dict, list):
                replacement = "wrong-type"
                if type(node) is str:
                    replacement = 1
                elif type(node) is bool:
                    replacement = int(node)
                elif type(node) is int:
                    replacement = bool(node)
                with self.subTest(location=location):
                    changed = copy.deepcopy(intent)
                    at_location(changed, location[:-1])[location[-1]] = replacement
                    with self.assertRaises(ValueError):
                        environment.validate_intent(changed)

        for replacement in (True, False, 2.0, "2", None, [], {}):
            with self.subTest(repetitions=replacement, kind=type(replacement).__name__):
                changed = copy.deepcopy(intent)
                changed["verification"]["repetitions"] = replacement
                before = environment.canonical_bytes(changed)
                with self.assertRaisesRegex(ValueError, "Contract field type differs"):
                    environment.validate_intent(changed)
                self.assertEqual(environment.canonical_bytes(changed), before)
                path = self.root / "wrong-repetitions-type.json"
                path.write_bytes(before)
                with self.assertRaisesRegex(ValueError, "Contract field type differs"):
                    environment.load_intent(path)
                self.assertEqual(path.read_bytes(), before)

    def test_pinned_leaf_values_cannot_be_silently_changed(self):
        intent = environment.default_intent()
        for location, node in locations(intent):
            if location and type(node) in (str, int, float, bool):
                if type(node) is str:
                    replacement = node + "unapproved"
                elif type(node) is bool:
                    replacement = not node
                else:
                    replacement = node + 1
                with self.subTest(location=location):
                    changed = copy.deepcopy(intent)
                    at_location(changed, location[:-1])[location[-1]] = replacement
                    with self.assertRaises(ValueError):
                        environment.validate_intent(changed)

        for replacement in (0, 1, 3, -1, -2):
            with self.subTest(repetitions=replacement):
                changed = copy.deepcopy(intent)
                changed["verification"]["repetitions"] = replacement
                before = environment.canonical_bytes(changed)
                path = self.root / "wrong-repetitions-value.json"
                path.write_bytes(before)
                with self.assertRaisesRegex(ValueError, "Unsupported v1 environment intent"):
                    environment.validate_intent(changed)
                with self.assertRaisesRegex(ValueError, "Unsupported v1 environment intent"):
                    environment.load_intent(path)
                self.assertEqual(environment.canonical_bytes(changed), before)
                self.assertEqual(path.read_bytes(), before)

    def test_nonfinite_duplicate_and_nonobject_json_are_rejected(self):
        invalid = (
            b'{"x":1,"x":2}', b'{"outer":{"x":1,"x":2}}', b'{"x":NaN}',
            b'{"x":Infinity}', b'{"x":-Infinity}', b'{"x":1e999}', b'[]', b'null', b'42',
        )
        for raw in invalid:
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    environment.strict_json(raw)
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    environment.canonical_bytes({"nested": [value]})

    def test_canonical_identity_is_order_independent_and_type_sensitive(self):
        self.assertEqual(environment.digest({"b": [1, 2], "a": 3}),
                         environment.digest({"a": 3, "b": [1, 2]}))
        self.assertNotEqual(environment.digest({"a": 1}), environment.digest({"a": True}))
        self.assertNotEqual(environment.digest({"a": 1}), environment.digest({"a": 1.0}))
        intent = environment.load_intent(INTENT_PATH)
        inventory = synthetic_inventory()
        self.assertEqual(inventory["intent_sha256"], environment.digest(intent))
        self.assertEqual(synthetic_lock()["intent_sha256"], environment.digest(intent))
        self.assertNotIn("repetitions", inventory["execution"])
        for replacement in (None, 1, 3):
            with self.subTest(repetitions=replacement):
                changed = copy.deepcopy(intent)
                if replacement is None:
                    del changed["verification"]["repetitions"]
                else:
                    changed["verification"]["repetitions"] = replacement
                self.assertNotEqual(environment.canonical_bytes(changed), environment.canonical_bytes(intent))
                self.assertNotEqual(environment.digest(changed), environment.digest(intent))
                changed_inventory = copy.deepcopy(inventory)
                changed_inventory["intent_sha256"] = environment.digest(changed)
                with self.assertRaises(ValueError):
                    environment.validate_seal(changed_inventory)
                changed_inventory = reseal(changed_inventory)
                environment.validate_seal(changed_inventory)
                with self.assertRaisesRegex(ValueError, "Inventory intent differs"):
                    environment.validate_inventory(changed_inventory)
                # Failed, in-memory identity inputs only; no accepted evidence is minted.
                original_record = verifier._empty_verification("cpu")
                original_record["inventory_sha256"] = environment.digest(inventory)
                changed_record = copy.deepcopy(original_record)
                changed_record["inventory_sha256"] = environment.digest(changed_inventory)
                self.assertNotEqual(verifier._verification_id(original_record),
                                    verifier._verification_id(changed_record))
                self.assertEqual(changed_inventory["execution"], inventory["execution"])

    def test_reference_has_exact_logical_fields_and_rejects_physical_paths(self):
        source = self.root / "archive.whl"
        source.write_bytes(b"synthetic archive bytes")
        reference = environment.file_reference(source, "wheels/archive.whl")
        self.assertEqual(reference, {
            "logical_path": "wheels/archive.whl", "byte_size": 23,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        })
        environment.validate_reference(reference)
        for logical in (str(source), "../archive.whl", "wheels/../archive.whl", "", "a//b"):
            with self.subTest(logical=logical):
                changed = dict(reference, logical_path=logical)
                with self.assertRaises(ValueError):
                    environment.validate_reference(changed)
        for changed in (dict(reference, byte_size=True), dict(reference, byte_size=-1),
                        dict(reference, sha256="f" * 63), dict(reference, extra=1)):
            with self.assertRaises(ValueError):
                environment.validate_reference(changed)

    def test_seal_detects_content_tampering(self):
        sealed = environment.seal({"schema_version": "synthetic_failure_test.v1", "status": "failed"})
        environment.validate_seal(sealed)
        changed = dict(sealed, status="changed")
        with self.assertRaises(ValueError):
            environment.validate_seal(changed)
        sealed_intent = environment.seal({"intent": environment.load_intent(INTENT_PATH)})
        sealed_intent["intent"]["verification"]["repetitions"] = 3
        with self.assertRaises(ValueError):
            environment.validate_seal(sealed_intent)
        consistent = environment.strict_json(environment.canonical_bytes(reseal(sealed_intent)))
        environment.validate_seal(consistent)
        with self.assertRaisesRegex(ValueError, "Unsupported v1 environment intent"):
            environment.validate_intent(consistent["intent"])


class PublicationTests(TemporaryCase):
    def assert_preserved_partial(self, partial, contents, mode):
        """Partial names are forensic files, never accepted manifest paths."""
        metadata = partial.lstat()
        self.assertTrue(stat.S_ISREG(metadata.st_mode))
        self.assertFalse(partial.is_symlink())
        self.assertEqual(stat.S_IMODE(metadata.st_mode), mode)
        self.assertEqual(metadata.st_mode & (stat.S_IRWXG | stat.S_IRWXO), 0)
        self.assertEqual(metadata.st_nlink, 1)
        self.assertEqual(metadata.st_size, len(contents))
        self.assertEqual(environment.read_regular(partial), contents)
        with patch.object(environment, "read_regular", side_effect=AssertionError("Partial manifest was opened")):
            with self.assertRaises(ValueError):
                environment.read_manifest(partial)

    def assert_successful_private_publication(self, ambient_umask):
        """Wrap real syscalls; raw os.write has no Python buffer to flush."""
        output = self.root / ("ordered-" + str(ambient_umask) + ".json")
        record = environment.seal({"status": "synthetic-ordering", "umask": ambient_umask})
        contents = environment.canonical_bytes(record) + b"\n"
        events = []
        state = {"descriptor": None, "written": 0, "closed": False, "file_synced": False, "reopened": 0}
        real_open, real_write, real_fchmod = os.open, os.write, os.fchmod
        real_fsync, real_close, real_link = os.fsync, os.close, os.link

        def opened(path, flags, mode=0o777, *, dir_fd=None):
            descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
            if flags & os.O_CREAT:
                state["descriptor"] = descriptor
                self.assertEqual(mode, 0o600)
                for required in (os.O_WRONLY, os.O_CREAT, os.O_EXCL,
                                 getattr(os, "O_CLOEXEC", 0), getattr(os, "O_NOFOLLOW", 0)):
                    self.assertEqual(flags & required, required)
                metadata = os.fstat(descriptor)
                self.assertTrue(stat.S_ISREG(metadata.st_mode))
                self.assertEqual(metadata.st_size, 0)
                self.assertEqual(metadata.st_nlink, 1)
                self.assertEqual(metadata.st_mode & (stat.S_IRWXG | stat.S_IRWXO), 0)
                self.assertEqual(stat.S_IMODE(metadata.st_mode), 0o600 & ~ambient_umask)
                events.append("create-0600")
            elif type(path) is str and path.startswith("." + output.name + ".partial-"):
                self.assertTrue(state["closed"])
                self.assertEqual(flags & os.O_ACCMODE, os.O_RDONLY)
                self.assertEqual(flags & getattr(os, "O_NOFOLLOW", 0), getattr(os, "O_NOFOLLOW", 0))
                self.assertEqual(stat.S_IMODE(os.fstat(descriptor).st_mode), 0o400)
                state["reopened"] += 1
            return descriptor

        def written(descriptor, data):
            self.assertEqual(descriptor, state["descriptor"])
            self.assertEqual(stat.S_IMODE(os.fstat(descriptor).st_mode), 0o600)
            self.assertIn("fchmod-0600", events)
            # Real short positive writes must be completed by the writer loop.
            count = real_write(descriptor, data[:7])
            state["written"] += count
            if state["written"] == len(contents):
                events.append("complete-unbuffered-write")
            return count

        def changed_mode(descriptor, mode):
            self.assertEqual(descriptor, state["descriptor"])
            if mode == 0o600:
                self.assertEqual(state["written"], 0)
            else:
                self.assertEqual(mode, 0o400)
                self.assertEqual(state["written"], len(contents))
                self.assertEqual(os.fstat(descriptor).st_size, len(contents))
                self.assertEqual(events[-1], "complete-unbuffered-write")
            real_fchmod(descriptor, mode)
            self.assertEqual(stat.S_IMODE(os.fstat(descriptor).st_mode), mode)
            events.append("fchmod-" + format(mode, "04o"))

        def synced(descriptor):
            if stat.S_ISREG(os.fstat(descriptor).st_mode):
                self.assertEqual(descriptor, state["descriptor"])
                self.assertEqual(stat.S_IMODE(os.fstat(descriptor).st_mode), 0o400)
                self.assertEqual(events[-1], "fchmod-0400")
                real_fsync(descriptor)
                state["file_synced"] = True
                events.append("file-fsync")
            else:
                self.assertEqual(events[-1], "exclusive-link")
                real_fsync(descriptor)
                events.append("parent-fsync")

        def closed(descriptor):
            if descriptor == state["descriptor"] and not state["closed"]:
                self.assertTrue(state["file_synced"])
                self.assertEqual(events[-1], "file-fsync")
                real_close(descriptor)
                state["closed"] = True
                events.append("writer-close")
            else:
                real_close(descriptor)

        def linked(source, destination, *, src_dir_fd=None, dst_dir_fd=None, follow_symlinks=True):
            self.assertTrue(state["closed"])
            self.assertEqual(state["reopened"], 1)
            self.assertEqual(events[-1], "writer-close")
            self.assertIsNotNone(src_dir_fd)
            self.assertEqual(src_dir_fd, dst_dir_fd)
            self.assertFalse(follow_symlinks)
            self.assertEqual(destination, output.name)
            self.assertTrue(source.startswith("." + output.name + ".partial-"))
            self.assertFalse(os.path.lexists(output))
            self.assertEqual(stat.S_IMODE(os.stat(source, dir_fd=src_dir_fd, follow_symlinks=False).st_mode), 0o400)
            real_link(source, destination, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd,
                      follow_symlinks=follow_symlinks)
            events.append("exclusive-link")

        previous_umask = os.umask(ambient_umask)
        try:
            with patch.object(os, "open", side_effect=opened), \
                 patch.object(os, "write", side_effect=written), \
                 patch.object(os, "fchmod", side_effect=changed_mode), \
                 patch.object(os, "fsync", side_effect=synced), \
                 patch.object(os, "close", side_effect=closed), \
                 patch.object(os, "link", side_effect=linked):
                environment.publish(output, record)
        finally:
            os.umask(previous_umask)
        self.assertEqual(events, ["create-0600", "fchmod-0600", "complete-unbuffered-write",
                                  "fchmod-0400", "file-fsync", "writer-close",
                                  "exclusive-link", "parent-fsync"])
        metadata = output.lstat()
        self.assertTrue(stat.S_ISREG(metadata.st_mode))
        self.assertFalse(output.is_symlink())
        self.assertEqual(stat.S_IMODE(metadata.st_mode), 0o400)
        self.assertEqual(metadata.st_size, len(contents))
        self.assertEqual(metadata.st_nlink, 1)
        self.assertEqual(output.read_bytes(), contents)
        self.assertEqual(hashlib.sha256(output.read_bytes()).hexdigest(), hashlib.sha256(contents).hexdigest())
        self.assertEqual(environment.read_manifest(output), record)
        self.assertEqual(list(self.root.glob("." + output.name + ".partial-*")), [])
        with patch.object(os, "fchmod", side_effect=AssertionError("Existing final was chmodded")):
            with self.assertRaises((FileExistsError, ValueError)):
                environment.publish(output, environment.seal({"status": "replacement"}))
        self.assertEqual(output.read_bytes(), contents)
        self.assertEqual(stat.S_IMODE(output.lstat().st_mode), 0o400)

    def test_exclusive_publication_preserves_prior_failure_bytes(self):
        output = self.root / "failed.json"
        record = environment.seal({"schema_version": "synthetic_failure_test.v1", "status": "failed"})
        environment.publish(output, record)
        original = output.read_bytes()
        with self.assertRaises((FileExistsError, ValueError)):
            environment.publish(output, dict(record, status="replacement"))
        self.assertEqual(output.read_bytes(), original)
        for ambient_umask in (0, 0o777):
            with self.subTest(ambient_umask=ambient_umask):
                self.assert_successful_private_publication(ambient_umask)

        concurrent = self.root / "concurrent.json"
        records = [environment.seal({"status": "synthetic-publisher", "publisher": index}) for index in range(2)]
        barrier = threading.Barrier(2, timeout=10)
        outcomes = [None, None]
        real_link = os.link

        def synchronized_link(source, destination, **kwargs):
            barrier.wait()
            return real_link(source, destination, **kwargs)

        def publisher(index):
            try:
                environment.publish(concurrent, records[index])
                outcomes[index] = "published"
            except BaseException as error:
                outcomes[index] = error

        with patch.object(os, "link", side_effect=synchronized_link):
            workers = [threading.Thread(target=publisher, args=(index,)) for index in range(2)]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(timeout=15)
            self.assertTrue(all(not worker.is_alive() for worker in workers))
        self.assertEqual(outcomes.count("published"), 1)
        winner = outcomes.index("published")
        loser = 1 - winner
        self.assertIsInstance(outcomes[loser], FileExistsError)
        self.assertEqual(environment.read_manifest(concurrent), records[winner])
        self.assertEqual(stat.S_IMODE(concurrent.lstat().st_mode), 0o400)
        retained = list(self.root.glob(".concurrent.json.partial-*"))
        self.assertEqual(len(retained), 1)
        self.assert_preserved_partial(retained[0], environment.canonical_bytes(records[loser]) + b"\n", 0o400)

    def test_symlink_destination_cannot_overwrite_its_target(self):
        target = self.root / "existing.json"
        target.write_bytes(b"preserve me")
        output = self.root / "output.json"
        output.symlink_to(target)
        with self.assertRaises((FileExistsError, ValueError)):
            environment.publish(output, environment.seal({"status": "failed"}))
        self.assertEqual(target.read_bytes(), b"preserve me")
        for kind in ("regular", "directory", "live-symlink", "dangling-symlink"):
            with self.subTest(existing_destination=kind):
                destination = self.root / (kind + ".json")
                if kind == "regular":
                    destination.write_bytes(b"existing final")
                elif kind == "directory":
                    destination.mkdir()
                else:
                    destination.symlink_to(target if kind == "live-symlink" else self.root / "absent-target")
                before = destination.lstat()
                with patch.object(os, "fchmod", side_effect=AssertionError("Existing destination was chmodded")), \
                     patch.object(os, "write", side_effect=AssertionError("Existing destination was written")):
                    with self.assertRaises((FileExistsError, ValueError)):
                        environment.publish(destination, environment.seal({"status": "replacement"}))
                after = destination.lstat()
                self.assertEqual((before.st_dev, before.st_ino, before.st_mode, before.st_size),
                                 (after.st_dev, after.st_ino, after.st_mode, after.st_size))
                self.assertEqual(target.read_bytes(), b"preserve me")
                self.assertFalse((self.root / "absent-target").exists())
                self.assertEqual(list(self.root.glob("." + destination.name + ".partial-*")), [])

        for kind in ("regular", "directory", "live-symlink", "dangling-symlink"):
            with self.subTest(existing_partial=kind):
                destination = self.root / ("partial-collision-" + kind + ".json")
                partial = self.root / ("." + destination.name + ".partial-" + "a" * 32)
                if kind == "regular":
                    partial.write_bytes(b"existing forensic partial")
                    partial.chmod(0o600)
                elif kind == "directory":
                    partial.mkdir()
                else:
                    partial.symlink_to(target if kind == "live-symlink" else self.root / "absent-target")
                before = partial.lstat()
                with patch.object(environment.secrets, "token_hex", return_value="a" * 32), \
                     patch.object(os, "fchmod", side_effect=AssertionError("Existing partial was chmodded")), \
                     patch.object(os, "write", side_effect=AssertionError("Existing partial was written")):
                    with self.assertRaises((FileExistsError, ValueError)):
                        environment.publish(destination, environment.seal({"status": "replacement"}))
                after = partial.lstat()
                self.assertEqual((before.st_dev, before.st_ino, before.st_mode, before.st_size),
                                 (after.st_dev, after.st_ino, after.st_mode, after.st_size))
                self.assertFalse(os.path.lexists(destination))
                self.assertEqual(target.read_bytes(), b"preserve me")
                if kind == "regular":
                    self.assertEqual(partial.read_bytes(), b"existing forensic partial")

    def test_partial_and_durability_unconfirmed_evidence_is_preserved(self):
        record = environment.seal({"schema_version": "synthetic_failure_test.v1", "status": "failed"})
        contents = environment.canonical_bytes(record) + b"\n"
        real_write, real_fchmod = os.write, os.fchmod
        for failure in ("initial-chmod", "before-write", "during-write", "zero-write", "negative-write", "oversized-write",
                        "boolean-write", "reported-complete-without-writing", "before-final-chmod"):
            with self.subTest(private_failure=failure):
                output = self.root / (failure + ".json")
                state = {"calls": 0}

                def failing_write(descriptor, data):
                    self.assertEqual(stat.S_IMODE(os.fstat(descriptor).st_mode), 0o600)
                    state["calls"] += 1
                    if failure == "before-write" or (failure == "during-write" and state["calls"] > 1):
                        raise OSError("synthetic private write failure")
                    if failure == "during-write":
                        return real_write(descriptor, data[:7])
                    if failure == "zero-write":
                        return 0
                    if failure == "negative-write":
                        return -1
                    if failure == "oversized-write":
                        return len(data) + 1
                    if failure == "boolean-write":
                        return True
                    if failure == "reported-complete-without-writing":
                        return len(data)
                    return real_write(descriptor, data)

                def failing_chmod(descriptor, mode):
                    if failure == "initial-chmod" and mode == 0o600:
                        self.assertEqual(stat.S_IMODE(os.fstat(descriptor).st_mode), 0o600)
                        self.assertEqual(os.fstat(descriptor).st_size, 0)
                        raise OSError("synthetic immediate chmod failure")
                    if failure == "before-final-chmod" and mode == 0o400:
                        self.assertEqual(os.fstat(descriptor).st_size, len(contents))
                        raise OSError("synthetic completed private chmod failure")
                    real_fchmod(descriptor, mode)

                previous_umask = os.umask(0)
                try:
                    with patch.object(os, "write", side_effect=failing_write), \
                         patch.object(os, "fchmod", side_effect=failing_chmod), \
                         patch.object(os, "link", side_effect=AssertionError("Incomplete partial was published")):
                        with self.assertRaises((OSError, ValueError)):
                            environment.publish(output, record)
                finally:
                    os.umask(previous_umask)
                self.assertFalse(os.path.lexists(output))
                retained = list(self.root.glob("." + output.name + ".partial-*"))
                self.assertEqual(len(retained), 1)
                expected = b""
                if failure == "before-final-chmod":
                    expected = contents
                elif failure == "during-write":
                    expected = contents[:7]
                self.assert_preserved_partial(retained[0], expected, 0o600)

        partial = self.root / "partial.json"
        with patch.object(os, "fsync", side_effect=OSError("synthetic file fsync failure")):
            with self.assertRaises(OSError):
                environment.publish(partial, record)
        self.assertFalse(partial.exists())
        preserved_partials = list(self.root.glob(".partial.json.partial-*"))
        self.assertEqual(len(preserved_partials), 1)
        self.assert_preserved_partial(preserved_partials[0], contents, 0o400)
        real_fsync = os.fsync
        for failure in ("link-failure", "unsupported-link", "line\nbreak"):
            with self.subTest(completed_failure=failure):
                output = self.root / (failure + ".json")
                cause = OSError(errno.ENOTSUP, "synthetic unsupported exclusive link") \
                    if failure == "unsupported-link" else OSError("synthetic pre-publication failure")
                with patch.object(os, "link", side_effect=cause), \
                     patch.object(os, "replace", side_effect=AssertionError("Publication used overwrite fallback")), \
                     patch.object(os, "rename", side_effect=AssertionError("Publication used rename fallback")):
                    with self.assertRaises(OSError):
                        environment.publish(output, record)
                self.assertFalse(os.path.lexists(output))
                retained = list(self.root.glob("." + output.name + ".partial-*"))
                self.assertEqual(len(retained), 1)
                first = retained[0]
                first_metadata = first.lstat()
                self.assert_preserved_partial(first, contents, 0o400)
                with patch.object(os, "open", side_effect=AssertionError("Reserved partial name was created")):
                    with self.assertRaises(ValueError):
                        environment.publish(first, record)
                    with self.assertRaises(ValueError):
                        environment.publish(self.root / (".unused\nname.partial-" + "f" * 32), record)
                with patch.object(environment.secrets, "token_hex", return_value=first.name.rsplit("-", 1)[1]), \
                     patch.object(os, "fchmod", side_effect=AssertionError("Failed partial was reused")):
                    with self.assertRaises(FileExistsError):
                        environment.publish(output, record)
                environment.publish(output, record)
                self.assertEqual(environment.read_manifest(output), record)
                self.assertEqual(stat.S_IMODE(output.lstat().st_mode), 0o400)
                self.assert_preserved_partial(first, contents, 0o400)
                after = first.lstat()
                self.assertEqual((first_metadata.st_dev, first_metadata.st_ino, first_metadata.st_mode,
                                  first_metadata.st_size, first_metadata.st_mtime_ns, first_metadata.st_ctime_ns),
                                 (after.st_dev, after.st_ino, after.st_mode, after.st_size, after.st_mtime_ns, after.st_ctime_ns))
                self.assertEqual(list(self.root.glob("." + output.name + ".partial-*")), [first])

        published = self.root / "published.json"
        sync_calls = []

        def directory_sync_failure(descriptor):
            metadata = os.fstat(descriptor)
            sync_calls.append("directory" if stat.S_ISDIR(metadata.st_mode) else "file")
            if stat.S_ISDIR(metadata.st_mode):
                raise OSError("synthetic directory fsync failure")
            self.assertEqual(stat.S_IMODE(metadata.st_mode), 0o400)
            real_fsync(descriptor)

        with patch.object(os, "fsync", side_effect=directory_sync_failure):
            with self.assertRaises(environment.PublishedDurabilityError):
                environment.publish(published, record)
        self.assertEqual(sync_calls, ["file", "directory"])
        self.assertEqual(published.read_bytes(), contents)
        self.assertEqual(stat.S_IMODE(published.lstat().st_mode), 0o400)
        self.assertEqual(environment.read_manifest(published), record)
        self.assertEqual(list(self.root.glob(".published.json.partial-*")), [])
        with self.assertRaises((FileExistsError, ValueError)):
            environment.publish(published, record)
        self.assertTrue(preserved_partials[0].exists())

        real_verify = environment._verify_completed_partial
        for corruption in ("mode", "content", "size", "extra-link", "inode", "symlink"):
            with self.subTest(prepublication_corruption=corruption):
                output = self.root / ("corrupt-" + corruption + ".json")
                retained = []

                def corrupted_before_verification(parent, name, completed, data):
                    path = self.root / name
                    retained.append(path)
                    self.assertEqual(stat.S_IMODE(path.lstat().st_mode), 0o400)
                    self.assertEqual(path.read_bytes(), contents)
                    if corruption == "mode":
                        path.chmod(0o600)
                    elif corruption in ("content", "size"):
                        path.chmod(0o600)
                        path.write_bytes(b"x" + contents[1:] if corruption == "content" else contents[:-1])
                        path.chmod(0o400)
                        # Bind the mutated inode to exercise size and bytes checks,
                        # independently of the earlier replacement/time checks.
                        completed = path.lstat()
                    elif corruption == "extra-link":
                        os.link(path, self.root / ("external-link-" + corruption))
                    else:
                        original = self.root / ("forensic-original-" + corruption)
                        path.rename(original)
                        if corruption == "inode":
                            path.write_bytes(contents)
                            path.chmod(0o400)
                        else:
                            path.symlink_to(original)
                    return real_verify(parent, name, completed, data)

                with patch.object(environment, "_verify_completed_partial", side_effect=corrupted_before_verification), \
                     patch.object(os, "replace", side_effect=AssertionError("Invalid partial was repaired")):
                    with self.assertRaises((OSError, ValueError)):
                        environment.publish(output, record)
                self.assertFalse(os.path.lexists(output))
                self.assertEqual(len(retained), 1)
                self.assertTrue(os.path.lexists(retained[0]))
                with self.assertRaises(ValueError):
                    environment.read_manifest(retained[0])

    def test_symlink_parent_is_rejected_without_writing(self):
        real = self.root / "real"
        real.mkdir()
        link = self.root / "link"
        link.symlink_to(real, target_is_directory=True)
        with self.assertRaises(ValueError):
            environment.publish(link / "output.json", environment.seal({"status": "failed"}))
        self.assertEqual(list(real.iterdir()), [])

    def test_acquisition_hash_failure_preserves_download_diagnostics_without_network(self):
        response = io.BytesIO(b"synthetic binary archive fixture")
        response.url = "https://files.pythonhosted.org/packages/synthetic.whl"
        output = self.root / "synthetic.whl"
        with patch.object(environment.urllib.request, "urlopen", return_value=response):
            with self.assertRaisesRegex(ValueError, "archive hash differs"):
                environment._download(response.url, output, {"algorithm": "sha256", "value": "f" * 64})
        self.assertEqual(output.read_bytes(), b"synthetic binary archive fixture")

    def test_installed_payload_must_match_acquired_wheel_record(self):
        prefix = self.root / "synthetic-payload-directory"
        logical = "synthetic_pkg/payload.py"
        payload = b"VALUE = 1\n"
        physical = prefix / logical
        physical.parent.mkdir(parents=True)
        physical.write_bytes(payload)
        checksum = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).decode().rstrip("=")
        record = logical + ",sha256=" + checksum + "," + str(len(payload)) + "\nsynthetic_pkg.dist-info/RECORD,,\n"
        archive_path = self.root / "synthetic.whl"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr(logical, payload)
            archive.writestr("synthetic_pkg.dist-info/RECORD", record)
        distribution = SimpleNamespace(locate_file=lambda relative: prefix / relative)
        environment._verify_wheel_archive(distribution, prefix, archive_path)
        physical.write_bytes(b"VALUE = 2\n")
        # A rewritten installed RECORD cannot replace acquired archive evidence.
        changed_hash = base64.urlsafe_b64encode(hashlib.sha256(physical.read_bytes()).digest()).decode().rstrip("=")
        metadata_path = prefix / "synthetic_pkg.dist-info/RECORD"
        metadata_path.parent.mkdir()
        metadata_path.write_text(logical + ",sha256=" + changed_hash + ",10\nsynthetic_pkg.dist-info/RECORD,,\n")
        with self.assertRaisesRegex(ValueError, "differs from acquired archive"):
            environment._verify_wheel_archive(distribution, prefix, archive_path)

    def test_conda_relocated_payload_uses_installed_hash_and_preserves_failed_bytes(self):
        prefix = self.root / "synthetic-conda-payload-directory"
        payload = prefix / "lib/relocated.txt"
        payload.parent.mkdir(parents=True)
        contents = ("prefix=" + str(prefix) + "\n").encode()
        payload.write_bytes(contents)
        record = {"_path": "lib/relocated.txt", "path_type": "hardlink",
                  "prefix_placeholder": "/synthetic/build-prefix", "sha256": "a" * 64,
                  "sha256_in_prefix": hashlib.sha256(contents).hexdigest()}
        installed = {"paths_data": {"paths": [record]}}
        environment._verify_conda_payload(installed, prefix)
        missing_relocated_hash = copy.deepcopy(installed)
        missing_relocated_hash["paths_data"]["paths"][0].pop("sha256_in_prefix")
        with self.assertRaisesRegex(ValueError, "Relocated Conda payload"):
            environment._verify_conda_payload(missing_relocated_hash, prefix)
        payload.write_bytes(contents + b"changed")
        with self.assertRaisesRegex(ValueError, "payload differs"):
            environment._verify_conda_payload(installed, prefix)
        self.assertEqual(payload.read_bytes(), contents + b"changed")

    def test_conda_duplicate_paths_and_escaping_symlinks_are_rejected(self):
        prefix = self.root / "synthetic-conda-payload-directory"
        prefix.mkdir()
        outside = self.root / "outside.txt"
        outside.write_bytes(b"preserved outside data")
        link = prefix / "escape.txt"
        link.symlink_to(outside)
        record = {"_path": "escape.txt", "path_type": "softlink"}
        installed = {"paths_data": {"paths": [record]}}
        with self.assertRaisesRegex(ValueError, "symlink escapes"):
            environment._verify_conda_payload(installed, prefix)
        self.assertTrue(link.is_symlink())
        self.assertEqual(outside.read_bytes(), b"preserved outside data")
        real = prefix / "inside.txt"
        real.write_bytes(b"inside")
        record = {"_path": "inside.txt", "path_type": "hardlink", "sha256": hashlib.sha256(b"inside").hexdigest()}
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            environment._verify_conda_payload({"paths_data": {"paths": [record, dict(record)]}}, prefix)


class AcquisitionAndInventoryTests(unittest.TestCase):
    def test_unverified_acquisition_schema_is_not_environment_acceptance(self):
        lock = synthetic_lock()
        environment.validate_acquisition_lock(lock, environment.default_intent())
        self.assertNotIn("status", lock)
        self.assertNotIn("verification_id", lock)
        intent = environment.load_intent(INTENT_PATH)
        pins = intended_resolver_pins(REQUIREMENTS_PATH.read_bytes())
        for incomplete in (pins, {"requirements": pins}):
            with self.subTest(incomplete=incomplete):
                with self.assertRaises(ValueError):
                    environment.validate_acquisition_lock(incomplete, intent)
                for validator in (environment.validate_inventory, verifier.validate_verification,
                                  verifier.validate_resolved):
                    with self.subTest(validator=validator.__name__):
                        with self.assertRaises(ValueError):
                            validator(incomplete)

        # validate_resolved checks, in order: seal, identity formats, inventory/
        # CPU/P100 reference envelopes, then the exact acceptance policy. It
        # does not dereference evidence or check its runtime provenance; those
        # operations belong to _installation/_evidence_pair/finalize/check.
        # All successful-looking records below are inert in-memory schema
        # fixtures. No device, installed environment, or source was observed.
        inventory = synthetic_inventory()
        payloads = synthetic_verification_payloads(inventory)
        fixture = payloads["cpu_verification"]["fixture"]
        software = payloads["cpu_verification"]["software"]

        references = {}
        for name, payload in payloads.items():
            data = environment.canonical_bytes(payload) + b"\n"
            references[name] = {"logical_path": "unpublished-schema-fixtures/" + name + ".json",
                                "byte_size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        resolved = environment.seal({
            "schema_version": verifier.RESOLVED_SCHEMA, "status": "accepted",
            "environment_id": inventory["environment_id"], "intent_sha256": inventory["intent_sha256"],
            **references, "acceptance_policy": copy.deepcopy(verifier.ACCEPTANCE_POLICY),
        })
        baseline_bytes = environment.canonical_bytes(resolved)
        payload_bytes = environment.canonical_bytes(payloads)

        # Exact messages distinguish reached branches from an earlier rejection.
        # All semantic cases are resealed; only integrity cases retain/change
        # the supplied seal. Every case changes one property of a fresh copy.
        cases = [
            ("seal.stale", ("status",), "failed", "Manifest hash differs.", False),
            ("seal.malformed", ("manifest_hash",), "g" * 64, "Missing manifest hash.", False),
            ("seal.missing_value", ("manifest_hash",), None, "Missing manifest hash.", False),
            ("identity.schema", ("schema_version",), "unapproved.v1", "Malformed resolved identity.", True),
            ("identity.status", ("status",), "failed", "Malformed resolved identity.", True),
            ("identity.environment_type", ("environment_id",), 1, "Malformed resolved identity.", True),
            ("identity.environment_prefix", ("environment_id",), "other_" + "a" * 64,
             "Malformed resolved identity.", True),
            ("identity.environment_digest", ("environment_id",), "env_" + "g" * 64,
             "Malformed resolved identity.", True),
            ("identity.intent_type", ("intent_sha256",), True, "Malformed resolved identity.", True),
            ("identity.intent_digest", ("intent_sha256",), "a" * 63, "Malformed resolved identity.", True),
        ]
        for name in ("inventory", "cpu_verification", "p100_verification"):
            for field, value, label, message in (
                ("logical_path", 1, "path_type", "Invalid logical path."),
                ("logical_path", "", "path_empty", "Invalid logical path."),
                ("logical_path", "/outside.json", "path_absolute", "Invalid logical path."),
                ("logical_path", "one\\two.json", "path_backslash", "Escaping logical path."),
                ("logical_path", "one//two.json", "path_empty_component", "Escaping logical path."),
                ("logical_path", "./two.json", "path_dot", "Escaping logical path."),
                ("logical_path", "../two.json", "path_parent", "Escaping logical path."),
                ("logical_path", "one\ntwo.json", "path_control", "Control character in logical path."),
                ("byte_size", -1, "size_negative", "Invalid file fingerprint."),
                ("byte_size", True, "size_boolean", "Invalid file fingerprint."),
                ("byte_size", 1.0, "size_float", "Invalid file fingerprint."),
                ("sha256", None, "sha_type", "Invalid file fingerprint."),
                ("sha256", "a" * 63, "sha_length", "Invalid file fingerprint."),
                ("sha256", "A" * 64, "sha_uppercase", "Invalid file fingerprint."),
                ("sha256", "g" * 64, "sha_nonhex", "Invalid file fingerprint."),
            ):
                cases.append((name + "." + label, (name, field), value, message, True))
            unknown = copy.deepcopy(resolved[name])
            unknown["unapproved"] = True
            missing = copy.deepcopy(resolved[name])
            del missing["sha256"]
            for label, value in (("unknown_field", unknown), ("missing_field", missing)):
                cases.append((name + "." + label, (name,), value, "File reference fields differ.", True))

        policy = resolved["acceptance_policy"]
        for name, value in policy.items():
            changed_value = not value if type(value) is bool else "unapproved"
            if name == "required_modes":
                changed_value = ["p100", "cpu"]
            cases.append(("policy." + name, ("acceptance_policy", name), changed_value,
                          "Acceptance policy differs.", True))
            if type(value) is bool:
                cases.append(("policy." + name + ".integer", ("acceptance_policy", name), int(value),
                              "Acceptance policy differs.", True))
        for label, value in (("missing_cpu", ["p100"]), ("missing_p100", ["cpu"]),
                             ("mocked_p100", ["cpu", "mocked_p100"]), ("local_cpu", ["local_cpu", "p100"]),
                             ("duplicate_mode", ["cpu", "cpu"])):
            cases.append(("policy." + label, ("acceptance_policy", "required_modes"), value,
                          "Acceptance policy differs.", True))
        unknown = copy.deepcopy(policy)
        unknown["allow_mocked"] = True
        missing = copy.deepcopy(policy)
        del missing["live_slurm_evidence"]
        for label, value in (("unknown_field", unknown), ("missing_field", missing)):
            cases.append(("policy." + label, ("acceptance_policy",), value, "Acceptance policy differs.", True))

        with tempfile.TemporaryDirectory(prefix="b4a-resolved-schema-") as directory:
            root = Path(directory)
            (root / "preserved.txt").write_bytes(b"existing temporary artifact\n")
            before_files = {path.name: (path.read_bytes(), path.stat()) for path in root.iterdir()}

            def validate_without_effects(validator, record, message=None):
                before = environment.canonical_bytes(record)
                with (
                    patch.object(environment, "publish", side_effect=AssertionError("Unexpected publication")) as publish,
                    patch.object(environment, "_exclusive_bytes", side_effect=AssertionError("Unexpected artifact write")) as writer,
                    patch.object(environment, "read_regular", side_effect=AssertionError("Unexpected evidence read")) as reader,
                    patch.object(verifier, "finalize_environment", side_effect=AssertionError("Unexpected finalize")) as finalize,
                    patch.object(builtins, "open", side_effect=AssertionError("Unexpected file open")) as builtin_open,
                    patch.object(io, "open", side_effect=AssertionError("Unexpected file open")) as io_open,
                    patch.object(os, "open", side_effect=AssertionError("Unexpected file open")) as os_open,
                ):
                    if message is None:
                        self.assertIsNone(validator(record))
                    else:
                        with self.assertRaises(environment.EnvironmentError) as raised:
                            validator(record)
                        self.assertIs(type(raised.exception), environment.EnvironmentError)
                        self.assertEqual(str(raised.exception), message)
                    for boundary in (publish, writer, reader, finalize, builtin_open, io_open, os_open):
                        boundary.assert_not_called()
                self.assertEqual(environment.canonical_bytes(record), before)
                self.assertEqual(environment.canonical_bytes(resolved), baseline_bytes)
                self.assertEqual(environment.canonical_bytes(payloads), payload_bytes)
                after_files = {path.name: (path.read_bytes(), path.stat()) for path in root.iterdir()}
                self.assertEqual(after_files, before_files)

            # Validate each referenced schema and the direct resolved baseline
            # before any malformed subcase; these calls collect no live facts.
            validate_without_effects(environment.validate_inventory, inventory)
            for name, mode in (("cpu_verification", "cpu"), ("p100_verification", "p100")):
                evidence = payloads[name]
                validate_without_effects(verifier.validate_verification, evidence)
                self.assertEqual(evidence["mode"], mode)
                self.assertEqual(evidence["status"], "successful")
                self.assertEqual(evidence["environment_id"], resolved["environment_id"])
                self.assertEqual(evidence["inventory_sha256"], resolved["inventory"]["sha256"])
                self.assertEqual(evidence["execution"], inventory["execution"])
                self.assertEqual(evidence["software"], software)
                self.assertEqual(evidence["fixture"], fixture)
            self.assertEqual(resolved["intent_sha256"], environment.digest(intent))
            self.assertEqual(len({reference["logical_path"] for reference in references.values()}), 3)
            validate_without_effects(verifier.validate_resolved, resolved)
            for label, location, value, message, recompute in cases:
                with self.subTest(resolved_branch=label):
                    changed = copy.deepcopy(resolved)
                    at_location(changed, location[:-1])[location[-1]] = copy.deepcopy(value)
                    if recompute:
                        changed = reseal(changed)
                        environment.validate_seal(changed)
                    validate_without_effects(verifier.validate_resolved, changed, message)
            validate_without_effects(verifier.validate_resolved, resolved)

        # Inspect the future procedure only; never execute acquisition or installation.
        shell = (PROJECT_ROOT / "scripts/carc/create_cnn_rc_environment.sh").read_text(encoding="utf-8")

        def assert_clean_module_baseline(script):
            self.assertTrue(script.startswith("#!/usr/bin/env bash\n"))
            active = [line.strip() for line in script.splitlines()
                      if line.strip() and not line.lstrip().startswith("#")]
            self.assertEqual(active[0], "set -euo pipefail")
            module_commands = [line for line in active if re.search(r"\bmodule\b", line)]
            self.assertEqual(module_commands, ["module purge", "module load conda/25.11.0"])
            purge_index = active.index("module purge")
            self.assertEqual(active[purge_index + 1], "module load conda/25.11.0")
            self.assertIn("\nmodule purge\nmodule load conda/25.11.0\n", script)
            prefix = active[:purge_index]
            allocation = ('[[ -n "${SLURM_JOB_ID:-}" ]] || { echo '
                          '"A real CPU Slurm compute allocation is required." >&2; exit 2; }')
            self.assertIn(allocation, prefix)
            # Closed, non-executing grammar for the existing input-validation
            # prelude. Unknown commands and software-initializing substitutions
            # fail instead of being classified by a growing command denylist.
            validation_forms = (
                r'set -euo pipefail',
                r'(spec|environment_prefix|cache_root|exports_root|inventory_output|expected_commit)=""',
                r'declare -A seen_arguments=\(\)',
                r'argument="\$1"',
                r'seen_arguments\[\$argument\]=1',
                r'while \(\( \$# > 0 \)\); do',
                r'if \(\( \$# < 2 \)\); then',
                r'if \[\[ -n "\$\{seen_arguments\[\$argument\]\+present\}" \]\]; then',
                r'echo "[^"]*" >&2',
                r'(fi|done|esac|shift 2|exit 2)',
                r'case "\$argument" in',
                r'--(spec|prefix|cache-root|exports-root|inventory-output|expected-software-commit)\) '
                r'(spec|environment_prefix|cache_root|exports_root|inventory_output|expected_commit)="\$2" ;;',
                r'\*\) echo "Unknown argument: \$argument" >&2; exit 2 ;;',
                r'for value in "\$spec" "\$environment_prefix" "\$cache_root" "\$exports_root" '
                r'"\$inventory_output"; do',
                r'\[\[ [^;\n]+ \]\] \|\| \{ echo "[^"]*" >&2; exit 2; \}',
            )
            blocks = []
            for line in prefix:
                self.assertNotIn("$(", line)
                self.assertNotIn("<(", line)
                self.assertNotIn(">(", line)
                self.assertNotIn("`", line)
                self.assertNotIn("\\", line)
                self.assertLessEqual(line.count("[["), 1)
                self.assertLessEqual(line.count("]]"), 1)
                self.assertRegex(line, "^(?:" + "|".join(validation_forms) + ")$")
                if line.startswith(("while ", "for ")):
                    blocks.append("done")
                elif line.startswith("if "):
                    blocks.append("fi")
                elif line.startswith("case "):
                    blocks.append("esac")
                elif line in ("done", "fi", "esac"):
                    self.assertTrue(blocks)
                    self.assertEqual(blocks.pop(), line)
                if line == allocation:
                    self.assertEqual(blocks, [])
            self.assertEqual(blocks, [], "Purge must be an unconditional top-level command.")
            self.assertEqual([line for line in active if line.startswith("set ")], ["set -euo pipefail"])
            self.assertNotRegex("\n".join(active), r"\b(?:LOADEDMODULES|MODULEPATH|trap)\b")

        assert_clean_module_baseline(shell)
        sequence = "module purge\nmodule load conda/25.11.0\n"
        malformed = {
            "Conda before purge": shell.replace(sequence, "module load conda/25.11.0\nmodule purge\n"),
            "missing purge": shell.replace("module purge\n", ""),
            "suppressed purge": shell.replace("module purge\n", "module purge || true\n"),
            "earlier module": shell.replace("module purge\n", "module load gcc/12\nmodule purge\n"),
            "commented purge": shell.replace("module purge\n", "# module purge\n"),
            "duplicate purge": shell.replace("module purge\n", "module purge\nmodule purge\n"),
            "redirected purge": shell.replace("module purge\n", "module purge 2>/dev/null\n"),
            "conditional purge": shell.replace("module purge\n", "if module purge; then :; fi\n"),
            "disabled errexit": shell.replace("module purge\n", "set +e\nmodule purge\n"),
            "wrong Conda version": shell.replace("module load conda/25.11.0", "module load conda/24.11.0"),
            "system CUDA": shell.replace(sequence, sequence + "module load cuda/12.6\n"),
            "inherited baseline": shell.replace("module purge\n", '[[ -n "${LOADEDMODULES:-}" ]]\nmodule purge\n'),
            "conditional module block": shell.replace(sequence, "if (( $# < 2 )); then\n" + sequence + "fi\n"),
            "missing allocation check": shell.replace(
                '[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "A real CPU Slurm compute allocation is required." >&2; exit 2; }\n',
                "",
            ),
        }
        for command in (
            "conda info", 'eval "$(conda shell.bash hook)"', 'conda activate "$environment_prefix"',
            "python -V", "pip --version", "conda create --dry-run", "pip install --dry-run synthetic",
            "curl https://example.invalid/synthetic", "wget https://example.invalid/synthetic",
            'result="$(python -V)"',
            '[[ -e <(python -V) ]] || { echo "synthetic" >&2; exit 2; }',
            '[[ -e >(python -V) ]] || { echo "synthetic" >&2; exit 2; }',
            '[[ -n "$spec" ]] && python -V && [[ -n "$spec" ]] || { echo "synthetic" >&2; exit 2; }',
        ):
            malformed["early " + command] = shell.replace("module purge\n", command + "\nmodule purge\n")
        for label, changed in malformed.items():
            with self.subTest(shell_contract=label):
                with self.assertRaises(AssertionError):
                    assert_clean_module_baseline(changed)
        self.assertNotIn("pip --user", shell)
        self.assertNotIn("sudo", shell)
        self.assertIn('conda activate "$environment_prefix"', shell)
        self.assertIn('python -B "$script_directory/cnn_rc_environment.py"', shell)
        self.assertIn('python -B -m scripts.carc.verify_cnn_rc_environment inventory', shell)
        self.assertIn("This script never accepts an environment. CPU and P100 evidence followed by", shell)
        self.assertIn("explicit finalize are separate future authorized verification operations.", shell)
        source = inspect.getsource(environment.acquire_environment)
        self.assertIn('intended["name"] + "==" + intended["version"]', source)
        self.assertIn('record["sha256"] == intended["advertised_sha256"]', source)
        calls = [node for node in ast.walk(ast.parse(source))
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]

        def matching_calls(function, required, excluded=()):
            matches = []
            for call in calls:
                strings = {node.value for node in ast.walk(call)
                           if isinstance(node, ast.Constant) and type(node.value) is str}
                if call.func.id == function and set(required) <= strings and not set(excluded) & strings:
                    matches.append(call)
            self.assertEqual(len(matches), 1, (function, required, excluded))
            return matches[0]

        closure = matching_calls("_run", ("pip", "install", "--dry-run", "--ignore-installed"), ("--no-deps",))
        validation = matching_calls("validate_acquisition_lock", ())
        publication = matching_calls("publish", ("acquisition-lock.json",))
        resolved_input = matching_calls("_exclusive_bytes", ("resolved-requirements.txt",))
        installation = matching_calls("_run", ("pip", "install"), ("--dry-run",))
        command = installation.args[0]
        self.assertIsInstance(command, ast.List)
        options = [node.value for node in command.elts if isinstance(node, ast.Constant)]
        for required in ("--no-index", "--no-deps", "--only-binary=:all:", "--require-hashes", "--requirement"):
            self.assertIn(required, options)
        self.assertEqual(ast.dump(command.elts[-1]),
                         ast.dump(ast.parse('str(cache / "resolved-requirements.txt")', mode="eval").body))
        self.assertLess(closure.lineno, validation.lineno)
        self.assertLess(validation.lineno, publication.lineno)
        self.assertLess(publication.lineno, resolved_input.lineno)
        self.assertLess(resolved_input.lineno, installation.lineno)
        bootstrap_lock = matching_calls("publish", ("conda-bootstrap-lock.json",))
        bootstrap_install = matching_calls("_run", ("conda", "create", "--offline"))
        self.assertLess(bootstrap_lock.lineno, bootstrap_install.lineno)

    def test_acquisition_rejects_missing_duplicate_or_changed_exact_packages(self):
        original = synthetic_lock()
        mutations = []
        changed = copy.deepcopy(original)
        changed["pip"].pop()
        mutations.append(changed)
        changed = copy.deepcopy(original)
        changed["pip"].append(copy.deepcopy(changed["pip"][0]))
        mutations.append(changed)
        for manager, field, value in (("conda", "version", "25.3"), ("pip", "sha256", "f" * 64),
                                      ("pip", "byte_size", True), ("pip", "version", ">=2.2.6"),
                                      ("pip", "platform_tags", []), ("pip", "cache_path", "../wheel.whl")):
            changed = copy.deepcopy(original)
            changed[manager][0][field] = value
            mutations.append(changed)
        for version in ("2.2.6", "2.2.5"):
            changed = copy.deepcopy(original)
            overlapping = copy.deepcopy(original["conda"][0])
            filename = "numpy-" + version + "-synthetic_0.conda"
            overlapping.update(name="NumPy", version=version, filename=filename,
                               origin_url="https://conda.anaconda.org/conda-forge/linux-64/" + filename,
                               cache_path="conda/" + filename)
            changed["conda"].insert(0, overlapping)
            mutations.append(changed)
        for changed in mutations:
            with self.subTest(changed=changed):
                with self.assertRaises(ValueError):
                    environment.validate_acquisition_lock(reseal(changed), environment.default_intent())

        # Keep the archive and installed metadata identities fixed: only the
        # declared Conda build disagrees with the valid inventory baseline.
        inventory = synthetic_inventory()
        inventory_before = copy.deepcopy(inventory)
        inventory_bytes = environment.canonical_bytes(inventory)
        changed = copy.deepcopy(inventory)
        self.assertEqual(changed["packages"][0]["manager"], "conda")
        changed["packages"][0]["build"] = "synthetic_1"
        self.assertEqual(
            [key for key in inventory["packages"][0]
             if changed["packages"][0][key] != inventory["packages"][0][key]],
            ["build"],
        )
        expected = copy.deepcopy(inventory)
        expected["packages"][0]["build"] = "synthetic_1"
        changed = reseal(changed)
        expected["manifest_hash"] = changed["manifest_hash"]
        self.assertEqual(changed, expected)
        self.assertNotEqual(changed["manifest_hash"], inventory["manifest_hash"])
        changed_before = copy.deepcopy(changed)
        changed_bytes = environment.canonical_bytes(changed)
        with patch.object(environment, "publish", side_effect=AssertionError("Inventory validation must not publish")) as writer:
            environment.validate_inventory(inventory)
            environment.validate_seal(changed)
            with self.assertRaises(environment.EnvironmentError) as rejected:
                environment.validate_inventory(changed)
            self.assertEqual(str(rejected.exception), "Conda filename does not bind its name, version and build.")
            writer.assert_not_called()
        self.assertEqual(inventory, inventory_before)
        self.assertEqual(environment.canonical_bytes(inventory), inventory_bytes)
        self.assertEqual(changed, changed_before)
        self.assertEqual(environment.canonical_bytes(changed), changed_bytes)

    def test_acquisition_rejects_source_archives_wrong_origins_and_unbound_tags(self):
        original = synthetic_lock()["pip"][0]
        changes = (
            {"origin_url": "https://unapproved.invalid/" + original["filename"]},
            {"origin_url": "http://files.pythonhosted.org/" + original["filename"]},
            {"filename": "numpy-2.2.6.tar.gz", "origin_url": "https://files.pythonhosted.org/numpy-2.2.6.tar.gz"},
            {"platform_tags": ["cp312-cp312-win_amd64"]},
            {"upstream_digest": {"algorithm": "sha256", "value": "f" * 64}},
        )
        for fields in changes:
            with self.subTest(fields=fields):
                with self.assertRaises(ValueError):
                    environment._validate_package(dict(original, **fields), acquisition=True)

    def test_semantic_identity_excludes_prefix_host_job_and_installed_metadata(self):
        inventory = synthetic_inventory()
        environment.validate_inventory(inventory)
        changed = copy.deepcopy(inventory)
        changed["execution"]["prefix"] = "/different/uninstalled/prefix"
        changed["execution"]["python_executable"] = "/different/uninstalled/prefix/bin/python"
        changed["execution"]["host"] = "another-compute"
        changed["execution"]["slurm"]["node"] = "another-compute"
        changed["execution"]["slurm"]["job_id"] = "456"
        for package in changed["packages"]:
            package["installed_metadata_sha256"] = "f" * 64
        changed = reseal(changed)
        environment.validate_inventory(changed)
        self.assertEqual(changed["environment_id"], inventory["environment_id"])
        self.assertNotEqual(changed["manifest_hash"], inventory["manifest_hash"])
        semantic = json.dumps(changed["semantic_environment"])
        self.assertNotIn("/different/", semantic)
        self.assertNotIn("another-compute", semantic)
        self.assertNotIn("installed_metadata_sha256", semantic)

    def test_inventory_recursive_unknown_missing_and_wrong_types_rejected(self):
        original = synthetic_inventory()
        environment.validate_inventory(original)
        for location, node in locations(original):
            if type(node) is dict:
                for operation in ("unknown", "missing"):
                    changed = copy.deepcopy(original)
                    selected = at_location(changed, location)
                    if operation == "unknown":
                        selected["unapproved"] = True
                    else:
                        selected.pop(next(key for key in selected if key != "manifest_hash"))
                    with self.subTest(location=location, operation=operation):
                        with self.assertRaises(ValueError):
                            environment.validate_inventory(reseal(changed))
        for path, value in ((["exports", 0, "byte_size"], True),
                            (["execution", "slurm", "job_id"], 123),
                            (["semantic_environment", "numerics", "torch_intraop_threads"], True)):
            changed = copy.deepcopy(original)
            at_location(changed, path[:-1])[path[-1]] = value
            with self.assertRaises(ValueError):
                environment.validate_inventory(reseal(changed))


class ActivationFailureTests(TemporaryCase):
    def test_local_context_fails_before_any_scheduler_or_installer_command(self):
        runtime = dict(os.environ)
        for name in tuple(runtime):
            if name.startswith("SLURM_"):
                runtime.pop(name)
        with (
            patch.dict(os.environ, runtime, clear=True),
            patch.object(subprocess, "run", side_effect=AssertionError("No external command before allocation checks")),
        ):
            with self.assertRaises(ValueError):
                environment.execution_context(Path(sys.prefix), mode="cpu")

    def test_relative_prefix_and_unsupported_modes_are_rejected(self):
        for prefix, mode in ((Path("relative"), "cpu"), (self.root, "mocked")):
            with self.subTest(prefix=prefix, mode=mode):
                with self.assertRaises(ValueError):
                    environment.execution_context(prefix, mode=mode)

    def test_wrong_module_platform_architecture_and_glibc_are_rejected(self):
        for modules in ("", "conda/25.10.0", "conda/25.11.0:cuda/12.6"):
            with self.subTest(modules=modules), patch.dict(os.environ, {"LOADEDMODULES": modules}):
                with self.assertRaises(ValueError):
                    environment._module_guard()
        for system, machine, libc in (("Darwin", "x86_64", ("glibc", "2.28")),
                                      ("Linux", "aarch64", ("glibc", "2.28")),
                                      ("Linux", "x86_64", ("glibc", "2.27")),
                                      ("Linux", "x86_64", ("musl", "1.2"))):
            with self.subTest(system=system, machine=machine, libc=libc):
                with (patch.object(environment.platform, "system", return_value=system),
                      patch.object(environment.platform, "machine", return_value=machine),
                      patch.object(environment.platform, "libc_ver", return_value=libc)):
                    with self.assertRaises(ValueError):
                        environment.platform_facts()

    def test_each_activation_policy_is_enforced_before_runtime_use(self):
        runtime = {
            "CONDA_PREFIX": str(self.root), "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            **{name: "1" for name in environment.THREAD_VARIABLES},
        }
        mutations = [("CONDA_PREFIX", "/another/prefix"), ("PYTHONNOUSERSITE", "0"),
                     ("PYTHONDONTWRITEBYTECODE", "0"), ("PYTHONPATH", "/unapproved/imports"),
                     ("PYTHONHOME", "/unapproved/python"), ("CUBLAS_WORKSPACE_CONFIG", ":16:8")]
        mutations.extend((name, "2") for name in environment.THREAD_VARIABLES)
        for name, value in mutations:
            changed = dict(runtime, **{name: value})
            with self.subTest(name=name):
                with (
                    patch.dict(os.environ, changed, clear=True),
                    patch.object(environment.platform, "python_implementation", return_value="CPython"),
                    patch.object(environment.platform, "python_version", return_value="3.11.14"),
                    patch.object(sys, "prefix", str(self.root)),
                    patch.object(sys, "executable", str(self.root / "bin/python")),
                    patch.object(environment.site, "ENABLE_USER_SITE", False),
                    patch.object(environment.importlib.metadata, "version", return_value="25.2"),
                ):
                    with self.assertRaises(ValueError):
                        environment.activation_guard(self.root)

    def test_slurm_wrong_owner_state_nodes_host_cluster_and_gpu_are_rejected(self):
        base = {"JobId": "123", "JobState": "RUNNING", "UserId": "test(" + str(os.getuid()) + ")",
                "NumNodes": "1", "NumTasks": "1", "NodeList": "synthetic-compute",
                "Partition": "synthetic", "AllocTRES": "cpu=1,mem=1G,node=1"}
        cases = [
            ("JobId", "456"), ("JobState", "PENDING"), ("UserId", "test(99999999)"),
            ("NumNodes", "2"), ("NumTasks", "2"), ("NodeList", "other-compute"),
            ("Partition", ""), ("AllocTRES", "cpu=1,gres/gpu=1"),
        ]
        for field, value in cases:
            record = dict(base, **{field: value})
            raw = " ".join(key + "=" + item for key, item in record.items()).encode()
            responses = [raw, record["NodeList"].encode(), b"ClusterName = synthetic\n"]
            with self.subTest(field=field):
                with (
                    patch.dict(os.environ, {"SLURM_JOB_ID": "123", "SLURM_CLUSTER_NAME": "synthetic", "CUDA_VISIBLE_DEVICES": ""}),
                    patch.object(environment.socket, "gethostname", return_value="synthetic-compute"),
                    patch.object(environment, "_run", side_effect=responses),
                ):
                    with self.assertRaises(ValueError):
                        environment._slurm_context("cpu")


class FixedNumericalTests(unittest.TestCase):
    def setUp(self):
        guard_b4a_imports(self)

    def test_fixture_exact_independent_one_hot_and_float32_targets(self):
        model, inputs, targets, fixture = verifier.fixed_fixture()
        expected_inputs = np.zeros((128, 14, 4), dtype=np.float32)
        expected_targets = np.zeros((128, 1), dtype=np.float32)
        for row in range(128):
            for position in range(14):
                text = FIXTURE_ID + "\0" + str(row) + "\0" + str(position)
                channel = hashlib.sha256(text.encode("utf-8")).digest()[0] % 4
                expected_inputs[row, position, channel] = 1
            expected_targets[row, 0] = (((37 * row) % 127) + 0.5) / 128
        self.assertEqual(inputs.device.type, "cpu")
        self.assertEqual(targets.device.type, "cpu")
        self.assertEqual(inputs.dtype, torch.float32)
        self.assertEqual(targets.dtype, torch.float32)
        np.testing.assert_array_equal(inputs.numpy(), expected_inputs)
        np.testing.assert_array_equal(targets.numpy(), expected_targets)
        encoded = json.dumps(fixture)
        self.assertIn(FIXTURE_ID, encoded)
        self.assertIn(hashlib.sha256(expected_inputs.tobytes()).hexdigest(), encoded)
        self.assertIn(hashlib.sha256(expected_targets.tobytes()).hexdigest(), encoded)
        self.assertEqual(fixture["input"], {
            "dtype": "torch.float32", "shape": [128, 14, 4],
            "sha256": hashlib.sha256(expected_inputs.tobytes()).hexdigest(),
        })
        self.assertEqual(fixture["targets"], {
            "dtype": "torch.float32", "shape": [128, 1],
            "sha256": hashlib.sha256(expected_targets.tobytes()).hexdigest(),
        })
        self.assertEqual(fixture["initial_state_sha256"], verifier._state_fingerprint(model.state_dict()))
        second_model, second_inputs, second_targets, second_fixture = verifier.fixed_fixture()
        self.assertEqual(second_fixture, fixture)
        self.assertEqual(verifier._state_fingerprint(second_model.state_dict()), fixture["initial_state_sha256"])
        self.assertTrue(torch.equal(inputs, second_inputs))
        self.assertTrue(torch.equal(targets, second_targets))
        from src.cnn_rc import CNNRC
        expected_model = CNNRC(seed=43001)
        for name, actual in model.state_dict().items():
            expected = expected_model.state_dict()[name]
            if isinstance(actual, torch.Tensor):
                self.assertTrue(torch.equal(actual, expected), name)
            else:
                self.assertEqual(actual, expected)
        config = environment.strict_json((PROJECT_ROOT / "configs/exd_hox_cnn_rc_v1.json").read_bytes())
        candidate = verifier._fixed_candidate()
        self.assertEqual(candidate, config["candidates"]["smoke_adam_v1"])
        self.assertEqual(candidate["regularization"], {"l1": 5e-6, "l2": 1e-5})
        self.assertEqual(candidate["optimizer"], {
            "name": "Adam", "learning_rate": 5e-5, "betas": [0.9, 0.999],
            "epsilon": 1e-8, "weight_decay": 0.0, "amsgrad": False,
            "foreach": False, "fused": False, "capturable": False,
            "differentiable": False, "maximize": False,
        })
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            path = root / "configs/exd_hox_cnn_rc_v1.json"
            path.parent.mkdir()
            for key, replacement in (
                ("name", "SGD"), ("learning_rate", 1e-3), ("betas", [0.8, 0.999]),
                ("epsilon", 1e-7), ("weight_decay", 0.01), ("amsgrad", True),
                ("foreach", True), ("fused", True), ("capturable", True),
                ("differentiable", True), ("maximize", True),
            ):
                for missing in (False, True):
                    with self.subTest(optimizer_key=key, missing=missing):
                        changed = copy.deepcopy(config)
                        optimizer = changed["candidates"]["smoke_adam_v1"]["optimizer"]
                        if missing:
                            del optimizer[key]
                        else:
                            optimizer[key] = replacement
                        path.write_bytes(environment.canonical_bytes(changed))
                        before = path.read_bytes()
                        with patch.object(environment, "PROJECT_ROOT", root):
                            with self.assertRaises(ValueError):
                                verifier._fixed_candidate()
                        self.assertEqual(path.read_bytes(), before)
            for key in ("l1", "l2"):
                changed = copy.deepcopy(config)
                changed["candidates"]["smoke_adam_v1"]["regularization"][key] *= 128
                path.write_bytes(environment.canonical_bytes(changed))
                with self.subTest(regularization=key), patch.object(environment, "PROJECT_ROOT", root):
                    with self.assertRaises(ValueError):
                        verifier._fixed_candidate()
            for mutation in ("candidate", "batch_size", "model_contract"):
                changed = copy.deepcopy(config)
                if mutation == "candidate":
                    changed["candidates"]["other_candidate"] = changed["candidates"].pop("smoke_adam_v1")
                elif mutation == "batch_size":
                    changed["candidates"]["smoke_adam_v1"]["batch_size"] = 64
                else:
                    changed["model_contract"] = "different_model"
                path.write_bytes(environment.canonical_bytes(changed))
                with self.subTest(candidate_mutation=mutation), patch.object(environment, "PROJECT_ROOT", root):
                    with self.assertRaises(ValueError):
                        verifier._fixed_candidate()

    def test_real_cpu_probe_does_not_initialize_cuda_or_publish(self):
        from src.cnn_rc import CNNRC

        repetitions = environment.load_intent(INTENT_PATH)["verification"]["repetitions"]
        original_trajectory = verifier._trajectory
        original_optimizer = verifier._make_optimizer
        original_serialization = verifier._serialization_probe
        original_penalty = CNNRC.convolution_kernel_penalty
        initial_states, models, optimizers, trajectories = [], [], [], []
        serialized = []
        penalties = []

        def observe_penalty(model, **coefficients):
            self.assertEqual(coefficients, {"l1": 5e-6, "l2": 1e-5})
            penalties.append(model)
            return original_penalty(model, **coefficients)

        def observe_serialization(state):
            reloaded = original_serialization(state)
            self.assertEqual(verifier._state_fingerprint(state), verifier._state_fingerprint(reloaded))
            serialized.append(reloaded)
            return reloaded

        def observe_optimizer(model, config):
            optimizer = original_optimizer(model, config)
            self.assertIs(type(optimizer), torch.optim.Adam)
            self.assertEqual(config, verifier._fixed_candidate()["optimizer"])
            self.assertEqual(verifier._state_fingerprint(model.state_dict()), initial_states[-1]["model"])
            self.assertEqual(verifier._state_fingerprint(verifier._capture_rng()), initial_states[-1]["rng"])
            self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
            self.assertEqual(optimizer.state_dict()["state"], {})
            self.assertTrue(all(model is not prior for prior in models))
            self.assertTrue(all(optimizer is not prior for prior in optimizers))
            models.append(model)
            optimizers.append(optimizer)
            return optimizer

        def observe_trajectory(template, inputs, targets, device, initial_rng):
            self.assertEqual(device, "cpu")
            self.assertTrue(all(parameter.grad is None for parameter in template.parameters()))
            initial = {
                "model": verifier._state_fingerprint(template.state_dict()),
                "rng": verifier._state_fingerprint(initial_rng),
                "inputs": verifier._state_fingerprint(inputs),
                "targets": verifier._state_fingerprint(targets),
            }
            initial_states.append(initial)
            result = original_trajectory(template, inputs, targets, device, initial_rng)
            self.assertIsNot(models[-1], template)
            self.assertEqual(verifier._state_fingerprint(template.state_dict()), initial["model"])
            self.assertTrue(all(parameter.grad is None for parameter in template.parameters()))
            self.assertEqual(set(result["tensors"]), set(verifier._comparison_names()))
            for state in optimizers[-1].state.values():
                self.assertEqual(state["step"].item(), 1)
            self.assertTrue(all(parameter.grad is None for parameter in models[-1].parameters()))
            trajectories.append(result)
            # Force RNG drift between complete calls: the next call must restore it.
            torch.rand(1)
            return result

        guards = (
            patch.object(torch.cuda, "init", side_effect=AssertionError("CUDA initialization forbidden")),
            patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA lazy initialization forbidden")),
            patch.object(torch.cuda, "device_count", side_effect=AssertionError("CUDA enumeration forbidden")),
            patch.object(environment, "publish", side_effect=AssertionError("Probe publication forbidden")),
        )
        for guard in guards:
            guard.start()
            self.addCleanup(guard.stop)
        with patch.object(verifier, "_trajectory", side_effect=observe_trajectory), \
                patch.object(verifier, "_make_optimizer", side_effect=observe_optimizer), \
                patch.object(verifier, "_serialization_probe", side_effect=observe_serialization), \
                patch.object(CNNRC, "convolution_kernel_penalty", autospec=True, side_effect=observe_penalty):
            result = verifier.run_fixed_probe("cpu")
        self.assertEqual(len(trajectories), repetitions)
        self.assertEqual(len(optimizers), repetitions)
        self.assertEqual(penalties, models)
        self.assertEqual(initial_states[0], initial_states[1])
        self.assertEqual(trajectories[0]["state_sha256"], trajectories[1]["state_sha256"])
        self.assertTrue(serialized)
        self.assertIs(result["checks"]["controlled_serialization_round_trip"], True)

        # Inspect the GPU branch without executing or mocking a successful P100 run.
        probe = ast.parse(inspect.getsource(verifier.run_fixed_probe))
        gpu_condition = ast.dump(ast.parse('device == "cuda:0"', mode="eval").body)
        gpu_branches = [node for node in ast.walk(probe)
                        if isinstance(node, ast.If) and ast.dump(node.test) == gpu_condition]
        self.assertEqual(len(gpu_branches), 1)
        gpu_calls = [node for node in ast.walk(gpu_branches[0])
                     if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                     and node.func.id == "_trajectory"]
        self.assertEqual(len(gpu_calls), repetitions)
        for call in gpu_calls:
            self.assertEqual([ast.dump(argument) for argument in call.args],
                             [ast.dump(ast.Name(id=name, ctx=ast.Load()))
                              for name in ("template", "inputs", "targets", "device", "gpu_rng")])
            self.assertEqual(call.keywords, [])
        self.assertEqual(set(result), {"fixture", "observations", "checks"})
        self.assertTrue(result["checks"] and all(value is True for value in result["checks"].values()))
        observations = result["observations"]
        self.assertEqual(observations["state_sha256"]["cpu_first"], observations["state_sha256"]["cpu_repeat"])
        self.assertEqual(len(observations["comparisons"]["repeat_cpu"]), 83)
        self.assertTrue(all(record["passed"] for record in observations["comparisons"]["repeat_cpu"]))
        self.assertNotIn("status", result)
        self.assertNotIn("manifest_hash", result)
        self.assertFalse(torch.cuda.is_initialized())
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())
        self.assertFalse(torch.backends.cudnn.benchmark)
        self.assertTrue(torch.backends.cudnn.deterministic)
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        self.assertFalse(torch.backends.cudnn.allow_tf32)
        self.assertEqual(torch.get_float32_matmul_precision(), "highest")
        self.assertEqual(torch.get_num_threads(), 1)
        self.assertEqual(torch.get_num_interop_threads(), 1)

    def test_real_trajectory_checks_gradients_bn_optimizer_and_their_nonfinite_values(self):
        verifier._configure_runtime("cpu")
        template, inputs, targets, unused = verifier.fixed_fixture()
        initial_rng = verifier._capture_rng()
        trajectory = verifier._trajectory(template, inputs, targets, "cpu", initial_rng)
        self.assertEqual(set(trajectory["tensors"]), set(verifier._comparison_names()))
        tensors = trajectory["tensors"]
        candidate = verifier._fixed_candidate()
        expected_penalty = template.convolution_kernel_penalty(**candidate["regularization"]).detach()
        self.assertTrue(torch.equal(tensors["loss.penalty"], expected_penalty))
        self.assertTrue(torch.equal(tensors["loss.mse"], (tensors["train.output"] - targets).square().mean()))
        self.assertTrue(torch.equal(tensors["loss.total"], tensors["loss.mse"] + tensors["loss.penalty"]))
        for name in ("gradient.W", "state.running_var", "optimizer.state.0.exp_avg"):
            with self.subTest(name=name):
                reference = {name: trajectory["tensors"][name]}
                changed = reference[name].clone()
                changed.reshape(-1)[0] += 0.1
                result = verifier.compare_tensors(reference, {name: changed})
                self.assertFalse(result[0]["passed"])
                changed.reshape(-1)[0] = float("nan")
                with self.assertRaises(ValueError):
                    verifier.compare_tensors(reference, {name: changed})
                with self.assertRaises(ValueError):
                    verifier._state_tensors({name: changed}, "state", {})

    def test_comparison_reports_each_tensor_and_detects_drift(self):
        reference = {
            "train.output": torch.tensor([0.25, 0.75], dtype=torch.float32),
            "gradient.W": torch.tensor([0.001, -0.01], dtype=torch.float32),
        }
        equal = {name: value.clone() for name, value in reference.items()}
        records = verifier.compare_tensors(reference, equal, exact=True)
        self.assertEqual(len(records), 2)
        self.assertTrue(all(record["passed"] for record in records))
        changed = {name: value.clone() for name, value in reference.items()}
        changed["gradient.W"][0] += 0.05
        records = verifier.compare_tensors(reference, changed)
        self.assertTrue(any(not record["passed"] for record in records))
        self.assertTrue(any(record["passed"] for record in records))

    def test_exact_repeat_distinguishes_signed_zero_bytes(self):
        positive = {"eval.output": torch.tensor([0.0], dtype=torch.float32)}
        negative = {"eval.output": torch.tensor([-0.0], dtype=torch.float32)}
        self.assertFalse(verifier.compare_tensors(positive, negative, exact=True)[0]["passed"])
        self.assertTrue(verifier.compare_tensors(positive, negative)[0]["passed"])
        self.assertNotEqual(verifier._state_fingerprint(positive), verifier._state_fingerprint(negative))
        original = {"weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
                    "step": 1, "flags": [False, None], "pair": (0.5, "controlled")}
        self.assertEqual(verifier._state_fingerprint(original), verifier._state_fingerprint(copy.deepcopy(original)))
        reordered = {key: original[key] for key in reversed(original)}
        self.assertEqual(verifier._state_fingerprint(original), verifier._state_fingerprint(reordered))
        noncontiguous = original["weight"].transpose(0, 1)
        self.assertFalse(noncontiguous.is_contiguous())
        self.assertEqual(verifier._state_fingerprint({"weight": noncontiguous}),
                         verifier._state_fingerprint({"weight": noncontiguous.contiguous()}))
        self.assertNotEqual(verifier._state_fingerprint({"step": 1}), verifier._state_fingerprint({"step": True}))
        self.assertNotEqual(verifier._state_fingerprint({1: "value"}), verifier._state_fingerprint({"1": "value"}))
        for changed in (
            {"renamed": original["weight"]}, {"weight": original["weight"].to(torch.int64)},
            {"weight": original["weight"].reshape(3, 2)}, {"weight": original["weight"] + 1},
        ):
            with self.subTest(changed=changed):
                self.assertNotEqual(verifier._state_fingerprint({"weight": original["weight"]}),
                                    verifier._state_fingerprint(changed))
        # B4a compatibility only: no run/index/recovery or scientific checkpoint identities.
        saved_paths = []
        original_save = torch.save
        original_load = torch.load

        def observe_save(value, path, *arguments, **keywords):
            saved_paths.append(Path(path))
            self.assertTrue(Path(path).parent.is_dir())
            return original_save(value, path, *arguments, **keywords)

        def observe_load(path, *arguments, **keywords):
            self.assertIs(keywords.get("weights_only"), True)
            self.assertEqual(keywords.get("map_location"), "cpu")
            return original_load(path, *arguments, **keywords)

        with patch.object(torch, "save", side_effect=observe_save), \
                patch.object(torch, "load", side_effect=observe_load):
            reloaded = verifier._serialization_probe(original)
        self.assertEqual(verifier._state_fingerprint(original), verifier._state_fingerprint(reloaded))
        self.assertTrue(saved_paths)
        self.assertTrue(all(not path.exists() and not path.parent.exists() for path in saved_paths))
        for unsupported in (object(), {1, 2}, b"arbitrary bytes", {"bad": object()},
                            {"bad": float("nan")}, {"bad": torch.tensor(float("inf"))},
                            {"bad": torch.tensor([1.0], dtype=torch.float64)}):
            with self.subTest(unsupported_type=type(unsupported).__name__):
                with self.assertRaises(ValueError):
                    verifier._state_fingerprint(unsupported)
                with patch.object(torch, "save", side_effect=AssertionError("Reject before serialization")):
                    with self.assertRaises(ValueError):
                        verifier._serialization_probe(unsupported)
        with patch.object(torch, "load", return_value={"unexpected": object()}):
            with self.assertRaises(ValueError):
                verifier._serialization_probe(original)

    def test_local_observation_schema_rejects_missing_tensors_and_rehashed_shapes(self):
        observations = verifier.run_fixed_probe("cpu")["observations"]
        verifier._validate_observations(observations, "cpu", intended=False)
        changed = copy.deepcopy(observations)
        changed["comparisons"]["repeat_cpu"].pop()
        with self.assertRaises(ValueError):
            verifier._validate_observations(changed, "cpu", intended=False)
        changed = copy.deepcopy(observations)
        comparison = changed["comparisons"]["repeat_cpu"][0]
        comparison["reference"]["shape"] = [1]
        comparison["observed"]["shape"] = [1]
        with self.assertRaises(ValueError):
            verifier._validate_observations(changed, "cpu", intended=False)
        changed = copy.deepcopy(observations)
        comparison = next(record for record in changed["comparisons"]["repeat_cpu"] if record["name"] == "eval.output")
        comparison["reference"]["sha256"] = "f" * 64
        comparison["observed"]["sha256"] = "f" * 64
        with self.assertRaises(ValueError):
            verifier._validate_observations(changed, "cpu", intended=False)

    def test_comparison_refuses_wrong_keys_shapes_dtypes_and_nonfinite_values(self):
        reference = {"output": torch.tensor([0.25, 0.75], dtype=torch.float32)}
        alternatives = (
            {}, {"different": reference["output"]},
            {"output": torch.zeros(2, 1)},
            {"output": reference["output"].double()},
            {"output": torch.tensor([float("nan"), 0.75])},
            {"output": torch.tensor([float("inf"), 0.75])},
        )
        for alternative in alternatives:
            with self.subTest(alternative=alternative):
                with self.assertRaises(ValueError):
                    verifier.compare_tensors(reference, alternative)
        with self.assertRaises(ValueError):
            verifier.compare_tensors({"output": torch.tensor([float("nan"), 0.75])}, reference)

    def test_p100_rejects_actual_local_runtime_before_device_execution(self):
        with patch.object(torch.cuda, "init", side_effect=AssertionError("Must fail before CUDA init")):
            with self.assertRaises(ValueError):
                verifier.require_p100_device()

    def test_mocked_gpu_failure_never_publishes_evidence(self):
        with (
            patch.object(torch, "__version__", "2.14.0+cu126"),
            patch.object(torch.version, "cuda", "12.6"),
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "device_count", return_value=2),
            patch.object(environment, "publish", side_effect=AssertionError("Mocked GPU publication forbidden")),
        ):
            with self.assertRaises(ValueError):
                verifier.require_p100_device()

        inventory = synthetic_inventory()
        payloads = synthetic_verification_payloads(inventory)
        cpu_schema = payloads["cpu_verification"]
        p100_schema = payloads["p100_verification"]
        _, inputs, _, _ = verifier.fixed_fixture()
        tensors = {}
        for name, shape in verifier._comparison_shapes().items():
            tensors[name] = inputs.clone() if name.endswith(".input") else torch.zeros(shape, dtype=torch.float32)
        forward_outputs = {name: tensors[name] for name in ("eval.output", "eval_after.output", "train.output")}
        reverse_outputs = copy.deepcopy(forward_outputs)
        trajectory = {
            "tensors": tensors,
            "rc": verifier.compare_tensors(forward_outputs, reverse_outputs),
            "state_sha256": environment.digest("mocked-unaccepted-trajectory"),
        }
        gpu_trajectory = copy.deepcopy(trajectory)
        gpu_trajectory["tensors"]["gradient.W"][0, 0, 0] = 1.0
        rc_outputs = copy.deepcopy(reverse_outputs)
        rc_outputs["eval.output"][0, 0] = 0.125
        rc_trajectory = copy.deepcopy(trajectory)
        rc_trajectory["rc"] = verifier.compare_tensors(forward_outputs, rc_outputs)
        for name in forward_outputs:
            self.assertEqual(rc_outputs[name].dtype, torch.float32)
            self.assertEqual(rc_outputs[name].shape, forward_outputs[name].shape)
            self.assertTrue(bool(torch.isfinite(rc_outputs[name]).all()))
            self.assertEqual(int((rc_outputs[name] != forward_outputs[name]).sum()),
                             1 if name == "eval.output" else 0)
        self.assertEqual(float(forward_outputs["eval.output"][0, 0]), 0.0)
        self.assertEqual(float(rc_outputs["eval.output"][0, 0]), 0.125)
        self.assertGreater(0.125, 1e-6 + 1e-5 * abs(float(forward_outputs["eval.output"][0, 0])))
        payload_bytes = environment.canonical_bytes(payloads)
        trajectory_inputs = [trajectory, gpu_trajectory, rc_trajectory, forward_outputs, reverse_outputs, rc_outputs]
        trajectory_hashes = verifier._state_fingerprint(trajectory_inputs)
        original_installation = verifier._installation
        original_finish = verifier._finish_verification
        reports = []

        def artifact_snapshot(root):
            snapshot = {}
            for path in root.iterdir():
                metadata = path.lstat()
                snapshot[path.name] = (
                    path.read_bytes(), metadata.st_dev, metadata.st_ino, metadata.st_mode,
                    metadata.st_nlink, metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns,
                )
            return snapshot

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            inventory_path = root / "inventory.json"
            inventory_path.write_bytes(environment.canonical_bytes(inventory) + b"\n")
            inventory_bytes = inventory_path.read_bytes()
            artifacts_before = artifact_snapshot(root)
            arguments = {
                "mode": "p100", "spec": INTENT_PATH, "prefix": root / "uninstalled-prefix",
                "inventory": inventory_path, "expected_environment_id": inventory["environment_id"],
                "acquisition_lock": root / "absent-acquisition-lock.json", "exports_root": root / "absent-exports",
                "expected_software_commit": p100_schema["software"]["runtime_commit"],
                "output": root / "unpublished-p100.json",
            }

            cases = (("p100", "baseline"), ("p100", "gradient"), ("cpu", "rc"), ("p100", "rc"))
            for mode, failure_kind in cases:
                arguments["mode"] = mode
                arguments["output"] = root / ("unpublished-" + mode + ".json")
                arguments_before = copy.deepcopy(arguments)
                schema = cpu_schema if mode == "cpu" else p100_schema
                installation_calls = []
                diagnostics = []
                captured_records = []

                def isolated_installation(*args, **kwargs):
                    installation_calls.append((args, kwargs))
                    if len(installation_calls) == 1:
                        return inventory, schema["execution"], schema["software"]
                    # Passing mocked kernels must still face the real installed
                    # environment gate, which this absent local prefix cannot pass.
                    return original_installation(*args, **kwargs)

                def mocked_trajectory(template, actual_inputs, targets, device, initial_rng):
                    self.assertEqual(verifier._tensor_record(actual_inputs), cpu_schema["fixture"]["input"])
                    self.assertEqual(device in ("cpu", "cuda:0"), True)
                    if failure_kind == "rc":
                        # Identical RC-only drift on both mocked devices keeps
                        # their forward/gradient and repeat comparisons valid.
                        return rc_trajectory
                    return gpu_trajectory if failure_kind == "gradient" and device == "cuda:0" else trajectory

                def failed_diagnostic_sink(path, record):
                    # The public operation permits failed diagnostics. Intercept
                    # that write; a successful mock must never reach this sink.
                    self.assertNotEqual(failure_kind, "rc", "RC subcase reached publication")
                    self.assertEqual(path, arguments["output"])
                    self.assertEqual(record["status"], "failed")
                    diagnostics.append(copy.deepcopy(record))

                def capture_rc_diagnostic(record, output):
                    if failure_kind != "rc":
                        return original_finish(record, output)
                    # Capture only the failed diagnostic before the production
                    # finisher's permitted diagnostic-publication boundary.
                    self.assertEqual(output, arguments["output"])
                    self.assertEqual(record["status"], "failed")
                    captured_records.append((record, environment.canonical_bytes(record)))
                    captured = copy.deepcopy(record)
                    captured["verification_id"] = verifier._verification_id(captured)
                    captured = environment.seal(captured)
                    verifier.validate_verification(captured)
                    diagnostics.append(copy.deepcopy(captured))
                    return captured

                runtime = {
                    "cpu": cpu_schema["observations"]["b3_environment"],
                    "cuda:0": p100_schema["observations"]["b3_environment"],
                }
                with (
                    self.subTest(mode=mode, failure_kind=failure_kind),
                    patch.object(verifier, "_installation", side_effect=isolated_installation),
                    patch.object(verifier, "_configure_runtime", side_effect=lambda device: runtime[device]),
                    patch.object(verifier, "require_p100_device", return_value=p100_schema["observations"]["gpu"]),
                    patch.object(verifier, "_trajectory", side_effect=mocked_trajectory) as trajectories,
                    patch.object(verifier, "run_fixed_probe", wraps=verifier.run_fixed_probe) as probe,
                    patch.object(verifier, "_finish_verification", side_effect=capture_rc_diagnostic) as finish,
                    patch.object(torch.cuda, "is_initialized", return_value=False),
                    patch.object(torch.cuda, "init", side_effect=AssertionError("Real CUDA initialization forbidden")),
                    patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("Real CUDA lazy initialization forbidden")),
                    patch.object(torch.cuda, "device_count", side_effect=AssertionError("Real CUDA enumeration forbidden")),
                    patch.object(torch.cuda, "synchronize"),
                    patch.object(torch.cuda, "reset_peak_memory_stats"),
                    patch.object(torch.cuda, "max_memory_allocated", return_value=1),
                    patch.object(torch.cuda, "max_memory_reserved", return_value=1),
                    patch.object(environment, "_run", side_effect=AssertionError("External command forbidden")),
                    patch.object(environment, "publish", side_effect=failed_diagnostic_sink) as publication,
                    patch.object(environment, "_exclusive_bytes", side_effect=AssertionError("Artifact write forbidden")) as artifact_writer,
                    patch.object(verifier, "finalize_environment", side_effect=AssertionError("Mock finalization forbidden")) as finalization,
                ):
                    report = verifier.verify_environment(**arguments)
                    verifier.validate_verification(report)
                    self.assertEqual([call.args[3] for call in trajectories.call_args_list],
                                     ["cpu", "cpu"] if mode == "cpu" else ["cpu", "cpu", "cuda:0", "cuda:0"])
                    probe.assert_called_once_with("cpu" if mode == "cpu" else "cuda:0")
                    finish.assert_called_once()
                    if failure_kind == "rc":
                        publication.assert_not_called()
                    else:
                        publication.assert_called_once()
                    artifact_writer.assert_not_called()
                    finalization.assert_not_called()
                self.assertEqual(diagnostics, [report])
                self.assertEqual(report["status"], "failed")
                self.assertEqual(report["mode"], mode)
                self.assertIs(report["checks"]["no_cuda_initialization" if mode == "cpu" else "device_contract"], True)
                failed_checks = [name for name, passed in report["checks"].items() if not passed]
                failed_comparisons = [
                    (group, item["name"]) for group, comparisons in report["observations"]["comparisons"].items()
                    for item in comparisons if not item["passed"]
                ]
                if failure_kind != "baseline":
                    self.assertEqual(len(installation_calls), 1)
                    self.assertEqual(report["failure"], {
                        "type": "EnvironmentError", "message": "Fixed-tensor numerical acceptance failed.",
                        "stage": "fixed_tensor_verification",
                    })
                    if failure_kind == "gradient":
                        self.assertEqual(failed_checks, ["cpu_gpu_tolerances"])
                        self.assertEqual(failed_comparisons, [("cpu_p100", "gradient.W")])
                        drift = next(item for item in report["observations"]["comparisons"]["cpu_p100"]
                                     if item["name"] == "gradient.W")
                        self.assertEqual(drift["mismatched_elements"], 1)
                        self.assertEqual(drift["max_absolute_error"], 1.0)
                        self.assertEqual((drift["atol"], drift["rtol"]), (1e-5, 1e-4))
                    else:
                        self.assertIs(report["checks"]["forward_rc_invariance"], False)
                        self.assertFalse(all(report["checks"].values()))
                        self.assertEqual(failed_checks, ["forward_rc_invariance"])
                        rc_groups = ("rc_cpu",) if mode == "cpu" else ("rc_cpu", "rc_p100")
                        self.assertEqual(failed_comparisons, [(group, "eval.output") for group in rc_groups])
                        comparisons = report["observations"]["comparisons"]
                        for group in rc_groups:
                            drift = next(item for item in comparisons[group] if item["name"] == "eval.output")
                            self.assertEqual(drift["mismatched_elements"], 1)
                            self.assertEqual(drift["max_absolute_error"], 0.125)
                            self.assertEqual((drift["atol"], drift["rtol"]), (1e-6, 1e-5))
                            self.assertEqual(drift["reference"], verifier._tensor_record(forward_outputs["eval.output"]))
                            self.assertEqual(drift["observed"], verifier._tensor_record(rc_outputs["eval.output"]))
                        if mode == "p100":
                            self.assertIs(report["checks"]["cpu_gpu_tolerances"], True)
                            self.assertEqual(comparisons["rc_cpu"], comparisons["rc_p100"])
                            self.assertEqual(len(comparisons["cpu_p100"]), len(verifier._comparison_names()))
                            for comparison in comparisons["cpu_p100"]:
                                self.assertIs(comparison["passed"], True)
                                self.assertEqual(comparison["reference"], comparison["observed"])
                                self.assertEqual(comparison["max_absolute_error"], 0.0)
                            baseline_comparisons = reports[0]["observations"]["comparisons"]
                            for group in ("repeat_cpu", "repeat_p100", "cpu_p100"):
                                self.assertEqual(comparisons[group], baseline_comparisons[group])
                else:
                    self.assertEqual(failed_checks, [])
                    self.assertEqual(failed_comparisons, [])
                    self.assertIs(report["checks"]["forward_rc_invariance"], True)
                    self.assertEqual(len(installation_calls), 2)
                    self.assertEqual(report["failure"]["stage"], "post_verification_identity")
                    self.assertEqual(report["failure"]["type"], "EnvironmentError")
                    self.assertEqual(report["failure"]["message"], "Linux x86_64 is required.")
                reports.append(report)
                self.assertEqual(inventory_path.read_bytes(), inventory_bytes)
                self.assertEqual(list(root.iterdir()), [inventory_path])
                self.assertEqual(artifact_snapshot(root), artifacts_before)
                self.assertEqual(environment.canonical_bytes(payloads), payload_bytes)
                self.assertEqual(verifier._state_fingerprint(trajectory_inputs), trajectory_hashes)
                self.assertEqual(arguments, arguments_before)
                for record, contents in captured_records:
                    self.assertEqual(environment.canonical_bytes(record), contents)

            baseline, failed, rc_cpu, rc_p100 = reports
            for report in (failed, rc_p100):
                for field in ("environment_id", "inventory_sha256", "software", "fixture", "execution"):
                    self.assertEqual(report[field], baseline[field])
                for field in ("b3_environment", "numpy_config_sha256", "gpu", "state_sha256", "memory"):
                    self.assertEqual(report["observations"][field], baseline["observations"][field])
            for report, failed_check in ((failed, "cpu_gpu_tolerances"),
                                         (rc_cpu, "forward_rc_invariance"), (rc_p100, "forward_rc_invariance")):
                failed_bytes = environment.canonical_bytes(report)
                for overwrite_failed_check in (False, True):
                    forged = copy.deepcopy(report)
                    forged.pop("manifest_hash")
                    forged["status"], forged["failure"] = "successful", None
                    if overwrite_failed_check:
                        forged["checks"][failed_check] = True
                    with self.subTest(mode=report["mode"], failed_check=failed_check,
                                      overwrite_failed_check=overwrite_failed_check), \
                            patch.object(environment, "publish") as publication, \
                            patch.object(environment, "_exclusive_bytes") as artifact_writer:
                        with self.assertRaisesRegex(ValueError, r"^Failed tensor comparison in successful evidence\.$"):
                            verifier._finish_verification(forged, arguments["output"])
                        publication.assert_not_called()
                        artifact_writer.assert_not_called()
                    self.assertEqual(environment.canonical_bytes(report), failed_bytes)
                    self.assertEqual(artifact_snapshot(root), artifacts_before)

            # Inert failed diagnostic fixtures only; real finalization must
            # reject their freshly sealed semantic evidence before publication.
            cpu_path = root / "mocked-rc-failed-cpu.json"
            p100_path = root / "mocked-rc-failed-p100.json"
            for path, report in ((cpu_path, rc_cpu), (p100_path, rc_p100)):
                verifier.validate_verification(report)
                path.write_bytes(environment.canonical_bytes(report) + b"\n")
                path.chmod(0o400)
            finalization_arguments = {key: value for key, value in arguments.items() if key not in ("mode", "output")}
            finalization_arguments.update(cpu_verification=cpu_path, p100_verification=p100_path,
                                          output=root / "never-resolved.json")
            finalization_before = copy.deepcopy(finalization_arguments)
            artifacts_before = artifact_snapshot(root)
            reports_before = environment.canonical_bytes(reports)
            loaded_records = []
            original_reader = environment.read_manifest

            def read_unchanged_record(path):
                record = original_reader(path)
                loaded_records.append((record, environment.canonical_bytes(record)))
                return record

            with (
                patch.object(verifier, "_installation", return_value=(inventory, cpu_schema["execution"],
                                                                      cpu_schema["software"])) as installation,
                patch.object(environment, "read_manifest", side_effect=read_unchanged_record),
                patch.object(verifier, "_evidence_pair", wraps=verifier._evidence_pair) as evidence_pair,
                patch.object(verifier, "validate_verification", wraps=verifier.validate_verification) as validation,
                patch.object(environment, "publish", side_effect=AssertionError("RC publication forbidden")) as publication,
                patch.object(environment, "_exclusive_bytes", side_effect=AssertionError("RC artifact write forbidden")) as artifact_writer,
                patch.object(verifier, "validate_resolved", side_effect=AssertionError("Mock acceptance forbidden")) as resolved,
                patch.object(verifier, "run_fixed_probe", side_effect=AssertionError("Numerical fallback forbidden")) as probe,
                patch.object(environment, "verify_software", side_effect=AssertionError("Source fallback forbidden")) as source,
                patch.object(environment, "revalidate_inventory", side_effect=AssertionError("Install repair forbidden")) as repair,
                patch.object(environment, "_run", side_effect=AssertionError("External command forbidden")) as command,
                patch.object(torch.cuda, "init", side_effect=AssertionError("Real CUDA initialization forbidden")) as cuda_init,
                patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("Real CUDA lazy initialization forbidden")) as cuda_lazy,
            ):
                with self.assertRaisesRegex(environment.EnvironmentError,
                                            r"^Finalization requires existing successful real CPU and P100 evidence\.$"):
                    verifier.finalize_environment(**finalization_arguments)
                installation.assert_called_once()
                evidence_pair.assert_called_once_with(inventory, inventory_path, cpu_path, p100_path,
                                                      arguments["expected_software_commit"])
                validation.assert_called_once_with(rc_cpu)
                for boundary in (publication, artifact_writer, resolved, probe, source, repair, command, cuda_init, cuda_lazy):
                    boundary.assert_not_called()
            self.assertEqual(len(loaded_records), 2)
            for record, contents in loaded_records:
                self.assertEqual(environment.canonical_bytes(record), contents)
            self.assertEqual(finalization_arguments, finalization_before)
            self.assertEqual(environment.canonical_bytes(reports), reports_before)
            self.assertEqual(environment.canonical_bytes(payloads), payload_bytes)
            self.assertEqual(verifier._state_fingerprint(trajectory_inputs), trajectory_hashes)
            self.assertEqual(artifact_snapshot(root), artifacts_before)
            for name in ("unpublished-cpu.json", "unpublished-p100.json", "never-resolved.json"):
                self.assertFalse(os.path.lexists(root / name))

    def test_p100_rejects_each_device_contract_mismatch(self):
        cases = (
            ("runtime", "12.8"), ("name", "Tesla V100-PCIE-16GB"),
            ("capability", (7, 0)), ("architectures", ["sm_70", "sm_80"]),
            ("available", False),
        )
        for field, value in cases:
            actual = {
                "runtime": "12.6", "name": "Tesla P100-PCIE-16GB",
                "capability": (6, 0), "architectures": ["sm_60"], "available": True,
            }
            actual[field] = value
            with self.subTest(field=field):
                with (
                    patch.object(torch, "__version__", "2.14.0+cu126"),
                    patch.object(torch.version, "cuda", actual["runtime"]),
                    patch.object(torch.cuda, "is_available", return_value=actual["available"]),
                    patch.object(torch.cuda, "device_count", return_value=1),
                    patch.object(torch.cuda, "get_device_name", return_value=actual["name"]),
                    patch.object(torch.cuda, "get_device_capability", return_value=actual["capability"]),
                    patch.object(torch.cuda, "get_arch_list", return_value=actual["architectures"]),
                    patch.object(environment, "publish", side_effect=AssertionError("Mocked GPU publication forbidden")),
                ):
                    with self.assertRaises(ValueError):
                        verifier.require_p100_device()


class FailedEvidenceTests(TemporaryCase):
    def setUp(self):
        super().setUp()
        guard_b4a_imports(self)

    def failed_record(self, mode="cpu"):
        record = verifier._empty_verification(mode)
        record["failure"] = {"type": "SyntheticFailure", "message": "No real installed CARC environment", "stage": "preflight"}
        record["verification_id"] = verifier._verification_id(record)
        return environment.seal(record)

    def update_record(self, record):
        changed = copy.deepcopy(record)
        changed.pop("manifest_hash", None)
        changed["verification_id"] = verifier._verification_id(changed)
        return environment.seal(changed)

    def test_failed_cpu_and_p100_evidence_has_strict_recursive_schema(self):
        for mode in ("cpu", "p100"):
            original = self.failed_record(mode)
            verifier.validate_verification(original)
            for location in ((), ("failure",)):
                for mutation in ("unknown", "missing"):
                    changed = copy.deepcopy(original)
                    target = at_location(changed, location)
                    if mutation == "unknown":
                        target["unapproved"] = True
                    else:
                        target.pop("mode" if not location else "stage")
                    with self.subTest(mode=mode, location=location, mutation=mutation):
                        with self.assertRaises(ValueError):
                            verifier.validate_verification(self.update_record(changed))
            changed = copy.deepcopy(original)
            changed["checks"] = {"mocked_gpu_accepted": True}
            with self.assertRaises(ValueError):
                verifier.validate_verification(self.update_record(changed))

    def test_rehashing_fake_success_does_not_create_valid_verification(self):
        for mode in ("cpu", "p100"):
            record = self.failed_record(mode)
            record["status"] = "successful"
            record["failure"] = None
            record["checks"] = {"device_contract": True}
            with self.subTest(mode=mode):
                with self.assertRaises(ValueError):
                    verifier.validate_verification(self.update_record(record))

    def test_finalize_rejects_existing_failed_synthetic_evidence_without_publication(self):
        inventory = synthetic_inventory()
        inventory_path = self.root / "inventory.json"
        cpu_path = self.root / "cpu-failed.json"
        p100_path = self.root / "p100-mocked-failed.json"
        environment.publish(inventory_path, inventory)
        environment.publish(cpu_path, self.failed_record("cpu"))
        environment.publish(p100_path, self.failed_record("p100"))
        output = self.root / "resolved.json"
        before = {path.name: path.read_bytes() for path in self.root.iterdir()}
        with patch.object(verifier, "_installation", return_value=(inventory, {}, {})), \
             patch.object(environment, "publish", side_effect=AssertionError("Unexpected publication")) as publish, \
             patch.object(environment, "_exclusive_bytes", side_effect=AssertionError("Unexpected artifact write")) as writer:
            with self.assertRaisesRegex(ValueError, "successful real CPU and P100"):
                verifier.finalize_environment(
                    spec=INTENT_PATH, prefix=self.root / "uninstalled-prefix", inventory=inventory_path,
                    expected_environment_id=inventory["environment_id"], acquisition_lock=self.root / "absent-lock.json",
                    exports_root=self.root / "absent-exports", expected_software_commit="a" * 40,
                    cpu_verification=cpu_path, p100_verification=p100_path, output=output,
                )
            publish.assert_not_called()
            writer.assert_not_called()
        self.assertFalse(output.exists())
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.root.iterdir()})

        # These success-status records are only schema fixtures. The installed
        # prefix and source facts are synthetic; no runtime verifier runs and
        # neither a resolved record nor accepted publication may be produced.
        baseline = synthetic_verification_payloads(inventory)
        baseline_bytes = environment.canonical_bytes(baseline)
        software = baseline["cpu_verification"]["software"]
        expected_commit = software["runtime_commit"]

        def write_schema_evidence(label, payloads):
            root = self.root / ("unaccepted-schema-only-" + label)
            root.mkdir()
            paths = {}
            for name, record in payloads.items():
                path = root / (name + ".json")
                path.write_bytes(environment.canonical_bytes(record) + b"\n")
                path.chmod(0o400)
                paths[name] = path
            return root, paths

        def evidence_snapshot(root):
            snapshot = {}
            for path in root.iterdir():
                metadata = path.lstat()
                snapshot[path.name] = (
                    path.read_bytes(), metadata.st_dev, metadata.st_ino,
                    metadata.st_mode, metadata.st_nlink, metadata.st_size,
                    metadata.st_mtime_ns, metadata.st_ctime_ns,
                )
            return snapshot

        def assert_finalization_rejected(label, payloads, runtime_software, message,
                                         pair_baseline=False):
            inputs_before = environment.canonical_bytes(payloads)
            installed_context = copy.deepcopy(payloads["inventory"]["execution"])
            runtime_before = environment.canonical_bytes([installed_context, runtime_software])
            root, paths = write_schema_evidence(label, payloads)
            files_before = evidence_snapshot(root)
            arguments = {
                "spec": INTENT_PATH, "prefix": self.root / "uninstalled-prefix",
                "inventory": paths["inventory"], "expected_environment_id": inventory["environment_id"],
                "acquisition_lock": self.root / "absent-lock.json", "exports_root": self.root / "absent-exports",
                "expected_software_commit": expected_commit, "cpu_verification": paths["cpu_verification"],
                "p100_verification": paths["p100_verification"], "output": root / "never-resolved.json",
            }
            arguments_before = copy.deepcopy(arguments)
            loaded_records = []
            real_reader = environment.read_manifest

            def read_unchanged_manifest(path):
                record = real_reader(path)
                loaded_records.append((record, environment.canonical_bytes(record)))
                return record

            with (
                patch.object(verifier, "_installation", return_value=(payloads["inventory"], installed_context,
                                                                      runtime_software)) as installation,
                patch.object(environment, "read_manifest", side_effect=read_unchanged_manifest),
                patch.object(verifier, "_evidence_pair", wraps=verifier._evidence_pair) as evidence_pair,
                patch.object(verifier, "validate_verification", wraps=verifier.validate_verification) as validation,
                patch.object(environment, "publish", side_effect=AssertionError("Unexpected publication")) as publish,
                patch.object(environment, "_exclusive_bytes", side_effect=AssertionError("Unexpected artifact write")) as writer,
                patch.object(verifier, "validate_resolved", side_effect=AssertionError("Unexpected resolved acceptance")) as resolved,
                patch.object(verifier, "run_fixed_probe", side_effect=AssertionError("Unexpected numerical fallback")) as probe,
                patch.object(environment, "verify_software", side_effect=AssertionError("Unexpected source fallback")) as source,
                patch.object(environment, "revalidate_inventory", side_effect=AssertionError("Unexpected install fallback")) as repair,
            ):
                environment.validate_inventory(payloads["inventory"])
                for name in ("cpu_verification", "p100_verification"):
                    record = payloads[name]
                    self.assertEqual(record["status"], "successful")
                    self.assertIsNone(verifier.validate_verification(record))
                if pair_baseline:
                    # The real pair validator accepts the matching schema
                    # relationships without creating any resolved environment.
                    self.assertEqual(verifier._evidence_pair(
                        payloads["inventory"], paths["inventory"], paths["cpu_verification"],
                        paths["p100_verification"], expected_commit,
                    ), (payloads["cpu_verification"], payloads["p100_verification"]))
                    evidence_pair.reset_mock()
                validation.reset_mock()
                with self.assertRaises(environment.EnvironmentError) as raised:
                    verifier.finalize_environment(**arguments)
                self.assertIs(type(raised.exception), environment.EnvironmentError)
                self.assertEqual(str(raised.exception), message)
                installation.assert_called_once()
                evidence_pair.assert_called_once_with(
                    payloads["inventory"], paths["inventory"], paths["cpu_verification"],
                    paths["p100_verification"], expected_commit,
                )
                self.assertEqual(validation.call_count, 2)
                for boundary in (publish, writer, resolved, probe, source, repair):
                    boundary.assert_not_called()
            self.assertEqual(arguments, arguments_before)
            self.assertEqual(environment.canonical_bytes(payloads), inputs_before)
            self.assertEqual(environment.canonical_bytes([installed_context, runtime_software]), runtime_before)
            self.assertEqual(environment.canonical_bytes(baseline), baseline_bytes)
            for record, contents in loaded_records:
                self.assertEqual(environment.canonical_bytes(record), contents)
            self.assertFalse(os.path.lexists(arguments["output"]))
            self.assertEqual(evidence_snapshot(root), files_before)

        # A separate finalizer-source disagreement keeps even the matching
        # CPU/P100 baseline ineligible for acceptance, after all pair checks.
        finalizer_software = copy.deepcopy(software)
        finalizer_software["source_inventory"][0]["sha256"] = "9" * 64
        assert_finalization_rejected(
            "matching-pair-finalizer-source", baseline, finalizer_software,
            "Executing finalizer source differs.", pair_baseline=True,
        )

        cases = (
            ("inventory-identity", ("environment_id",), "env_" + "9" * 64,
             "Verification does not bind the exact installed inventory."),
            ("inventory-bytes", ("inventory_sha256",), "9" * 64,
             "Verification does not bind the exact installed inventory."),
            ("installation-executable", ("execution", "python_executable"),
             inventory["execution"]["prefix"] + "/bin/python3.11",
             "CPU/P100 verification belongs to a different installation."),
            ("producer-commit", ("software", "runtime_commit"), "9" * 40,
             "Verification producer commit differs."),
            ("source-inventory", ("software", "source_inventory"), software["source_inventory"][1:],
             "CPU/P100 source or fixture evidence differs."),
            ("source-hash", ("software", "source_inventory", 0, "sha256"), "9" * 64,
             "CPU/P100 source or fixture evidence differs."),
            ("fixture-fingerprint", ("fixture", "initial_state_sha256"), "9" * 64,
             "CPU/P100 source or fixture evidence differs."),
        )
        for label, location, value, message in cases:
            with self.subTest(finalize_relationship=label):
                payloads = copy.deepcopy(baseline)
                changed = payloads["p100_verification"]
                self.assertNotEqual(at_location(changed, location), value)
                at_location(changed, location[:-1])[location[-1]] = copy.deepcopy(value)
                # Only this property and its derived verification/seal hashes
                # change; fresh file references bind the recomputed bytes.
                payloads["p100_verification"] = self.update_record(changed)
                expected_change = copy.deepcopy(baseline["p100_verification"])
                at_location(expected_change, location[:-1])[location[-1]] = copy.deepcopy(value)
                for field in ("verification_id", "manifest_hash"):
                    self.assertNotEqual(payloads["p100_verification"][field], expected_change[field])
                    expected_change[field] = payloads["p100_verification"][field]
                self.assertEqual(payloads["p100_verification"], expected_change)
                assert_finalization_rejected(label, payloads, copy.deepcopy(software), message)

    def test_changed_reference_bytes_are_not_accepted_as_original_evidence(self):
        artifact = self.root / "failed.json"
        environment.publish(artifact, self.failed_record())
        reference = environment.file_reference(artifact)
        artifact.chmod(0o644)
        artifact.write_bytes(artifact.read_bytes() + b" ")
        with self.assertRaises(ValueError):
            verifier._reference_from_sibling(self.root, reference)


class CommittedProvenanceTests(TemporaryCase):
    """Public gates run against real committed source bytes in temporary trees."""

    def setUp(self):
        super().setUp()
        self.checkout = self.root / "checkout"
        self.checkout.mkdir()
        for relative in environment.SOURCE_PATHS:
            self.assertFalse(relative.startswith(("data/", "results/", "checkpoints/", "plots/")))
            destination = self.checkout / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PROJECT_ROOT / relative, destination)
        self.inert_sources = (
            "src/exd_hox_dataset.py", "src/downstream_run.py",
            "src/downstream_checkpoint.py", "src/cnn_rc_training.py",
        )
        for relative in self.inert_sources:
            (self.checkout / relative).write_text(
                "raise AssertionError('Inert provenance source must never execute: " + relative + "')\n"
            )
        self.git("init", "--quiet")
        self.git("config", "user.name", "Synthetic B4a Test")
        self.git("config", "user.email", "synthetic-b4a@example.invalid")
        self.git("add", "--", ".")
        self.git("commit", "--quiet", "-m", "Temporary B4a source provenance fixture")
        self.commit = self.git("rev-parse", "HEAD").strip()
        self.sentinels = (
            "data/raw/pbm/sentinel.txt", "data/raw/flex_tables/sentinel.yaml",
            "data/processed/sentinel.txt", "data/sealed/sentinel.txt", "plots/sentinel.txt",
        )
        for relative in self.sentinels:
            destination = self.checkout / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"Synthetic unrelated sentinel: never opened by B4a.\n")

    def git(self, *arguments):
        result = subprocess.run(
            ["git", "-C", str(self.checkout), *arguments],
            capture_output=True, text=True, check=True, timeout=30,
        )
        return result.stdout

    def run_script(self, source):
        runtime = dict(os.environ)
        runtime.pop("PYTHONPATH", None)
        runtime["PYTHONDONTWRITEBYTECODE"] = "1"
        runtime["PYTHONNOUSERSITE"] = "1"
        runtime["CUDA_VISIBLE_DEVICES"] = ""
        runtime["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            runtime[name] = "1"
        for name in tuple(runtime):
            if name.startswith("SLURM_"):
                runtime.pop(name)
        harness = textwrap.dedent("""\
            import builtins
            import importlib
            import json
            import os
            from pathlib import Path
            import runpy
            import sys
            forbidden_modules = ('src.downstream_run', 'src.downstream_checkpoint', 'src.cnn_rc_training',
                                 'src.exd_hox_dataset', 'src.exd_hox_splits', 'src.sealed_test_access')
            def forbidden_import_name(name):
                return any(name.lstrip('.') == module or name.lstrip('.') == module.split('.')[-1]
                           or name.lstrip('.').startswith(module + '.') for module in forbidden_modules)
            original_import = builtins.__import__
            original_dynamic_import = importlib.import_module
            def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
                requested = [name] + [name + '.' + item for item in (fromlist or ()) if isinstance(item, str)]
                if any(forbidden_import_name(item) for item in requested):
                    raise AssertionError('Forbidden B4a import: ' + repr(requested))
                return original_import(name, globals, locals, fromlist, level)
            def guarded_dynamic_import(name, package=None):
                resolved = importlib.util.resolve_name(name, package) if name.startswith('.') else name
                if forbidden_import_name(resolved):
                    raise AssertionError('Forbidden B4a dynamic import: ' + resolved)
                return original_dynamic_import(name, package)
            class ForbiddenFinder:
                def find_spec(self, fullname, path=None, target=None):
                    if forbidden_import_name(fullname):
                        raise AssertionError('Forbidden B4a module loading: ' + fullname)
                    return None
            builtins.__import__ = guarded_import
            importlib.import_module = guarded_dynamic_import
            sys.meta_path.insert(0, ForbiddenFinder())
            assert not any(forbidden_import_name(name) for name in sys.modules)
            def deny_external_access(event, arguments):
                if event.startswith('socket.'):
                    raise AssertionError('Network access forbidden in B4a tests.')
                if event == 'open' and isinstance(arguments[0], (str, bytes, os.PathLike)):
                    path = Path(os.fsdecode(arguments[0]))
                    absolute = Path(os.path.abspath(path))
                    if path.suffix.lower() in {'.h5', '.hdf5'} or 'sealed' in path.parts:
                        raise AssertionError('Biological data access forbidden in B4a tests.')
                    for root in forbidden_data_roots:
                        if absolute == root or root in absolute.parents:
                            raise AssertionError('Biological data access forbidden in B4a tests.')
            sys.addaudithook(deny_external_access)
            """)
        roots = [str(PROJECT_ROOT / "data"), str(self.checkout / "data"), str(self.checkout / "plots")]
        harness = "from pathlib import Path\nforbidden_data_roots = tuple(Path(value) for value in " + repr(roots) + ")\n" + harness
        result = subprocess.run(
            [sys.executable, "-B", "-c", harness + "\n" + textwrap.dedent(source)
             + "\nassert not any(forbidden_import_name(name) for name in sys.modules)\n"],
            cwd=self.checkout, env=runtime, capture_output=True, text=True,
            check=False, timeout=120,
        )
        for diagnostic in ("Forbidden B4a", "Biological data access forbidden", "Network access forbidden",
                           "Inert provenance source must never execute"):
            self.assertNotIn(diagnostic, result.stdout + result.stderr)
        return result

    def cli(self, arguments):
        source = "sys.argv = " + repr([VERIFIER_MODULE] + list(arguments)) + "\n"
        source += "runpy.run_module(" + repr(VERIFIER_MODULE) + ", run_name='__main__')\n"
        return self.run_script(source)

    def test_clean_actual_committed_sources_are_verifiable(self):
        source = "expected_commit = " + repr(self.commit) + "\n"
        source += "allowlist = " + repr(self.inert_sources[:1] + self.inert_sources[2:]) + "\n"
        source += textwrap.dedent("""\
            from scripts.carc import cnn_rc_environment as env
            from unittest.mock import patch
            software = env.verify_software(expected_commit)
            original_read = env.read_regular
            original_run = env.subprocess.run
            opened, commands = [], []
            def allowlisted_read(path):
                relative = Path(path).relative_to(Path.cwd()).as_posix()
                assert relative in allowlist, relative
                opened.append(relative)
                return original_read(path)
            def observe_git(command, **keywords):
                assert type(command) is list and command[0] == 'git'
                assert not keywords.get('shell', False)
                assert 'ls-files' not in command and 'grep' not in command
                if 'ls-tree' in command:
                    assert command[-1] in allowlist
                    assert command[-2] == '--'
                commands.append(command)
                return original_run(command, **keywords)
            with patch.object(env, 'read_regular', side_effect=allowlisted_read), \
                 patch.object(env.subprocess, 'run', side_effect=observe_git), \
                 patch.object(Path, 'rglob', side_effect=AssertionError('Broad traversal forbidden')), \
                 patch.object(os, 'walk', side_effect=AssertionError('Broad traversal forbidden')):
                small = env._b4a_verify_runtime_sources(Path.cwd(), expected_commit, allowlist)
                env._b4a_verify_historical_sources(Path.cwd(), small, allowlist)
            assert opened == sorted(allowlist)
            assert commands
            print(json.dumps({'software': software, 'small': small}))
            """)
        result = self.run_script(source)
        self.assertEqual(result.returncode, 0, result.stderr)
        evidence = json.loads(result.stdout)
        software = evidence["software"]
        self.assertIn(self.commit, json.dumps(software))
        self.assertNotIn(str(self.checkout), json.dumps(software))
        self.assertEqual([record["path"] for record in software["source_inventory"]], sorted(environment.SOURCE_PATHS))
        self.assertEqual(len(evidence["small"]["source_inventory"]), 3)
        for record in evidence["small"]["source_inventory"]:
            raw = (self.checkout / record["path"]).read_bytes()
            self.assertEqual(set(record), {"path", "git_blob", "byte_size", "sha256"})
            self.assertEqual(record["git_blob"], self.git("rev-parse", self.commit + ":" + record["path"]).strip())
            self.assertEqual(record["byte_size"], len(raw))
            self.assertEqual(record["sha256"], hashlib.sha256(raw).hexdigest())
            self.assertTrue(raw.startswith(b"raise AssertionError"))

    def test_wrong_commit_dirty_or_untracked_source_cannot_claim_provenance(self):
        source = "from scripts.carc import cnn_rc_environment as env\n"
        wrong = self.run_script(source + "env.verify_software('" + "0" * 40 + "')")
        self.assertNotEqual(wrong.returncode, 0)
        candidate = self.checkout / "scripts/carc/cnn_rc_environment.py"
        original = candidate.read_bytes()
        candidate.write_bytes(original + b"\n# Dirty test-only candidate\n")
        dirty = self.run_script(source + "env.verify_software(" + repr(self.commit) + ")")
        self.assertNotEqual(dirty.returncode, 0)
        candidate.write_bytes(original)
        allowlist = list(self.inert_sources)
        collect = environment._b4a_verify_runtime_sources
        historical = environment._b4a_verify_historical_sources
        software = collect(self.checkout, self.commit, allowlist)
        for commit in (self.commit[:12], "A" * 40, "0" * 40, self.commit + "\n"):
            with self.subTest(commit=commit):
                with self.assertRaises(ValueError):
                    collect(self.checkout, commit, allowlist)
        for path in ("../outside.py", str(candidate), "src//exd_hox_dataset.py", "src/./exd_hox_dataset.py",
                     "src/../src/exd_hox_dataset.py", "src/missing.py"):
            with self.subTest(source_path=path):
                with self.assertRaises(ValueError):
                    collect(self.checkout, self.commit, [path])
        with self.assertRaises(ValueError):
            collect(self.checkout, self.commit, allowlist + [allowlist[0]])
        untracked = self.checkout / "src/untracked.py"
        untracked.write_text("raise AssertionError('untracked source must not execute')\n")
        with self.assertRaises(ValueError):
            collect(self.checkout, self.commit, ["src/untracked.py"])

        selected = self.checkout / allowlist[0]
        original = selected.read_bytes()
        selected.write_bytes(original + b"# staged synthetic mutation\n")
        self.git("add", "--", allowlist[0])
        selected.write_bytes(original)
        with self.assertRaisesRegex(ValueError, "Dirty tracked"):
            collect(self.checkout, self.commit, allowlist)
        self.git("add", "--", allowlist[0])
        original_mode = selected.stat().st_mode
        self.git("config", "core.fileMode", "false")
        selected.chmod(0o755)
        with self.assertRaisesRegex(ValueError, "mode differs"):
            collect(self.checkout, self.commit, allowlist)
        selected.chmod(original_mode)
        selected.unlink()
        selected.symlink_to(self.checkout / allowlist[1])
        with self.assertRaises(ValueError):
            collect(self.checkout, self.commit, allowlist)
        selected.unlink()
        selected.write_bytes(original)
        selected.chmod(original_mode)
        with self.assertRaisesRegex(ValueError, "Shadowed"):
            collect(self.checkout, self.commit, allowlist,
                    executing_paths={allowlist[0]: self.root / "shadowed.py"})
        for field, value in (("git_blob", "f" * 40), ("sha256", "f" * 64), ("byte_size", len(original) + 1)):
            changed = copy.deepcopy(software)
            changed["source_inventory"][0][field] = value
            with self.subTest(historical_field=field):
                with self.assertRaises(ValueError):
                    historical(self.checkout, changed, allowlist)
        missing = copy.deepcopy(software)
        missing["runtime_commit"] = "0" * 40
        with self.assertRaisesRegex(ValueError, "unavailable|invalid"):
            historical(self.checkout, missing, allowlist)
        # Historical committed bytes remain verifiable after the checkout changes.
        selected.write_bytes(original + b"# new temporary historical revision\n")
        self.git("add", "--", allowlist[0])
        self.git("commit", "--quiet", "-m", "Temporary source revision for historical evidence")
        historical(self.checkout, software, allowlist)
        with self.assertRaisesRegex(ValueError, "Wrong runtime HEAD"):
            collect(self.checkout, self.commit, allowlist)
        for output in (b"", b"not-newline-terminated", b"one\ntwo\n", b"bad\x00\n", b"\xff\n"):
            with self.subTest(plumbing_output=output):
                with self.assertRaises(ValueError):
                    environment._b4a_git_line(output)
        with patch.object(environment.subprocess, "run", return_value=SimpleNamespace(returncode=1, stdout=b"bad")):
            with self.assertRaises(ValueError):
                environment._b4a_git(self.checkout, "rev-parse", "HEAD")
        current_commit = self.git("rev-parse", "HEAD").strip()
        original_git = environment._b4a_git
        for mutation in ("top", "prefix", "tree_mode", "tree_path", "blob_hash", "blob_bytes"):
            def changed_git(root, *arguments, **keywords):
                output = original_git(root, *arguments, **keywords)
                if mutation == "top" and arguments == ("rev-parse", "--show-toplevel"):
                    return (str(self.root / "unrelated") + "\n").encode()
                if mutation == "prefix" and arguments == ("rev-parse", "--show-prefix"):
                    return b"unexpected/\n"
                if arguments[0] == "ls-tree" and mutation == "tree_mode":
                    return output.replace(b"100644 blob", b"120000 blob", 1)
                if arguments[0] == "ls-tree" and mutation == "tree_path":
                    return output.replace(b"\t", b"\tdifferent/", 1)
                if arguments[0] == "hash-object" and mutation == "blob_hash":
                    return b"ffffffffffffffffffffffffffffffffffffffff\n"
                if arguments[:2] == ("cat-file", "blob") and mutation == "blob_bytes":
                    return output + b"# changed plumbing bytes\n"
                return output
            with self.subTest(provenance_mutation=mutation), \
                    patch.object(environment, "_b4a_git", side_effect=changed_git):
                with self.assertRaises((ValueError, OSError)):
                    collect(self.checkout, current_commit, allowlist)

    def test_public_cli_has_exact_five_modes_and_no_test_bypass(self):
        for path in ("scripts/carc/cnn_rc_environment.py", "scripts/carc/verify_cnn_rc_environment.py"):
            with self.subTest(production_source=path):
                self.assertEqual(forbidden_imports((PROJECT_ROOT / path).read_text()), [])
        for snippet in (
            "import src.downstream_run as runs", "from src import downstream_checkpoint as checkpoints",
            "from .src import cnn_rc_training", "from .. import exd_hox_dataset",
            "import importlib as loader; loader.import_module('src.exd_hox_splits')",
            "from importlib import import_module as load; load('.sealed_test_access', 'src')",
            "__import__('src.downstream_checkpoint')",
            "__import__('src', fromlist=['downstream_run'])",
            "__import__('src', globals(), locals(), ['exd_hox_dataset'])",
            "import subprocess; subprocess.run(['python', '-c', 'from src import cnn_rc_training'])",
        ):
            with self.subTest(forbidden_syntax=snippet):
                self.assertTrue(forbidden_imports(snippet))
        self.assertEqual(forbidden_imports("SOURCE_PATHS = ('src/downstream_run.py', 'src/exd_hox_dataset.py')"), [])
        schema = self.run_script("""\
            from scripts.carc import cnn_rc_environment as env
            from scripts.carc import verify_cnn_rc_environment as verifier
            intent = env.load_intent(Path.cwd() / 'environments/carc_cnn_rc_v1.json')
            assert intent['verification']['repetitions'] == 2
            candidate = verifier._fixed_candidate()
            assert candidate['optimizer']['name'] == 'Adam'
            failed = verifier._empty_verification('cpu')
            assert failed['status'] == 'failed'
            assert 'torch' not in sys.modules and 'numpy' not in sys.modules
            """)
        self.assertEqual(schema.returncode, 0, schema.stderr)
        result = self.cli(["--help"])
        self.assertEqual(result.returncode, 0, result.stderr)
        for mode in ("inventory", "cpu", "p100", "check", "finalize"):
            self.assertIn(mode, result.stdout)
        for forbidden in ("--mock", "--synthetic", "--skip", "--test-only", "--allow-local"):
            self.assertNotIn(forbidden, result.stdout)
        result = self.cli(["cpu", "--specification", str(INTENT_PATH)])
        self.assertNotEqual(result.returncode, 0)

    def test_local_public_cpu_p100_and_finalize_never_publish_accepted_evidence(self):
        for mode in ("inventory", "cpu", "p100", "check", "finalize"):
            with self.subTest(mode=mode):
                output = self.root / (mode + ".json")
                arguments = [
                    mode, "--spec", str(self.checkout / "environments/carc_cnn_rc_v1.json"),
                    "--prefix", str(Path(sys.prefix)), "--expected-software-commit", self.commit,
                    "--acquisition-lock", str(self.root / "missing-lock.json"),
                    "--exports-root", str(self.root), "--output", str(output),
                ]
                if mode != "inventory":
                    arguments.extend([
                        "--inventory", str(self.root / "missing-inventory.json"),
                        "--expected-environment-id", "environment_" + "a" * 64,
                    ])
                if mode == "check":
                    arguments.extend([
                        "--environment", str(self.root / "missing-environment.json"), "--device", "cpu",
                    ])
                if mode == "finalize":
                    arguments.extend([
                        "--cpu-verification", str(self.root / "missing-cpu.json"),
                        "--p100-verification", str(self.root / "missing-p100.json"),
                    ])
                result = self.cli(arguments)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                if output.exists():
                    manifest = json.loads(output.read_bytes())
                    self.assertEqual(manifest.get("status"), "failed")

    def test_runtime_guards_block_network_and_biological_files(self):
        result = self.run_script("""\
            import socket
            for operation in (lambda: socket.socket(), lambda: Path('synthetic.hdf5').read_bytes(),
                              lambda: Path('sealed/target').read_bytes(), lambda: Path('data/anything').read_bytes(),
                              lambda: Path('plots/sentinel.txt').read_bytes(),
                              lambda: __import__('src.downstream_run'),
                              lambda: __import__('src', fromlist=['downstream_checkpoint']),
                              lambda: importlib.import_module('.cnn_rc_training', 'src')):
                try:
                    operation()
                except AssertionError:
                    pass
                else:
                    raise AssertionError('Forbidden operation was accepted.')
            """)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_real_cpu_probe_with_network_and_biological_access_tripwires(self):
        result = self.run_script("""\
            from scripts.carc import verify_cnn_rc_environment as verifier
            import torch
            from unittest.mock import patch
            model, inputs, targets, fixture = verifier.fixed_fixture()
            assert fixture['initial_state_sha256'] == verifier._state_fingerprint(model.state_dict())
            assert inputs.shape == (128, 14, 4) and targets.shape == (128, 1)
            with patch.object(torch.cuda, 'init', side_effect=AssertionError('CUDA must stay uninitialized')), \
                 patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('CUDA must stay uninitialized')):
                result = verifier.run_fixed_probe('cpu')
            assert set(result) == {'fixture', 'observations', 'checks'}
            assert 'status' not in result and 'manifest_hash' not in result
            assert result['checks']['controlled_serialization_round_trip'] is True
            assert all(result['checks'].values())
            assert not torch.cuda.is_initialized()
            """)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_unavailable_bootstrap_stops_and_preserves_cache_without_installing(self):
        source = "synthetic_root = Path(" + repr(str(self.root)) + ")\n"
        source += "expected_commit = " + repr(self.commit) + "\n"
        source += textwrap.dedent("""\
            from scripts.carc import cnn_rc_environment as env
            from unittest.mock import patch
            original_run = env._run
            calls = []
            def unavailable_binary(command, **keywords):
                if command[0] == 'git':
                    return original_run(command, **keywords)
                calls.append(command)
                assert command[:3] == ['conda', 'create', '--dry-run']
                raise env.EnvironmentError('Synthetic unavailable intended bootstrap binary')
            prefix = synthetic_root / 'uninstalled-prefix'
            cache = synthetic_root / 'preserved-cache'
            exports = synthetic_root / 'unwritten-exports'
            with patch.object(env, 'platform_facts', return_value={}), \
                 patch.object(env, '_module_guard', return_value=[]), \
                 patch.object(env, '_slurm_context', return_value={}), \
                 patch.object(env, '_run', side_effect=unavailable_binary):
                try:
                    env.acquire_environment(Path.cwd() / 'environments/carc_cnn_rc_v1.json', prefix,
                                            cache, exports, expected_commit)
                except env.EnvironmentError as error:
                    assert 'unavailable intended bootstrap' in str(error), str(error)
                else:
                    raise AssertionError('Unavailable package was accepted')
                assert not prefix.exists() and not exports.exists()
                assert sorted(path.name for path in cache.iterdir()) == ['conda', 'pip']
                assert len(calls) == 1
                try:
                    env.acquire_environment(Path.cwd() / 'environments/carc_cnn_rc_v1.json', prefix,
                                            cache, exports, expected_commit)
                except env.EnvironmentError as error:
                    assert 'Fresh prefix' in str(error), str(error)
                else:
                    raise AssertionError('Existing failed cache was repaired or reused')
                assert len(calls) == 1
                assert sorted(path.name for path in cache.iterdir()) == ['conda', 'pip']
            """)
        result = self.run_script(source)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
