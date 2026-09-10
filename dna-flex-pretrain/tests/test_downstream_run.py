"""Hermetic B3a identities, paired orientation and Git/blob evidence."""

import ast
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch

from src import downstream_run as runs
from src import downstream_checkpoint as checkpoints
from src.cnn_rc import CNNRC
from tests.cnn_rc_synthetic_support import SyntheticCase, rehash_run


class RunTests(SyntheticCase):
    def test_repeated_identity_and_only_public_data_entry(self):
        with patch.object(runs, "open_public_tf_data", wraps=runs.open_public_tf_data) as opened:
            self.assertEqual(self.resolve(), self.run)
        opened.assert_called_once_with(self.stage, transcription_factor="Ubx", expected_stage_id=self.stage_id)
        self.assertEqual(self.run["identity"]["selection"]["actual_logical_example_count"], 129)
        self.assertEqual(self.run["identity"]["budget"]["batches_per_epoch"], 2)
        self.assertEqual(self.run["identity"]["budget"]["maximum_updates"], 4)

    def test_relocated_stage_checkout_and_attempt_facts_preserve_identity(self):
        relocated_stage = self.root / "relocated-stage"
        relocated_checkout = self.root / "relocated-checkout"
        shutil.copytree(self.stage, relocated_stage)
        shutil.copytree(self.checkout, relocated_checkout)
        other = self.resolve(stage_root=relocated_stage, checkout_root=relocated_checkout,
                             config_path=relocated_checkout / "config.json")
        self.assertEqual(other, self.run)
        first = runs.attempt_id(self.run["run_id"], "1" * 32)
        second = runs.attempt_id(self.run["run_id"], "2" * 32)
        self.assertNotEqual(first, second)
        encoded = json.dumps(self.run)
        for forbidden in (str(self.root), "hostname", "slurm", "hardware", "attempt_id"):
            self.assertNotIn(forbidden, encoded)

    def test_seeds_levels_and_verified_code_change_identity(self):
        self.assertNotEqual(self.resolve(downstream_seed=33002)["run_id"], self.run["run_id"])
        self.assertNotEqual(self.resolve(level_id=self.level_ids[2])["run_id"], self.run["run_id"])
        (self.checkout / "entry.py").write_text("# Changed synthetic implementation.\n")
        self.git("add", "entry.py")
        self.git("commit", "--quiet", "-m", "Synthetic source change")
        commit = self.git("rev-parse", "HEAD").decode().strip()
        changed = self.resolve(expected_software_commit=commit)
        self.assertNotEqual(changed["run_id"], self.run["run_id"])

    def test_every_identity_leaf_is_hash_bound_or_rejected(self):
        def leaves(value, path=()):
            if type(value) is dict:
                for key, item in value.items():
                    yield from leaves(item, path + (key,))
            elif type(value) is list:
                for index, item in enumerate(value):
                    yield from leaves(item, path + (index,))
            else:
                yield path, value
        for path, value in leaves(self.run["identity"]):
            changed = copy.deepcopy(self.run)
            target = changed["identity"]
            for key in path[:-1]:
                target = target[key]
            replacements = {str: "changed", int: 999999, float: 0.123456789, bool: not value, type(None): "changed"}
            target[path[-1]] = replacements[type(value)]
            with self.subTest(path=path):
                self.assertNotEqual(runs.domain_hash("downstream_scientific_run.v1", changed["identity"]), self.run["run_id"][4:])
                with self.assertRaises((ValueError, TypeError, KeyError)):
                    runs.validate_run(changed)

    def test_strict_config_and_nested_schema_no_circular_commit(self):
        for raw in (b'{"a":1,"a":2}', b'{"x":NaN}', b'{"x":1e999}', b'[]', b'\xff'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                runs.strict_json(raw)
        cases = []
        for field in ("runtime_commit", "future_implementation_commit", "stage_root"):
            changed = copy.deepcopy(self.config)
            changed[field] = "unexpected"
            cases.append(changed)
        changed = copy.deepcopy(self.config)
        del changed["orientation"]
        cases.append(changed)
        changed = copy.deepcopy(self.config)
        changed["batching"]["num_workers"] = False
        cases.append(changed)
        changed = copy.deepcopy(self.config)
        changed["candidates"]["smoke_adam_v1"]["optimizer"]["weight_decay"] = float("nan")
        cases.append(changed)
        for changed in cases:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                runs.validate_config(changed)

    def test_new_tracked_config_matches_approved_default(self):
        path = Path(__file__).resolve().parents[1] / "configs/exd_hox_cnn_rc_v1.json"
        self.assertEqual(runs.strict_json(path.read_bytes()), self.original_config)

    def test_historical_sources_do_not_require_current_head_or_clean_files(self):
        (self.checkout / "entry.py").write_text("# Second version.\n")
        self.git("add", "entry.py")
        self.git("commit", "--quiet", "-m", "New historical fixture HEAD")
        (self.checkout / "entry.py").write_text("# Dirty current bytes.\n")
        runs.verify_historical_sources(self.checkout, self.run["identity"]["software"])
        with self.assertRaises(ValueError):
            runs.verify_run_sources(self.run, self.checkout)
        changed = copy.deepcopy(self.run["identity"]["software"])
        changed["source_inventory"][0]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            runs.verify_historical_sources(self.checkout, changed)

    def test_git_subdirectory_prefix_inventory(self):
        nested = self.checkout / "project"
        nested.mkdir()
        (nested / "source.py").write_text("# nested\n")
        self.git("add", "project/source.py")
        self.git("commit", "--quiet", "-m", "Nested fixture")
        commit = self.git("rev-parse", "HEAD").decode().strip()
        result = runs.verify_runtime_sources(nested, commit, ["source.py"])
        self.assertEqual(result["source_inventory"][0]["path"], "source.py")
        runs.verify_historical_sources(nested, result)

    def test_dirty_staged_untracked_inventory_wrong_head_shadow_and_symlink(self):
        with self.assertRaises(ValueError):
            runs.verify_runtime_sources(self.checkout, "0" * 40, self.source_paths)
        with self.assertRaises(ValueError):
            runs.verify_runtime_sources(self.checkout, self.commit, self.source_paths,
                                        executing_paths={"entry.py": self.root / "shadow.py"})
        with self.assertRaises(ValueError):
            runs.verify_runtime_sources(self.checkout, self.commit, ["../escape"])
        (self.checkout / "untracked.py").write_text("# not tracked\n")
        with self.assertRaises(ValueError):
            runs.verify_runtime_sources(self.checkout, self.commit, ["untracked.py"])
        entry = self.checkout / "entry.py"
        original = entry.read_bytes()
        entry.write_bytes(original + b"# dirty\n")
        with self.assertRaisesRegex(ValueError, "Dirty"):
            self.resolve()
        self.git("add", "entry.py")
        with self.assertRaisesRegex(ValueError, "Dirty"):
            self.resolve()
        entry.write_bytes(original)
        self.git("add", "entry.py")
        outside = self.root / "outside.py"
        outside.write_bytes(original)
        entry.unlink()
        entry.symlink_to(outside)
        with self.assertRaises((ValueError, OSError)):
            self.resolve()

    def test_regular_reader_rejects_symlink_ancestor_fifo_and_replacement(self):
        alias = self.root / "alias"
        alias.symlink_to(self.checkout, target_is_directory=True)
        with self.assertRaises(OSError):
            runs.read_regular(alias / "entry.py")
        fifo = self.root / "fifo"
        os.mkfifo(fifo)
        with self.assertRaises(ValueError):
            runs.read_regular(fifo)
        original_read = os.read
        target = self.checkout / "entry.py"
        replaced = False
        def replace(descriptor, count):
            nonlocal replaced
            content = original_read(descriptor, count)
            if not replaced:
                replaced = True
                target.unlink()
                target.write_bytes(content)
            return content
        with patch.object(runs.os, "read", side_effect=replace), self.assertRaisesRegex(ValueError, "changed"):
            runs.read_regular(target)

    def test_inventory_schema_missing_future_files_not_required(self):
        software = self.run["identity"]["software"]
        self.assertEqual([entry["path"] for entry in software["source_inventory"]], self.source_paths)
        runs.validate_software(software)
        for mutation in ("duplicate", "missing", "type", "extra"):
            changed = copy.deepcopy(software)
            if mutation == "duplicate":
                changed["source_inventory"].append(changed["source_inventory"][0])
            elif mutation == "missing":
                del changed["source_inventory"][0]["git_blob"]
            elif mutation == "type":
                changed["source_inventory"][0]["byte_size"] = True
            else:
                changed["source_inventory"][0]["runtime_path"] = "/irrelevant"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                runs.validate_software(changed)

    def test_named_seed_independent_reference_and_rng_isolation(self):
        record = runs.seed_record(33001)
        values = []
        before = checkpoints.state_fingerprint(checkpoints.capture_rng())
        for component in runs.COMPONENTS:
            value = {"parent_seed": 33001, "component": component}
            raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
            expected = int(hashlib.sha256(b"downstream_named_seed.v1\0" + raw).hexdigest()[:16], 16) % 2**63
            self.assertEqual(record["derived_seeds"][component], expected)
            self.assertNotEqual(expected, runs.named_seed(33002, component))
            values.append(expected)
        self.assertEqual(len(values), len(set(values)))
        self.assertEqual(before, checkpoints.state_fingerprint(checkpoints.capture_rng()))
        self.assertEqual(runs.worker_seed(values[3], 1, 0), runs.worker_seed(values[3], 1, 0))
        self.assertNotEqual(runs.worker_seed(values[3], 1, 0), runs.worker_seed(values[3], 1, 1))
        with self.assertRaises(ValueError):
            runs.seed_record(True)

    def test_initialization_and_order_repeat_without_global_rng_consumption(self):
        seed = self.run["identity"]["seeds"]["derived_seeds"]["model_initialization"]
        first = CNNRC(seed=seed)
        second = CNNRC(seed=seed)
        self.assertEqual(checkpoints.state_fingerprint(first.state_dict()), checkpoints.state_fingerprint(second.state_dict()))
        self.assertNotEqual(checkpoints.state_fingerprint(first.state_dict()), checkpoints.state_fingerprint(CNNRC(seed=seed+1).state_dict()))
        ids = [self.training[index].metadata.logical_example_id for index in range(len(self.training))]
        first_order = runs.epoch_order(ids, 123, 0)
        self.assertEqual(first_order, runs.epoch_order(ids, 123, 0))
        self.assertNotEqual(first_order, runs.epoch_order(ids, 124, 0))
        self.assertEqual(set(first_order), set(range(129)))
        self.assertEqual(tuple(map(len, runs.remaining_batches(first_order, 128))), (128, 1))
        self.assertEqual(runs.remaining_batches(first_order, 128, 1), (first_order[128:],))
        self.assertEqual(runs.remaining_batches(first_order, 128, 2), ())
        with self.assertRaises(ValueError):
            runs.remaining_batches(first_order, 128, 3)
        with patch.object(runs, "domain_hash", return_value="0" * 64):
            tied = runs.epoch_order(ids, 1, 1)
        self.assertEqual([ids[index] for index in tied], sorted(ids))

    def test_orientation_cross_model_cross_commit_roots_hardware_and_attempt(self):
        signature = tuple(inspect.signature(runs.training_orientation).parameters)
        self.assertEqual(signature, ("orientation_seed", "training_membership_hash", "epoch", "logical_example_id"))
        identity = self.run["identity"]
        ids = [self.training[index].metadata.logical_example_id for index in range(len(self.training))]
        def schedule(run):
            seeds = run["identity"]["seeds"]["derived_seeds"]
            membership = run["identity"]["selection"]["training_membership_hash"]
            return tuple(runs.training_orientation(seeds["training_orientation"], membership, epoch, logical_id)
                         for epoch in range(3) for logical_id in ids)
        expected = schedule(self.run)
        for family in ("cnn_rc", "random_transformer", "s0_transformer", "s1_transformer"):
            simulated = copy.deepcopy(self.run)
            simulated["identity"]["model"]["family"] = family
            simulated["identity"]["model"]["contract_id"] = family + ".simulated"
            simulated["identity"]["software"]["runtime_commit"] = "f" * 40
            rehash_run(simulated)
            simulated["execution"] = {"root": "/simulated/elsewhere", "hardware": "different", "attempt": family, "slurm": "123"}
            with self.subTest(family=family):
                self.assertEqual(schedule(simulated), expected)
        alias = self.resolve(level_id=self.level_ids[1])
        self.assertNotEqual(alias["run_id"], self.run["run_id"])
        self.assertEqual(alias["identity"]["selection"]["training_membership_hash"], identity["selection"]["training_membership_hash"])
        self.assertEqual(schedule(alias), expected)

    def test_orientation_each_allowed_input_can_change_and_matches_reference(self):
        base = {"orientation_seed": 100, "training_membership_hash": "a" * 64, "epoch": 0,
                "logical_example_id": "lex_" + "b" * 64}
        expected = runs.training_orientation(**base)
        raw = json.dumps(base, sort_keys=True, separators=(",", ":")).encode()
        self.assertEqual(expected, int(hashlib.sha256(b"downstream_training_orientation.v1\0" + raw).hexdigest(), 16) % 2 == 1)
        for field in base:
            observations = []
            for index in range(1, 33):
                changed = dict(base)
                if field in ("orientation_seed", "epoch"):
                    changed[field] = index
                else:
                    changed[field] = ("lex_" if field == "logical_example_id" else "") + hashlib.sha256(str(index).encode()).hexdigest()
                observations.append(runs.training_orientation(**changed))
            self.assertIn(not expected, observations, field)
        source = inspect.getsource(runs.training_orientation)
        for prohibited in ("random.", "hash(", "run_id", "model_family", "runtime_commit"):
            self.assertNotIn(prohibited, source.replace("domain_hash(", "H("))

    def test_python_hash_seed_has_no_effect(self):
        code = "from src.downstream_run import training_orientation; print(training_orientation(100, 'a'*64, 0, 'lex_'+'b'*64))"
        results = []
        for seed in ("1", "999"):
            environment = dict(os.environ, PYTHONHASHSEED=seed, PYTHONDONTWRITEBYTECODE="1")
            result = subprocess.run([sys.executable, "-B", "-c", code], check=True, capture_output=True, env=environment)
            results.append(result.stdout)
        self.assertEqual(results[0], results[1])

    def test_module_has_no_training_or_sealed_imports(self):
        for module in (runs, checkpoints):
            tree = ast.parse(inspect.getsource(module))
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imports.append(node.module or "")
            for name in imports:
                self.assertFalse(any(word in name for word in ("sealed_test", "h5py", "cnn_rc_training", "scripts.carc", "plot")), name)

    def test_attempt_receipts_bind_execution_facts_without_changing_run(self):
        environment_hash = checkpoints.environment_hash(self.environment)
        receipt = {
            "schema_version": "downstream_execution_attempt.v1", "run_id": self.run["run_id"],
            "attempt_id": self.attempt, "nonce": "1" * 32, "operation": "training",
            "parent_attempt_id": None, "resume_checkpoint_id": None,
            "physical_roots": {"stage": str(self.stage), "output": str(self.run_root), "attempt": str(self.root / "attempt")},
            "host": "synthetic-host", "device": "cpu", "hardware": "synthetic-cpu",
            "requested_resources": {"tasks": 1, "cpus": 1, "gpus": 0, "memory_bytes": None},
            "observed_resources": {"tasks": 1, "cpus": 1, "gpus": 0, "memory_bytes": None},
            "slurm": {"job_id": None, "array_job_id": None, "array_task_id": None, "account": None, "partition": None},
            "environment": self.environment,
            "environment_hash": runs.domain_hash("downstream_attempt_environment.v1", self.environment),
            "resume_compatibility_hash": environment_hash,
            "started_at": "2000-01-01T00:00:00Z", "ended_at": None, "status": "running", "exit_code": None,
            "failure_reason": None,
        }
        receipt["manifest_hash"] = runs.domain_hash("downstream_execution_attempt_manifest.v1", receipt)
        runs.validate_attempt(receipt)
        terminal = copy.deepcopy(receipt)
        terminal.update(status="failed", exit_code=1, ended_at="2000-01-01T00:00:01Z", failure_reason="synthetic interruption")
        del terminal["manifest_hash"]
        terminal["manifest_hash"] = runs.domain_hash("downstream_execution_attempt_manifest.v1", terminal)
        runs.validate_attempt(terminal)
        self.assertEqual(terminal["attempt_id"], receipt["attempt_id"])
        self.assertEqual(self.resolve(), self.run)
        for name in receipt:
            changed = copy.deepcopy(receipt)
            del changed[name]
            with self.subTest(name=name), self.assertRaises(ValueError):
                runs.validate_attempt(changed)

    def test_fresh_imports_cannot_load_sealed_or_raw_modules(self):
        code = '''
import builtins
original = builtins.__import__
def checked(name, globals=None, locals=None, fromlist=(), level=0):
    if any(item in name for item in ('sealed_test', 'h5py', 'cnn_rc_training', 'scripts.carc')):
        raise AssertionError('Forbidden import: ' + name)
    return original(name, globals, locals, fromlist, level)
builtins.__import__ = checked
import src.downstream_run
import src.downstream_checkpoint
'''
        subprocess.run([sys.executable, "-B", "-c", code], check=True, capture_output=True)


if __name__ == "__main__":
    unittest.main()
