"""Hermetic recovery/publication tests; no training or evaluation loop."""

from concurrent.futures import ThreadPoolExecutor
import copy
import ctypes
import errno
import hashlib
import io
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import unittest
from unittest.mock import patch
from unittest import mock

import numpy as np
import torch

from src import downstream_checkpoint as checkpoints
from src import downstream_run as runs
from tests.cnn_rc_synthetic_support import SyntheticCase, refresh_tensor_inventory, rehash_run


class CheckpointTests(SyntheticCase):
    def test_complete_state_roundtrip_populated_adam_bn_and_rng(self):
        runs_before = copy.deepcopy(self.run)
        payload = self.payload(next_batch=1)
        fingerprint = checkpoints.state_fingerprint(payload)
        with checkpoints.run_writer(self.run_root, self.run["run_id"]) as writer:
            path = self.publish(payload, writer)
            restored, envelope = self.load(path)
            self.assertEqual(envelope["semantic_hash"], fingerprint)
            self.assertEqual(checkpoints.state_fingerprint(restored), fingerprint)
            expected_draws = (random.random(), np.random.random(), torch.rand(3))
            model, optimizer = checkpoints.restore_checkpoint(
                restored, run=self.run, environment=self.environment, training=self.training,
                writer=writer, checkout_root=self.checkout)
            self.assertEqual(random.random(), expected_draws[0])
            self.assertEqual(np.random.random(), expected_draws[1])
            self.assertTrue(torch.equal(torch.rand(3), expected_draws[2]))
            self.assertEqual(checkpoints.state_fingerprint(model.state_dict()), checkpoints.state_fingerprint(payload["model_state"]))
            self.assertEqual(checkpoints.state_fingerprint(optimizer.state_dict()), checkpoints.state_fingerprint(payload["optimizer_state"]))
        self.assertEqual(self.run, runs_before)
        self.assertEqual(payload["rng"]["torch_cuda"], None)
        self.assertEqual(payload["scheduler_state"], None)
        self.assertEqual(payload["position"]["next_batch_index"], 1)

    def test_initial_and_both_epoch_boundaries(self):
        for payload in (self.payload(), self.payload(2, phase="validation_pending"),
                        self.payload(2, phase="epoch_complete", history=[self.event()])):
            self.validate_payload(payload)
        pending = self.payload(2, phase="validation_pending")
        self.assertEqual(pending["consumed_validation_event_count"], 0)
        complete = self.payload(2, phase="epoch_complete", history=[self.event()])
        self.assertEqual(complete["consumed_validation_event_count"], 1)
        self.assertEqual(complete["selection_state"]["checkpoint_ref"], {"kind": "self"})

    def test_epoch_history_tie_break_and_carry_forward(self):
        reference = {"kind": "checkpoint", "checkpoint_id": "ckpt_" + "a" * 64}
        first = self.event(reference=reference)
        second = self.event(epoch=1)
        payload = self.payload(2, epoch=1, phase="epoch_complete", history=[first, second])
        self.assertEqual(payload["selection_state"]["epoch"], 0)
        self.assertEqual(payload["consumed_selection_count"], 2)
        self.validate_payload(payload)
        changed = copy.deepcopy(payload)
        changed["selection_state"]["epoch"] = 1
        with self.assertRaisesRegex(ValueError, "Best-selection"):
            self.validate_payload(changed)
        better = self.event(epoch=1, mse=0.015625, r2=0.5)
        payload = self.payload(2, epoch=1, phase="epoch_complete", history=[first, better])
        self.assertEqual(payload["selection_state"]["epoch"], 1)

    def test_tensor_state_codec_types_endianness_and_storage_layout(self):
        tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4).T
        value = {"tuple": (1, -0.0, None), 3: [True, tensor]}
        tree, tensors, inventory = checkpoints.semantic_state(value)
        decoded = checkpoints._decode(tree, tensors, set())
        self.assertEqual(checkpoints.state_fingerprint(decoded), checkpoints.state_fingerprint(value))
        self.assertEqual(checkpoints.state_fingerprint(tensor), checkpoints.state_fingerprint(tensor.contiguous()))
        self.assertNotEqual(checkpoints.state_fingerprint([1]), checkpoints.state_fingerprint((1,)))
        self.assertNotEqual(checkpoints.state_fingerprint(1), checkpoints.state_fingerprint(True))
        self.assertNotEqual(checkpoints.state_fingerprint(0.0), checkpoints.state_fingerprint(-0.0))
        expected = hashlib.sha256(tensor.numpy().astype("<f4").tobytes(order="C")).hexdigest()
        self.assertEqual(inventory[0]["sha256"], expected)
        for invalid in (object(), np.zeros(2), {True: 1}, float("inf"), torch.tensor([float("nan")])):
            with self.subTest(kind=type(invalid)), self.assertRaises(ValueError):
                checkpoints.semantic_state(invalid)

    def test_tensor_names_shapes_dtypes_and_content_before_load(self):
        original = self.payload(1)
        for name in original["model_state"]:
            if type(original["model_state"][name]) is torch.Tensor:
                for mutation in ("shape", "dtype", "content", "missing"):
                    changed = copy.deepcopy(original)
                    if mutation == "shape":
                        changed["model_state"][name] = changed["model_state"][name].reshape(-1)[:0]
                    elif mutation == "dtype":
                        changed["model_state"][name] = changed["model_state"][name].double()
                    elif mutation == "content":
                        changed["model_state"][name].reshape(-1)[0] += 1
                    else:
                        del changed["model_state"][name]
                    with self.subTest(name=name, mutation=mutation), self.assertRaises(ValueError):
                        self.validate_payload(changed)
        changed = copy.deepcopy(original)
        changed["model_state"]["_extra_state"]["architecture_settings"]["convolution"]["stride"] = 2
        refresh_tensor_inventory(changed)
        with self.assertRaises(ValueError):
            self.validate_payload(changed)

    def test_adam_moments_parameter_order_and_scheduler_rejected(self):
        original = self.payload(1)
        mutations = []
        changed = copy.deepcopy(original)
        changed["optimizer_parameter_names_in_order"].reverse()
        mutations.append(changed)
        for field in ("step", "exp_avg", "exp_avg_sq"):
            changed = copy.deepcopy(original)
            changed["optimizer_state"]["state"][0][field] = torch.tensor(-1.0)
            mutations.append(changed)
        changed = copy.deepcopy(original)
        del changed["optimizer_state"]["state"][0]
        mutations.append(changed)
        changed = copy.deepcopy(original)
        changed["optimizer_state"]["param_groups"][0]["weight_decay"] = 0.1
        mutations.append(changed)
        changed = copy.deepcopy(original)
        changed["scheduler_state"] = {}
        mutations.append(changed)
        for changed in mutations:
            refresh_tensor_inventory(changed)
            with self.assertRaises(ValueError):
                self.validate_payload(changed)

    def test_all_missing_unknown_and_mismatched_checkpoint_contract_fields(self):
        payload = self.payload(1)
        for field in checkpoints.PAYLOAD_FIELDS:
            changed = copy.deepcopy(payload)
            del changed[field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate_payload(changed)
        changed = copy.deepcopy(payload)
        changed["unknown"] = 1
        with self.assertRaises(ValueError):
            self.validate_payload(changed)
        for field in ("run_id", "resolved_run_hash", "model_contract", "software_identity", "seed_record",
                      "data_identity", "orientation_policy", "optimizer_definition", "environment_compatibility"):
            changed = copy.deepcopy(payload)
            changed[field] = "wrong"
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate_payload(changed)

    def test_run_data_model_seed_code_and_level_mismatch(self):
        payload = self.payload(1)
        paths = [
            ("data", "stage_id"), ("data", "split_identity_hash"), ("data", "subset_set_manifest_hash"),
            ("selection", "transcription_factor"), ("selection", "requested_level_id"),
            ("selection", "canonical_level_id"), ("selection", "training_membership_hash"),
            ("model", "contract_id"), ("seeds", "parent_seed"), ("software", "runtime_commit"),
        ]
        for section, field in paths:
            changed_run = copy.deepcopy(self.run)
            old = changed_run["identity"][section][field]
            changed_run["identity"][section][field] = 33002 if type(old) is int else "changed"
            rehash_run(changed_run)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate_payload(payload, run=changed_run)
        alias = self.data.dataset("training", level_id=self.level_ids[1])
        with self.assertRaisesRegex(ValueError, "selection"):
            self.validate_payload(payload, training=alias)

    def test_position_permutation_accumulators_and_budget_invariants(self):
        payload = self.payload(1)
        for field, value in (("current_epoch", 2), ("global_update", 0), ("next_batch_index", 3),
                             ("completed_epoch_count", 1), ("phase", "evaluation"), ("epoch_permutation_hash", "a" * 64)):
            changed = copy.deepcopy(payload)
            changed["position"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate_payload(changed)
        for field, value in (("example_count", 1), ("update_count", 3), ("total_loss_sum", float("nan"))):
            changed = copy.deepcopy(payload)
            changed["training_accumulators"][field] = value
            with self.assertRaises(ValueError):
                self.validate_payload(changed)
        changed = copy.deepcopy(payload)
        changed["consumed_selection_count"] = 1
        with self.assertRaises(ValueError):
            self.validate_payload(changed)

    def test_validation_history_and_invariance_fail_closed(self):
        complete = self.payload(2, phase="epoch_complete", history=[self.event()])
        for mutation in ("duplicate", "event_id", "nan", "undefined", "count", "invariance", "best", "self"):
            changed = copy.deepcopy(complete)
            if mutation == "duplicate":
                changed["validation_history"].append(copy.deepcopy(changed["validation_history"][0]))
            elif mutation == "event_id":
                changed["validation_history"][0]["event_id"] = "wrong"
            elif mutation == "nan":
                changed["validation_history"][0]["metrics"]["mean"]["r2"] = float("nan")
            elif mutation == "undefined":
                changed["validation_history"][0]["metrics"]["mean"]["r2"] = None
            elif mutation == "count":
                changed["validation_history"][0]["metrics"]["mean"]["sample_count"] = 1
            elif mutation == "invariance":
                changed["validation_history"][0]["rc_diagnostic"]["violation_count"] = 1
            elif mutation == "best":
                changed["selection_state"] = None
            else:
                changed["validation_history"][0]["checkpoint_ref"] = {"kind": "self", "checkpoint_id": "wrong"}
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.validate_payload(changed)
        event = self.event()
        for metrics in event["metrics"].values():
            metrics["pearson"] = None
            metrics["spearman"] = None
            metrics["undefined_reasons"] = {"pearson": "constant_predictions", "spearman": "constant_predictions"}
        self.validate_payload(self.payload(2, phase="epoch_complete", history=[event]))

    def test_rng_schema_corruption_and_reserved_streams(self):
        payload = self.payload()
        for mutation in ("python", "numpy", "torch_cpu", "torch_cuda", "component_generators"):
            changed = copy.deepcopy(payload)
            changed["rng"][mutation] = {"invalid": 1}
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.validate_payload(changed)
        before = checkpoints.state_fingerprint(checkpoints.capture_rng())
        self.validate_payload(payload)
        self.assertEqual(before, checkpoints.state_fingerprint(checkpoints.capture_rng()))
        adapter = checkpoints.seed_runtime(self.run["identity"]["seeds"])
        expected = self.run["identity"]["seeds"]["derived_seeds"]["numpy_runtime"] % 2**32
        self.assertEqual(adapter, {"numpy_legacy_seed_mod_2_32": expected})

    def test_environment_compatibility_and_actual_resume_environment(self):
        payload = self.payload(1)
        for field in ("python", "numpy", "torch", "torch_build", "machine", "threads", "backend"):
            changed = copy.deepcopy(self.environment)
            changed[field] = changed[field] + 1 if type(changed[field]) is int else "different"
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate_payload(payload, environment=changed)
        with checkpoints.run_writer(self.run_root, self.run["run_id"]) as writer:
            with patch.object(checkpoints, "capture_environment", return_value={}):
                with self.assertRaisesRegex(ValueError, "Actual resume"):
                    checkpoints.restore_checkpoint(payload, run=self.run, environment=self.environment,
                                                   training=self.training, writer=writer, checkout_root=self.checkout)

    def test_byte_corruption_is_rejected_before_torch_load(self):
        payload = self.payload()
        with checkpoints.run_writer(self.run_root, self.run["run_id"]) as writer:
            path = self.publish(payload, writer)
        state = path / "state.pt"
        os.chmod(state, 0o600)
        raw = state.read_bytes()
        state.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
        with patch.object(torch, "load", side_effect=AssertionError("must verify bytes first")):
            with self.assertRaisesRegex(ValueError, "Raw checkpoint"):
                self.load(path)

    def test_semantic_corruption_with_valid_raw_envelope_is_rejected(self):
        payload = self.payload()
        with checkpoints.run_writer(self.run_root, self.run["run_id"]) as writer:
            path = self.publish(payload, writer)
        envelope_path = path / "manifest.json"
        state_path = path / "state.pt"
        envelope = runs.strict_json(envelope_path.read_bytes())
        archive = torch.load(io.BytesIO(state_path.read_bytes()), weights_only=True)
        tensor = next(value for value in archive["tensors"].values() if value.dtype == torch.float32 and value.numel() > 1)
        tensor.reshape(-1)[0] += 1
        buffer = io.BytesIO()
        torch.save(archive, buffer)
        raw = buffer.getvalue()
        envelope["file"]["byte_size"] = len(raw)
        envelope["file"]["sha256"] = hashlib.sha256(raw).hexdigest()
        content = dict(envelope)
        del content["manifest_hash"]
        envelope["manifest_hash"] = runs.domain_hash("downstream_checkpoint_publication.v1", content)
        for file in (state_path, envelope_path):
            os.chmod(file, 0o600)
        state_path.write_bytes(raw)
        envelope_path.write_bytes(runs.canonical_json_bytes(envelope) + b"\n")
        with self.assertRaisesRegex(ValueError, "Tensor fingerprint"):
            self.load(path)

    def test_weights_only_loader_and_exact_bundle_inventory(self):
        with checkpoints.run_writer(self.run_root, self.run["run_id"]) as writer:
            path = self.publish(self.payload(), writer)
        original = torch.load
        with patch.object(torch, "load", wraps=original) as loaded:
            self.load(path)
        self.assertIs(loaded.call_args.kwargs["weights_only"], True)
        os.chmod(path, 0o700)
        (path / "extra").write_text("unexpected")
        with self.assertRaisesRegex(ValueError, "inventory"):
            self.load(path)

    def test_exclusive_checkpoint_publication_and_private_output_rejection(self):
        payload = self.payload()
        with checkpoints.run_writer(self.run_root, self.run["run_id"]) as writer:
            path = self.publish(payload, writer)
            before = (path / "state.pt").read_bytes()
            with self.assertRaises(FileExistsError):
                self.publish(payload, writer)
            self.assertEqual((path / "state.pt").read_bytes(), before)
        for private in self.run_root.glob(".b3-private-*"):
            with self.assertRaisesRegex(ValueError, "Private"):
                self.load(private)

    def test_failures_at_write_sync_rename_are_not_finalized(self):
        for helper in ("_write_file", "_exclusive_rename"):
            destination = self.root / ("fail-" + helper)
            with patch.object(checkpoints, helper, side_effect=OSError("injected")):
                with self.assertRaises(OSError):
                    checkpoints.publish_bundle(destination, {"receipt.json": b"{}"})
            self.assertFalse(destination.exists())
        with patch.object(checkpoints.os, "fsync", side_effect=OSError("sync injected")):
            with self.assertRaises(OSError):
                checkpoints.publish_bundle(self.root / "sync-fail", {"receipt.json": b"{}"})
        self.assertFalse((self.root / "sync-fail").exists())

    def test_final_name_invisible_until_publication_and_post_sync_failure(self):
        destination = self.root / "complete"
        original = checkpoints._exclusive_rename
        published = False
        def rename(parent, source, target):
            nonlocal published
            self.assertFalse(destination.exists())
            self.assertEqual((self.root / source / "file").read_bytes(), b"complete bytes")
            original(parent, source, target)
            published = True
        sync = os.fsync
        def fsync(descriptor):
            if published:
                raise OSError("post-publication sync")
            sync(descriptor)
        with patch.object(checkpoints, "_exclusive_rename", side_effect=rename):
            with patch.object(checkpoints.os, "fsync", side_effect=fsync):
                with self.assertRaises(checkpoints.PublishedDurabilityError) as error:
                    checkpoints.publish_bundle(destination, {"file": b"complete bytes"})
        self.assertEqual(error.exception.destination, destination)
        self.assertEqual((destination / "file").read_bytes(), b"complete bytes")

    def test_concurrent_publishers_one_winner_without_wall_clock_assertion(self):
        destination = self.root / "race"
        barrier = threading.Barrier(2)
        original = checkpoints._exclusive_rename
        def rename(*arguments):
            barrier.wait()
            original(*arguments)
        with patch.object(checkpoints, "_exclusive_rename", side_effect=rename):
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(checkpoints.publish_bundle, destination, {"file": b"complete"}) for _ in range(2)]
                outcomes = []
                for future in futures:
                    try:
                        future.result()
                        outcomes.append("published")
                    except FileExistsError:
                        outcomes.append("exists")
        self.assertEqual(sorted(outcomes), ["exists", "published"])

    def test_same_run_lock_competition_process_exit_and_finalized_refusal(self):
        with checkpoints.run_writer(self.run_root, self.run["run_id"]):
            with self.assertRaises(BlockingIOError):
                with checkpoints.run_writer(self.run_root, self.run["run_id"]):
                    self.fail("Second writer acquired lock")
        # A killed process releases its OS lock; there is no stale-file deletion.
        code = "import os,sys; from pathlib import Path; from src.downstream_checkpoint import run_writer; "
        code += "context=run_writer(Path(sys.argv[1]),sys.argv[2]); context.__enter__(); os._exit(0)"
        subprocess.run([sys.executable, "-B", "-c", code, str(self.run_root), self.run["run_id"]], check=True, capture_output=True)
        with checkpoints.run_writer(self.run_root, self.run["run_id"]) as writer:
            self.assertTrue(writer.active)
        (self.run_root / "completion").mkdir()
        with self.assertRaisesRegex(ValueError, "Finalized"):
            with checkpoints.run_writer(self.run_root, self.run["run_id"]):
                self.fail("Finalized writer acquired")

    def test_parent_symlinks_existing_objects_unsupported_platform_and_filesystem(self):
        link = self.root / "linked-root"
        link.symlink_to(self.run_root, target_is_directory=True)
        with self.assertRaises(OSError):
            checkpoints.publish_bundle(link / "output", {"file": b"x"})
        for kind in ("file", "empty", "nonempty", "dangling"):
            destination = self.root / kind
            if kind == "file":
                destination.write_bytes(b"keep")
            elif kind == "dangling":
                destination.symlink_to(self.root / "missing")
            else:
                destination.mkdir()
                if kind == "nonempty":
                    (destination / "keep").write_bytes(b"keep")
            inode = destination.lstat().st_ino
            with self.assertRaises(OSError):
                checkpoints.publish_bundle(destination, {"file": b"x"})
            self.assertEqual(destination.lstat().st_ino, inode)

        # Obtain a real checkpoint name and bundle without duplicating the
        # publication algorithm, then attempt that checkpoint at a live link.
        payload = self.payload(1)
        with checkpoints.run_writer(self.run_root, self.run["run_id"]) as writer:
            published = self.publish(payload, writer)
        self.assertEqual(checkpoints.state_fingerprint(self.load(published)[0]),
                         checkpoints.state_fingerprint(payload))

        def entry_identity(path):
            status = path.lstat()
            return (status.st_dev, status.st_ino, status.st_mode, status.st_nlink,
                    status.st_size, status.st_mtime_ns, status.st_ctime_ns)

        published_identity = entry_identity(published)
        published_files = {}
        for path in published.iterdir():
            published_files[path.name] = (entry_identity(path), path.read_bytes())
        published_entries = sorted(path.name for path in self.run_root.iterdir())

        attempt_root = self.root / "live-destination" / self.run["run_id"]
        attempt_root.mkdir(parents=True)
        target = attempt_root / "live-target"
        target.mkdir()
        sentinel = target / "sentinel.bin"
        sentinel_bytes = b"preserve live checkpoint destination\n\x00\xff"
        sentinel.write_bytes(sentinel_bytes)
        destination = attempt_root / published.name
        self.assertTrue(target.is_dir())
        self.assertTrue(sentinel.is_file())
        self.assertEqual(sentinel.read_bytes(), sentinel_bytes)
        target_entries = sorted(path.name for path in target.iterdir())
        self.assertEqual(target_entries, ["sentinel.bin"])
        target_identity = entry_identity(target)
        sentinel_identity = entry_identity(sentinel)
        self.assertFalse(os.path.lexists(destination))
        link_text = "live-target"
        destination.symlink_to(link_text, target_is_directory=True)
        self.assertTrue(destination.is_symlink())
        self.assertEqual(os.readlink(destination), link_text)
        self.assertEqual(destination.resolve(strict=True), target)
        link_identity = entry_identity(destination)

        with checkpoints.run_writer(attempt_root, self.run["run_id"]) as writer:
            entries_before = set(path.name for path in attempt_root.iterdir())
            with self.assertRaises(FileExistsError) as error:
                self.publish(payload, writer)
            self.assertEqual(error.exception.errno, errno.EEXIST)
            self.assertEqual(error.exception.filename, destination.name)
            self.assertTrue(destination.is_symlink())
            self.assertEqual(entry_identity(destination), link_identity)
            self.assertEqual(os.readlink(destination), link_text)
            self.assertEqual(destination.resolve(strict=True), target)
            self.assertTrue(target.is_dir())
            self.assertFalse(target.is_symlink())
            self.assertEqual(entry_identity(target), target_identity)
            self.assertTrue(sentinel.is_file())
            self.assertFalse(sentinel.is_symlink())
            self.assertEqual(entry_identity(sentinel), sentinel_identity)
            self.assertEqual(sentinel.read_bytes(), sentinel_bytes)
            self.assertEqual(sorted(path.name for path in target.iterdir()), target_entries)
            with self.assertRaises(OSError):
                self.load(destination)
            self.assertEqual(list(attempt_root.glob("index-*")), [])
            entries_after = set(path.name for path in attempt_root.iterdir())
            self.assertTrue(entries_before <= entries_after)
            for name in entries_after - entries_before:
                self.assertTrue(name.startswith(".b3-private-"))
                private = attempt_root / name
                self.assertTrue(private.is_dir())
                self.assertFalse(private.is_symlink())
                with self.assertRaisesRegex(ValueError, "Private output is not a checkpoint"):
                    self.load(private)

        self.assertEqual(entry_identity(published), published_identity)
        self.assertEqual(sorted(path.name for path in published.iterdir()), sorted(published_files))
        for path in published.iterdir():
            self.assertEqual((entry_identity(path), path.read_bytes()), published_files[path.name])
        self.assertEqual(sorted(path.name for path in self.run_root.iterdir()), published_entries)
        self.assertEqual(checkpoints.state_fingerprint(self.load(published)[0]),
                         checkpoints.state_fingerprint(payload))
        with patch.object(checkpoints.sys, "platform", "unsupported"):
            with self.assertRaises(ValueError):
                checkpoints.publish_bundle(self.root / "unsupported", {"file": b"x"})

    def test_parent_replacement_does_not_publish_into_decoy(self):
        parent = self.root / "parent"
        parent.mkdir()
        moved = self.root / "moved"
        original = checkpoints._assert_directory
        replaced = False
        def replace(path, descriptor):
            nonlocal replaced
            if not replaced:
                replaced = True
                parent.rename(moved)
                parent.mkdir()
                (parent / "keep").write_bytes(b"keep")
            original(path, descriptor)
        with patch.object(checkpoints, "_assert_directory", side_effect=replace):
            with self.assertRaisesRegex(ValueError, "replaced"):
                checkpoints.publish_bundle(parent / "output", {"file": b"x"})
        self.assertFalse((parent / "output").exists())
        self.assertEqual((parent / "keep").read_bytes(), b"keep")

    def test_publication_and_resume_reverify_code(self):
        payload = self.payload()
        (self.checkout / "entry.py").write_text("# dirty after resolving run\n")
        with checkpoints.run_writer(self.run_root, self.run["run_id"]) as writer:
            with self.assertRaisesRegex(ValueError, "Dirty"):
                self.publish(payload, writer)
            with self.assertRaisesRegex(ValueError, "Dirty"):
                checkpoints.restore_checkpoint(payload, run=self.run, environment=self.environment,
                                               training=self.training, writer=writer, checkout_root=self.checkout)

    def test_immutable_index_chain_and_best_references(self):
        with checkpoints.run_writer(self.run_root, self.run["run_id"]) as writer:
            first = self.publish(self.payload(), writer)
            first_path, first_index = checkpoints.publish_index(writer=writer, checkpoint_paths=[first], run=self.run,
                                                                 environment=self.environment, training=self.training, checkout_root=self.checkout)
            selected = self.publish(self.payload(2, phase="epoch_complete", history=[self.event()]), writer)
            second_path, second_index = checkpoints.publish_index(writer=writer, checkpoint_paths=[first, selected], run=self.run,
                                                         environment=self.environment, training=self.training, checkout_root=self.checkout, previous=first_index)
            self.assertEqual(second_index["predecessor_hash"], first_index["manifest_hash"])
            self.assertEqual(second_index["selected_best_checkpoint_id"], self.load(selected)[1]["checkpoint_id"])
            with self.assertRaises(ValueError):
                checkpoints.publish_index(writer=writer, checkpoint_paths=[selected], run=self.run,
                                           environment=self.environment, training=self.training, checkout_root=self.checkout, previous=first_index)
            with self.assertRaises((ValueError, FileExistsError)):
                checkpoints.publish_index(writer=writer, checkpoint_paths=[first], run=self.run,
                                           environment=self.environment, training=self.training, checkout_root=self.checkout)

            # Prepare a legitimate next checkpoint before corrupting the current
            # index; all later operations must preserve these published bundles.
            selected_id = self.load(selected)[1]["checkpoint_id"]
            recovery = self.publish(self.payload(
                1, epoch=1, history=[self.event(reference={
                    "kind": "checkpoint", "checkpoint_id": selected_id,
                })],
            ), writer)
            checkpoint_paths = [first, selected, recovery]
            for path in checkpoint_paths:
                self.load(path)
            checkpoints.validate_index(first_index)
            checkpoints.validate_index(second_index)
            index_file = second_path / "index.json"
            original_bytes = checkpoints.read_regular(index_file)
            self.assertEqual(checkpoints.strict_json(original_bytes), second_index)
            self.assertEqual(second_index["revision"], 1)
            self.assertEqual(second_index["predecessor_hash"], first_index["manifest_hash"])
            self.assertEqual(second_index["latest_recovery_checkpoint_id"], selected_id)
            self.assertEqual(second_index["selected_best_checkpoint_id"], selected_id)
            indexes_before = copy.deepcopy([first_index, second_index])
            run_before = copy.deepcopy(self.run)

            def published_snapshot():
                inventory = {}
                for path in sorted(self.run_root.rglob("*")):
                    status = path.lstat()
                    record = {
                        "identity": (status.st_dev, status.st_ino, status.st_mode,
                                     status.st_size, status.st_mtime_ns, status.st_ctime_ns),
                    }
                    if path.is_file():
                        content = path.read_bytes()
                        record.update(bytes=content, byte_size=len(content),
                                      sha256=hashlib.sha256(content).hexdigest())
                    inventory[path.relative_to(self.run_root).as_posix()] = record
                return inventory

            before = published_snapshot()
            corrupted = copy.deepcopy(second_index)
            fingerprint = corrupted["entries"][0]["file"]["sha256"]
            replacement = "0" if fingerprint[0] != "0" else "1"
            corrupted["entries"][0]["file"]["sha256"] = replacement + fingerprint[1:]
            unchanged_fields = copy.deepcopy(corrupted)
            unchanged_fields["entries"][0]["file"]["sha256"] = fingerprint
            self.assertEqual(unchanged_fields, second_index)
            self.assertEqual(corrupted["manifest_hash"], second_index["manifest_hash"])
            corrupted_bytes = runs.canonical_json_bytes(corrupted) + b"\n"
            self.assertEqual(len(corrupted_bytes), len(original_bytes))
            self.assertNotEqual(hashlib.sha256(corrupted_bytes).hexdigest(),
                                hashlib.sha256(original_bytes).hexdigest())
            original_mode = index_file.lstat().st_mode & 0o777
            os.chmod(index_file, 0o600)
            index_file.write_bytes(corrupted_bytes)
            os.chmod(index_file, original_mode)
            after_corruption = published_snapshot()
            index_name = index_file.relative_to(self.run_root).as_posix()
            self.assertEqual(set(after_corruption), set(before))
            self.assertEqual([name for name in before if before[name] != after_corruption[name]],
                             [index_name])

            # Exercise the production reader and validator on the corrupted
            # published JSON, with observers that retain the real operations.
            with (
                patch.object(checkpoints, "load_checkpoint", wraps=checkpoints.load_checkpoint) as loaded,
                patch.object(checkpoints, "restore_checkpoint", wraps=checkpoints.restore_checkpoint) as resumed,
                patch.object(checkpoints, "publish_bundle", wraps=checkpoints.publish_bundle) as published,
            ):
                corrupted_from_disk = checkpoints.strict_json(checkpoints.read_regular(index_file))
                self.assertEqual(corrupted_from_disk, corrupted)
                with self.assertRaisesRegex(runs.RunContractError, r"^Index hash differs\.$"):
                    checkpoints.validate_index(corrupted_from_disk)
                loaded.assert_not_called()
                resumed.assert_not_called()
                published.assert_not_called()
            self.assertEqual(published_snapshot(), after_corruption)
            self.assertEqual(corrupted_from_disk, corrupted)

            # Also refuse a cached intact predecessor while its published file
            # is corrupt; neither path may fall back, repair, or advance it.
            for previous, message in (
                (corrupted_from_disk, r"^Index hash differs\.$"),
                (second_index, r"^Unpublished predecessor index\.$"),
            ):
                with (
                    self.subTest(rejection=message),
                    patch.object(checkpoints, "validate_index", wraps=checkpoints.validate_index) as validated,
                    patch.object(checkpoints, "load_checkpoint", wraps=checkpoints.load_checkpoint) as loaded,
                    patch.object(checkpoints, "restore_checkpoint", wraps=checkpoints.restore_checkpoint) as resumed,
                    patch.object(checkpoints, "publish_bundle", wraps=checkpoints.publish_bundle) as published,
                    patch.object(checkpoints, "read_regular", wraps=checkpoints.read_regular) as read,
                ):
                    with self.assertRaisesRegex(runs.RunContractError, message):
                        checkpoints.publish_index(
                            writer=writer, checkpoint_paths=checkpoint_paths, run=self.run,
                            environment=self.environment, training=self.training,
                            checkout_root=self.checkout, previous=previous,
                        )
                    validated.assert_called_once_with(previous)
                    self.assertEqual([call.args[0] for call in loaded.call_args_list],
                                     [first, selected, recovery, selected])
                    index_reads = [call.args[0] for call in read.call_args_list
                                   if Path(call.args[0]).name == "index.json"]
                    self.assertEqual(index_reads, [index_file] if previous is second_index else [])
                    resumed.assert_not_called()
                    published.assert_not_called()
                    self.assertEqual(published_snapshot(), after_corruption)
                    self.assertEqual(index_file.read_bytes(), corrupted_bytes)
                    self.assertFalse(os.path.lexists(self.run_root / "index-00000002"))
                    self.assertEqual(sorted(path.name for path in self.run_root.glob("index-*")),
                                     [first_path.name, second_path.name])
                    self.assertEqual([first_index, second_index], indexes_before)
                    self.assertEqual(corrupted_from_disk, corrupted)
                    self.assertEqual(self.run, run_before)
        self.assertEqual(runs.strict_json((first_path / "index.json").read_bytes()), first_index)

    def test_linux_exclusive_rename_flags_and_unsupported_filesystem(self):
        library = mock.Mock()
        library.renameat2 = mock.Mock(return_value=0)
        with patch.object(checkpoints.sys, "platform", "linux"):
            with patch.object(checkpoints.ctypes, "CDLL", return_value=library):
                checkpoints._exclusive_rename(123, "private", "final")
        library.renameat2.assert_called_once_with(123, b"private", 123, b"final", 1)
        def unsupported(*arguments):
            ctypes.set_errno(errno.ENOTSUP)
            return -1
        library.renameat2.side_effect = unsupported
        with patch.object(checkpoints.sys, "platform", "linux"):
            with patch.object(checkpoints.ctypes, "CDLL", return_value=library):
                with self.assertRaisesRegex(ValueError, "Filesystem"):
                    checkpoints._exclusive_rename(123, "private", "final")

    def test_valid_raw_bytes_with_wrong_semantic_hash_fail(self):
        with checkpoints.run_writer(self.run_root, self.run["run_id"]) as writer:
            path = self.publish(self.payload(), writer)
        manifest_path = path / "manifest.json"
        envelope = runs.strict_json(manifest_path.read_bytes())
        envelope["semantic_hash"] = "a" * 64
        envelope["checkpoint_id"] = "ckpt_" + "a" * 64
        del envelope["manifest_hash"]
        envelope["manifest_hash"] = runs.domain_hash("downstream_checkpoint_publication.v1", envelope)
        os.chmod(manifest_path, 0o600)
        manifest_path.write_bytes(runs.canonical_json_bytes(envelope) + b"\n")
        with self.assertRaisesRegex(ValueError, "semantic hash"):
            self.load(path)

    def test_invalid_capture_does_not_serialize_or_execute_model(self):
        with patch.object(torch.nn.Module, "__call__", side_effect=AssertionError("No model execution")):
            with patch.object(torch.optim.Adam, "step", side_effect=AssertionError("No optimization")):
                payload = self.payload(1)
                self.validate_payload(payload)
        original = checkpoints.CNNRC.load_state_dict
        with patch.object(checkpoints.CNNRC, "load_state_dict", wraps=original) as loaded:
            bad = copy.deepcopy(payload)
            bad["model_state"]["W"] = bad["model_state"]["W"].double()
            refresh_tensor_inventory(bad)
            with checkpoints.run_writer(self.run_root, self.run["run_id"]) as writer:
                with self.assertRaises(ValueError):
                    checkpoints.restore_checkpoint(bad, run=self.run, environment=self.environment,
                                                   training=self.training, writer=writer, checkout_root=self.checkout)
        loaded.assert_not_called()

    def test_recovery_state_resolves_self_without_mutation_or_budget_reset(self):
        payload = self.payload(2, phase="epoch_complete", history=[self.event()])
        original = checkpoints.state_fingerprint(payload)
        state = checkpoints.recovery_state(payload)
        expected = {"kind": "checkpoint", "checkpoint_id": "ckpt_" + original}
        self.assertEqual(state["selection_state"]["checkpoint_ref"], expected)
        self.assertEqual(state["validation_history"][0]["checkpoint_ref"], expected)
        self.assertEqual(state["consumed_validation_event_count"], 1)
        self.assertEqual(state["consumed_selection_count"], 1)
        self.assertEqual(checkpoints.state_fingerprint(payload), original)
        self.validate_payload(self.payload(0, epoch=1, history=state["validation_history"]))

    def test_cuda_driver_query_contract_without_gpu_access(self):
        library = mock.Mock()
        def version(pointer):
            pointer._obj.value = 12080
            return 0
        library.cuDriverGetVersion = mock.Mock(side_effect=version)
        library.nvmlInit_v2 = mock.Mock(return_value=0)
        library.nvmlShutdown = mock.Mock(return_value=0)
        def driver_release(buffer, length):
            buffer.value = b"570.124.06"
            return 0
        library.nvmlSystemGetDriverVersion = mock.Mock(side_effect=driver_release)
        with patch.object(checkpoints.ctypes, "CDLL", return_value=library):
            self.assertEqual(checkpoints._cuda_driver_version(), '{"cuda_driver_api":12080,"nvidia_driver_release":"570.124.06"}')
        library.nvmlShutdown.assert_called_once_with()
        library.cuDriverGetVersion.side_effect = None
        library.cuDriverGetVersion.return_value = 1
        with patch.object(checkpoints.ctypes, "CDLL", return_value=library):
            with self.assertRaisesRegex(ValueError, "driver identity"):
                checkpoints._cuda_driver_version()
        environment = copy.deepcopy(self.environment)
        environment.update(backend="cuda", device_count=1, driver="12080", cuda="12.8",
                           cublas_workspace_config=":4096:8", device_class="synthetic GPU")
        checkpoints.validate_environment(environment)
        environment["driver"] = None
        with self.assertRaises(ValueError):
            checkpoints.validate_environment(environment)

        # Establish the unchanged CPU path before introducing mocked CUDA.
        with patch.multiple(
            checkpoints.torch.cuda,
            is_initialized=mock.Mock(return_value=False),
            get_rng_state_all=mock.Mock(),
            set_rng_state_all=mock.Mock(),
            init=mock.Mock(side_effect=AssertionError("Real CUDA initialization forbidden")),
            _lazy_init=mock.Mock(side_effect=AssertionError("Real CUDA initialization forbidden")),
        ):
            cpu_payload = self.payload(1)
            self.assertIsNone(cpu_payload["rng"]["torch_cuda"])
            with checkpoints.run_writer(self.run_root, self.run["run_id"]) as writer:
                model, optimizer = checkpoints.restore_checkpoint(
                    cpu_payload, run=self.run, environment=self.environment,
                    training=self.training, writer=writer, checkout_root=self.checkout,
                )
            self.assertEqual(checkpoints.state_fingerprint(model.state_dict()),
                             checkpoints.state_fingerprint(cpu_payload["model_state"]))
            self.assertEqual(checkpoints.state_fingerprint(optimizer.state_dict()),
                             checkpoints.state_fingerprint(cpu_payload["optimizer_state"]))
            checkpoints.torch.cuda.get_rng_state_all.assert_not_called()
            checkpoints.torch.cuda.set_rng_state_all.assert_not_called()
            checkpoints.torch.cuda.init.assert_not_called()
            checkpoints.torch.cuda._lazy_init.assert_not_called()

        source_states = [torch.tensor([0, 17, 34, 51, 68, 85, 102, 119], dtype=torch.uint8)]
        expected_state = source_states[0].clone()
        second_state = torch.tensor([255, 238, 221, 204, 187, 170, 153, 136], dtype=torch.uint8)
        cpu_rng_before = checkpoints.capture_rng()
        observed_environment = dict(os.environ)
        observed_environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        real_cdll = checkpoints.ctypes.CDLL
        library.cuDriverGetVersion.side_effect = version

        def hardware_library(name, *args, **kwargs):
            if name in ("libcuda.so.1", "libnvidia-ml.so.1"):
                return library
            return real_cdll(name, *args, **kwargs)

        def assert_cpu_rng_unchanged(expected):
            current = checkpoints.capture_rng()
            for field in ("python", "numpy", "torch_cpu", "component_generators"):
                self.assertEqual(checkpoints.state_fingerprint(current[field]),
                                 checkpoints.state_fingerprint(expected[field]))

        # Mock environment observations, not the process environment or validators.
        with (
            patch.multiple(
                checkpoints.torch.cuda,
                is_initialized=mock.Mock(return_value=True),
                is_available=mock.Mock(return_value=True),
                device_count=mock.Mock(return_value=1),
                get_device_name=mock.Mock(return_value="synthetic GPU"),
                get_rng_state_all=mock.Mock(return_value=source_states),
                set_rng_state_all=mock.Mock(),
                init=mock.Mock(side_effect=AssertionError("Real CUDA initialization forbidden")),
                _lazy_init=mock.Mock(side_effect=AssertionError("Real CUDA initialization forbidden")),
            ),
            patch.object(checkpoints.torch.version, "cuda", "12.8"),
            patch.object(checkpoints.os, "environ", observed_environment),
            patch.object(checkpoints.ctypes, "CDLL", side_effect=hardware_library),
        ):
            cuda_environment = checkpoints.capture_environment()
            checkpoints.validate_environment(cuda_environment)
            self.assertEqual(cuda_environment["backend"], "cuda")
            self.assertEqual(cuda_environment["device_count"], 1)
            cuda_payload = checkpoints.capture_checkpoint(
                run=self.run, model=model, optimizer=optimizer, training=self.training,
                environment=cuda_environment, position=cpu_payload["position"],
                training_accumulators=cpu_payload["training_accumulators"],
                validation_history=cpu_payload["validation_history"],
                selection_state=cpu_payload["selection_state"],
            )
            captured_rng = cuda_payload["rng"]
            checkpoints.validate_rng(captured_rng)
            self.assertIsNotNone(captured_rng["torch_cuda"])
            self.assertEqual(len(captured_rng["torch_cuda"]), 1)
            checkpoints.torch.cuda.get_rng_state_all.assert_called_once_with()
            actual = captured_rng["torch_cuda"][0]
            self.assertEqual(actual.device.type, "cpu")
            self.assertEqual(actual.dtype, torch.uint8)
            self.assertEqual(actual.ndim, 1)
            self.assertEqual(tuple(actual.shape), (8,))
            self.assertTrue(torch.equal(actual, expected_state))
            source_states[0].zero_()
            self.assertTrue(torch.equal(actual, expected_state))
            for field in ("python", "numpy", "torch_cpu", "component_generators"):
                self.assertEqual(checkpoints.state_fingerprint(captured_rng[field]),
                                 checkpoints.state_fingerprint(cpu_rng_before[field]))
            assert_cpu_rng_unchanged(cpu_rng_before)

            payload_before = checkpoints.state_fingerprint(cuda_payload)
            events = []
            validate_checkpoint = checkpoints.validate_checkpoint
            load_model = checkpoints.CNNRC.load_state_dict
            load_optimizer = checkpoints.torch.optim.Adam.load_state_dict

            def validated_checkpoint(*args, **kwargs):
                checkpoints.torch.cuda.set_rng_state_all.assert_not_called()
                result = validate_checkpoint(*args, **kwargs)
                events.append("checkpoint validated")
                return result

            def loaded_model(instance, *args, **kwargs):
                result = load_model(instance, *args, **kwargs)
                events.append("model loaded")
                return result

            def mocked_transfer(instance, device):
                self.assertEqual(device, "cuda:0")
                events.append("device transfer")
                return instance

            def loaded_optimizer(instance, *args, **kwargs):
                result = load_optimizer(instance, *args, **kwargs)
                events.append("optimizer loaded")
                return result

            def restored_cuda(states):
                self.assertEqual(events, ["checkpoint integrity verified", "checkpoint validated",
                                          "model loaded", "device transfer", "optimizer loaded"])
                self.assertEqual(type(states), list)
                self.assertEqual(len(states), 1)
                self.assertEqual(states[0].dtype, torch.uint8)
                self.assertEqual(tuple(states[0].shape), (8,))
                self.assertTrue(torch.equal(states[0], expected_state))
                source_states[:] = [states[0].clone()]

            checkpoints.torch.cuda.set_rng_state_all.side_effect = restored_cuda
            with checkpoints.run_writer(self.run_root, self.run["run_id"]) as writer:
                path = self.publish(cuda_payload, writer, environment=cuda_environment)
                restored, envelope = checkpoints.load_checkpoint(
                    path, run=self.run, environment=cuda_environment, training=self.training,
                )
                self.assertEqual(envelope["semantic_hash"], payload_before)
                self.assertEqual(checkpoints.state_fingerprint(restored), payload_before)
                checkpoints.torch.cuda.set_rng_state_all.assert_not_called()
                events.append("checkpoint integrity verified")
                with (
                    patch.object(checkpoints, "validate_checkpoint", side_effect=validated_checkpoint),
                    patch.object(checkpoints.CNNRC, "load_state_dict", autospec=True, side_effect=loaded_model),
                    patch.object(checkpoints.CNNRC, "to", autospec=True, side_effect=mocked_transfer) as transfer,
                    patch.object(checkpoints.torch.optim.Adam, "load_state_dict", autospec=True, side_effect=loaded_optimizer),
                ):
                    model, optimizer = checkpoints.restore_checkpoint(
                        restored, run=self.run, environment=cuda_environment, training=self.training,
                        writer=writer, checkout_root=self.checkout,
                    )
                transfer.assert_called_once_with(model, "cuda:0")
            checkpoints.torch.cuda.set_rng_state_all.assert_called_once()
            self.assertEqual(checkpoints.state_fingerprint(model.state_dict()),
                             checkpoints.state_fingerprint(cuda_payload["model_state"]))
            self.assertEqual(checkpoints.state_fingerprint(optimizer.state_dict()),
                             checkpoints.state_fingerprint(cuda_payload["optimizer_state"]))
            assert_cpu_rng_unchanged(captured_rng)

            # Re-verification must neither consume nor restore the mocked state.
            checkpoints.load_checkpoint(path, run=self.run, environment=cuda_environment, training=self.training)
            checkpoints.validate_checkpoint(restored, self.run, cuda_environment, self.training)
            self.assertTrue(torch.equal(source_states[0], expected_state))
            self.assertEqual(checkpoints.state_fingerprint(restored), payload_before)
            self.assertEqual(checkpoints.state_fingerprint(cuda_payload), payload_before)
            checkpoints.torch.cuda.set_rng_state_all.assert_called_once()
            assert_cpu_rng_unchanged(captured_rng)
            checkpoints.torch.cuda.init.assert_not_called()
            checkpoints.torch.cuda._lazy_init.assert_not_called()

            def rng_with_cuda(states):
                value = copy.deepcopy(captured_rng)
                value["torch_cuda"] = states
                return value

            two_device_environment = dict(cuda_environment, device_count=2)
            zero_device_environment = dict(cuda_environment, device_count=0)
            checkpoints.torch.cuda.is_initialized.return_value = False
            cpu_environment = checkpoints.capture_environment()
            checkpoints.torch.cuda.is_initialized.return_value = True
            missing_cuda = copy.deepcopy(captured_rng)
            del missing_cuda["torch_cuda"]
            extra_field = dict(copy.deepcopy(captured_rng), unexpected=None)
            altered_cuda = copy.deepcopy(captured_rng)
            altered_cuda["torch_cuda"][0][0] = 1
            cases = [
                ("two devices and two states", two_device_environment,
                 rng_with_cuda([expected_state, second_state]), "Incomplete single-GPU", True),
                ("two-device environment with one state", two_device_environment,
                 captured_rng, "Incomplete single-GPU", True),
                ("zero-device environment", zero_device_environment,
                 captured_rng, "Incomplete single-GPU", True),
                ("one device with two states", cuda_environment,
                 rng_with_cuda([expected_state, second_state]), "CUDA RNG count differs", True),
                ("one device with zero states", cuda_environment,
                 rng_with_cuda([]), "Malformed CUDA RNG list", True),
                ("CPU with CUDA state", cpu_environment,
                 captured_rng, "CUDA RNG/backend mismatch", True),
                ("CUDA with null state", cuda_environment,
                 rng_with_cuda(None), "CUDA RNG/backend mismatch", True),
                ("wrong container", cuda_environment,
                 rng_with_cuda((expected_state,)), "Malformed CUDA RNG list", True),
                ("wrong element", cuda_environment,
                 rng_with_cuda(["invalid"]), "Malformed RNG tensor", True),
                ("wrong dtype", cuda_environment,
                 rng_with_cuda([expected_state.to(torch.float32)]), "Malformed RNG tensor", True),
                ("wrong rank", cuda_environment,
                 rng_with_cuda([expected_state.reshape(2, 4)]), "Malformed RNG tensor", True),
                ("empty tensor", cuda_environment,
                 rng_with_cuda([torch.empty(0, dtype=torch.uint8)]), "Malformed RNG tensor", True),
                ("missing CUDA field", cuda_environment, missing_cuda, "RNG fields differ", True),
                ("unexpected RNG field", cuda_environment, extra_field, "RNG fields differ", True),
                ("altered CUDA bytes after fingerprint", cuda_environment,
                 altered_cuda, "Tensor inventory differs", False),
            ]

            # Distinguish live state from checkpoint state so partial restoration
            # cannot pass the non-mutation assertions by writing identical values.
            random.random()
            np.random.random()
            torch.rand(1)
            source_states[0].zero_()
            model.running_mean.add_(1)
            next(iter(optimizer.state.values()))["exp_avg"].add_(1)
            model_before = checkpoints.state_fingerprint(model.state_dict())
            optimizer_before = checkpoints.state_fingerprint(optimizer.state_dict())
            rejection_rng_before = checkpoints.capture_rng()
            for name, rejected_environment, rng, error, refresh_inventory in cases:
                bad = copy.deepcopy(cuda_payload)
                bad["rng"] = copy.deepcopy(rng)
                bad["environment_compatibility"] = copy.deepcopy(rejected_environment)
                if refresh_inventory:
                    refresh_tensor_inventory(bad)
                bad_before = checkpoints.state_fingerprint(bad)
                checkpoints.torch.cuda.is_initialized.return_value = rejected_environment["backend"] == "cuda"
                checkpoints.torch.cuda.device_count.return_value = rejected_environment["device_count"]
                checkpoints.torch.cuda.set_rng_state_all.reset_mock()
                with (
                    self.subTest(case=name),
                    patch.object(checkpoints.CNNRC, "load_state_dict") as model_load,
                    patch.object(checkpoints.CNNRC, "to") as transfer,
                    patch.object(checkpoints.torch.optim.Adam, "load_state_dict") as optimizer_load,
                    patch.object(checkpoints.random, "setstate", wraps=random.setstate) as python_restore,
                    patch.object(checkpoints.np.random, "set_state", wraps=np.random.set_state) as numpy_restore,
                    patch.object(checkpoints.torch, "set_rng_state", wraps=torch.set_rng_state) as cpu_restore,
                    checkpoints.run_writer(self.run_root, self.run["run_id"]) as writer,
                ):
                    with self.assertRaisesRegex(ValueError, error):
                        checkpoints.restore_checkpoint(
                            bad, run=self.run, environment=rejected_environment, training=self.training,
                            writer=writer, checkout_root=self.checkout,
                        )
                    checkpoints.torch.cuda.set_rng_state_all.assert_not_called()
                    model_load.assert_not_called()
                    transfer.assert_not_called()
                    optimizer_load.assert_not_called()
                    python_restore.assert_not_called()
                    numpy_restore.assert_not_called()
                    cpu_restore.assert_not_called()
                    self.assertEqual(checkpoints.state_fingerprint(model.state_dict()), model_before)
                    self.assertEqual(checkpoints.state_fingerprint(optimizer.state_dict()), optimizer_before)
                    self.assertEqual(checkpoints.state_fingerprint(bad), bad_before)
                    self.assertEqual(checkpoints.state_fingerprint(source_states),
                                     checkpoints.state_fingerprint(rejection_rng_before["torch_cuda"]))
                    assert_cpu_rng_unchanged(rejection_rng_before)
                checkpoints.torch.cuda.init.assert_not_called()
                checkpoints.torch.cuda._lazy_init.assert_not_called()


if __name__ == "__main__":
    unittest.main()
