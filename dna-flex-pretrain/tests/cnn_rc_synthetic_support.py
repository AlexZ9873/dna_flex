"""B3-only test harness: synthetic B2 stages and temporary tracked inventories.

No production data or environment override is exposed by application code.
Optimizer moments below are assigned literals, never produced by optimization.
"""

import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src import downstream_run as runs
from src import downstream_checkpoint as checkpoints
from src import exd_hox_dataset as datasets
from src.cnn_rc import CNNRC
from tests.test_exd_hox_dataset import SyntheticPublicFiles, _stage_manifest, _json_bytes


class SyntheticCase(unittest.TestCase):
    """Share independent fixture setup, with complete cleanup of test state."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.addCleanup(self.cleanup_files)
        self.saved_rng = checkpoints.capture_rng()
        self.addCleanup(checkpoints.restore_rng, self.saved_rng)
        settings = (
            torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
            torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.get_num_threads(),
        )
        self.addCleanup(self.restore_settings, settings)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_num_threads(1)
        self.environment = checkpoints.capture_environment()
        self.stage = self.root / "stage"
        self.stage.mkdir()
        fixture = SyntheticPublicFiles(self.stage, overshoot=True)
        self.contract = fixture.write()
        self.level_ids = [row["level_id"] for row in fixture.level_rows]
        manifest = _stage_manifest(self.contract)
        (self.stage / datasets.STAGE_MANIFEST_FILENAME).write_bytes(_json_bytes(manifest) + b"\n")
        self.stage_id = "exd_hox_training_stage_" + manifest["manifest_hash"]
        self.original_config = runs.default_config()
        self.config = copy.deepcopy(self.original_config)
        for name in ("split_identity_hash", "split_manifest_hash", "subset_set_manifest_hash"):
            self.config["data"][name] = self.contract[name]
        self.config["data"]["staging_config_sha256"] = hashlib.sha256(_json_bytes(self.contract) + b"\n").hexdigest()
        config_patch = patch.object(runs, "default_config", side_effect=lambda: copy.deepcopy(self.config))
        config_patch.start()
        self.addCleanup(config_patch.stop)
        contract_patch = patch.object(datasets, "_load_contract", return_value=self.contract)
        contract_patch.start()
        self.addCleanup(contract_patch.stop)
        self.data = datasets.open_public_tf_data(self.stage, transcription_factor="Ubx", expected_stage_id=self.stage_id)
        self.training = self.data.dataset("training", level_id=self.level_ids[0])
        self.checkout = self.root / "checkout"
        self.checkout.mkdir()
        self.config_path = self.checkout / "config.json"
        self.config_path.write_bytes(_json_bytes(self.config) + b"\n")
        (self.checkout / "entry.py").write_text("# Synthetic tracked source; never imported.\n")
        self.source_paths = ["config.json", "entry.py"]
        self.git("init", "--quiet")
        self.git("add", "--", *self.source_paths)
        self.git("commit", "--quiet", "-m", "Synthetic B3 source inventory")
        self.commit = self.git("rev-parse", "HEAD").decode().strip()
        self.run = self.resolve()
        self.attempt = runs.attempt_id(self.run["run_id"], "1" * 32)
        self.run_root = self.root / self.run["run_id"]
        self.run_root.mkdir()

    def cleanup_files(self):
        for current, directories, files in os.walk(self.root):
            os.chmod(current, 0o700)
        self.temporary.cleanup()

    def restore_settings(self, settings):
        torch.use_deterministic_algorithms(settings[0], warn_only=settings[1])
        torch.backends.cudnn.benchmark = settings[2]
        torch.backends.cudnn.deterministic = settings[3]
        torch.backends.cuda.matmul.allow_tf32 = settings[4]
        torch.backends.cudnn.allow_tf32 = settings[5]
        torch.set_num_threads(settings[6])

    def git(self, *arguments):
        return subprocess.run([
            "git", "-c", "user.name=Synthetic Fixture", "-c", "user.email=fixture@example.invalid",
            "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", "-C", str(self.checkout), *arguments,
        ], check=True, capture_output=True).stdout

    def resolve(self, **changes):
        arguments = {
            "config_path": self.config_path, "stage_root": self.stage, "transcription_factor": "Ubx",
            "level_id": self.level_ids[0], "expected_stage_id": self.stage_id, "downstream_seed": 33001,
            "candidate_id": "smoke_adam_v1", "checkout_root": self.checkout,
            "expected_software_commit": self.commit, "source_paths": self.source_paths,
        }
        arguments.update(changes)
        return runs.resolve_run(**arguments)

    def payload(self, next_batch=0, epoch=0, phase="train", history=None):
        identity = self.run["identity"]
        seeds = identity["seeds"]["derived_seeds"]
        model = CNNRC(seed=seeds["model_initialization"])
        optimizer = checkpoints.make_optimizer(model, identity["configuration"]["resolved_candidate"]["optimizer"])
        batches = identity["budget"]["batches_per_epoch"]
        update = epoch * batches + next_batch
        if update:
            # Literal populated Adam state and distinct BN buffers, no step.
            for parameter in model.parameters():
                optimizer.state[parameter] = {
                    "step": torch.tensor(float(update), dtype=torch.float32),
                    "exp_avg": torch.full_like(parameter, 0.125),
                    "exp_avg_sq": torch.full_like(parameter, 0.25),
                }
            model.running_mean.fill_(0.375)
            model.running_var.fill_(0.625)
        ids = [self.training[index].metadata.logical_example_id for index in range(len(self.training))]
        order = runs.epoch_order(ids, seeds["training_data_order"], epoch)
        position = {
            "completed_epoch_count": epoch + int(phase == "epoch_complete"), "current_epoch": epoch,
            "next_batch_index": next_batch, "global_update": update, "phase": phase,
            "epoch_permutation_hash": runs.permutation_hash(ids, order),
        }
        count = min(len(self.training), next_batch * 128)
        accumulators = {"example_count": count, "update_count": next_batch,
                        "squared_error_sum": float(count) / 16,
                        "regularization_sum": float(next_batch) / 128,
                        "total_loss_sum": float(next_batch) / 8}
        if history is None:
            history = []
        best = None
        if history:
            event = max(history, key=lambda event: (event["metrics"]["mean"]["r2"], -event["metrics"]["mean"]["rmse"], -event["global_update"]))
            best = {"epoch": event["epoch"], "global_update": event["global_update"],
                    "metrics": event["metrics"]["mean"], "checkpoint_ref": event["checkpoint_ref"]}
        return checkpoints.capture_checkpoint(
            run=self.run, model=model, optimizer=optimizer, training=self.training, environment=self.environment,
            position=position, training_accumulators=accumulators, validation_history=history, selection_state=best,
        )

    def event(self, epoch=0, reference=None, mse=0.0625, r2=0.0):
        update = (epoch + 1) * self.run["identity"]["budget"]["batches_per_epoch"]
        metrics = {"sample_count": 2, "unique_rc_group_count": 2, "mse": mse, "rmse": mse**0.5,
                   "r2": r2, "pearson": 1.0, "spearman": 1.0, "undefined_reasons": {}}
        return {
            "event_id": "validation_event_" + runs.domain_hash("downstream_validation_event.v1", {
                "run_id": self.run["run_id"], "epoch": epoch, "global_update": update}),
            "epoch": epoch, "global_update": update, "checkpoint_ref": {"kind": "self"} if reference is None else reference,
            "metrics": {"forward": copy.deepcopy(metrics), "reverse_complement": copy.deepcopy(metrics), "mean": copy.deepcopy(metrics)},
            "rc_diagnostic": {"atol": 1e-6, "rtol": 1e-5, "maximum_absolute_difference": 0.0,
                              "maximum_normalized_difference": 0.0, "violation_count": 0},
        }

    def validate_payload(self, payload, **changes):
        arguments = {"run": self.run, "environment": self.environment, "training": self.training}
        arguments.update(changes)
        checkpoints.validate_checkpoint(payload, **arguments)

    def publish(self, payload, writer, **changes):
        arguments = {"writer": writer, "attempt_id": self.attempt, "run": self.run,
                     "environment": self.environment, "training": self.training, "checkout_root": self.checkout}
        arguments.update(changes)
        return checkpoints.publish_checkpoint(payload, **arguments)

    def load(self, path):
        return checkpoints.load_checkpoint(path, run=self.run, environment=self.environment, training=self.training)


def rehash_run(run):
    run["run_id"] = "run_" + runs.domain_hash("downstream_scientific_run.v1", run["identity"])
    content = {key: run[key] for key in ("schema_version", "run_id", "identity")}
    run["manifest_hash"] = runs.domain_hash("downstream_resolved_run_manifest.v1", content)


def refresh_tensor_inventory(payload):
    base = dict(payload)
    del base["tensor_inventory"]
    payload["tensor_inventory"] = checkpoints.semantic_state(base)[2]
