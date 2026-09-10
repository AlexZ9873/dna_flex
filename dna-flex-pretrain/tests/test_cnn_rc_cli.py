"""Hermetic CLI checks against actual, temporary committed B3b source bytes."""

import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
import unittest

from tests.cnn_rc_synthetic_support import SyntheticCase
from tests.test_exd_hox_dataset import _json_bytes


TRAIN_MODULE = "scripts.downstream.train_cnn_rc"
VALIDATE_MODULE = "scripts.downstream.validate_cnn_rc"
CANDIDATE_PATHS = (
    "src/cnn_rc_training.py", "scripts/downstream/train_cnn_rc.py",
    "scripts/downstream/validate_cnn_rc.py", "tests/test_cnn_rc_training.py",
    "tests/test_cnn_rc_cli.py",
)


class B3bSyntheticCase(SyntheticCase):
    """Run unchanged candidate modules from a fully committed temporary tree."""

    def setUp(self):
        super().setUp()
        from src import cnn_rc_training as training_module

        self.project_root = Path(__file__).resolve().parents[1]
        self.source_paths = list(training_module.SOURCE_PATHS)
        support_paths = (
            "tests/cnn_rc_synthetic_support.py", "tests/test_exd_hox_dataset.py",
        )
        for relative in sorted(set(self.source_paths + list(CANDIDATE_PATHS) + list(support_paths))):
            destination = self.checkout / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.project_root / relative, destination)
        self.config_path = self.checkout / "configs/exd_hox_cnn_rc_v1.json"
        self.config_path.write_bytes(_json_bytes(self.config) + b"\n")
        (self.checkout / "configs/carc_exd_hox_training_staging_v1.json").write_bytes(
            _json_bytes(self.contract) + b"\n")
        self.git("add", "--", ".")
        self.git("commit", "--quiet", "-m", "Synthetic committed B3b candidate")
        self.commit = self.git("rev-parse", "HEAD").decode().strip()
        self.run = self.resolve()
        self.output_root = self.root / "output"
        self.attempt_root = self.root / "attempts"
        self.harness = textwrap.dedent("""\
            import builtins
            import copy
            import json
            import os
            from pathlib import Path
            import runpy
            import sys
            import torch
            from src import downstream_run as runs
            from src import exd_hox_dataset as datasets
            torch.set_num_threads(1)
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision('highest')
            fixture_config = json.loads(Path('configs/exd_hox_cnn_rc_v1.json').read_text())
            fixture_contract = json.loads(Path('configs/carc_exd_hox_training_staging_v1.json').read_text())
            runs.default_config = lambda: copy.deepcopy(fixture_config)
            datasets._load_contract = lambda: copy.deepcopy(fixture_contract)
            original_import = builtins.__import__
            def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
                forbidden = {'src.sealed_test_access', 'src.exd_hox_splits'}
                requested = {name}
                if fromlist:
                    requested.update(name + '.' + item for item in fromlist)
                if requested & forbidden:
                    raise AssertionError('Forbidden transitive import: ' + name)
                return original_import(name, globals, locals, fromlist, level)
            builtins.__import__ = guarded_import
            def deny_external_access(event, arguments):
                if event.startswith('socket.'):
                    raise AssertionError('Network access is forbidden in the synthetic harness.')
                if event == 'open' and isinstance(arguments[0], (str, bytes, os.PathLike)):
                    path = Path(os.fsdecode(arguments[0]))
                    if path.suffix.lower() in {'.h5', '.hdf5'} or 'sealed' in path.parts:
                        raise AssertionError('Raw or sealed data access is forbidden.')
            sys.addaudithook(deny_external_access)
            """)
        self.harness += "\nproduction_data_root = Path(" + repr(str(self.project_root / "data")) + ")\n"
        self.harness += textwrap.dedent("""\
            def deny_production_data(event, arguments):
                if event == 'open' and isinstance(arguments[0], (str, bytes, os.PathLike)):
                    path = Path(os.path.abspath(os.fsdecode(arguments[0])))
                    if path == production_data_root or production_data_root in path.parents:
                        raise AssertionError('Production biological artifacts are forbidden.')
            sys.addaudithook(deny_production_data)
            """)

    def training_keywords(self, **changes):
        arguments = {
            "stage_root": self.stage, "expected_stage_id": self.stage_id,
            "config": self.config_path, "tf": "Ubx", "level_id": self.level_ids[0],
            "downstream_seed": 33001, "candidate_id": "smoke_adam_v1",
            "output_root": self.output_root, "attempt_root": self.attempt_root,
            "expected_software_commit": self.commit, "device": "cpu",
        }
        arguments.update(changes)
        return arguments

    def train_arguments(self, **changes):
        arguments = []
        for name, value in self.training_keywords(**changes).items():
            if value is not None:
                arguments.extend(("--" + name.replace("_", "-"), str(value)))
        return arguments

    def run_script(self, source):
        """Execute test instrumentation without adding any application bypass."""
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["OMP_NUM_THREADS"] = "1"
        environment["MKL_NUM_THREADS"] = "1"
        environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        environment.pop("PYTHONPATH", None)
        return subprocess.run(
            [sys.executable, "-c", self.harness + "\n" + textwrap.dedent(source)],
            cwd=self.checkout, env=environment, capture_output=True, text=True,
            check=False, timeout=120,
        )

    def cli(self, module, arguments, injection=""):
        source = injection + "\nsys.argv = " + repr([module] + list(arguments))
        source += "\nrunpy.run_module(" + repr(module) + ", run_name='__main__')\n"
        return self.run_script(source)

    def successful_training(self, **changes):
        result = self.cli(TRAIN_MODULE, self.train_arguments(**changes))
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        return json.loads(result.stdout)

    def assert_failure(self, result, message=None):
        self.assertNotEqual(result.returncode, 0, result.stdout)
        if message is not None:
            self.assertIn(message, result.stderr)


class ArgumentInterfaceTests(B3bSyntheticCase):
    """Reject unsupported scientific choices before any successful run."""

    def test_exact_public_options_and_no_hidden_modes(self):
        expected = {
            TRAIN_MODULE: {
                "--help", "--stage-root", "--expected-stage-id", "--config", "--tf",
                "--level-id", "--downstream-seed", "--candidate-id", "--output-root",
                "--attempt-root", "--expected-software-commit", "--device", "--resume-checkpoint",
            },
            VALIDATE_MODULE: {
                "--help", "--stage-root", "--expected-stage-id", "--checkpoint",
                "--output-root", "--attempt-root", "--expected-software-commit", "--device",
            },
        }
        for module, options in expected.items():
            source = "from " + module + " import argument_parser\n"
            source += "parser = argument_parser()\n"
            source += "print(json.dumps({'abbreviation': parser.allow_abbrev, 'options': sorted(name for action in parser._actions for name in action.option_strings if name.startswith('--'))}))"
            result = self.run_script(source)
            self.assertEqual(result.returncode, 0, result.stderr)
            parsed = json.loads(result.stdout)
            self.assertFalse(parsed["abbreviation"])
            self.assertEqual(set(parsed["options"]), options)

    def test_canonical_source_inventory_and_forbidden_imports(self):
        expected = sorted((
            "configs/carc_exd_hox_training_staging_v1.json", "configs/exd_hox_cnn_rc_v1.json",
            "scripts/downstream/train_cnn_rc.py", "scripts/downstream/validate_cnn_rc.py",
            "src/__init__.py", "src/cnn_rc.py", "src/cnn_rc_training.py",
            "src/downstream_checkpoint.py", "src/downstream_fingerprints.py",
            "src/downstream_metrics.py", "src/downstream_run.py", "src/exd_hox_dataset.py",
        ))
        self.assertEqual(self.source_paths, expected)
        forbidden = {"src.sealed_test_access", "src.exd_hox_splits"}
        for relative in self.source_paths:
            if relative.endswith(".py"):
                tree = ast.parse((self.checkout / relative).read_text())
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        self.assertFalse({item.name for item in node.names} & forbidden)
                    elif isinstance(node, ast.ImportFrom):
                        self.assertNotIn(node.module, forbidden)
                        if node.module == "src":
                            self.assertFalse({"src." + item.name for item in node.names} & forbidden)

    def test_exact_accepted_argument_values_and_types(self):
        source = "from scripts.downstream.train_cnn_rc import argument_parser\n"
        source += "arguments = vars(argument_parser().parse_args(" + repr(self.train_arguments()) + "))\n"
        source += "assert type(arguments['downstream_seed']) is int\n"
        source += "assert all(isinstance(arguments[name], Path) for name in ('stage_root', 'config', 'output_root', 'attempt_root'))\n"
        source += "assert arguments.pop('resume_checkpoint') is None\n"
        source += "print(json.dumps(arguments, default=str, sort_keys=True))\n"
        result = self.run_script(source)
        self.assertEqual(result.returncode, 0, result.stderr)
        expected = {}
        for key, value in self.training_keywords().items():
            expected[key] = str(value) if isinstance(value, Path) else value
        self.assertEqual(json.loads(result.stdout), expected)

    def test_test_harness_blocks_network_raw_sealed_and_production_data(self):
        result = self.run_script("""\
            import socket
            forbidden_operations = (
                lambda: socket.socket(),
                lambda: Path('raw.hdf5').read_bytes(),
                lambda: Path('sealed/targets').read_bytes(),
                lambda: (production_data_root / 'unread').read_bytes(),
                lambda: __import__('src.sealed_test_access'),
                lambda: __import__('src.exd_hox_splits'),
            )
            for operation in forbidden_operations:
                try:
                    operation()
                except AssertionError:
                    pass
                else:
                    raise AssertionError('Forbidden operation was accepted.')
            assert 'src.sealed_test_access' not in sys.modules
            assert 'src.exd_hox_splits' not in sys.modules
            print('all access guards active')
            """)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "all access guards active")

    def test_abbreviations_unknown_and_forbidden_options_rejected(self):
        # Parsing all cases within a child avoids a new torch import per flag.
        forbidden = (
            "--sta", "--unknown", "--test-split", "--sealed-target-path",
            "--raw-input-table", "--hdf5-path", "--sample-list", "--requested-count",
            "--batch-size", "--learning-rate", "--regularization", "--callbacks",
            "--plot", "--num-workers", "--amp", "--compile", "--distributed",
            "--skip-verification", "--fixture", "--fixture-mode", "--test",
        )
        validation = [
            "--stage-root", str(self.stage), "--expected-stage-id", self.stage_id,
            "--checkpoint", "absent", "--output-root", str(self.output_root),
            "--attempt-root", str(self.attempt_root), "--expected-software-commit", self.commit,
            "--device", "cpu",
        ]
        source = "from scripts.downstream import train_cnn_rc, validate_cnn_rc\n"
        source += "cases = [(train_cnn_rc, " + repr(self.train_arguments()) + ", " + repr(forbidden) + "), (validate_cnn_rc, " + repr(validation) + ", " + repr(forbidden + ("--tf", "--level-id", "--config", "--candidate-id", "--downstream-seed")) + ")]\n"
        source += textwrap.dedent("""\
            for module, base_arguments, flags in cases:
                for flag in flags:
                    try:
                        module.argument_parser().parse_args(base_arguments + [flag, 'forbidden'])
                    except SystemExit as error:
                        assert error.code == 2, (flag, error.code)
                    else:
                        raise AssertionError('Unexpected accepted option: ' + flag)
            print('all forbidden options rejected')
            """)
        result = self.run_script(source)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("all forbidden options rejected", result.stdout)

    def test_unsupported_tf_level_seed_and_candidate(self):
        for changes in ({"tf": "not_a_tf"}, {"level_id": "lvl_" + "0" * 64},
                        {"downstream_seed": 17}, {"candidate_id": "unapproved"}):
            with self.subTest(changes=changes):
                self.assert_failure(self.cli(TRAIN_MODULE, self.train_arguments(**changes)))
        self.assertFalse(self.output_root.exists())

    def test_wrong_commit_and_stage_identity(self):
        for changes in ({"expected_software_commit": "0" * 40},
                        {"expected_software_commit": self.commit[:12]},
                        {"expected_stage_id": "exd_hox_training_stage_" + "0" * 64}):
            with self.subTest(changes=changes):
                self.assert_failure(self.cli(TRAIN_MODULE, self.train_arguments(**changes)))

    def test_dirty_tracked_checkout_and_altered_production_source(self):
        for relative in ("entry.py", "src/cnn_rc.py"):
            path = self.checkout / relative
            original = path.read_bytes()
            path.write_bytes(original + b"\n# Deliberate hermetic dirt.\n")
            self.assert_failure(self.cli(TRAIN_MODULE, self.train_arguments()), "Dirty tracked state")
            path.write_bytes(original)

    def test_untracked_inventory_source_rejected(self):
        relative = "scripts/downstream/validate_cnn_rc.py"
        self.git("rm", "--cached", "--", relative)
        self.git("commit", "--quiet", "-m", "Synthetic missing production inventory source")
        self.commit = self.git("rev-parse", "HEAD").decode().strip()
        self.assertTrue((self.checkout / relative).exists())
        self.assert_failure(self.cli(TRAIN_MODULE, self.train_arguments()))

    def test_untracked_namespace_initializers_rejected(self):
        for relative in ("scripts/__init__.py", "scripts/downstream/__init__.py"):
            path = self.checkout / relative
            path.write_bytes(b"# Synthetic namespace-shadowing source.\n")
            self.assert_failure(self.cli(TRAIN_MODULE, self.train_arguments()))
            path.unlink()

    def test_overlapping_roots_and_symlink_confinement(self):
        cases = (
            {"output_root": self.stage}, {"attempt_root": self.stage},
            {"output_root": self.stage / "inside"}, {"attempt_root": self.root},
            {"attempt_root": self.output_root},
            {"attempt_root": self.output_root / "inside"},
        )
        for changes in cases:
            with self.subTest(changes=changes):
                self.assert_failure(self.cli(TRAIN_MODULE, self.train_arguments(**changes)))
        link = self.root / "stage-link"
        link.symlink_to(self.stage, target_is_directory=True)
        self.assert_failure(self.cli(TRAIN_MODULE, self.train_arguments(output_root=link)))

    def test_only_one_cpu_or_cuda_device_option(self):
        for device in ("cuda", "cuda:1", "cpu,cuda:0", "mps"):
            with self.subTest(device=device):
                self.assert_failure(self.cli(TRAIN_MODULE, self.train_arguments(device=device)), "invalid choice")

    def test_publication_uncertainty_has_explicit_exit_status(self):
        for module, function in ((TRAIN_MODULE, "train_cnn_rc"), (VALIDATE_MODULE, "verify_cnn_rc")):
            source = "from " + module + " import main\n"
            source += "import " + module + " as entry\n"
            source += "from src.downstream_checkpoint import PublishedDurabilityError\n"
            source += "def uncertain(**arguments):\n    raise PublishedDurabilityError(Path('published'))\n"
            source += "entry." + function + " = uncertain\n"
            arguments = self.train_arguments()
            if module == VALIDATE_MODULE:
                arguments = ["--stage-root", str(self.stage), "--expected-stage-id", self.stage_id,
                             "--checkpoint", "bundle", "--output-root", str(self.output_root),
                             "--attempt-root", str(self.attempt_root), "--expected-software-commit", self.commit,
                             "--device", "cpu"]
            source += "raise SystemExit(main(" + repr(arguments) + "))\n"
            result = self.run_script(source)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertEqual(json.loads(result.stderr)["status"], "published_but_durability_unconfirmed")


class SyntheticCommandTests(B3bSyntheticCase):
    """Exercise actual training, recovery and verification entry points."""

    def snapshot_files(self, root):
        result = {}
        for path in sorted(root.rglob("*")):
            if path.is_file():
                result[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        return result

    def verification_arguments(self, checkpoint, output_name="verification"):
        return [
            "--stage-root", str(self.stage), "--expected-stage-id", self.stage_id,
            "--checkpoint", str(checkpoint), "--output-root", str(self.root / output_name),
            "--attempt-root", str(self.root / (output_name + "-attempts")),
            "--expected-software-commit", self.commit, "--device", "cpu",
        ]

    def assert_bundle_fingerprints(self, path):
        envelope = json.loads((path / "manifest.json").read_text())
        self.assertEqual(set(envelope), {
            "schema_version", "run_id", "producing_attempt_id", "files", "manifest_hash",
        })
        self.assertEqual(envelope["schema_version"], "cnn_rc_result_publication.v1")
        paths = []
        for fingerprint in envelope["files"]:
            self.assertEqual(set(fingerprint), {
                "path", "byte_size", "sha256", "schema_version", "semantic_hash",
            })
            paths.append(fingerprint["path"])
            raw = (path / fingerprint["path"]).read_bytes()
            self.assertEqual(len(raw), fingerprint["byte_size"])
            self.assertEqual(hashlib.sha256(raw).hexdigest(), fingerprint["sha256"])
            record = json.loads(raw)
            self.assertEqual(record["manifest_hash"], fingerprint["semantic_hash"])
            self.assertEqual(record["schema_version"], fingerprint["schema_version"])
        self.assertEqual(paths, sorted(set(paths)))
        self.assertEqual({entry.name for entry in path.iterdir()}, set(paths) | {"manifest.json"})

    def test_two_epoch_training_immutable_artifacts_and_repeated_completion(self):
        from src import cnn_rc_training as training_module

        result = self.successful_training()
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["run_id"], self.run["run_id"])
        self.assertEqual(result["global_update"], 4)
        self.assertEqual(result["consumed_validation_event_count"], 2)
        self.assertEqual(result["consumed_selection_count"], 2)
        root = self.output_root / result["run_id"]
        completion = training_module.read_artifacts(root / "completion", self.run)
        self.assertEqual(set(completion), {
            "completion.json", "validation-summary.json", "inventory.json", "attempt-terminal.json",
        })
        self.assertEqual(completion["completion.json"], result)
        summary = completion["validation-summary.json"]
        self.assertEqual(summary["selection"], self.run["identity"]["selection"])
        self.assertEqual(summary["selection"]["actual_logical_example_count"], 129)
        self.assertEqual(summary["rc_diagnostic"]["violation_count"], 0)
        self.assertEqual(set(summary["metrics"]), {"forward", "reverse_complement", "mean"})
        for values in summary["metrics"].values():
            self.assertEqual(values["sample_count"], 2)
            self.assertEqual(values["unique_rc_group_count"], 2)
        self.assertEqual(len(list(root.glob("validation_event_*"))), 2)
        self.assertEqual(len(list(root.glob("update-*"))), 4)
        for epoch in (0, 1):
            record = training_module.read_artifacts(root / ("epoch-" + str(epoch)), self.run)["record.json"]
            self.assertEqual(record["training"]["example_count"], 129)
            self.assertEqual(record["training"]["update_count"], 2)
            self.assertEqual(record["training"]["mse_example_weighted"],
                             record["training"]["squared_error_sum"] / 129)
        for path in root.iterdir():
            if path.is_dir() and (path / "manifest.json").exists():
                envelope = json.loads((path / "manifest.json").read_text())
                if envelope["schema_version"] == "cnn_rc_result_publication.v1":
                    self.assert_bundle_fingerprints(path)
                    training_module.read_artifacts(path, self.run)
        for fingerprint in completion["inventory.json"]["files"]:
            raw = (root / fingerprint["path"]).read_bytes()
            self.assertEqual(len(raw), fingerprint["byte_size"])
            self.assertEqual(hashlib.sha256(raw).hexdigest(), fingerprint["sha256"])
        before = self.snapshot_files(root)
        self.assert_failure(self.cli(TRAIN_MODULE, self.train_arguments()), "completed runs are immutable")
        terminal = next(root.glob("update-*" + result["terminal_recovery_checkpoint_id"]))
        self.assert_failure(self.cli(TRAIN_MODULE, self.train_arguments(resume_checkpoint=terminal)),
                            "Completed run cannot be resumed")
        self.assertEqual(self.snapshot_files(root), before)
        self.assertFalse(any(path.name == "sealed" for path in self.root.rglob("*")))

    def test_cli_batch_boundary_resume_matches_uninterrupted_checkpoints(self):
        reference = self.successful_training(output_root=self.root / "reference",
                                             attempt_root=self.root / "reference-attempts")
        injection = textwrap.dedent("""\
            from src import cnn_rc_training as training_module
            original_step = training_module._TrainingRun.step
            def interrupt_after_batch(session):
                original_step(session)
                if session.state['position']['global_update'] == 1:
                    raise KeyboardInterrupt('synthetic nonterminal batch interruption')
            training_module._TrainingRun.step = interrupt_after_batch
            """)
        interrupted = self.cli(TRAIN_MODULE, self.train_arguments(), injection)
        self.assert_failure(interrupted)
        root = self.output_root / self.run["run_id"]
        checkpoints = list(root.glob("update-*-train-ckpt_*"))
        self.assertEqual(len(checkpoints), 1)
        self.assertFalse((root / "completion").exists())
        resumed = self.successful_training(resume_checkpoint=checkpoints[0])
        for key in ("run_id", "selected_checkpoint_id", "terminal_recovery_checkpoint_id",
                    "validation_event_id", "global_update", "consumed_validation_event_count",
                    "consumed_selection_count", "status"):
            self.assertEqual(resumed[key], reference[key])
        reference_root = self.root / "reference" / reference["run_id"]
        for key in ("selected_checkpoint_id", "terminal_recovery_checkpoint_id"):
            resumed_path = next(root.glob("update-*" + resumed[key]))
            reference_path = next(reference_root.glob("update-*" + reference[key]))
            self.assertEqual((resumed_path / "state.pt").read_bytes(),
                             (reference_path / "state.pt").read_bytes())

    def test_standalone_verification_preserves_selection_and_never_evaluates(self):
        result = self.successful_training()
        source = self.output_root / result["run_id"]
        selected = next(source.glob("update-*" + result["selected_checkpoint_id"]))
        before = self.snapshot_files(source)
        arguments = self.verification_arguments(selected)
        injection = textwrap.dedent("""\
            from src import cnn_rc_training as training_module
            from src.cnn_rc import CNNRC
            def forbidden_evaluation(*arguments, **keywords):
                raise AssertionError('Standalone verification must reuse committed validation.')
            training_module.evaluate_validation = forbidden_evaluation
            CNNRC.forward = forbidden_evaluation
            """)
        verification = self.cli(VALIDATE_MODULE, arguments, injection)
        self.assertEqual(verification.returncode, 0, verification.stderr)
        report = json.loads(verification.stdout)
        self.assertEqual(report["status"], "succeeded")
        self.assertEqual(report["checkpoint_id"], result["selected_checkpoint_id"])
        self.assertEqual(report["source_completion_hash"], result["manifest_hash"])
        self.assertTrue(report["selection_unchanged"])
        self.assertEqual(report["additional_selection_evaluations"], 0)
        self.assertEqual(self.snapshot_files(source), before)
        self.assert_bundle_fingerprints(self.root / "verification" / result["run_id"] / "completion")
        self.assert_failure(self.cli(VALIDATE_MODULE, arguments), "Verification output already exists")
        self.assertEqual(self.snapshot_files(source), before)

    def test_artifact_schema_and_raw_fingerprint_corruption_fail_closed(self):
        from src import cnn_rc_training as training_module
        from src import downstream_run as runs

        result = self.successful_training()
        root = self.output_root / result["run_id"]
        for change in ("unknown", "missing", "nonfinite"):
            malformed = copy.deepcopy(result)
            if change == "unknown":
                malformed["arbitrary_notes"] = "forbidden"
            elif change == "missing":
                del malformed["global_update"]
            else:
                malformed["global_update"] = float("nan")
            if change != "nonfinite":
                content = dict(malformed)
                del content["manifest_hash"]
                malformed["manifest_hash"] = runs.domain_hash(malformed["schema_version"], content)
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    training_module.validate_artifact(malformed, self.run)
        bundle = root / "completion"
        path = bundle / "validation-summary.json"
        original = path.read_bytes()
        path.chmod(0o600)
        path.write_bytes(original + b" ")
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            training_module.read_artifacts(bundle, self.run)
        path.write_bytes(original)
        self.assertEqual(training_module.read_artifacts(bundle, self.run)["completion.json"], result)
        envelope_path = bundle / "manifest.json"
        envelope_original = envelope_path.read_bytes()
        envelope = json.loads(envelope_original)
        envelope["producing_attempt_id"] = "attempt_" + "0" * 64
        self.assertNotEqual(envelope["producing_attempt_id"],
                            json.loads(envelope_original)["producing_attempt_id"])
        del envelope["manifest_hash"]
        envelope["manifest_hash"] = runs.domain_hash(envelope["schema_version"], envelope)
        internal_before = {item.name: item.read_bytes() for item in bundle.iterdir()
                           if item.name != "manifest.json"}
        envelope_path.chmod(0o600)
        envelope_path.write_bytes(_json_bytes(envelope) + b"\n")
        try:
            with self.assertRaisesRegex(ValueError, "producer differs"):
                training_module.read_artifacts(bundle, self.run)
            self.assertEqual({item.name: item.read_bytes() for item in bundle.iterdir()
                              if item.name != "manifest.json"}, internal_before)
        finally:
            envelope_path.write_bytes(envelope_original)
        self.assertEqual(training_module.read_artifacts(bundle, self.run)["completion.json"], result)

        # A completed run cannot resume. Verify its selected checkpoint through
        # the public CLI so the checkpoint integrity check is reached directly.
        selected = next(root.glob("update-*" + result["selected_checkpoint_id"]))
        copied_output = self.root / "corrupted-checkpoint-source"
        shutil.copytree(self.output_root, copied_output)
        corrupted = copied_output / result["run_id"] / selected.name
        arguments = self.verification_arguments(corrupted, "corrupted-checkpoint-verification")
        source = "arguments = " + repr(arguments) + "\n"
        source += "baseline_output = Path(" + repr(str(self.output_root)) + ")\n"
        source += "baseline_attempts = Path(" + repr(str(self.attempt_root)) + ")\n"
        source += "expected_run_id = " + repr(result["run_id"]) + "\n"
        source += "expected_checkpoint_id = " + repr(result["selected_checkpoint_id"]) + "\n"
        source += '''
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import hashlib
import io
import random
import stat
import struct
import zipfile
from unittest.mock import patch
import numpy as np
from scripts.downstream import validate_cnn_rc as entry
from src import cnn_rc_training as implementation
from src import downstream_checkpoint as checkpoints

options = entry.argument_parser().parse_args(arguments)
checkpoint_path = options.checkpoint
original_path = baseline_output / expected_run_id / checkpoint_path.name
copied_output = checkpoint_path.parent.parent
run = runs.strict_json(runs.read_regular(checkpoint_path.parent / 'resolved-run/record.json'))
assert run['run_id'] == expected_run_id
assert run['identity']['software']['runtime_commit'] == options.expected_software_commit
data = datasets.open_public_tf_data(options.stage_root, transcription_factor='Ubx',
                                    expected_stage_id=options.expected_stage_id)
training = data.dataset('training', level_id=run['identity']['selection']['requested_level_id'])
environment = implementation.configure_runtime('cpu')
with patch.object(checkpoints, 'validate_checkpoint', wraps=checkpoints.validate_checkpoint) as validator:
    original_payload, original_envelope = checkpoints.load_checkpoint(
        original_path, run=run, environment=environment, training=training)
    payload, envelope = checkpoints.load_checkpoint(
        checkpoint_path, run=run, environment=environment, training=training)
    assert validator.call_count == 2
assert envelope == original_envelope
assert envelope['checkpoint_id'] == expected_checkpoint_id
assert payload['run_id'] == expected_run_id
assert payload['software_identity'] == run['identity']['software']
assert payload['position']['phase'] == 'epoch_complete'
assert checkpoints.state_fingerprint(payload) == checkpoints.state_fingerprint(original_payload)

def inventory(directory):
    records = {}
    for path in [directory] + sorted(directory.rglob('*')):
        assert not path.is_symlink(), path
        raw = path.read_bytes() if path.is_file() else None
        records[path.relative_to(directory).as_posix()] = (path.stat().st_mode, raw)
    return records

def fingerprints(records):
    return {name: {'byte_size': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
            for name, (_, raw) in records.items() if raw is not None}

baseline_before = inventory(baseline_output)
attempts_before = inventory(baseline_attempts)
copy_before = inventory(copied_output)
assert copy_before == baseline_before
bundle_before = fingerprints(inventory(checkpoint_path))
index_paths = sorted(checkpoint_path.parent.glob('index-*/index.json'))
assert index_paths
latest_index = runs.strict_json(runs.read_regular(index_paths[-1]))
checkpoints.validate_index(latest_index)
assert latest_index['selected_best_checkpoint_id'] == expected_checkpoint_id
assert any(item['checkpoint_id'] == expected_checkpoint_id and item['path'] == checkpoint_path.name
           and item['file'] == envelope['file'] for item in latest_index['entries'])
result_fingerprints = fingerprints(inventory(checkpoint_path.parent))
assert any(name.startswith('validation_event_') for name in result_fingerprints)
assert 'completion/completion.json' in result_fingerprints
assert 'completion/validation-summary.json' in result_fingerprints
assert not options.output_root.exists() and not options.attempt_root.exists()

# Flip one tensor-storage byte without changing archive structure or any
# envelope/index bytes. The accepted raw hash guard must precede torch.load.
state_path = checkpoint_path / 'state.pt'
raw = state_path.read_bytes()
with zipfile.ZipFile(io.BytesIO(raw)) as archive:
    storage = next(item for item in archive.infolist()
                   if '/data/' in item.filename and item.file_size > 0)
    assert storage.compress_type == zipfile.ZIP_STORED
    assert raw[storage.header_offset:storage.header_offset + 4] == b'PK' + bytes((3, 4))
    name_length, extra_length = struct.unpack_from('<HH', raw, storage.header_offset + 26)
    offset = storage.header_offset + 30 + name_length + extra_length
changed = bytearray(raw)
changed[offset] ^= 1
assert len(changed) == envelope['file']['byte_size']
assert sum(first != second for first, second in zip(raw, changed)) == 1
mode = stat.S_IMODE(state_path.stat().st_mode)
state_path.chmod(0o600)
state_path.write_bytes(changed)
state_path.chmod(mode)
copy_corrupted = inventory(copied_output)
relative_payload = state_path.relative_to(copied_output).as_posix()
assert set(copy_corrupted) == set(copy_before)
assert [name for name in copy_before if copy_before[name] != copy_corrupted[name]] == [relative_payload]
assert (checkpoint_path / 'manifest.json').read_bytes() == (original_path / 'manifest.json').read_bytes()
with zipfile.ZipFile(io.BytesIO(changed)) as corrupted_archive:
    assert any(item.filename == storage.filename and item.header_offset == storage.header_offset
               and item.file_size == storage.file_size for item in corrupted_archive.infolist())

calls = []
actual_verify = implementation.verify_cnn_rc
actual_load = checkpoints.load_checkpoint
def observed_verify(**keywords):
    assert keywords['checkpoint'] == checkpoint_path
    calls.append('B3b.verify_cnn_rc')
    return actual_verify(**keywords)

def observed_load(path, **keywords):
    assert calls == ['B3b.verify_cnn_rc'], calls
    assert path == checkpoint_path and keywords['run'] == run
    assert keywords['environment'] == environment
    calls.append('B3a.load_checkpoint')
    try:
        return actual_load(path, **keywords)
    except runs.RunContractError as error:
        assert str(error) == 'Raw checkpoint hash differs.', str(error)
        calls.append('RunContractError: ' + str(error))
        raise

# Observe every mutation boundary, including CPU and mocked CUDA RNG setters.
# AssertionError is intentionally outside the CLI's caught exception classes.
guard_targets = [
    (torch, 'load'), (implementation.models.CNNRC, '__init__'),
    (implementation.models.CNNRC, 'load_state_dict'), (implementation.models.CNNRC, 'forward'),
    (torch.optim.Adam, 'load_state_dict'), (torch.optim.Adam, 'step'),
    (random, 'setstate'), (np.random, 'set_state'), (torch, 'set_rng_state'),
    (torch.cuda, 'set_rng_state'), (torch.cuda, 'set_rng_state_all'), (torch.cuda, 'init'),
    (checkpoints, 'restore_checkpoint'), (checkpoints, 'restore_rng'),
    (checkpoints, 'seed_runtime'), (checkpoints, 'make_optimizer'),
    (checkpoints, 'recovery_state'), (checkpoints, 'publish_checkpoint'),
    (checkpoints, 'publish_index'), (checkpoints, 'publish_bundle'), (checkpoints, 'run_writer'),
    (implementation, 'training_step'), (implementation, 'evaluate_validation'),
    (implementation, 'select_best'), (implementation, '_load_indexes'),
    (implementation, '_attempt'), (implementation, '_mkdir'), (implementation, 'train_cnn_rc'),
    (implementation._TrainingRun, 'initialize'), (implementation._TrainingRun, 'step'),
    (implementation._TrainingRun, 'validate_epoch'), (implementation._TrainingRun, 'commit_validation'),
    (implementation._TrainingRun, 'finish'), (runs, 'resolve_run'),
]
stdout, stderr = io.StringIO(), io.StringIO()
rng_before = checkpoints.state_fingerprint(checkpoints.capture_rng())
operation_active = False
def forbid_operation_writes_and_test_access(event, values):
    if operation_active and event == 'open':
        assert not values[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC), values
        if isinstance(values[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(values[0]))
            assert not {'test', 'sealed'} & set(path.parts), path
            assert not path.name.startswith(('test.', 'test_')), path
sys.addaudithook(forbid_operation_writes_and_test_access)
with ExitStack() as stack:
    verify_call = stack.enter_context(patch.object(entry, 'verify_cnn_rc', side_effect=observed_verify))
    load_call = stack.enter_context(patch.object(checkpoints, 'load_checkpoint', side_effect=observed_load))
    guards = []
    for owner, name in guard_targets:
        guards.append(stack.enter_context(patch.object(
            owner, name, side_effect=AssertionError('Forbidden checkpoint side effect: ' + name))))
    stack.enter_context(redirect_stdout(stdout))
    stack.enter_context(redirect_stderr(stderr))
    operation_active = True
    try:
        status = entry.main(arguments)
    finally:
        operation_active = False
    verify_call.assert_called_once()
    load_call.assert_called_once()
    for guard in guards:
        guard.assert_not_called()
assert status == 1 and stdout.getvalue() == '', (status, stdout.getvalue(), stderr.getvalue())
assert json.loads(stderr.getvalue()) == {'status': 'failed', 'error': 'Raw checkpoint hash differs.'}
assert calls == ['B3b.verify_cnn_rc', 'B3a.load_checkpoint', 'RunContractError: Raw checkpoint hash differs.']
assert checkpoints.state_fingerprint(checkpoints.capture_rng()) == rng_before
assert inventory(baseline_output) == baseline_before
assert inventory(baseline_attempts) == attempts_before
assert inventory(copied_output) == copy_corrupted
assert not options.output_root.exists() and not options.attempt_root.exists()
assert state_path.read_bytes() == bytes(changed)
print(json.dumps({
    'checkpoint_id': expected_checkpoint_id, 'phase': payload['position']['phase'],
    'run_id': expected_run_id, 'software_commit': options.expected_software_commit,
    'checkpoint_path': str(checkpoint_path), 'payload_path': str(state_path),
    'envelope_path': str(checkpoint_path / 'manifest.json'),
    'index_paths': [str(path) for path in index_paths], 'bundle_before': bundle_before,
    'corrupted_byte_offset': offset, 'cli_status': status, 'boundary_calls': calls,
    'forbidden_call_count': sum(guard.call_count for guard in guards),
    'output_file_count': len(fingerprints(baseline_before)),
    'attempt_file_count': len(fingerprints(attempts_before)),
    'preserved': True,
}, sort_keys=True))
'''
        with self.subTest(change="checkpoint_payload_integrity_at_b3b_public_boundary"):
            verification = self.run_script(source)
            self.assertEqual(verification.returncode, 0, verification.stderr + verification.stdout)
            evidence = json.loads(verification.stdout)
            self.assertEqual(evidence["checkpoint_id"], result["selected_checkpoint_id"])
            self.assertEqual(evidence["cli_status"], 1)
            self.assertEqual(evidence["forbidden_call_count"], 0)
            self.assertTrue(evidence["preserved"])
            print("checkpoint_corruption_evidence=" + json.dumps(evidence, sort_keys=True), flush=True)

    def test_verification_rejects_unknown_files_and_rehashed_selection_mismatch(self):
        from src import cnn_rc_training as training_module
        from src import downstream_run as runs

        result = self.successful_training()
        root = self.output_root / result["run_id"]
        selected = next(root.glob("update-*" + result["selected_checkpoint_id"]))
        unknown = root / "unlisted-result.json"
        unknown.write_bytes(b"{}\n")
        self.assert_failure(self.cli(VALIDATE_MODULE, self.verification_arguments(selected, "unknown-result")))
        unknown.unlink()
        bundle = root / "completion"
        original = {path.name: path.read_bytes() for path in bundle.iterdir()}
        records = training_module.read_artifacts(bundle, self.run)
        summary = records["validation-summary.json"]
        summary["selected_checkpoint_id"] = "ckpt_" + "0" * 64
        del summary["manifest_hash"]
        summary["manifest_hash"] = runs.domain_hash(summary["schema_version"], summary)
        fingerprints = []
        for name in sorted(set(records) - {"completion.json"}):
            raw = _json_bytes(records[name]) + b"\n"
            fingerprints.append({"path": name, "byte_size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
        completion = records["completion.json"]
        completion["artifact_fingerprints"] = fingerprints
        del completion["manifest_hash"]
        completion["manifest_hash"] = runs.domain_hash(completion["schema_version"], completion)
        producer = json.loads(original["manifest.json"])["producing_attempt_id"]
        rewritten = training_module._bundle_files(records, self.run, producer)
        for name, raw in rewritten.items():
            path = bundle / name
            path.chmod(0o600)
            path.write_bytes(raw)
        # Every local schema, semantic hash and raw-byte envelope is valid;
        # only the cross-artifact selected-checkpoint relationship is wrong.
        self.assertEqual(training_module.read_artifacts(bundle, self.run), records)
        self.assert_bundle_fingerprints(bundle)
        self.assert_failure(self.cli(VALIDATE_MODULE, self.verification_arguments(selected, "bad-selection")))
        for name, raw in original.items():
            (bundle / name).write_bytes(raw)
        self.assertEqual(training_module.read_artifacts(bundle, self.run)["completion.json"], result)


if __name__ == "__main__":
    unittest.main()
