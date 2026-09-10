"""Hermetic byte-inventory and atomic-publication tests using temporary Git repos."""

from concurrent.futures import ThreadPoolExecutor
import copy
import ctypes
import errno
import gzip
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import threading
import unittest
from unittest import mock

from scripts.carc import verify_exd_hox_staging as staging
from src import exd_hox_dataset as dataset
from src.downstream_fingerprints import canonical_json_bytes


METADATA_PATHS = (
    "configs/exd_hox_primary_split_v1.yaml",
    "configs/exd_hox_test_access_policy_v1.yaml",
    "data/processed/exd_hox_nested_subsets_v1/exd_hox_nested_subset_levels_v1.tsv",
    "data/processed/exd_hox_nested_subsets_v1/exd_hox_subset_set_manifest_v1.json",
    "data/processed/exd_hox_primary_split_v1/exd_hox_primary_split_manifest_v1.json",
    "data/processed/exd_hox_selex_audit_v1/exd_hox_audit_manifest_v1.json",
    "data/processed/exd_hox_selex_audit_v1/exd_hox_source_manifest_v1.json",
)
PAYLOAD_PATHS = (
    "data/processed/exd_hox_nested_subsets_v1/exd_hox_nested_subset_ordering_v1.tsv.gz",
    "data/processed/exd_hox_primary_split_v1/exd_hox_logical_examples_v1.tsv.gz",
)
FORBIDDEN_PATH_CLASSES = [
    "data/sealed/**", "data/raw/**", "**/*.h5", "**/*.hdf5",
    "**/SELEX_canonical/**", "**/SELEX_RCmodel/**",
    "**/exd_hox_public_test_inputs_v1.tsv.gz",
    "**/exd_hox_source_occurrence_provenance_v1.tsv.gz",
    "**/exd_hox_global_rc_groups_v1.tsv.gz",
    "**/exd_hox_primary_split_assignments_v1.tsv.gz",
    "**/exd_hox_sealed_test_targets_v1.tsv.gz",
    "**/exd_hox_sealed_test_target_manifest_v1.json",
    "results/**", "plots/**", "checkpoints/**", "**/authorization*", "**/test_access_record*",
]
LOGICAL_FIELDS = (
    "logical_example_id", "transcription_factor", "sequence", "sequence_sha256",
    "reverse_complement_canonical_sequence", "reverse_complement_canonical_sha256",
    "global_rc_group_id", "primary_split", "target_value_float32",
    "target_bits_big_endian_hex", "target_commitment_sha256", "source_occurrence_count",
)
ORDERING_FIELDS = (
    "transcription_factor", "rank_one_based", "global_rc_group_id", "logical_example_id",
    "training_affinity_bin", "deterministic_order_sha256",
)
LEVEL_FIELDS = (
    "transcription_factor", "level_id", "request_type", "request_value",
    "unaliased_requested_logical_example_count", "alias_absolute_anchor",
    "canonical_requested_logical_example_count", "actual_logical_example_count",
    "actual_rc_group_count", "inclusive_maximum_rank",
)


def synthetic_tables() -> dict[str, bytes]:
    def domain_hash(domain: str, *parts: str) -> str:
        return hashlib.sha256((domain + "\0" + "\0".join(parts)).encode("ascii")).hexdigest()

    logical_rows = []
    ordering_rows = []
    for index, suffix in enumerate(("AC", "AG", "AT", "CA")):
        sequence = "A" * 12 + suffix
        bits = "3e800000"
        logical_id = "lex_" + domain_hash("exd_hox_logical_example.v1", "Ubx", sequence, bits)
        sequence_hash = hashlib.sha256(sequence.encode("ascii")).hexdigest()
        group = "rcg_" + domain_hash("exd_hox_global_rc_group.v1", sequence)
        split = "training" if index < 2 else ("validation" if index == 2 else "test")
        logical_rows.append((
            logical_id, "Ubx", sequence, sequence_hash, sequence, sequence_hash, group, split,
            "0.25" if split != "test" else "", bits if split != "test" else "",
            domain_hash("exd_hox_target_commitment.v1", logical_id, bits), "1",
        ))
        if split == "training":
            ordering_rows.append((
                "Ubx", str(index + 1), group, logical_id, "0",
                domain_hash("exd_hox_nested_subset_group_order.v1", "32001", "Ubx", group),
            ))
    levels = (("Ubx", "lvl_" + "5" * 64, "fraction", "1.0", "2", "", "2", "2", "2", "2"),)
    tables = {}
    for path, fields, rows in (
        (PAYLOAD_PATHS[1], LOGICAL_FIELDS, sorted(logical_rows)),
        (PAYLOAD_PATHS[0], ORDERING_FIELDS, ordering_rows),
        (METADATA_PATHS[2], LEVEL_FIELDS, levels),
    ):
        text = "\t".join(fields) + "\n"
        for row in rows:
            text += "\t".join(row) + "\n"
        tables[path] = text.encode("ascii")
    return tables


def synthetic_contract() -> tuple[dict, dict[str, bytes]]:
    """Build independent bytes with actual public table schemas."""

    files = {}
    metadata_records = []
    payload_records = []
    for path in METADATA_PATHS:
        files[path] = canonical_json_bytes({"synthetic_metadata": path}) + b"\n"
    tables = synthetic_tables()
    files[METADATA_PATHS[2]] = tables[METADATA_PATHS[2]]
    for path in PAYLOAD_PATHS:
        buffer = io.BytesIO()
        with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0, compresslevel=9) as compressed:
            compressed.write(tables[path])
        files[path] = buffer.getvalue()
    for path, content in sorted(files.items()):
        record = {"path": path, "byte_size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        if path in METADATA_PATHS:
            metadata_records.append(record)
        else:
            payload_records.append(record)
    per_split = {
        "training": {"logical_examples": 2, "rc_groups": 2},
        "validation": {"logical_examples": 1, "rc_groups": 1},
        "test": {"logical_examples": 1, "rc_groups": 1},
    }
    contract = {
        "schema_version": "carc_exd_hox_training_staging_config.v1",
        "purpose": "training_validation_only",
        "staging_policy_identifier": "exd_hox_public_training_stage.v1",
        "foundation_commit": "1" * 40,
        "dataset_identifier": "wang_etal_exd_hox_selex_canonical.v1",
        "split_identity_hash": "2" * 64,
        "split_manifest_hash": "3" * 64,
        "subset_set_manifest_hash": "4" * 64,
        "allowed_dataset_selections": ["training", "validation"],
        "transcription_factors": ["Ubx"],
        "tracked_metadata": metadata_records,
        "transferred_public_payloads": payload_records,
        "forbidden_path_classes": FORBIDDEN_PATH_CLASSES.copy(),
        "expected_counts": {
            "source_occurrences": 4, "logical_examples": 4,
            "ordering_logical_examples": 2, "level_rows": 1,
            "split_logical_examples": {"training": 2, "validation": 1, "test": 1},
            "global_rc_groups": {"training": 2, "validation": 1, "test": 1},
            "per_tf": {"Ubx": per_split},
        },
    }
    return contract, files


class StagingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.checkout = self.root / "checkout"
        self.incoming = self.root / "incoming"
        self.output = self.root / "completed"
        self.checkout.mkdir()
        self.incoming.mkdir()
        self.contract, self.files = synthetic_contract()
        for path, content in self.files.items():
            physical_root = self.checkout if path in METADATA_PATHS else self.incoming
            target = physical_root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        (self.checkout / dataset.CONFIG_PATH).write_bytes(canonical_json_bytes(self.contract) + b"\n")
        self.git("init", "--quiet")
        self.git("add", "--all")
        self.git("commit", "--quiet", "-m", "Synthetic public stage fixture")
        self.commit = self.git("rev-parse", "HEAD").decode("ascii").strip()

    def tearDown(self) -> None:
        for current, directory_names, file_names in os.walk(self.root, followlinks=False):
            os.chmod(current, 0o700)
        self.temporary.cleanup()

    def git(self, *arguments: str) -> bytes:
        result = subprocess.run(
            ["git", "-c", "user.name=Synthetic Fixture", "-c", "user.email=fixture@example.invalid",
             "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", "-C", str(self.checkout), *arguments],
            capture_output=True, check=True,
        )
        return result.stdout

    def assemble(self, output: Path | None = None):
        return staging._assemble_stage(
            self.incoming, self.checkout, self.output if output is None else output,
            expected_commit=self.commit, contract=self.contract,
        )

    def assert_unpublished(self) -> None:
        self.assertFalse(os.path.lexists(self.output))
        self.assertEqual(list(self.root.glob(".exd-hox-stage-*")), [])

    def test_incoming_verification_hashes_only_and_does_not_write(self) -> None:
        before = sorted(path.relative_to(self.incoming).as_posix() for path in self.incoming.rglob("*"))
        with mock.patch("gzip.GzipFile", side_effect=AssertionError("decompression forbidden")):
            staging._verify_incoming(self.incoming, self.contract)
        self.assertEqual(before, sorted(path.relative_to(self.incoming).as_posix() for path in self.incoming.rglob("*")))

    def test_missing_payload_refused(self) -> None:
        (self.incoming / PAYLOAD_PATHS[0]).unlink()
        with self.assertRaises((ValueError, OSError)):
            staging._verify_incoming(self.incoming, self.contract)

    def test_extra_hidden_forbidden_files_and_empty_directories_refused(self) -> None:
        cases = (".hidden", "unrelated", "data/raw/extra.h5", "data/sealed/extra", "results/authorization.json")
        for name in cases:
            with self.subTest(name=name):
                extra = self.incoming / name
                extra.parent.mkdir(parents=True, exist_ok=True)
                extra.write_bytes(b"unlisted")
                with self.assertRaises((ValueError, OSError)):
                    staging._verify_incoming(self.incoming, self.contract)
                extra.unlink()
                for parent in (extra.parent, *extra.parent.parents):
                    if parent != self.incoming and self.incoming in parent.parents and not list(parent.iterdir()):
                        parent.rmdir()
        (self.incoming / "empty").mkdir()
        with self.assertRaises((ValueError, OSError)):
            staging._verify_incoming(self.incoming, self.contract)

    def test_size_and_same_size_hash_mismatch_refused(self) -> None:
        path = self.incoming / PAYLOAD_PATHS[0]
        original = path.read_bytes()
        for content in (original + b"x", bytes([original[0] ^ 1]) + original[1:]):
            with self.subTest(size=len(content)):
                path.write_bytes(content)
                with self.assertRaisesRegex(ValueError, "[Ff]ingerprint|[Ss]ize|[Hh]ash|SHA"):
                    staging._verify_incoming(self.incoming, self.contract)
        path.write_bytes(original)

    def test_symlink_file_ancestor_root_and_fifo_refused(self) -> None:
        path = self.incoming / PAYLOAD_PATHS[0]
        path.unlink()
        outside = self.root / "outside"
        outside.write_bytes(self.files[PAYLOAD_PATHS[0]])
        path.symlink_to(outside)
        with self.assertRaises((ValueError, OSError)):
            staging._verify_incoming(self.incoming, self.contract)
        path.unlink()
        os.mkfifo(path)
        with self.assertRaises((ValueError, OSError)):
            staging._verify_incoming(self.incoming, self.contract)
        path.unlink()
        path.write_bytes(self.files[PAYLOAD_PATHS[0]])
        link = self.root / "linked-incoming"
        link.symlink_to(self.incoming, target_is_directory=True)
        with self.assertRaises((ValueError, OSError)):
            staging._verify_incoming(link, self.contract)
        link.unlink()
        parent = path.parent
        renamed = self.root / "moved-subdirectory"
        parent.rename(renamed)
        parent.symlink_to(renamed, target_is_directory=True)
        with self.assertRaises((ValueError, OSError)):
            staging._verify_incoming(self.incoming, self.contract)

    def test_contract_paths_types_unknown_and_duplicate_fields_refused(self) -> None:
        for invalid in ("../escape", "/absolute", "data//escape", "data/./escape", "data/../escape", "data\\escape"):
            candidate = copy.deepcopy(self.contract)
            candidate["transferred_public_payloads"][0]["path"] = invalid
            with self.subTest(path=invalid), self.assertRaises(ValueError):
                staging._verify_incoming(self.incoming, candidate)
        candidate = copy.deepcopy(self.contract)
        candidate["transferred_public_payloads"][0]["byte_size"] = True
        with self.assertRaises(ValueError):
            staging._verify_incoming(self.incoming, candidate)
        candidate = copy.deepcopy(self.contract)
        candidate["unknown"] = 1
        with self.assertRaises(ValueError):
            staging._verify_incoming(self.incoming, candidate)
        with self.assertRaises(ValueError):
            dataset._strict_json(b'{"purpose":"one","purpose":"two"}')

    def test_wrong_dirty_staged_or_untracked_metadata_checkout_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "exact approved"):
            staging._verify_checkout(self.checkout, "0" * 40, self.contract)
        metadata = self.checkout / METADATA_PATHS[0]
        original = metadata.read_bytes()
        metadata.write_bytes(original + b" ")
        with self.assertRaisesRegex(ValueError, "dirty tracked"):
            self.assemble()
        self.git("add", "--", METADATA_PATHS[0])
        with self.assertRaisesRegex(ValueError, "dirty tracked"):
            self.assemble()
        metadata.write_bytes(original)
        self.git("add", "--", METADATA_PATHS[0])
        self.git("rm", "--cached", "--", METADATA_PATHS[0])
        self.git("commit", "--quiet", "-m", "Remove one tracked fixture record")
        self.commit = self.git("rev-parse", "HEAD").decode("ascii").strip()
        with self.assertRaisesRegex(ValueError, "Git verification|tracked"):
            self.assemble()
        self.assert_unpublished()

    def test_committed_metadata_or_config_fingerprint_mismatch_refused(self) -> None:
        for path in (METADATA_PATHS[0], dataset.CONFIG_PATH):
            with self.subTest(path=path):
                target = self.checkout / path
                original = target.read_bytes()
                target.write_bytes(original + b" ")
                self.git("add", "--", path)
                self.git("commit", "--quiet", "-m", "Changed fixture bytes")
                self.commit = self.git("rev-parse", "HEAD").decode("ascii").strip()
                with self.assertRaises(ValueError):
                    self.assemble()
                target.write_bytes(original)
                self.git("add", "--", path)
                self.git("commit", "--quiet", "-m", "Restore fixture bytes")
                self.commit = self.git("rev-parse", "HEAD").decode("ascii").strip()
        self.assert_unpublished()

    def test_assembly_is_exact_readonly_copied_and_deterministic(self) -> None:
        first = self.assemble()
        second_root = self.root / "second-completed"
        second_checkout = self.root / "relocated-checkout"
        second_incoming = self.root / "relocated-incoming"
        shutil.copytree(self.checkout, second_checkout)
        shutil.copytree(self.incoming, second_incoming)
        second = staging._assemble_stage(
            second_incoming, second_checkout, second_root,
            expected_commit=self.commit, contract=self.contract,
        )
        self.assertEqual(first.stage_id, second.stage_id)
        self.assertEqual(first.manifest_hash, second.manifest_hash)
        manifest_path = self.output / dataset.STAGE_MANIFEST_FILENAME
        self.assertEqual(manifest_path.read_bytes(), (second_root / dataset.STAGE_MANIFEST_FILENAME).read_bytes())
        manifest = json.loads(manifest_path.read_bytes())
        self.assertEqual(manifest_path.read_bytes(), canonical_json_bytes(manifest) + b"\n")
        self.assertNotIn(str(self.root), manifest_path.read_text())
        self.assertNotIn(self.commit, manifest_path.read_text())
        self.assertEqual(first.runtime_commit, self.commit)
        files = sorted(path.relative_to(self.output).as_posix() for path in self.output.rglob("*") if path.is_file())
        self.assertEqual(files, sorted((*METADATA_PATHS, *PAYLOAD_PATHS, dataset.STAGE_MANIFEST_FILENAME)))
        for path in self.files:
            source = (self.checkout if path in METADATA_PATHS else self.incoming) / path
            target = self.output / path
            self.assertEqual(target.read_bytes(), self.files[path])
            self.assertNotEqual(source.stat().st_ino, target.stat().st_ino)
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o444)
        self.assertEqual(stat.S_IMODE(self.output.stat().st_mode), 0o555)

    def test_existing_file_empty_directory_nonempty_directory_and_dangling_link_refused(self) -> None:
        for kind in ("file", "empty", "nonempty", "dangling"):
            with self.subTest(kind=kind):
                output = self.root / kind
                if kind == "file":
                    output.write_bytes(b"preserve")
                elif kind == "dangling":
                    output.symlink_to(self.root / "missing")
                else:
                    output.mkdir()
                    if kind == "nonempty":
                        (output / "preserve").write_bytes(b"preserve")
                before = output.lstat()
                with self.assertRaises(OSError):
                    self.assemble(output)
                self.assertEqual(output.lstat().st_ino, before.st_ino)
                self.assertEqual(list(self.root.glob(".exd-hox-stage-*")), [])

    def test_concurrent_publishers_have_exactly_one_winner(self) -> None:
        barrier = threading.Barrier(2, timeout=10)
        original_rename = staging._exclusive_rename

        def simultaneous(parent, source, destination):
            barrier.wait()
            original_rename(parent, source, destination)

        with mock.patch.object(staging, "_exclusive_rename", side_effect=simultaneous):
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(self.assemble), executor.submit(self.assemble)]
                successes = []
                failures = []
                for future in futures:
                    try:
                        successes.append(future.result())
                    except FileExistsError as error:
                        failures.append(error)
        self.assertEqual(len(successes), 1)
        self.assertEqual(len(failures), 1)
        dataset._verify_records(self.output, dataset._records(self.contract))
        self.assertEqual(list(self.root.glob(".exd-hox-stage-*")), [])

    def test_final_name_is_invisible_until_complete_atomic_publication(self) -> None:
        original_rename = staging._exclusive_rename

        def inspect_publication(parent, source, destination):
            self.assertFalse(os.path.lexists(self.output))
            temporary = self.root / source
            dataset._require_inventory(temporary, (*METADATA_PATHS, *PAYLOAD_PATHS, dataset.STAGE_MANIFEST_FILENAME))
            dataset._verify_records(temporary, dataset._records(self.contract))
            original_rename(parent, source, destination)
            self.assertTrue((self.output / dataset.STAGE_MANIFEST_FILENAME).is_file())
            dataset._verify_records(self.output, dataset._records(self.contract))

        with mock.patch.object(staging, "_exclusive_rename", side_effect=inspect_publication):
            self.assemble()

    def test_source_mutation_and_same_bytes_replacement_during_copy_refused(self) -> None:
        original_copy = staging._copy_stream
        source_path = self.checkout / METADATA_PATHS[0]
        original_bytes = source_path.read_bytes()
        for replacement in (False, True):
            changed = False

            def change_source(source, destination):
                nonlocal changed
                result = original_copy(source, destination)
                if not changed:
                    changed = True
                    if replacement:
                        source_path.unlink()
                        source_path.write_bytes(original_bytes)
                    else:
                        source_path.write_bytes(original_bytes[:-1] + b" ")
                return result

            with self.subTest(replacement=replacement):
                with mock.patch.object(staging, "_copy_stream", side_effect=change_source):
                    with self.assertRaisesRegex(ValueError, "changed|replaced"):
                        self.assemble()
                source_path.write_bytes(original_bytes)
                self.assert_unpublished()

    def test_prepublication_failure_removes_only_private_output(self) -> None:
        for helper in ("_copy_record", "_freeze_and_flush", "_exclusive_rename"):
            with self.subTest(helper=helper):
                with mock.patch.object(staging, helper, side_effect=OSError("injected failure")):
                    with self.assertRaisesRegex(OSError, "injected"):
                        self.assemble()
                self.assert_unpublished()
                staging._verify_incoming(self.incoming, self.contract)

    def test_unsupported_platform_and_filesystem_fail_closed(self) -> None:
        with mock.patch.object(staging.sys, "platform", "unsupported"):
            with self.assertRaisesRegex(ValueError, "unsupported"):
                staging._rename_function()

        def unsupported(*arguments):
            ctypes.set_errno(errno.ENOTSUP)
            return -1

        with mock.patch.object(staging, "_rename_function", return_value=(unsupported, 1)):
            with self.assertRaisesRegex(ValueError, "lacks exclusive"):
                self.assemble()
        self.assert_unpublished()

    def test_linux_rename_wrapper_uses_noreplace_flag_and_no_fallback(self) -> None:
        library = mock.Mock()
        library.renameat2 = mock.Mock(return_value=0)
        with mock.patch.object(staging.sys, "platform", "linux"):
            with mock.patch.object(staging.ctypes, "CDLL", return_value=library):
                staging._exclusive_rename(123, "source", "destination")
        library.renameat2.assert_called_once_with(123, b"source", 123, b"destination", 1)
        self.assertEqual(library.renameat2.restype, ctypes.c_int)
        self.assertEqual(len(library.renameat2.argtypes), 5)
        library.renameat2 = None
        with mock.patch.object(staging.sys, "platform", "linux"):
            with mock.patch.object(staging.ctypes, "CDLL", return_value=library):
                with self.assertRaisesRegex(ValueError, "unsupported"):
                    staging._exclusive_rename(123, "source", "destination")

    def test_destination_parent_symlink_is_refused(self) -> None:
        linked_parent = self.root / "linked-parent"
        linked_parent.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(OSError):
            self.assemble(linked_parent / "completed")
        self.assert_unpublished()

    def test_parent_replacement_cleanup_stays_with_original_directory(self) -> None:
        parent = self.root / "publisher-parent"
        parent.mkdir()
        moved_parent = self.root / "moved-publisher-parent"
        original_assert = staging._assert_parent_identity
        replaced = False
        decoy = None

        def replace_parent(physical_parent, descriptor):
            nonlocal replaced, decoy
            if not replaced:
                replaced = True
                temporary_name = next(parent.glob(".exd-hox-stage-*")).name
                parent.rename(moved_parent)
                parent.mkdir()
                decoy = parent / temporary_name
                decoy.mkdir()
                (decoy / "preserve").write_bytes(b"unrelated")
            original_assert(physical_parent, descriptor)

        with mock.patch.object(staging, "_assert_parent_identity", side_effect=replace_parent):
            with self.assertRaisesRegex(ValueError, "parent changed"):
                self.assemble(parent / "completed")
        self.assertEqual((decoy / "preserve").read_bytes(), b"unrelated")
        self.assertEqual(list(moved_parent.iterdir()), [])

    def test_postpublication_failure_reports_publication_and_preserves_stage(self) -> None:
        original_rename = staging._exclusive_rename
        original_fsync = os.fsync
        published = False

        def publish(*arguments):
            nonlocal published
            original_rename(*arguments)
            published = True

        def flush(descriptor):
            if published:
                raise OSError("injected durability failure")
            original_fsync(descriptor)

        with mock.patch.object(staging, "_exclusive_rename", side_effect=publish):
            with mock.patch.object(staging.os, "fsync", side_effect=flush):
                with self.assertRaises(staging.StagePublishedError) as caught:
                    self.assemble()
        self.assertTrue(self.output.is_dir())
        self.assertEqual(caught.exception.result.destination, str(self.output))
        dataset._verify_records(self.output, dataset._records(self.contract))
        self.assertEqual(list(self.root.glob(".exd-hox-stage-*")), [])

    def test_public_api_has_no_contract_override_and_requires_executing_checkout(self) -> None:
        self.assertEqual(tuple(inspect.signature(staging.verify_incoming).parameters), ("incoming_root",))
        self.assertEqual(
            tuple(inspect.signature(staging.assemble_stage).parameters),
            ("incoming_root", "checkout_root", "destination", "expected_commit"),
        )
        with self.assertRaisesRegex(ValueError, "execute from"):
            staging.assemble_stage(self.incoming, self.checkout, self.output, expected_commit=self.commit)


if __name__ == "__main__":
    unittest.main()
