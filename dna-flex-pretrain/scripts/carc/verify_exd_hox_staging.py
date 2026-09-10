"""Verify public Exd-Hox inputs and exclusively publish immutable stages.

Incoming verification hashes compressed bytes without opening their contents.
Only an explicitly invoked assembly command writes a stage. Execution facts
are returned separately from the deterministic scientific stage manifest.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict, dataclass
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, BinaryIO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.downstream_fingerprints import canonical_json_bytes
from src.exd_hox_dataset import (
    CONFIG_PATH,
    STAGE_MANIFEST_FILENAME,
    ExdHoxDatasetError,
    _build_stage_manifest,
    _load_contract,
    _open_regular,
    _read_small,
    _records,
    _require_inventory,
    _stage_id,
    _validate_contract,
    _verify_records,
)


@dataclass(frozen=True)
class StageResult:
    """Execution receipt; never stored inside the scientific inventory."""

    stage_id: str
    manifest_hash: str
    destination: str
    runtime_commit: str


class StagePublishedError(ExdHoxDatasetError):
    """Publication succeeded but the final durability operation failed."""

    def __init__(self, result: StageResult) -> None:
        self.result = result
        super().__init__(
            "Stage was published; final parent-directory durability failed. "
            "Do not delete, replace, or automatically retry: " + result.stage_id
        )


def _paths(records: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> tuple[str, ...]:
    return tuple(record["path"] for record in records)


def _verify_incoming(incoming_root: Path, contract: dict[str, Any]) -> None:
    """Private contract-parametrized boundary for independent synthetic tests."""

    _validate_contract(contract)
    records = contract["transferred_public_payloads"]
    _require_inventory(incoming_root, _paths(records))
    _verify_records(incoming_root, records)
    _require_inventory(incoming_root, _paths(records))


def verify_incoming(incoming_root: Path) -> None:
    """Hash exactly the two pinned incoming payloads, without decompression."""

    _verify_incoming(Path(incoming_root), _load_contract())


def _git(checkout_root: Path, *arguments: str) -> bytes:
    command = ["git", "-C", str(checkout_root), *arguments]
    completed = subprocess.run(command, capture_output=True, check=False)
    if completed.returncode != 0:
        raise ExdHoxDatasetError("Accepted checkout Git verification failed.")
    return completed.stdout


def _verify_checkout(
    checkout_root: Path, expected_commit: str, contract: dict[str, Any]
) -> None:
    if type(expected_commit) is not str or re.fullmatch("[0-9a-f]{40}", expected_commit) is None:
        raise ExdHoxDatasetError("Expected software commit must be a full lowercase Git SHA-1.")
    head = _git(checkout_root, "rev-parse", "HEAD").decode("ascii").strip()
    if head != expected_commit:
        raise ExdHoxDatasetError("Checkout is not the exact approved software commit.")
    if _git(checkout_root, "status", "--porcelain=v1", "--untracked-files=no"):
        raise ExdHoxDatasetError("Accepted checkout has dirty tracked state.")
    metadata_paths = _paths(contract["tracked_metadata"])
    tracked = _git(checkout_root, "ls-files", "--error-unmatch", "-z", "--", *metadata_paths, CONFIG_PATH)
    if set(tracked.decode("utf-8").rstrip("\0").split("\0")) != set((*metadata_paths, CONFIG_PATH)):
        raise ExdHoxDatasetError("Staging metadata and config must be tracked regular files.")
    _verify_records(checkout_root, contract["tracked_metadata"])
    contract_bytes = canonical_json_bytes(contract) + b"\n"
    if _read_small(checkout_root, CONFIG_PATH) != contract_bytes:
        raise ExdHoxDatasetError("Tracked staging config differs from the trusted contract.")


def _file_identity(file_stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        file_stat.st_dev, file_stat.st_ino, file_stat.st_size,
        file_stat.st_mtime_ns, file_stat.st_ctime_ns,
    )


def _copy_stream(source: BinaryIO, destination: BinaryIO) -> tuple[int, str]:
    """Hash precisely the bytes copied, with bounded memory."""

    digest = hashlib.sha256()
    byte_count = 0
    for block in iter(lambda: source.read(1024 * 1024), b""):
        destination.write(block)
        digest.update(block)
        byte_count += len(block)
    return byte_count, digest.hexdigest()


def _copy_record(source_root: Path, temporary_root: Path, record: dict[str, Any]) -> None:
    target = temporary_root / record["path"]
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    # Context managers are necessary here to release secure descriptors on all failures.
    with _open_regular(source_root, record["path"]) as source:
        before = _file_identity(os.fstat(source.fileno()))
        with open(target, "xb") as destination:
            byte_count, digest = _copy_stream(source, destination)
            destination.flush()
            os.fsync(destination.fileno())
        after = _file_identity(os.fstat(source.fileno()))
        with _open_regular(source_root, record["path"]) as reopened:
            current = _file_identity(os.fstat(reopened.fileno()))
        if before != after or before != current:
            raise ExdHoxDatasetError("Source file changed or was replaced during staging.")
    if byte_count != record["byte_size"] or digest != record["sha256"]:
        raise ExdHoxDatasetError("Copied source fingerprint mismatch.")
    _verify_records(temporary_root, (record,))


def _open_directory(path: Path) -> int:
    """Open a physical directory through non-symlink ancestors."""

    absolute = Path(os.path.abspath(path))
    descriptor = os.open(absolute.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in absolute.parts[1:]:
            next_descriptor = os.open(
                component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor
            )
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        # This handler only releases a descriptor, and always reraises.
        os.close(descriptor)
        raise


def _assert_parent_identity(parent: Path, descriptor: int) -> None:
    current = _open_directory(parent)
    try:
        original_stat = os.fstat(descriptor)
        current_stat = os.fstat(current)
        if (original_stat.st_dev, original_stat.st_ino) != (current_stat.st_dev, current_stat.st_ino):
            raise ExdHoxDatasetError("Destination parent changed during assembly.")
    finally:
        os.close(current)


def _rename_function() -> tuple[Any, int]:
    """Resolve the OS exclusive-rename primitive; unsupported systems fail closed.

    Linux renameat2 uses RENAME_NOREPLACE=1. The macOS SDK declares
    renameatx_np(int, const char *, int, const char *, unsigned int) and
    RENAME_EXCL=0x00000004 in sys/stdio.h. Neither path permits replacement.
    """

    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "linux":
        function = getattr(library, "renameat2", None)
        flag = 1
    elif sys.platform == "darwin":
        function = getattr(library, "renameatx_np", None)
        flag = 4
    else:
        function = None
        flag = 0
    if function is None:
        raise ExdHoxDatasetError("Exclusive atomic directory rename is unsupported.")
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    return function, flag


def _exclusive_rename(parent_descriptor: int, source_name: str, destination_name: str) -> None:
    function, flag = _rename_function()
    result = function(
        parent_descriptor, os.fsencode(source_name),
        parent_descriptor, os.fsencode(destination_name), flag,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in (errno.ENOSYS, errno.EINVAL, errno.ENOTSUP):
            raise ExdHoxDatasetError("Destination filesystem lacks exclusive atomic rename support.")
        raise OSError(error_number, os.strerror(error_number), destination_name)


def _freeze_and_flush(root: Path) -> None:
    """Make completed files read-only, then flush directory entries bottom-up."""

    directories = []
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        directories.append(current_path)
        for file_name in file_names:
            path = current_path / file_name
            os.chmod(path, 0o444, follow_symlinks=False)
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    for directory in reversed(directories):
        os.chmod(directory, 0o555, follow_symlinks=False)
        descriptor = _open_directory(directory)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _remove_private_temporary(
    parent_descriptor: int, name: str, identity: tuple[int, int]
) -> None:
    """Remove only unpublished temporary output owned by this invocation."""

    def make_directories_writable(descriptor: int) -> None:
        os.fchmod(descriptor, 0o700)
        for child_name in os.listdir(descriptor):
            details = os.stat(child_name, dir_fd=descriptor, follow_symlinks=False)
            if stat.S_ISDIR(details.st_mode):
                child = os.open(
                    child_name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
                try:
                    make_directories_writable(child)
                finally:
                    os.close(child)

    descriptor = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_descriptor)
    try:
        details = os.fstat(descriptor)
        if (details.st_dev, details.st_ino) != identity:
            raise ExdHoxDatasetError("Private temporary directory was replaced; cleanup refused.")
        make_directories_writable(descriptor)
        shutil.rmtree(name, dir_fd=parent_descriptor)
    finally:
        os.close(descriptor)


def _assemble_stage(
    incoming_root: Path,
    checkout_root: Path,
    destination: Path,
    *,
    expected_commit: str,
    contract: dict[str, Any],
) -> StageResult:
    """Private assembly core; production callers use the installed contract."""

    incoming_root = Path(incoming_root)
    checkout_root = Path(checkout_root)
    destination = Path(os.path.abspath(destination))
    _validate_contract(contract)
    _verify_checkout(checkout_root, expected_commit, contract)
    _verify_incoming(incoming_root, contract)
    _rename_function()
    parent_descriptor = _open_directory(destination.parent)
    temporary_root = None
    temporary_identity = None
    published = False
    try:
        temporary_root = Path(tempfile.mkdtemp(prefix=".exd-hox-stage-", dir=destination.parent))
        temporary_stat = os.stat(temporary_root.name, dir_fd=parent_descriptor, follow_symlinks=False)
        temporary_identity = (temporary_stat.st_dev, temporary_stat.st_ino)
        _assert_parent_identity(destination.parent, parent_descriptor)
        for record in contract["tracked_metadata"]:
            _copy_record(checkout_root, temporary_root, record)
        for record in contract["transferred_public_payloads"]:
            _copy_record(incoming_root, temporary_root, record)
        _verify_checkout(checkout_root, expected_commit, contract)
        _verify_incoming(incoming_root, contract)
        manifest = _build_stage_manifest(contract)
        manifest_bytes = canonical_json_bytes(manifest) + b"\n"
        with open(temporary_root / STAGE_MANIFEST_FILENAME, "xb") as manifest_file:
            manifest_file.write(manifest_bytes)
            manifest_file.flush()
            os.fsync(manifest_file.fileno())
        records = _records(contract)
        _require_inventory(temporary_root, (*_paths(records), STAGE_MANIFEST_FILENAME))
        _verify_records(temporary_root, records)
        if _read_small(temporary_root, STAGE_MANIFEST_FILENAME) != manifest_bytes:
            raise ExdHoxDatasetError("Completed stage manifest bytes changed.")
        _freeze_and_flush(temporary_root)
        _assert_parent_identity(destination.parent, parent_descriptor)
        result = StageResult(_stage_id(manifest), manifest["manifest_hash"], str(destination), expected_commit)
        _exclusive_rename(parent_descriptor, temporary_root.name, destination.name)
        published = True
        try:
            os.fsync(parent_descriptor)
        except OSError as error:
            raise StagePublishedError(result) from error
        return result
    finally:
        try:
            if temporary_root is not None and temporary_identity is not None and not published:
                _remove_private_temporary(parent_descriptor, temporary_root.name, temporary_identity)
        finally:
            os.close(parent_descriptor)


def assemble_stage(
    incoming_root: Path,
    checkout_root: Path,
    destination: Path,
    *,
    expected_commit: str,
) -> StageResult:
    """Publish from the exact approved executing checkout and pinned inventory."""

    if Path(checkout_root).resolve() != PROJECT_ROOT:
        raise ExdHoxDatasetError("Assembly must execute from the approved checkout itself.")
    _git(
        Path(checkout_root), "ls-files", "--error-unmatch", "--",
        "scripts/carc/verify_exd_hox_staging.py", "src/exd_hox_dataset.py",
        "src/downstream_fingerprints.py",
    )
    return _assemble_stage(
        incoming_root, checkout_root, destination,
        expected_commit=expected_commit, contract=_load_contract(),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    incoming = commands.add_parser("incoming", help="Verify exactly the two incoming compressed payloads.")
    incoming.add_argument("--incoming-root", type=Path, required=True)
    assembly = commands.add_parser("assemble", help="Exclusively publish an approved public training stage.")
    assembly.add_argument("--incoming-root", type=Path, required=True)
    assembly.add_argument("--checkout-root", type=Path, required=True)
    assembly.add_argument("--destination", type=Path, required=True)
    assembly.add_argument("--expected-commit", required=True)
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "incoming":
            verify_incoming(arguments.incoming_root)
            print(json.dumps({"incoming_verified": True}, sort_keys=True))
        else:
            result = assemble_stage(
                arguments.incoming_root, arguments.checkout_root, arguments.destination,
                expected_commit=arguments.expected_commit,
            )
            print(json.dumps(asdict(result), sort_keys=True))
    except StagePublishedError as error:
        print(json.dumps({"published": True, "error": str(error), **asdict(error.result)}, sort_keys=True), file=sys.stderr)
        return 2
    except (ExdHoxDatasetError, OSError) as error:
        print("Staging refused: " + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
