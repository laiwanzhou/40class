"""Verify the immutable source snapshot before exposing any upstream symbol."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import inspect
import json
from pathlib import Path, PurePosixPath
import re
import sys


SOURCE_FILE_COUNT = 1167
SOURCE_MANIFEST_SHA256 = "a80a20a0f6fe23a5b3288f51a9e4a02daf83157c4b6a598bfb6fa0ec8b73649a"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SourceVerification:
    root: Path
    manifest_path: Path
    manifest_sha256: str
    file_count: int
    files: tuple[str, ...]


def verify_teammate_source(root: Path, manifest_path: Path, *,
                           expected_count: int | None = None,
                           expected_sha256: str | None = None) -> SourceVerification:
    root, manifest_path = Path(root).resolve(), Path(manifest_path).resolve()
    digest = sha256_file(manifest_path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("source manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("files")
    if not isinstance(entries, list) or type(manifest.get("file_count")) is not int:
        raise ValueError("invalid source manifest")
    if len(entries) != manifest["file_count"] or (expected_count is not None and len(entries) != expected_count):
        raise ValueError("source manifest count mismatch")
    listed = set()
    for entry in entries:
        name = entry.get("path", "")
        rel = PurePosixPath(name)
        if not name or rel.is_absolute() or ".." in rel.parts or "\\" in name or ":" in name:
            raise ValueError("invalid relative source manifest path")
        if name in listed:
            raise ValueError("duplicate source manifest path")
        listed.add(name)
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"source file missing: {name}")
        if type(entry.get("bytes")) is not int or path.stat().st_size != entry["bytes"] or sha256_file(path) != entry.get("sha256"):
            raise ValueError(f"source file mismatch: {name}")
    return SourceVerification(root, manifest_path, digest, len(entries), tuple(sorted(listed)))


def load_teammate_symbol(report: SourceVerification, module_name: str, symbol_name: str):
    if not re.fullmatch(r"[A-Za-z_]\w*", module_name) or not re.fullmatch(r"[A-Za-z_]\w*", symbol_name):
        raise ValueError("source module and symbol must be simple identifiers")
    current = verify_teammate_source(report.root, report.manifest_path,
                                    expected_count=report.file_count,
                                    expected_sha256=report.manifest_sha256)
    relative = f"aligned_multimodal/{module_name}.py"
    if relative not in current.files:
        raise ValueError("requested module is not in the verified manifest")
    expected = (report.root / relative).resolve()
    module_paths = {}
    verified_paths = {(report.root / name).resolve() for name in current.files}
    for name in current.files:
        parts = PurePosixPath(name).parts
        if not parts or parts[0] != "aligned_multimodal" or not name.endswith(".py"):
            continue
        components = list(parts[1:-1])
        if parts[-1] != "__init__.py":
            components.append(parts[-1][:-3])
        if components and all(re.fullmatch(r"[A-Za-z_]\w*", c) for c in components):
            module_paths[".".join(components)] = (report.root / name).resolve()

    def validate_cached_origins():
        for name, path in module_paths.items():
            cached = sys.modules.get(name)
            if cached is not None:
                origin = getattr(cached, "__file__", None)
                if not origin or Path(origin).resolve() != path:
                    raise ImportError(f"source dependency origin collision: {name}")

    # The top-level file can be correct while Python silently reuses a helper
    # imported earlier from a different worktree. Check both sides of import.
    validate_cached_origins()
    old_path, old_bytecode = list(sys.path), sys.dont_write_bytecode
    try:
        sys.path.insert(0, str(expected.parent))
        sys.dont_write_bytecode = True
        module = importlib.import_module(module_name)
        if Path(module.__file__).resolve() != expected:
            raise ImportError(f"source module origin mismatch: {module_name}")
        validate_cached_origins()
        symbol = getattr(module, symbol_name)
        if callable(symbol):
            try:
                origin = Path(inspect.getfile(symbol)).resolve()
            except (TypeError, OSError) as error:
                raise ImportError("source callable origin cannot be verified") from error
            if origin not in verified_paths:
                raise ImportError("exported callable origin is outside the verified source snapshot")
        return symbol
    finally:
        sys.path[:] = old_path
        sys.dont_write_bytecode = old_bytecode
