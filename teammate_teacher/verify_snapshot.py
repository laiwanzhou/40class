from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / "project"
MANIFEST = ROOT / "source_manifest.json"
FORBIDDEN_SUFFIXES = {
    ".bin", ".ckpt", ".npy", ".npz", ".pt", ".pth", ".safetensors", ".zip"
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    listed = {entry["path"]: entry for entry in manifest["files"]}
    actual = {
        path.relative_to(PROJECT).as_posix(): path
        for path in PROJECT.rglob("*")
        if path.is_file()
    }
    missing = sorted(set(listed) - set(actual))
    unlisted = sorted(set(actual) - set(listed))
    mismatches = sorted(
        relative
        for relative in set(listed) & set(actual)
        if sha256(actual[relative]) != listed[relative]["sha256"]
        or actual[relative].stat().st_size != listed[relative]["bytes"]
    )
    syntax_errors: list[str] = []
    for relative, path in actual.items():
        if path.suffix == ".py":
            try:
                ast.parse(path.read_text(encoding="utf-8-sig"), filename=relative)
            except (OSError, SyntaxError, UnicodeError) as error:
                syntax_errors.append(f"{relative}: {error}")
    forbidden = sorted(
        path.relative_to(ROOT.parent).as_posix()
        for path in ROOT.parent.rglob("*")
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES
    )
    oversized = sorted(
        path.relative_to(ROOT.parent).as_posix()
        for path in ROOT.parent.rglob("*")
        if path.is_file() and ".git" not in path.parts and path.stat().st_size >= 100_000_000
    )
    report = {
        "manifest_entries": len(listed),
        "actual_project_files": len(actual),
        "missing": missing,
        "unlisted": unlisted,
        "hash_or_size_mismatches": mismatches,
        "python_files": sum(path.suffix == ".py" for path in actual.values()),
        "syntax_errors": syntax_errors,
        "forbidden_artifacts": forbidden,
        "oversized_files": oversized,
        "source_manifest_training_executed": manifest.get("training_executed"),
        "source_manifest_exclusions": manifest.get("excluded", []),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if any((missing, unlisted, mismatches, syntax_errors, forbidden, oversized)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
