"""Explicit logical addresses for an approved project-root rename.

Only paths beneath one verified directory alias are translated. Content hashes,
protocol parameters and serialized historical records are never rewritten.
"""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re


@dataclass(frozen=True)
class RootRelocation:
    historical_root: Path
    runtime_root: Path

    def __post_init__(self):
        historical = Path(self.historical_root).absolute()
        runtime = Path(self.runtime_root).resolve()
        if (historical == runtime or not historical.is_dir() or not runtime.is_dir()
                or not historical.samefile(runtime)):
            raise ValueError('relocation alias must identify the same physical directory')
        object.__setattr__(self, 'historical_root', historical)
        object.__setattr__(self, 'runtime_root', runtime)

    def logical(self, path):
        physical = Path(path).resolve()
        if physical.is_relative_to(self.runtime_root):
            return self.historical_root / physical.relative_to(self.runtime_root)
        return physical


def load_relocation(manifest):
    manifest = Path(manifest).resolve()
    body = json.loads(manifest.read_text(encoding='utf-8'))
    if body.get('schema_version') != 1:
        raise ValueError('unsupported path relocation schema')
    relocation = RootRelocation(Path(body['historical_root']), Path(body['runtime_root']))
    anchors = body.get('anchors')
    if not manifest.is_relative_to(relocation.runtime_root) or not isinstance(anchors, dict) or not anchors:
        raise ValueError('relocation requires local content anchors')
    for name, expected in anchors.items():
        rel = Path(name)
        if (rel.is_absolute() or '..' in rel.parts or not name
                or not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected)):
            raise ValueError('invalid relative relocation anchor')
        file = (relocation.runtime_root / rel).resolve()
        if (not file.is_relative_to(relocation.runtime_root) or not file.is_file()
                or hashlib.sha256(file.read_bytes()).hexdigest() != expected):
            raise ValueError('relocation content anchor mismatch')
    return relocation


@lru_cache(maxsize=1)
def active_relocation():
    manifest = Path(__file__).resolve().parents[2] / 'configs/experiments/project_root_relocation.json'
    return load_relocation(manifest) if manifest.is_file() else None


def logical_path(path):
    relocation = active_relocation()
    return relocation.logical(path) if relocation else Path(path).resolve()


def path_entry(mapping, path):
    """Find a recorded file address through the verified alias, rejecting ambiguity."""
    physical = Path(path).resolve()
    matches = [value for key, value in mapping.items() if Path(key).resolve() == physical]
    if len(matches) > 1:
        raise ValueError('ambiguous relocated file addresses')
    return matches[0] if matches else None
