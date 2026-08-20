from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Mapping

ALLOWED_DEVELOPMENT_SPLITS = {"train12", "val_user6_user7"}
IMAGE_SUFFIXES = {".jpg", ".jpeg"}


def _natural_key(path: Path) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    )


def image_files(trial_path: Path) -> tuple[Path, ...]:
    if not trial_path.is_dir():
        return ()
    return tuple(
        sorted(
            (
                path
                for path in trial_path.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            ),
            key=_natural_key,
        )
    )


def load_canonical_thermal_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("thermal_data_audit", {}).get("canonical_trial_records")
    if not isinstance(records, list):
        raise ValueError("T0 report lacks canonical Thermal trial records")
    output: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    for value in records:
        if not isinstance(value, dict):
            raise ValueError("canonical Thermal record must be a mapping")
        split = value.get("development_split")
        if split not in ALLOWED_DEVELOPMENT_SPLITS:
            raise ValueError(
                f"record outside the fixed development population: {value.get('sample_id')}"
            )
        sample_id = str(value.get("sample_id", ""))
        if not sample_id or sample_id in sample_ids:
            raise ValueError("canonical Thermal sample IDs must be non-empty and unique")
        sample_ids.add(sample_id)
        output.append(dict(value))
    return output


def resolve_thermal_trial_path(
    data_root: Path, record: Mapping[str, Any]
) -> Path:
    class_id = int(record["class_id"])
    action_name = str(record["action_name"])
    user_id = str(record["user_id"])
    trial_id = str(record["trial_id"])
    return data_root / "Thermal" / f"{class_id}_{action_name}" / user_id / trial_id
