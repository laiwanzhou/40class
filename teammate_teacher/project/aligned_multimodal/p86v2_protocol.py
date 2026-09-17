from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = PROJECT_DIR / "configs/p86v2_train_only_protocol.json"


@dataclass(frozen=True)
class P86V2Split:
    name: str
    training_indices: tuple[int, ...]
    holdout_indices: tuple[int, ...]
    embargo_indices: tuple[int, ...]
    training_subjects: tuple[str, ...]
    holdout_subjects: tuple[str, ...]
    sample_fingerprint: str


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict:
    value = json.loads(path.resolve().read_text(encoding="utf-8"))
    if value.get("stage") != "P86-v2":
        raise ValueError("not a P86-v2 protocol")
    return value


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.resolve().open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"sample_id", "user_id", "class_id"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"rows must contain {sorted(required)}")
    return rows


def _fingerprint(sample_ids: Iterable[str]) -> str:
    payload = "\n".join(sorted(map(str, sample_ids))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_split(
    rows: list[dict[str, str]],
    split_name: str,
    protocol: dict | None = None,
) -> P86V2Split:
    protocol = load_protocol() if protocol is None else protocol
    if split_name not in {"development", "confirmation"}:
        raise ValueError("split_name must be development or confirmation")
    embargo = set(map(str, protocol["embargo"]["subjects"]))
    allowed = set(map(str, protocol["allowed_train_subjects"]))
    holdout = set(map(str, protocol[split_name]["holdout_subjects"]))
    observed = {str(row["user_id"]) for row in rows}
    if observed != embargo | allowed:
        raise RuntimeError(
            "training universe subjects differ from the frozen protocol: "
            f"missing={sorted((embargo | allowed) - observed)}, "
            f"extra={sorted(observed - (embargo | allowed))}"
        )
    if holdout & embargo or not holdout <= allowed:
        raise RuntimeError("holdout is not a subset of the allowed non-embargo subjects")

    training_indices: list[int] = []
    holdout_indices: list[int] = []
    embargo_indices: list[int] = []
    for index, row in enumerate(rows):
        user = str(row["user_id"])
        if user in embargo:
            embargo_indices.append(index)
        elif user in holdout:
            holdout_indices.append(index)
        elif user in allowed:
            training_indices.append(index)
        else:  # pragma: no cover - guarded by observed equality above
            raise RuntimeError(f"unclassified subject: {user}")

    expected_holdout = int(protocol[split_name]["expected_samples"])
    expected_embargo = int(protocol["embargo"]["expected_samples"])
    if len(holdout_indices) != expected_holdout:
        raise RuntimeError(
            f"{split_name} holdout has {len(holdout_indices)} rows, expected {expected_holdout}"
        )
    if len(embargo_indices) != expected_embargo:
        raise RuntimeError(
            f"embargo has {len(embargo_indices)} rows, expected {expected_embargo}"
        )
    holdout_classes = {int(rows[index]["class_id"]) for index in holdout_indices}
    expected_classes = int(protocol[split_name]["expected_classes_present"])
    if len(holdout_classes) != expected_classes:
        raise RuntimeError(
            f"{split_name} contains {len(holdout_classes)} classes, expected {expected_classes}"
        )
    known_absent = set(map(int, protocol[split_name].get("known_absent_classes", [])))
    if set(range(40)) - holdout_classes != known_absent:
        raise RuntimeError("holdout absent-class set differs from the frozen protocol")

    train_subjects = tuple(sorted({rows[index]["user_id"] for index in training_indices}))
    holdout_subjects = tuple(sorted({rows[index]["user_id"] for index in holdout_indices}))
    if set(train_subjects) & set(holdout_subjects):
        raise RuntimeError("subject-disjoint isolation failed")
    used_ids = [rows[index]["sample_id"] for index in training_indices + holdout_indices]
    return P86V2Split(
        name=str(protocol[split_name]["name"]),
        training_indices=tuple(training_indices),
        holdout_indices=tuple(holdout_indices),
        embargo_indices=tuple(embargo_indices),
        training_subjects=train_subjects,
        holdout_subjects=holdout_subjects,
        sample_fingerprint=_fingerprint(used_ids),
    )


def assert_no_forbidden_path(path: Path) -> None:
    """Reject resources whose purpose is Test/submission feedback.

    Train caches may physically contain embargo rows because they predate P86-v2;
    row-level filtering is enforced by :func:`build_split`. This guard covers
    resources that have no legitimate role in Train-only development.
    """

    normalized = str(path.resolve()).replace("\\", "/").lower()
    forbidden_tokens = (
        "test_manifest",
        "test_union",
        "test_submission",
        "/submission",
        "leaderboard",
        "pseudo_label",
        "pseudo-label",
    )
    matched = [token for token in forbidden_tokens if token in normalized]
    if matched:
        raise RuntimeError(f"P86-v2 Train-only guard rejected {path}: {matched}")
