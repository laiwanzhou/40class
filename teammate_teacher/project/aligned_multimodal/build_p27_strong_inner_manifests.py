from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
SOURCE_MANIFEST = PROJECT_DIR / "data" / "manifest.csv"
P27_MANIFEST = PROJECT_DIR / "runs" / "p27_0_audit" / "p27_train_manifest.csv"
OUTPUT_DIR = PROJECT_DIR / "data" / "p27_strong_inner"

OUTER_FOLD = 0
INNER_HELD_SUBJECTS = {
    0: ("user1", "user17", "user18", "user5"),
    1: ("user16", "user2", "user21", "user7"),
    2: ("user19", "user23", "user6", "user8"),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    source_rows = read_csv(SOURCE_MANIFEST)
    p27_rows = read_csv(P27_MANIFEST)
    p27_by_id = {row["sample_id"]: row for row in p27_rows}
    if len(p27_by_id) != len(p27_rows):
        raise RuntimeError("P27 manifest sample_id is not unique")
    missing = [row["sample_id"] for row in source_rows if row["sample_id"] not in p27_by_id]
    if missing:
        raise RuntimeError(f"P27 manifest misses {len(missing)} common rows")

    outer_train_subjects = sorted(
        {
            row["user_id"]
            for row in p27_rows
            if int(row["subject_fold"]) != OUTER_FOLD
        }
    )
    outer_held_subjects = sorted(
        {
            row["user_id"]
            for row in p27_rows
            if int(row["subject_fold"]) == OUTER_FOLD
        }
    )
    held_union = sorted(
        {subject for values in INNER_HELD_SUBJECTS.values() for subject in values}
    )
    if held_union != outer_train_subjects:
        raise RuntimeError(
            "Inner held groups do not partition fold-0 outer-train subjects"
        )

    audit: dict[str, object] = {
        "protocol": "p27-strong-inner-manifests-v1",
        "source_manifest": str(SOURCE_MANIFEST.resolve()),
        "source_manifest_sha256": sha256(SOURCE_MANIFEST),
        "p27_manifest": str(P27_MANIFEST.resolve()),
        "p27_manifest_sha256": sha256(P27_MANIFEST),
        "outer_fold": OUTER_FOLD,
        "outer_train_subjects": outer_train_subjects,
        "outer_held_subjects_excluded": outer_held_subjects,
        "outer_held_predictions_generated": False,
        "folds": {},
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for fold, held_subjects in INNER_HELD_SUBJECTS.items():
        held = set(held_subjects)
        rows: list[dict[str, str]] = []
        for source in source_rows:
            subject = source["user_id"]
            if subject not in outer_train_subjects:
                continue
            row = dict(source)
            row["split"] = "val" if subject in held else "train"
            rows.append(row)
        path = OUTPUT_DIR / f"fold_{fold}.csv"
        write_csv(path, rows)
        train_rows = [row for row in rows if row["split"] == "train"]
        val_rows = [row for row in rows if row["split"] == "val"]
        if set(row["user_id"] for row in train_rows) & set(held_subjects):
            raise RuntimeError(f"fold {fold} subject leakage")
        audit["folds"][str(fold)] = {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "train_samples": len(train_rows),
            "val_samples": len(val_rows),
            "train_subjects": sorted({row["user_id"] for row in train_rows}),
            "val_subjects": sorted({row["user_id"] for row in val_rows}),
            "train_class_counts": dict(
                sorted(Counter(row["class_id"] for row in train_rows).items())
            ),
            "val_class_counts": dict(
                sorted(Counter(row["class_id"] for row in val_rows).items())
            ),
        }
    audit_path = OUTPUT_DIR / "audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
