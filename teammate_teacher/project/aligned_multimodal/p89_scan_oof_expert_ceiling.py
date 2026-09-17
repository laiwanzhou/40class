from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

import p89_full40_scale_invariant_transfer as full40
from audit_p87_sequence_decoder import classification_metrics
from p88_train_depth_residual import rescue_harm


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT = PROJECT_DIR / "runs/p89_oof_expert_ceiling_scan_v1"
SIGNAL = re.compile(r"(logit|prob|prediction|score)", re.IGNORECASE)
IDENTIFIER = re.compile(r"(^|_)(sample_?ids?|ids?)$", re.IGNORECASE)
FORBIDDEN_VALUE = re.compile(
    r"(label|target|oracle|ground.?truth|fold|mask)", re.IGNORECASE
)


def prediction(values: np.ndarray) -> np.ndarray | None:
    values = np.asarray(values)
    if values.ndim == 1 and values.dtype.kind in "iub" and len(values):
        result = values.astype(np.int64)
        return result if result.min() >= 0 and result.max() < 40 else None
    if values.ndim == 2 and values.shape[1] == 40:
        return np.argmax(values, axis=1).astype(np.int64)
    if values.ndim == 3 and values.shape[-1] == 40:
        return np.argmax(values.astype(np.float64).mean(axis=1), axis=1).astype(np.int64)
    return None


def possible_id_keys(keys: list[str], value_key: str) -> list[str]:
    prefixes = []
    for suffix in ("_prediction", "_predictions", "_probability", "_probabilities", "_logits", "_scores"):
        if value_key.lower().endswith(suffix):
            prefixes.append(value_key[: -len(suffix)])
    candidates = []
    for prefix in prefixes:
        candidates.extend((f"{prefix}_sample_ids", f"{prefix}_ids"))
    candidates.extend(
        (
            "sample_ids",
            "oof_sample_ids",
            "validation_sample_ids",
            "ids",
        )
    )
    candidates.extend(key for key in keys if IDENTIFIER.search(key))
    return list(dict.fromkeys(key for key in candidates if key in keys))


def evaluate(
    source_ids: np.ndarray,
    source_prediction: np.ndarray,
    target_ids: np.ndarray,
    labels: np.ndarray,
    safe: np.ndarray,
) -> dict[str, object] | None:
    if len(source_ids) != len(source_prediction):
        return None
    lookup = {str(value): index for index, value in enumerate(source_ids.astype(str))}
    if not all(str(value) in lookup for value in target_ids):
        return None
    rows = np.asarray([lookup[str(value)] for value in target_ids], dtype=np.int64)
    candidate = source_prediction[rows]
    audit = rescue_harm(labels, safe, candidate)
    safe_errors = safe != labels
    rescued_safe_errors = int(np.sum(safe_errors & (candidate == labels)))
    candidate_changes = candidate != safe
    wins = int(np.sum(candidate_changes & (candidate == labels) & (safe != labels)))
    losses = int(np.sum(candidate_changes & (candidate != labels) & (safe == labels)))
    return {
        "metrics": classification_metrics(labels, candidate),
        "rescue_harm_vs_safe": audit,
        "safe_error_rescues": rescued_safe_errors,
        "oracle_safe_or_expert_correct": int(
            np.sum((safe == labels) | (candidate == labels))
        ),
        "disagreement_wins": wins,
        "disagreement_losses": losses,
        "disagreement_win_rate": wins / max(wins + losses, 1),
    }


def main() -> None:
    h1 = full40.protocol(full40.H1_RUN, full40.H1_USERS)
    h2 = full40.protocol(full40.H2_RUN, full40.H2_USERS)
    with np.load(
        PROJECT_DIR / "runs/p89_imu_probability_blend_v1/validation_predictions.npz"
    ) as source:
        h1_safe = source["h1_prediction"].astype(np.int64)
        h2_safe = source["h2_prediction"].astype(np.int64)
    splits = {
        "H1": (h1[0], h1[1], h1_safe),
        "H2": (h2[0], h2[1], h2_safe),
    }
    files = sorted(
        path
        for path in (PROJECT_DIR / "runs").rglob("*.npz")
        if re.search(r"(oof|logit|prediction|score)", path.name, re.IGNORECASE)
    )
    records = []
    failures = []
    for path in files:
        try:
            with np.load(path, allow_pickle=False) as source:
                keys = list(source.files)
                for value_key in keys:
                    if (
                        not SIGNAL.search(value_key)
                        or IDENTIFIER.search(value_key)
                        or FORBIDDEN_VALUE.search(value_key)
                    ):
                        continue
                    candidate = prediction(source[value_key])
                    if candidate is None:
                        continue
                    for id_key in possible_id_keys(keys, value_key):
                        ids = np.asarray(source[id_key])
                        if ids.ndim != 1 or len(ids) != len(candidate):
                            continue
                        result = {
                            "path": str(path.relative_to(PROJECT_DIR)),
                            "value_key": value_key,
                            "id_key": id_key,
                            "shape": list(np.asarray(source[value_key]).shape),
                        }
                        matched = False
                        for split, (target_ids, labels, safe) in splits.items():
                            metrics = evaluate(ids, candidate, target_ids, labels, safe)
                            if metrics is not None:
                                result[split] = metrics
                                matched = True
                        if matched:
                            records.append(result)
                            break
        except Exception as error:  # Diagnostic scan must survive stale artifacts.
            failures.append({"path": str(path.relative_to(PROJECT_DIR)), "error": str(error)})
    unique = {}
    for record in records:
        key = (record["path"], record["value_key"], record["id_key"])
        unique[key] = record
    records = list(unique.values())
    records.sort(
        key=lambda item: (
            "H1" in item and "H2" in item,
            min(
                item.get(split, {}).get("disagreement_win_rate", 0.0)
                for split in ("H1", "H2")
                if split in item
            ),
            sum(item.get(split, {}).get("safe_error_rescues", 0) for split in ("H1", "H2")),
            sum(item.get(split, {}).get("oracle_safe_or_expert_correct", 0) for split in ("H1", "H2")),
        ),
        reverse=True,
    )
    report = {
        "stage": "P89_existing_OOF_expert_complementarity_scan_v1",
        "protocol": (
            "Read-only inventory. Metrics only identify potentially complementary "
            "artifacts; every candidate still requires a leakage/protocol audit and "
            "a frozen H1-to-H2 gate before it can be used."
        ),
        "files_scanned": len(files),
        "aligned_signals": len(records),
        "failures": failures,
        "top_candidates": records[:100],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "files_scanned": len(files),
                "aligned_signals": len(records),
                "failures": len(failures),
                "top_candidates": records[:20],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
