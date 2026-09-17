"""Formal mechanism and complementarity audit for P103-B3 local visual OOF."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from analyze_p102_b0_mechanisms import outcome_breakdown, per_value_net
from audit_p102_session_closure import load_npz


HERE = Path(__file__).resolve().parent
DEFAULT_B2 = HERE / "runs/p103_b2_rich_visual_oof_v1/b2_oof_predictions.npz"
DEFAULT_B3 = HERE / "runs/p103_b3_local_visual_oof_v1/b3_oof_predictions.npz"
DEFAULT_SUMMARY = HERE / "runs/p103_b3_local_visual_oof_v1/summary.json"
DEFAULT_HARD = HERE / "runs/p102_hard_set_v1/hard_set.npz"
DEFAULT_OUTPUT = HERE / "runs/p103_b3_local_visual_oof_v1/mechanism_audit.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b2", type=Path, default=DEFAULT_B2)
    parser.add_argument("--b3", type=Path, default=DEFAULT_B3)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--hard-set", type=Path, default=DEFAULT_HARD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def normalized_entropy(values: np.ndarray) -> np.ndarray:
    probability = np.asarray(values, dtype=np.float64)
    probability /= np.maximum(probability.sum(axis=-1, keepdims=True), 1e-12)
    return -np.sum(probability * np.log(np.maximum(probability, 1e-12)), axis=-1) / np.log(
        probability.shape[-1]
    )


def attention_structure(
    attention: np.ndarray, candidates: np.ndarray, names: np.ndarray
) -> dict[str, Any]:
    lookup = {str(name): index for index, name in enumerate(names.astype(str))}
    axes = {
        "encoder": [lookup["encoder_videomaev2"], lookup["encoder_vjepa2"]],
        "roi": [lookup[f"roi_{name}"] for name in ("workspace", "left_hand", "right_hand", "interaction")],
        "window": [
            lookup[f"window_{name}"]
            for name in ("full", "early", "middle", "late", "motion_peak")
        ],
        "type": [lookup["type_feature"], lookup["type_action"]],
    }
    valid = candidates >= 0
    queries = attention[valid]
    result: dict[str, Any] = {}
    for axis, indices in axes.items():
        entropy = normalized_entropy(queries[:, indices])
        dispersion = []
        for row in range(len(attention)):
            selected = attention[row, valid[row]][:, indices]
            dispersion.append(float(np.mean(np.std(selected, axis=0))))
        result[axis] = {
            "mean_normalized_entropy": float(entropy.mean()),
            "p10_normalized_entropy": float(np.quantile(entropy, 0.10)),
            "mean_candidate_query_dispersion": float(np.mean(dispersion)),
        }
    return result


def confusion_delta(
    labels: np.ndarray,
    a_prediction: np.ndarray,
    aligned: np.ndarray,
    zero: np.ndarray,
    selected: np.ndarray,
) -> dict[str, list[dict[str, int]]]:
    rows = []
    for (truth, base), count in Counter(
        zip(labels[selected].tolist(), a_prediction[selected].tolist(), strict=True)
    ).items():
        mask = selected & (labels == truth) & (a_prediction == base)
        aligned_correct = int(np.sum(aligned[mask] == labels[mask]))
        zero_correct = int(np.sum(zero[mask] == labels[mask]))
        rows.append(
            {
                "true": int(truth),
                "a_prediction": int(base),
                "rows": int(count),
                "aligned_correct": aligned_correct,
                "zero_correct": zero_correct,
                "aligned_minus_zero": aligned_correct - zero_correct,
            }
        )
    return {
        "best": sorted(rows, key=lambda row: (-row["aligned_minus_zero"], -row["rows"]))[:15],
        "worst": sorted(rows, key=lambda row: (row["aligned_minus_zero"], -row["rows"]))[:15],
    }


def main() -> None:
    args = parse_args()
    b2 = load_npz(args.b2.resolve())
    b3 = load_npz(args.b3.resolve())
    hard = load_npz(args.hard_set.resolve())
    summary = json.loads(args.summary.resolve().read_text(encoding="utf-8"))
    for archive, name in ((b2, "B2"), (hard, "hard")):
        if not np.array_equal(b3["sample_ids"].astype(str), archive["sample_ids"].astype(str)):
            raise RuntimeError(f"B3/{name} sample order differs")
    labels = np.asarray(b3["labels"], dtype=np.int64)
    users = b3["users"].astype(str)
    folds = np.asarray(b3["fold_ids"], dtype=np.int64)
    candidates = np.asarray(b3["candidate_ids"], dtype=np.int64)
    evaluated = np.asarray(b3["evaluated"], dtype=bool)
    if len(labels) != 1941 or not evaluated.all():
        raise RuntimeError("formal B3 OOF coverage is incomplete")
    if summary["data"]["h3_rows_selected"] != 0 or summary["data"]["h3_users_loaded"]:
        raise RuntimeError("H3 contract failed")

    a_probability = np.asarray(b3["a_session_probability"], dtype=np.float64)
    a_prediction = a_probability.argmax(axis=1)
    probabilities = {
        name: np.asarray(b3[f"{name}_probability"], dtype=np.float64)
        for name in (
            "full_local",
            "full_session",
            "shuffle_local",
            "shuffle_session",
            "zero_local",
            "zero_session",
        )
    }
    predictions = {name: value.argmax(axis=1) for name, value in probabilities.items()}
    selected = (a_prediction != labels) & np.any(candidates == labels[:, None], axis=1)
    if int(selected.sum()) != 289:
        raise RuntimeError("B3 attackable set drifted")
    aligned = predictions["full_session"]
    zero = predictions["zero_session"]

    rescue_sets: dict[str, dict[str, set[int]]] = {}
    for variant in ("full_local", "full_session"):
        rescue_sets[variant] = {
            stage: set(
                np.flatnonzero(
                    (a_prediction != labels)
                    & (np.asarray(archive[f"{variant}_probability"]).argmax(axis=1) == labels)
                ).tolist()
            )
            for stage, archive in (("b2", b2), ("b3", b3))
        }
    complementarity = {
        variant: {
            "b2_rescues": len(sets["b2"]),
            "b3_rescues": len(sets["b3"]),
            "overlap": len(sets["b2"] & sets["b3"]),
            "union": len(sets["b2"] | sets["b3"]),
        }
        for variant, sets in rescue_sets.items()
    }
    attention = np.asarray(b3["aligned_attention_groups"], dtype=np.float32)
    result = {
        "status": "complete",
        "integrity": {
            "rows": 1941,
            "subjects": sorted(set(users.tolist())),
            "folds": sorted(set(map(int, folds.tolist()))),
            "h3_rows_selected": 0,
            "h3_users_loaded": [],
        },
        "level1": summary["level1_conditional"],
        "level2": summary["level2_full_system"],
        "outcomes": {
            name: outcome_breakdown(
                labels, a_prediction, prediction, np.ones(len(labels), dtype=bool)
            )
            for name, prediction in predictions.items()
        },
        "fold": per_value_net(folds.astype(str), labels, a_prediction, aligned),
        "subject": per_value_net(users, labels, a_prediction, aligned),
        "b2_b3_rescue_complementarity": complementarity,
        "confusion_aligned_vs_zero": confusion_delta(
            labels, a_prediction, aligned, zero, selected
        ),
        "attention_structure": attention_structure(
            attention, candidates, b3["attention_group_names"].astype(str)
        ),
        "mechanism_decision": {
            "b3_pass": False,
            "student_authorized": False,
            "hard_case_b_teacher_route_exhausted": False,
            "b_teacher_hypothesis_closed": False,
            "failure_type": "local visual label direction is confusion-specific and not cross-subject stable",
            "evidence": [
                "B3 full-session conditional Top-1 is 52/289 (17.99%) and net is -74.",
                "Aligned full-session is 7 rows worse than shuffled local visual and 58 worse than zero local visual.",
                "All four fold nets are negative (-6/-25/-28/-15) after about 99% source training accuracy.",
                "B3 post-Session Top-2 improves to 167/289 and Top-5 reaches 1825, showing rank information without reliable Top-1 direction.",
                "B2 and B3 full-session rescues overlap on only 30 rows and union to 86, so local views changed evidence rather than reproducing B2.",
                "B3 strongly helps 37->6 and 8->9 but harms 24->27, 22->21, and 8->10 relative to zero; a motion/geometry disambiguator is the next causal representation gap.",
            ],
            "authorized_revision": (
                "P103-B4 staged candidate-conditioned local-visual -> Skeleton -> IMU interaction. "
                "Use raw 2x16x5 motion tokens, explicit same-subject wrong-label motion "
                "correspondence loss, and aligned/shuffle/zero/reverse motion controls. Keep "
                "the frozen candidate, hard population, optimizer, epochs, and Session recipe."
            ),
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["mechanism_decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
