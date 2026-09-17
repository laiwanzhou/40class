"""Train-cache diagnostics and label-free Test input audit; never select a model.

Historical scores are explicitly exploratory. No production artifacts are changed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

from stable_routing_protocol import paired_subject_bootstrap, register_experiment

HERE = Path(__file__).resolve().parent
CHAMPION = HERE / "runs/p315_final_kaggle_candidate_v1/submission_p315_compact_student_raw.csv"
CHAMPION_HASH = "a9796cde28c8a999623a1ee388047c97d2d1553fd9bf10a9ed30f8ca7768561c"
DEFAULT_OUT = HERE / "runs/p415_stable_routing_audit_v1"
NONVISUAL_FALLBACKS = {
    "p90_deep_imu", "p90_motionbert_3view", "expanded_p12_imu",
    "expanded_motionbert_front", "expanded_skeleton_invariant",
}
ALIASES = {"expanded_thermal": "p12_thermal_candidate", "p88_latent_prefix": "p88_repeat"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def funnel(part, k):
    y, b = part["labels"], part["base"]
    visual, group = part["visual_references"]["strong_visual_mean"], part["group_probability"]
    wrong = b != y
    locked = visual.argmax(1) == b
    in_visual = (np.argsort(-visual, axis=1, kind="stable")[:, :k] == y[:, None]).any(1)
    in_group = (np.argsort(-group, axis=1, kind="stable")[:, :k] == y[:, None]).any(1)
    reachable = wrong & ~locked & in_visual & in_group
    bank_support = (part["bank"].argmax(2) == y[:, None]).any(1)
    return {
        "rows": len(y), "base_correct": int((~wrong).sum()), "errors": int(wrong.sum()),
        "locked_rows": int(locked.sum()), "locked_errors": int((locked & wrong).sum()),
        "locked_errors_with_any_bank_top1_correct": int((locked & wrong & bank_support).sum()),
        "reachable_errors": int(reachable.sum()),
        "additional_reachable_if_union": int((wrong & ~locked & (in_visual | in_group) & ~(in_visual & in_group)).sum()),
        "oracle_correct": int((~wrong).sum() + reachable.sum()),
    }


def provenance_examples(parts):
    """Source-code-derived ancestry; not an artifact-level DAG certificate."""
    result = []
    for outer in parts:
        held_users = set(parts[outer]["users"].tolist())
        for source in parts:
            if source == outer:
                continue
            upstream_training = set().union(*[
                set(part["users"].tolist()) for name, part in parts.items() if name != source
            ])
            result.append({
                "router_outer_held": outer, "router_source_features": source,
                "cached_group_probability": f"P307/{source}",
                "upstream_supervised_cohorts": [name for name in parts if name != source],
                "forbidden_held_subjects_in_upstream_training": sorted(held_users & upstream_training),
                "outer_pure": not bool(held_users & upstream_training),
                "evidence_level": "source_code_derived_not_artifact_certified",
            })
    return result


def historical_uncertainty(parts):
    y = np.concatenate([p["labels"] for p in parts.values()])
    b = np.concatenate([p["base"] for p in parts.values()])
    users = np.concatenate([p["users"] for p in parts.values()])
    ids = np.concatenate([p["ids"] for p in parts.values()])
    runs = {
        "P410": HERE / "runs/p410_target_aware_weighted_raw_detail_oof_v1/oof_predictions.npz",
        "P414_p12_imu": HERE / "runs/p414_single_matched_replacement_ablation_v1/replace_p12_imu/oof_predictions.npz",
    }
    report = {}
    for name, path in runs.items():
        if not path.exists():
            report[name] = {"status": "missing"}
            continue
        with np.load(path, allow_pickle=False) as saved:
            if "sample_ids" not in saved.files:
                report[name] = {
                    "artifact_sha256": sha256(path),
                    "status": "subject_ci_unavailable_missing_sample_ids",
                    "warning": "Matching labels/base alone does not prove row identity. No subject CI or per-subject gain is certified.",
                }
                continue
            saved_ids = saved["sample_ids"].astype(str)
            if len(set(saved_ids)) != len(saved_ids) or set(saved_ids) != set(ids):
                raise ValueError(f"historical sample IDs mismatch: {path}")
            lookup = {sample_id: i for i, sample_id in enumerate(saved_ids)}
            order = np.array([lookup[sample_id] for sample_id in ids])
            if not np.array_equal(saved["labels"][order], y) or not np.array_equal(saved["base_prediction"][order], b):
                raise ValueError(f"historical array order mismatch: {path}")
            if "users" in saved.files and not np.array_equal(saved["users"][order].astype(str), users):
                raise ValueError(f"historical subject order mismatch: {path}")
            pred = saved["prediction"][order].copy()
        report[name] = {
            "artifact_sha256": sha256(path),
            "status": "historical_exploratory_not_confirmation",
            "net": int((pred == y).sum() - (b == y).sum()),
            "rescue": int(((pred == y) & (b != y)).sum()),
            "harm": int(((pred != y) & (b == y)).sum()),
            "per_subject_net": {str(u): int(((pred == y).astype(int) - (b == y).astype(int))[users == u].sum()) for u in np.unique(users)},
            "sample_weighted_ci": asdict(paired_subject_bootstrap(users, y, pred, b, n_bootstrap=10000, seed=20260907)),
            "subject_mean_ci": asdict(paired_subject_bootstrap(users, y, pred, b, n_bootstrap=10000, seed=20260907, weighting="subject_mean")),
            "warning": "Descriptive interval only: reused subjects, model selection and shared fitted folds invalidate a fresh-confirmation interpretation.",
        }
    return report


def label_free_test_audit():
    # This loader obtains cached probabilities and raw covariates; labels are -1.
    import p400_candidate_conditioned_raw_detail_test as deployment
    import p399_candidate_conditioned_raw_detail_head_oof as candidate
    from p386_visual_scope_nonvisual_competence_gate import NONVISUAL_TEACHERS
    with np.load(deployment.P310, allow_pickle=False) as saved:
        ids = saved["sample_ids"].astype(str)
    test = deployment.test_part(ids)
    if np.any(test["labels"] != -1):
        raise ValueError("Test loader must expose no labels")
    nvp = test["nonvisual_probability"]
    names = list(NONVISUAL_TEACHERS)
    rows = []
    for i, name in enumerate(names):
        status = "safe_fallback" if name in NONVISUAL_FALLBACKS else "alias" if name in ALIASES else "not_yet_certified"
        identical = [other for j, other in enumerate(names) if j != i and np.allclose(nvp[:, i], nvp[:, j], rtol=0, atol=1e-7)]
        rows.append({"expert": name, "status": status, "alias_of": ALIASES.get(name), "numerically_identical_slots": identical})
    candidate_counts = {}
    for k in (3, 5):
        sizes = np.array([len(c) for c in candidate.candidates(test, k)])
        candidate_counts[str(k)] = {"rows_with_alternative": int((sizes > 1).sum()), "max_size": int(sizes.max()), "mean_size": float(sizes.mean())}
    return {
        "rows": len(ids), "test_labels_read": False, "model_fits": 0,
        "nonvisual_slots": rows, "candidate_counts": candidate_counts,
        "evidence": "p216.test_lookup default safe fill; p214.test_candidate_split aliases; numerical equality independently checked",
        "full_contract_certified": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--include-unlabeled-test-contract", action="store_true")
    args = parser.parse_args()
    if sha256(CHAMPION) != CHAMPION_HASH:
        raise RuntimeError("frozen champion hash mismatch")
    if args.output_dir.exists():
        raise FileExistsError(f"audit outputs are immutable: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    spec = {
        "stage": "P415", "kind": "diagnostic_only_no_model_selection",
        "candidate_k": [3, 5], "bootstrap_draws": 10000, "seed": 20260907,
        "test_covariates_allowed": args.include_unlabeled_test_contract,
        "test_labels_allowed": False, "champion_sha256": CHAMPION_HASH,
        "source_code_sha256": sha256(__file__),
        "guard_code_sha256": sha256(HERE / "stable_routing_protocol.py"),
    }
    digest = register_experiment(args.output_dir / "registration.json", spec, [args.output_dir / "summary.json"])
    print("P415 audits historical evidence and feature contracts; it does not select or train a candidate.", flush=True)
    import p399_candidate_conditioned_raw_detail_head_oof as source
    with threadpool_limits(limits=4):
        parts = source.load_data()
        report = {
            "stage": "P415_stable_routing_audit", "registration_sha256": digest,
            "environment": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__},
            "historical_funnel": {name: {str(k): funnel(p, k) for k in (3, 5)} for name, p in parts.items()},
            "source_code_derived_nested_ancestry_risks": provenance_examples(parts),
            "ancestry_evidence_hashes": {name: sha256(HERE / name) for name in (
                "p165_deployable_group_teacher.py", "p307_union_repeat_group_sequence_audit.py",
                "p361_bidirectional_existing_teacher_gate_oof.py", "p399_candidate_conditioned_raw_detail_head_oof.py",
            )},
            "historical_uncertainty": historical_uncertainty(parts),
            "test_contract": label_free_test_audit() if args.include_unlabeled_test_contract else {"status": "not_requested"},
            "decision": "rebuild_outer_pure_inputs_before_testing_new_router",
            "kaggle_target_achieved": False, "new_models_trained": 0,
        }
    if sha256(CHAMPION) != CHAMPION_HASH:
        raise RuntimeError("champion changed during audit")
    with (args.output_dir / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    (args.output_dir / "notes.txt").write_text("All diagnostics retained. No thresholds selected, no labels inferred from Test, no model promotion.\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), "decision": report["decision"], "historical_uncertainty": report["historical_uncertainty"], "test_contract": report["test_contract"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
