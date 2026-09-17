"""Current P310 pair specialists on cached Skeleton+IMU motion statistics."""
from __future__ import annotations

import json
from pathlib import Path

import p313_physical_topk_pair_reranker_oof as pair
from p364_p310_hard_pair_specialists_oof import build_current
from p366_cached_feature_hard_target_verifier_oof import motion_features
from p90_teacher_common import load_protocol


HERE = Path(__file__).resolve().parent
OUT = HERE / "runs/p379_motion_pair_specialists_oof_v1"


def cached_motion():
    protocol = load_protocol()
    return protocol.sample_ids, motion_features(protocol)


def main():
    print(
        "P379 tests pair-specific Ridge heads on cached Skeleton+IMU motion statistics "
        "within current P310 Top-3/Top-5 candidates.",
        flush=True,
    )
    pair.O = OUT
    pair.KS = (3, 5)
    pair.ALPHAS = (100.0, 1000.0, 3000.0)
    pair.build = build_current
    pair.physical = cached_motion
    pair.main()
    summary_path = OUT / "summary.json"
    report = json.loads(summary_path.read_text(encoding="utf-8"))
    report["stage"] = "P379_P310_Skeleton_IMU_motion_pair_specialists_OOF"
    report["protocol"]["base"] = "P310/P315 label-identical teacher"
    report["protocol"]["features"] = "cached Skeleton relation/motion plus IMU temporal/global statistics"
    report["aggregate"]["net_vs_p310"] = report["aggregate"].pop("net_vs_p245")
    fold_nets = report["aggregate"]["fold_nets"]
    strict = bool(
        all(net > 0 for net in fold_nets)
        or (sum(net > 0 for net in fold_nets) >= 2 and min(fold_nets) >= -1)
    )
    report["aggregate"]["strict_gate_pass"] = strict
    report["aggregate"]["decision"] = "eligible_for_test_audit" if strict else "reject_before_test"
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "notes.txt").write_text(
        "Run 1: cached Skeleton+IMU pair specialists on the current P310 base.\n"
        + json.dumps(report["aggregate"], ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
