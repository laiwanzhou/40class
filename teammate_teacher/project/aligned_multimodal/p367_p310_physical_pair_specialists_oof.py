"""Run cached physical-token pair specialists against the current P310 base."""
from __future__ import annotations

import json
from pathlib import Path

import p313_physical_topk_pair_reranker_oof as physical_pair
from p364_p310_hard_pair_specialists_oof import build_current


HERE = Path(__file__).resolve().parent
OUT = HERE / "runs/p367_p310_physical_pair_specialists_oof_v1"


def main():
    print(
        "P367 tests cached physical-token binary specialists on current P310 Top-3/Top-5 "
        "confusions; no encoder or Test data is run.",
        flush=True,
    )
    physical_pair.O = OUT
    physical_pair.KS = (3, 5)
    physical_pair.build = build_current
    physical_pair.main()
    summary_path = OUT / "summary.json"
    report = json.loads(summary_path.read_text(encoding="utf-8"))
    report["stage"] = "P367_P310_cached_physical_TopK_pair_specialists_OOF"
    report["protocol"]["base"] = "P310/P315 label-identical teacher"
    report["protocol"]["posterior"] = "P307 matched 30-teacher group posterior"
    report["aggregate"]["net_vs_p310"] = report["aggregate"].pop("net_vs_p245")
    fold_nets = report["aggregate"]["fold_nets"]
    strict_pass = bool(
        all(net > 0 for net in fold_nets)
        or (sum(net > 0 for net in fold_nets) >= 2 and min(fold_nets) >= -1)
    )
    report["aggregate"]["strict_gate_pass"] = strict_pass
    report["aggregate"]["decision"] = "eligible_for_test_audit" if strict_pass else "reject_before_test"
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "notes.txt").write_text(
        "Run 1: cached physical-token pair specialists on the current P310 base.\n"
        + json.dumps(report["aggregate"], ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
