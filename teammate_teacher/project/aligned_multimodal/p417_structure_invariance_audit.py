"""Input-order audit of repeat-group structure. No fitting or accuracy selection."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from aligned_multimodal.audit_p87_sequence_decoder import RecordingMetadata, align_metadata
from aligned_multimodal.stable_routing_protocol import register_experiment

METADATA = HERE / "data/p85_recording_metadata/train_recording_metadata.csv"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def permute_metadata(meta, order):
    return RecordingMetadata(meta.sample_ids[order], meta.users[order], meta.dates[order], meta.starts[order])


def compare(reference, reordered, order):
    delta = np.max(np.abs(reference - reordered[np.argsort(order)]), axis=1)
    return {"changed_rows_at_1e_6": int((delta > 1e-6).sum()), "maximum_abs_difference": float(delta.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    from p361_bidirectional_existing_teacher_gate_oof import load_parts
    from p137_group_classifier_selector import group_features
    from aligned_multimodal.stable_routing_structure import build_group_features
    spec = {"stage": "P417", "purpose": "label_free_structural_invariance_not_model_selection",
            "permutations": ["reverse", "random_seed_20260907"], "tolerance": 1e-6,
            "new_rule": "ambiguous_or_missing_timing_rows_have_no_peers_but_remain_in_output",
            "script_sha256": sha(__file__), "metadata_sha256": sha(METADATA),
            "structure_sha256": sha(HERE / "stable_routing_structure.py"),
            "legacy_sha256": {name: sha(HERE / name) for name in (
                "p137_group_classifier_selector.py", "p134_frozen_repeat_consensus.py",
                "p89_global_repeat_decoder.py", "p88_aligned_repeat_holdout.py")}}
    register_experiment(args.output_dir / "registration.json", spec)
    snapshot = args.output_dir / "source_snapshot"
    snapshot.mkdir(exist_ok=False)
    for path in (Path(__file__), HERE / "stable_routing_structure.py"):
        (snapshot / path.name).write_bytes(path.read_bytes())
    parts, names = load_parts()
    ids = np.concatenate([part["ids"] for part in parts.values()])
    bank = np.concatenate([part["bank"] for part in parts.values()])
    # IDs join metadata only, never become a model feature or chronological tie-break.
    meta = align_metadata(METADATA, ids)
    base = bank.mean(axis=1).argmax(axis=1)
    lookup = {sample_id: value for sample_id, value in zip(ids, bank, strict=True)}
    old = group_features(ids, base, lookup, metadata_path=METADATA)
    new, geometry = build_group_features(bank, meta)
    permutations = {"reverse": np.arange(len(ids))[::-1],
                    "random_seed_20260907": np.random.default_rng(20260907).permutation(len(ids))}
    report = {"rows": len(ids), "experts": names, "new_geometry": geometry, "transformations": {}}
    for name, order in permutations.items():
        old_p = group_features(ids[order], base[order], lookup, metadata_path=METADATA)
        new_p, _ = build_group_features(bank[order], permute_metadata(meta, order))
        report["transformations"][name] = {"legacy": compare(old, old_p, order), "safe": compare(new, new_p, order)}
    report["new_vs_legacy_feature_difference"] = compare(old, new, np.arange(len(ids)))
    report["interpretation"] = (
        "Within-implementation permutation changes diagnose input-order sensitivity, "
        "not accuracy gain or Test label leakage proof. Cross-implementation differences "
        "also include mean-probability versus legacy vote-based grouping evidence; "
        "they cannot be attributed solely to geometry repairs."
    )
    report["new_model_fits"] = 0
    report["test_rows_loaded"] = 0
    report["target_achieved"] = False
    with (args.output_dir / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
