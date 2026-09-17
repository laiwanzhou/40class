from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_C1 = PROJECT_DIR / "runs/p87s_fusion_holdout1_c1_emission_v1"
DEFAULT_C2 = PROJECT_DIR / "runs/p87s_fusion_holdout1_c2_structured_v1"
DEFAULT_TARGETS = (
    PROJECT_DIR / "runs/p87s_holdout1_structured_targets_v1/structured_targets.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one deployable checkpoint by interpolating the paired C1/C2 "
            "adaptation deltas. No second model is required at inference."
        )
    )
    parser.add_argument("--emission-run", type=Path, default=DEFAULT_C1)
    parser.add_argument("--structured-run", type=Path, default=DEFAULT_C2)
    parser.add_argument("--structured-targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument(
        "--structured-weight",
        type=float,
        help="Explicit structured delta weight. Omit to use mean target confidence.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    emission_dir = args.emission_run.resolve()
    structured_dir = args.structured_run.resolve()
    emission_path = emission_dir / "unified_student.pt"
    structured_path = structured_dir / "unified_student.pt"
    emission = torch.load(emission_path, map_location="cpu", weights_only=False)
    structured = torch.load(structured_path, map_location="cpu", weights_only=False)
    if emission.get("base_checkpoint") != structured.get("base_checkpoint"):
        raise ValueError("C1 and C2 do not share an identical base checkpoint")
    if emission.get("modality") != structured.get("modality"):
        raise ValueError("C1 and C2 modalities differ")

    if args.structured_weight is None:
        targets = np.load(args.structured_targets.resolve(), allow_pickle=False)
        mask = targets["target_mask"].astype(bool)
        weight = float(targets["structured_distillation_weight"][mask].mean())
        weight_source = "mean label-free structured distillation confidence"
    else:
        weight = float(args.structured_weight)
        weight_source = "explicit"
    if not 0.0 <= weight <= 1.0:
        raise ValueError("structured weight must be between zero and one")

    first = emission["model_state"]
    second = structured["model_state"]
    if first.keys() != second.keys():
        raise ValueError("C1 and C2 state keys differ")
    merged: dict[str, torch.Tensor] = {}
    changed = unchanged = 0
    maximum_delta = 0.0
    for key in first:
        left = first[key]
        right = second[key]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"C1/C2 tensor contract differs at {key}")
        if left.is_floating_point():
            merged[key] = torch.lerp(left, right, weight)
            delta = float((left.float() - right.float()).abs().max())
            if delta > 0:
                changed += 1
                maximum_delta = max(maximum_delta, delta)
            else:
                unchanged += 1
        else:
            if not torch.equal(left, right):
                raise ValueError(f"non-floating C1/C2 state differs at {key}")
            merged[key] = left.clone()
            unchanged += 1

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint: dict[str, Any] = dict(structured)
    checkpoint.update(
        {
            "stage": "P87S_adaptation_delta_soup",
            "target": "emission_structured_delta_soup",
            "model_state": merged,
            "emission_checkpoint": str(emission_path),
            "structured_checkpoint": str(structured_path),
            "structured_weight": weight,
            "structured_weight_source": weight_source,
        }
    )
    output_checkpoint = output / "unified_student.pt"
    torch.save(checkpoint, output_checkpoint)
    # Reuse row identity only. Logits/predictions must be regenerated from the merged
    # checkpoint by the normal evaluator; copying either branch output would be invalid.
    summary = {
        "stage": "P87S_adaptation_delta_soup",
        "status": "checkpoint_only_requires_evaluation",
        "protocol": (
            "C1 and C2 start from the same C0 and update the same parameter subset. "
            "Interpolate paired fine-tuning endpoints into one checkpoint, preserving "
            "the 89.88 MiB deployment footprint."
        ),
        "emission_checkpoint": str(emission_path),
        "emission_checkpoint_sha256": sha256(emission_path),
        "structured_checkpoint": str(structured_path),
        "structured_checkpoint_sha256": sha256(structured_path),
        "structured_weight": weight,
        "structured_weight_source": weight_source,
        "changed_float_tensors": changed,
        "unchanged_tensors": unchanged,
        "maximum_endpoint_tensor_delta": maximum_delta,
        "output_checkpoint": str(output_checkpoint),
        "parameters": sum(value.numel() for value in merged.values()),
        "fp32_mib": sum(value.numel() * value.element_size() for value in merged.values())
        / 1024**2,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
