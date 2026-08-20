from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.probe_thermal_backbones import DeploymentAsset, build_deployment_ledger
from src.data.ir_primary_full_sequence_dataset import class_map_hash
from src.data.thermal_native_dataset import (
    ThermalNativeDataset,
    collate_thermal_trials,
    load_development_records,
)
from src.models.thermal_iformer_tsm import build_pretrained_iformer_t_expert
from src.train_thermal_native_expert import (
    T1BRecipe,
    build_optimizer,
    run_epoch,
    seed_everything,
    train_development,
)


DEFAULT_DATA_ROOT = PROJECT_ROOT.parent / "datasets/Small-Model-Track/train"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/thermal_iformer_t_tsm_train12_val2_seed20260715"
AUDIT_PATH = PROJECT_ROOT / "reports/thermal_stage0_data_alignment_audit.json"
SPLIT_PATH = PROJECT_ROOT / "metadata/splits/train12_val2_user6_user7_development.json"
CONFIG_PATH = PROJECT_ROOT / "configs/experiments/thermal_iformer_t_tsm_train12_val2.yaml"
RESULT_PATH = PROJECT_ROOT / "reports/thermal_iformer_t_tsm_train12_val2.json"
T1A_REPORT = PROJECT_ROOT / "reports/thermal_backbone_environment_probe.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _class_hash(records: list[Any]) -> str:
    rows = pd.DataFrame(
        sorted(
            {(record.class_id, record.action_name) for record in records},
            key=lambda item: item[0],
        ),
        columns=["class_id", "action_name"],
    )
    if len(rows) != 40:
        raise ValueError("Thermal development inventory must contain all 40 classes")
    return class_map_hash(rows)


def _deployment_ledger(checkpoint_path: Path) -> dict[str, Any]:
    prior = json.loads(T1A_REPORT.read_text(encoding="utf-8"))
    retained = [
        DeploymentAsset(**asset)
        for asset in prior["deployment_ledger"]["current_retained_inference_assets"]["assets"]
    ]
    retained.extend(
        [
            DeploymentAsset.from_file("thermal_iformer_t_tsm", checkpoint_path),
            DeploymentAsset(
                name="calibration_fusion_reserved_upper_bound",
                identity="f" * 64,
                serialized_bytes=3_000_000,
                sha256="f" * 64,
                status="reserved_not_yet_serialized",
            ),
        ]
    )
    return build_deployment_ledger(retained)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run authorized Thermal iFormer-T T1-B")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Frozen T1-B recipe requires CUDA bfloat16 AMP")
    device = torch.device("cuda")
    recipe = T1BRecipe()
    seed_everything(recipe.seed)
    train_canonical, validation_canonical = load_development_records(
        AUDIT_PATH, SPLIT_PATH, args.data_root
    )
    train_usable = [record for record in train_canonical if record.usable]
    validation_usable = [record for record in validation_canonical if record.usable]
    counts = {
        "train_canonical": len(train_canonical),
        "train_thermal_usable": len(train_usable),
        "train_unavailable_excluded_from_batchnorm_and_loss": len(train_canonical) - len(train_usable),
        "validation_canonical": len(validation_canonical),
        "validation_thermal_usable": len(validation_usable),
        "validation_unavailable_excluded_from_selection_metrics": len(validation_canonical) - len(validation_usable),
    }
    expected = (2039, 1922, 388, 377)
    actual = (
        counts["train_canonical"],
        counts["train_thermal_usable"],
        counts["validation_canonical"],
        counts["validation_thermal_usable"],
    )
    if actual != expected:
        raise ValueError(f"Frozen Thermal population changed: expected={expected}, actual={actual}")
    class_hash = _class_hash(train_canonical + validation_canonical)
    smoke_dataset = ThermalNativeDataset(train_usable[:4], training=True, seed=recipe.seed)
    smoke_loader = DataLoader(
        smoke_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_thermal_trials,
    )
    smoke_model = build_pretrained_iformer_t_expert().to(device)
    smoke_optimizer = build_optimizer(smoke_model, recipe)
    torch.cuda.reset_peak_memory_stats(device)
    smoke_started = time.perf_counter()
    smoke = run_epoch(
        smoke_model,
        smoke_loader,
        device=device,
        recipe=recipe,
        optimizer=smoke_optimizer,
        max_batches=1,
    )
    smoke_report = {
        "passed": True,
        "batch_shape": [4, 16, 3, 224, 224],
        "finite_loss": bool(torch.isfinite(torch.tensor(smoke.loss))),
        "loss": smoke.loss,
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "wall_seconds": time.perf_counter() - smoke_started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "cuda_smoke.json").write_text(
        json.dumps(smoke_report, indent=2), encoding="utf-8"
    )
    del smoke_model, smoke_optimizer
    torch.cuda.empty_cache()
    if args.smoke_only:
        print(json.dumps(smoke_report, indent=2))
        return

    train_dataset = ThermalNativeDataset(train_usable, training=True, seed=recipe.seed)
    validation_dataset = ThermalNativeDataset(validation_usable, training=False, seed=recipe.seed)
    model = build_pretrained_iformer_t_expert()
    training = train_development(
        model,
        train_dataset,
        validation_dataset,
        device=device,
        output_dir=args.output_dir,
        class_map_hash=class_hash,
        recipe=recipe,
    )
    checkpoint = Path(training["checkpoint_path"])
    result = {
        "schema_version": "thermal-t1b-development-v1",
        "stage": "thermal_t1b",
        "status": "completed_development_only",
        "branch": "experiment/thermal-iformer-t-t1b",
        "scientific_baseline_commit": "c42bb43091c79903e5fde5655c2846c87305895a",
        "t1a_parent_commit": "d97f9c940cc16da470ddb5a05ed70b06897a6e39",
        "authorization": "explicit_user_instruction_2026_08_20",
        "data_access": {
            "population": "train12_fit_user6_user7_validation_only",
            "heldout_labels_accessed": False,
            "competition_test_accessed": False,
            "quarantined_evidence_accessed": False,
            "ir_x3d_modified": False,
        },
        "inputs": {
            "audit_path": str(AUDIT_PATH.relative_to(PROJECT_ROOT)),
            "audit_sha256": _sha256(AUDIT_PATH),
            "split_path": str(SPLIT_PATH.relative_to(PROJECT_ROOT)),
            "split_sha256": _sha256(SPLIT_PATH),
            "config_path": str(CONFIG_PATH.relative_to(PROJECT_ROOT)),
            "config_sha256": _sha256(CONFIG_PATH),
            "class_map_hash": class_hash,
            "route": "full_frame",
            "sampling": "16_uniform_thermal_native_normalized_time",
        },
        "population": counts,
        "cuda_smoke": smoke_report,
        "training": training,
        "deployment_ledger": _deployment_ledger(checkpoint),
        "limitations": [
            "Development selection only; final candidate retention requires shared train-14 OOF.",
            "Single-user absent classes are not interpreted as zero model capability.",
            "The complete six-modal package remains unevaluable until other experts are retained.",
        ],
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"result_path": str(RESULT_PATH), "best_epoch": training["best_epoch"]}, indent=2))


if __name__ == "__main__":
    main()
