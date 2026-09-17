"""Train one fixed Visual+Skeleton A18 Teacher on all 18 source subjects."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from a18_full_teacher_data import (
    A18_SOURCE_USER_SET,
    load_a18_data,
    write_a18_contract,
)
from p100a_global_teacher_data import (
    FoldNormalizer,
    P100ADataset,
    class_user_sample_weights,
    within_subject_permutation,
)
from p100a_global_teacher_model import P100AGlobalTeacher, P100AModelConfig
from train_p100a_global_teacher_oof import (
    classification_metrics,
    evaluate_model,
    make_loader,
    paired_comparison,
    set_seed,
    softmax_numpy,
    train_model,
)


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_CONFIG = HERE / "configs/a18_full_teacher.json"
DEFAULT_OUTPUT = HERE / "runs/a18_full_teacher_v1"
MODALITIES = ("visual", "skeleton")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
    ).strip()


def evaluate_counterfactual(
    model: P100AGlobalTeacher,
    data,
    indices: np.ndarray,
    normalizer: FoldNormalizer,
    batch_size: int,
    device: torch.device,
    seed: int,
    kind: str,
) -> np.ndarray:
    if kind == "zero_skeleton":
        dataset = P100ADataset(
            data,
            indices,
            normalizer,
            MODALITIES,
            zero_modalities=("skeleton",),
            cross_available=False,
        )
    elif kind == "shuffle_skeleton":
        source = within_subject_permutation(data, indices, "skeleton", seed + 1000)
        dataset = P100ADataset(
            data,
            indices,
            normalizer,
            MODALITIES,
            skeleton_source=source,
            cross_available=False,
        )
    else:
        raise ValueError(kind)
    result = evaluate_model(
        model,
        make_loader(dataset, batch_size, shuffle=False, seed=seed),
        device,
    )
    if not np.array_equal(result["rows"], indices):
        raise RuntimeError(f"A18 {kind} row order changed")
    return np.asarray(result["logits"], dtype=np.float32)


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_bytes = config_path.read_bytes()
    config: dict[str, Any] = json.loads(config_bytes.decode("utf-8"))
    if args.seed is not None:
        config["training"]["seed"] = int(args.seed)
    if config.get("variant") != "VS":
        raise ValueError("A18 route is fixed to Visual+Skeleton")
    epochs = int(args.epochs or config["training"]["epochs"])
    if args.smoke:
        epochs = 1
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    torch.set_float32_matmul_precision("high")
    data = load_a18_data()
    indices = data.full_indices()
    if set(data.users.tolist()) != A18_SOURCE_USER_SET:
        raise RuntimeError("A18 full source allow-list changed")
    write_a18_contract(output / "data_contract.json", data)
    manifest = {
        "status": "smoke" if args.smoke else "formal_started",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "config": str(config_path),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "epochs": epochs,
        "device": str(device),
        "data": data.summary(),
        "validation_split": False,
        "held_data_loaded": False,
        "held_label_used_for_checkpoint_or_recipe_selection": False,
        "h3_rows_selected": 0,
        "h3_confirmation_run": False,
        "b_teacher_started": False,
        "router_started": False,
        "confusion_family_loaded": False,
        "sweep_started": False,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    checkpoint_path = output / "a18_full_final.pt"
    prediction_path = output / "full_fit_predictions.npz"
    if args.resume and checkpoint_path.exists() and prediction_path.exists():
        print("A18 full refit already complete", flush=True)
        return

    normalizer = FoldNormalizer.fit(data, indices)
    weights = class_user_sample_weights(data, indices)
    seed = int(config["training"]["seed"])
    set_seed(seed)
    train_loader = make_loader(
        P100ADataset(
            data, indices, normalizer, MODALITIES, sample_weights=weights
        ),
        int(config["training"]["batch_size"]),
        shuffle=True,
        seed=seed,
    )
    model_config = P100AModelConfig(modalities=MODALITIES, **config["model"])
    model = P100AGlobalTeacher(model_config).to(device)
    print(
        json.dumps(
            {
                "stage": "A18_full_refit",
                "train_subjects": sorted(A18_SOURCE_USER_SET),
                "train_rows": len(indices),
                "validation_rows": 0,
                "epochs": epochs,
                "parameters": model.parameter_count,
            }
        ),
        flush=True,
    )
    history = train_model(model, train_loader, device, config["training"], epochs)

    # Fixed-final checkpoint is saved before any fit-set metric is computed.
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": model_config.__dict__,
            "normalizer_means": normalizer.means,
            "normalizer_stds": normalizer.stds,
            "epochs": epochs,
            "history": history,
            "train_users": sorted(A18_SOURCE_USER_SET),
            "validation_users": [],
            "metric_used_for_selection": None,
        },
        checkpoint_path,
    )

    eval_batch_size = int(config["training"]["eval_batch_size"])
    direct = evaluate_model(
        model,
        make_loader(
            P100ADataset(data, indices, normalizer, MODALITIES),
            eval_batch_size,
            shuffle=False,
            seed=seed,
        ),
        device,
    )
    if not np.array_equal(direct["rows"], indices):
        raise RuntimeError("A18 full-fit row order changed")
    direct_logits = np.asarray(direct["logits"], dtype=np.float32)
    zero_logits = evaluate_counterfactual(
        model, data, indices, normalizer, eval_batch_size, device, seed, "zero_skeleton"
    )
    shuffle_logits = evaluate_counterfactual(
        model,
        data,
        indices,
        normalizer,
        eval_batch_size,
        device,
        seed,
        "shuffle_skeleton",
    )
    direct_probability = softmax_numpy(direct_logits)
    zero_probability = softmax_numpy(zero_logits)
    shuffle_probability = softmax_numpy(shuffle_logits)
    np.savez_compressed(
        prediction_path,
        sample_ids=data.sample_ids,
        users=data.users,
        labels=data.labels,
        direct_logits=direct_logits,
        direct_probability=direct_probability,
        zero_skeleton_logits=zero_logits,
        zero_skeleton_probability=zero_probability,
        shuffle_skeleton_logits=shuffle_logits,
        shuffle_skeleton_probability=shuffle_probability,
    )
    systems = {
        "direct": classification_metrics(direct_probability, data.labels, data.users),
        "zero_skeleton": classification_metrics(
            zero_probability, data.labels, data.users
        ),
        "shuffle_skeleton": classification_metrics(
            shuffle_probability, data.labels, data.users
        ),
        "direct_minus_zero_skeleton": paired_comparison(
            direct_probability, zero_probability, data.labels, data.users
        ),
        "direct_minus_shuffle_skeleton": paired_comparison(
            direct_probability, shuffle_probability, data.labels, data.users
        ),
    }
    summary = {
        "status": "complete",
        "protocol": (
            "Single A18 Visual+Skeleton full refit on all 18 source subjects; fixed "
            "final epoch; metrics are fit-set diagnostics, not subject-disjoint OOF"
        ),
        "git_commit": git_commit(),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "checkpoint": str(checkpoint_path),
        "epochs": epochs,
        "data": data.summary(),
        "systems": systems,
        "metric_scope": "training_fit_set_only",
        "generalization_claim_allowed": False,
        "leakage_audit": {
            "held_data_loaded": False,
            "held_label_used_for_checkpoint_or_recipe_selection": False,
            "historical_40class_expert_probability_loaded": False,
            "h3_rows_selected": 0,
            "h3_confirmation_run": False,
            "b_teacher_started": False,
            "router_started": False,
            "confusion_family_loaded": False,
            "sweep_started": False,
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(systems["direct"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
