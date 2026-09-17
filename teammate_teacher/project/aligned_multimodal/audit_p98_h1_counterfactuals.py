"""Run H1-only modality counterfactuals on frozen P98 OOF checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from p98_four_modal_teacher_data import (
    build_four_modal_source_data,
    counterfactual_data,
)
from train_p98_four_modal_teacher import (
    build_model,
    infer_outputs,
    load_config,
    mean_output,
    subset_audit,
)


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_RUN = PROJECT / "runs/p98_four_modal_teacher_p91_matched_t2_h1_v1"
DEFAULT_CONFIG = HERE / "configs/p98_four_modal_teacher_p91_matched_t2.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run = args.run.resolve()
    raw_config, model_config, training = load_config(args.config.resolve())
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    if summary["stage"] != "h1_cv" or summary["config"] != raw_config:
        raise ValueError("run summary does not match the frozen H1 configuration")
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    raw = build_four_modal_source_data(model_config.modalities, include_h2=False)
    h1 = raw.boundaries["H1_selection"]
    users = summary["evaluated_users"]
    scenarios = ["direct"] + [
        f"{mode}_{modality}"
        for modality in model_config.modalities
        for mode in ("zero", "shuffle")
    ]
    logits = {
        name: np.full((len(raw.labels), 40), np.nan, dtype=np.float32)
        for name in scenarios
    }
    for fold_index, held_user in enumerate(users):
        target = h1[raw.users[h1] == held_user]
        fold_outputs: dict[str, list[dict[str, Any]]] = {
            name: [] for name in scenarios
        }
        for seed in training.seeds:
            path = run / f"h1_fold{fold_index}_{held_user}_seed{seed}.pt"
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            data = checkpoint["preprocessor"].transform(raw)
            model = build_model(model_config, data).to(device)
            model.load_state_dict(checkpoint["state_dict"])
            fold_outputs["direct"].append(
                infer_outputs(model, data, target, device, training.batch_size)
            )
            for modality in model_config.modalities:
                for mode in ("zero", "shuffle"):
                    name = f"{mode}_{modality}"
                    changed = counterfactual_data(data, modality, mode, seed=17)
                    fold_outputs[name].append(
                        infer_outputs(
                            model, changed, target, device, training.batch_size
                        )
                    )
        for name, values in fold_outputs.items():
            logits[name][target] = mean_output(values)["logits"]
        print(
            f"[H1 counterfactual] completed fold={fold_index + 1}/{len(users)} "
            f"held_user={held_user}",
            flush=True,
        )
    with np.load(run / "h1_oof_predictions.npz", allow_pickle=False) as source:
        saved_ids = source["sample_ids"].astype(str)
        saved_logits = source["direct_logits"].astype(np.float32)
    if not np.array_equal(saved_ids, raw.sample_ids[h1]):
        raise ValueError("saved H1 IDs differ from reconstructed H1 IDs")
    reproduction_error = float(np.max(np.abs(saved_logits - logits["direct"][h1])))
    if reproduction_error > 1e-4:
        raise ValueError(f"direct checkpoint reproduction error: {reproduction_error}")
    direct = subset_audit(
        raw.labels[h1], logits["direct"][h1], raw.base_prediction[h1], raw.users[h1]
    )
    audits: dict[str, Any] = {}
    for name in scenarios[1:]:
        item = subset_audit(
            raw.labels[h1], logits[name][h1], raw.base_prediction[h1], raw.users[h1]
        )
        item["delta_correct_vs_direct"] = int(
            item["top1_correct"] - direct["top1_correct"]
        )
        audits[name] = item
    result = {
        "stage": "P98 H1 frozen-checkpoint modality counterfactual audit",
        "scope": "H1 OOF only; H2/H3 unread; same checkpoints for direct/zero/shuffle",
        "config_sha256": summary["config_sha256"],
        "direct_reproduction_max_abs_error": reproduction_error,
        "direct": direct,
        "counterfactuals": audits,
    }
    write_json(run / "h1_counterfactual_audit.json", result)
    concise = {
        "direct": {
            key: direct[key]
            for key in ("top1_correct", "top1_accuracy", "rescue", "harm", "net")
        },
        "counterfactuals": {
            name: {
                key: value[key]
                for key in (
                    "top1_correct",
                    "top1_accuracy",
                    "rescue",
                    "harm",
                    "net",
                    "delta_correct_vs_direct",
                )
            }
            for name, value in audits.items()
        },
        "direct_reproduction_max_abs_error": reproduction_error,
    }
    print(json.dumps(concise, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
