"""Token-level P92 fusion with the twelve-view V-JEPA 2 visual teacher.

This runner uses the current P91 champion only as the must-be-beaten comparison
and a source-selected conservative stabilizer.  V-JEPA tokens enter the
Transformer before the 40-class decision; this is not a late-logit-only test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from p91_hierarchical_multimodal_teacher import (
    FusionData,
    HierarchicalTeacher,
    Preprocessor,
    audit,
    blend_prediction,
    build_data,
    infer_full,
    train_model,
)
from p91_subject_domain_normalization_probe import champion_predictions


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_OUTPUT = PROJECT / "runs/p92_vjepa2_token_fusion_h3_v1"
VJEPA_RUN = PROJECT / "runs/p92_vjepa2_vitl_ssv2_12view_fold0_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model-dim", type=int, default=192)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.22)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-3)
    parser.add_argument("--statistics-dim", type=int, default=64)
    parser.add_argument("--seeds", default="17")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def align(ids: np.ndarray, values: np.ndarray, target_ids: np.ndarray) -> np.ndarray:
    lookup = {str(sample_id): row for row, sample_id in enumerate(ids)}
    return np.asarray([values[lookup[str(sample_id)]] for sample_id in target_ids])


def build_vjepa_data() -> FusionData:
    raw = build_data()
    done = np.load(VJEPA_RUN / "done.npy", mmap_mode="r")
    if not bool(np.asarray(done).all()):
        raise RuntimeError(
            f"V-JEPA extraction incomplete: {int(np.asarray(done).sum())}/{len(done)}"
        )
    features = np.load(VJEPA_RUN / "features.npy", mmap_mode="r")
    # The V-JEPA cache follows the P90 master protocol order.
    from p90_teacher_common import load_protocol

    protocol = load_protocol()
    vjepa = align(
        protocol.sample_ids.astype(str),
        np.asarray(features, dtype=np.float32),
        raw.sample_ids,
    )
    # Official HD-GCN transfer was already rejected on H3.  Keep the stronger
    # MotionBERT plus task-specific raw graph/IMU encoders and replace that weak
    # frozen stream with V-JEPA rather than increasing noise and token count.
    raw.streams.pop("hdgcn", None)
    raw.streams["vjepa"] = vjepa
    return raw


class VJEPAFusionTeacher(HierarchicalTeacher):
    STREAM_NAMES = (
        "vmae", "iv2", "depth", "thermal", "hand", "motionbert", "vjepa"
    )

    def __init__(
        self,
        statistics_dim: int,
        model_dim: int,
        layers: int,
        heads: int,
        dropout: float,
    ) -> None:
        dimensions = {name: 768 for name in self.STREAM_NAMES}
        dimensions["vjepa"] = 1024
        super().__init__(
            statistics_dim, model_dim, layers, heads, dropout,
            stream_dims=dimensions,
        )


def user_audit(
    labels: np.ndarray, base: np.ndarray, prediction: np.ndarray, users: np.ndarray
) -> dict[str, Any]:
    accuracy = {}
    nets = []
    for user in np.unique(users):
        selected = users == user
        accuracy[str(user)] = float(np.mean(prediction[selected] == labels[selected]))
        nets.append(
            int(np.sum(prediction[selected] == labels[selected]))
            - int(np.sum(base[selected] == labels[selected]))
        )
    return {
        "per_user_accuracy": accuracy,
        "worst_user_accuracy": float(min(accuracy.values())),
        "negative_users": int(np.sum(np.asarray(nets) < 0)),
        "worst_user_net": int(min(nets, default=0)),
    }


def select_blend(
    logits: np.ndarray,
    base: np.ndarray,
    labels: np.ndarray,
    users: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grid = []
    for weight in np.linspace(0.0, 1.0, 41):
        prediction = blend_prediction(logits, base, float(weight))
        grid.append(
            {
                "weight": float(weight),
                **audit(labels, base, prediction),
                **user_audit(labels, base, prediction, users),
            }
        )
    eligible = [
        row for row in grid if row["negative_users"] <= 1 and row["worst_user_net"] >= -1
    ]
    selected = max(
        eligible or grid,
        key=lambda row: (
            row["correct"], -row["harm"], -row["negative_users"],
            row["worst_user_net"], -row["weight"],
        ),
    )
    return selected, grid


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw = build_vjepa_data()
    h1 = raw.boundaries["H1_selection"]
    h2 = raw.boundaries["H2_confirmation"]
    embargo = raw.boundaries["E0_p87_sequence_source"]
    h3 = raw.boundaries["H3_independent_fold0"]
    inner_train = np.concatenate((h1, embargo))
    final_train = np.concatenate((h1, h2, embargo))
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    h2_champion, h3_champion = champion_predictions(raw, h2, h3)

    inner_pre = Preprocessor(args.statistics_dim, seeds[0]).fit(raw, inner_train)
    inner = inner_pre.transform(raw)
    inner_logits = []
    inner_reliability = []
    inner_audits = []
    epochs = []
    for seed in seeds:
        print(f"inner seed={seed}", flush=True)
        _, model_audit, logits, reliability = train_model(
            args, inner, inner_train, h2, seed, model_type=VJEPAFusionTeacher
        )
        if logits is None or reliability is None:
            raise RuntimeError("missing inner predictions")
        inner_logits.append(logits)
        inner_reliability.append(reliability)
        inner_audits.append(model_audit)
        epochs.append(int(model_audit["best_epoch"]))
    h2_logits = np.mean(inner_logits, axis=0)
    selected_blend, blend_grid = select_blend(
        h2_logits, h2_champion, inner.labels[h2], inner.users[h2]
    )
    fixed_epochs = max(1, int(round(float(np.median(epochs)))))
    print(
        f"selected epochs={epochs} fixed={fixed_epochs} "
        f"champion blend={selected_blend['weight']} correct={selected_blend['correct']}",
        flush=True,
    )

    final_pre = Preprocessor(args.statistics_dim, seeds[0] + 1000).fit(raw, final_train)
    final = final_pre.transform(raw)
    target_logits = []
    target_reliability = []
    final_audits = []
    for seed in seeds:
        print(f"final seed={seed} epochs={fixed_epochs}", flush=True)
        model, model_audit, _, _ = train_model(
            args, final, final_train, None, seed + 10000,
            fixed_epochs=fixed_epochs, model_type=VJEPAFusionTeacher,
        )
        device = torch.device(
            args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
        )
        logits, reliability = infer_full(model, final, h3, device, args.batch_size)
        target_logits.append(logits)
        target_reliability.append(reliability)
        final_audits.append(model_audit)
        torch.save(
            {"state_dict": model.state_dict(), "audit": model_audit},
            output / f"seed_{seed}_teacher.pt",
        )
    h3_logits = np.mean(target_logits, axis=0)
    direct = h3_logits.argmax(axis=1)
    blended = blend_prediction(
        h3_logits, h3_champion, float(selected_blend["weight"])
    )
    report = {
        "protocol": (
            "H1+embargo train/H2 epoch and champion-blend selection; "
            "H1+H2+embargo refit; one frozen H3 audit; V-JEPA tokens enter pre-classification."
        ),
        "streams": {name: list(values.shape[1:]) for name, values in raw.streams.items()},
        "inner": {
            "seed_audits": inner_audits,
            "direct_vs_p90": audit(
                inner.labels[h2], inner.teacher_prediction[h2], h2_logits.argmax(1)
            ),
            "direct_vs_champion": audit(
                inner.labels[h2], h2_champion, h2_logits.argmax(1)
            ),
            "selected_blend": selected_blend,
            "top_blends": sorted(blend_grid, key=lambda row: row["correct"], reverse=True)[:10],
        },
        "final": {
            "fixed_epochs": fixed_epochs,
            "seed_audits": final_audits,
            "direct_vs_p90": audit(
                final.labels[h3], final.teacher_prediction[h3], direct
            ),
            "direct_vs_champion": audit(final.labels[h3], h3_champion, direct),
            "source_selected_vs_champion": {
                **audit(final.labels[h3], h3_champion, blended),
                **user_audit(final.labels[h3], h3_champion, blended, final.users[h3]),
            },
        },
    }
    np.savez_compressed(
        output / "predictions.npz",
        sample_ids=final.sample_ids[h3], labels=final.labels[h3],
        champion_prediction=h3_champion, logits=h3_logits.astype(np.float32),
        direct_prediction=direct, blended_prediction=blended,
        reliability_logits=np.mean(target_reliability, axis=0).astype(np.float32),
        selected_blend_weight=np.asarray(selected_blend["weight"], dtype=np.float32),
    )
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    main()
