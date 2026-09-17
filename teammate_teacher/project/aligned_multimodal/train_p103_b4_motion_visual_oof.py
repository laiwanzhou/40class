"""Train P103-B4 staged local-visual/Skeleton/IMU candidate Teacher OOF."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from analyze_p102_hard_set import (
    CandidateRecipe,
    build_candidate_ids,
    confusion_neighbors,
    source_crossfit_session_probability,
)
from audit_p102_session_closure import classification_metrics, load_npz
from audit_p87_sequence_decoder import DecoderConfig, align_metadata
from p100a_global_teacher_data import FOLD_USERS, H3_USERS
from p101_finegrained_teacher_data import (
    BOOLEAN_MOTION_FIELDS,
    IMU_FIELDS,
    SKELETON_FIELDS,
    TEMPORAL_MOTION_FIELDS,
    P101Data,
    canonical_time,
    load_p101_data,
    within_subject_wrong_label_source,
)
from p103_local_feature_data import P103LocalData, load_p103_local_data
from p103_motion_candidate_model import (
    MotionCandidateConfig,
    P103MotionCandidateTeacher,
    trainable_parameter_audit,
)
from train_p102_b1_visual_listwise_session_oof import session_decode_candidate_probability
from train_p102_candidate_reranker_oof import evaluate_variant
from train_p103_b2_rich_visual_oof import (
    MAX_CANDIDATES,
    build_hard_population,
    candidate_context,
    candidate_list_loss,
    conditional_metrics,
    set_seed,
    shuffle_feature_source,
)


HERE = Path(__file__).resolve().parent
DEFAULT_SESSION = HERE / "runs/p102_session_closure_v1/session_oof_predictions.npz"
DEFAULT_SESSION_SUMMARY = HERE / "runs/p102_session_closure_v1/summary.json"
DEFAULT_HARD = HERE / "runs/p102_hard_set_v1/hard_set.npz"
DEFAULT_HARD_SUMMARY = HERE / "runs/p102_hard_set_v1/summary.json"
DEFAULT_NESTED = HERE / "runs/p101_f1_coarse_anchor_oof_v1/nested_coarse_vs"
DEFAULT_METADATA = HERE / "data/p85_recording_metadata/train_recording_metadata.csv"
DEFAULT_OUTPUT = HERE / "runs/p103_b4_motion_visual_oof_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-oof", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--session-summary", type=Path, default=DEFAULT_SESSION_SUMMARY)
    parser.add_argument("--hard-set", type=Path, default=DEFAULT_HARD)
    parser.add_argument("--hard-summary", type=Path, default=DEFAULT_HARD_SUMMARY)
    parser.add_argument("--nested-root", type=Path, default=DEFAULT_NESTED)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, nargs="+", default=(0, 1, 2, 3))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


class MotionVisualDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        local: P103LocalData,
        motion: P101Data,
        rows: np.ndarray,
        candidate_ids: np.ndarray,
        context: np.ndarray,
        labels: np.ndarray,
        sample_weights: np.ndarray | None = None,
        motion_source: np.ndarray | None = None,
        negative_motion_source: np.ndarray | None = None,
        reverse_motion: bool = False,
        skeleton_scale: float = 1.0,
        imu_scale: float = 1.0,
    ) -> None:
        self.local = local
        self.motion = motion
        self.rows = np.asarray(rows, dtype=np.int64)
        self.candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
        self.context = np.asarray(context, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.sample_weights = sample_weights
        identity = np.arange(len(labels), dtype=np.int64)
        self.motion_source = identity if motion_source is None else np.asarray(motion_source, dtype=np.int64)
        self.negative_motion_source = negative_motion_source
        self.reverse_motion = bool(reverse_motion)
        self.skeleton_scale = float(skeleton_scale)
        self.imu_scale = float(imu_scale)

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def tensor(values: np.ndarray, boolean: bool = False) -> torch.Tensor:
        copied = np.asarray(values).copy()
        if boolean:
            return torch.from_numpy(copied.astype(bool, copy=False))
        return torch.from_numpy(copied.astype(np.float32, copy=False))

    def add_motion(
        self,
        item: dict[str, torch.Tensor],
        source: int,
        prefix: str = "",
        reverse: bool = False,
    ) -> None:
        cache_row = int(self.motion.motion_rows[source])
        for field in (*SKELETON_FIELDS, *IMU_FIELDS):
            values = self.motion.motion[field][cache_row]
            if reverse and field in TEMPORAL_MOTION_FIELDS:
                values = np.asarray(values)[:, ::-1]
            item[prefix + field] = self.tensor(values, field in BOOLEAN_MOTION_FIELDS)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        row = int(self.rows[item])
        candidates = self.candidate_ids[row]
        hits = np.flatnonzero(candidates == self.labels[row])
        target = int(hits[0]) if len(hits) == 1 else -1
        weight = 1.0 if self.sample_weights is None else float(self.sample_weights[row])
        output = {
            "row": torch.tensor(row, dtype=torch.long),
            "vmae_features": self.tensor(self.local.vmae_features[row]),
            "vmae_actions": self.tensor(self.local.vmae_actions[row]),
            "vjepa_features": self.tensor(self.local.vjepa_features[row]),
            "vjepa_actions": self.tensor(self.local.vjepa_actions[row]),
            "candidate_ids": torch.from_numpy(candidates.copy()).long(),
            "a_context": torch.from_numpy(self.context[row].copy()),
            "target_position": torch.tensor(target, dtype=torch.long),
            "sample_weight": torch.tensor(weight, dtype=torch.float32),
            "motion_time": self.tensor(canonical_time(2, 16)),
            "skeleton_scale": torch.tensor(self.skeleton_scale, dtype=torch.float32),
            "imu_scale": torch.tensor(self.imu_scale, dtype=torch.float32),
        }
        self.add_motion(
            output, int(self.motion_source[row]), reverse=self.reverse_motion
        )
        if self.negative_motion_source is not None:
            self.add_motion(
                output, int(self.negative_motion_source[row]), prefix="negative_"
            )
        return output


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if key != "row"
    }


def train_model(
    local: P103LocalData,
    motion: P101Data,
    rows: np.ndarray,
    candidate_ids: np.ndarray,
    context: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    device: torch.device,
    seed: int,
) -> tuple[P103MotionCandidateTeacher, dict[str, Any]]:
    set_seed(seed)
    negative_source = within_subject_wrong_label_source(motion, rows, seed + 500)
    negative_self = int(np.sum(negative_source[rows] == rows))
    if np.any(
        (negative_source[rows] != rows)
        & (motion.users[negative_source[rows]] != motion.users[rows])
    ):
        raise RuntimeError("B4 negative motion crossed subjects")
    if np.any(
        (negative_source[rows] != rows)
        & (motion.labels[negative_source[rows]] == motion.labels[rows])
    ):
        raise RuntimeError("B4 negative motion kept the same label")
    config = MotionCandidateConfig()
    model = P103MotionCandidateTeacher(config).to(device)
    dataset = MotionVisualDataset(
        local,
        motion,
        rows,
        candidate_ids,
        context,
        labels,
        sample_weights=weights,
        negative_motion_source=negative_source,
    )
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    epochs = 60
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history: list[dict[str, float | int]] = []
    for epoch in range(epochs):
        model.train()
        sums = {"loss": 0.0, "ce": 0.0, "pair": 0.0, "correspondence": 0.0}
        correct = 0
        count = 0
        for batch in loader:
            moved = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            aligned = model(moved)
            negative = model(moved, motion_prefix="negative_")
            list_loss, components = candidate_list_loss(
                aligned["candidate_scores"],
                moved["candidate_ids"],
                moved["target_position"],
                moved["sample_weight"],
            )
            aligned_true = aligned["candidate_scores"].gather(
                1, moved["target_position"][:, None]
            ).squeeze(1)
            negative_true = negative["candidate_scores"].gather(
                1, moved["target_position"][:, None]
            ).squeeze(1)
            row_correspondence = F.relu(0.5 - aligned_true + negative_true)
            correspondence = torch.sum(
                moved["sample_weight"] * row_correspondence
            ) / moved["sample_weight"].sum().clamp_min(1e-6)
            loss = list_loss + 0.25 * correspondence
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            size = len(moved["target_position"])
            sums["loss"] += float(loss.detach()) * size
            sums["ce"] += components["ce"] * size
            sums["pair"] += components["pair"] * size
            sums["correspondence"] += float(correspondence.detach()) * size
            correct += int(
                (aligned["candidate_scores"].argmax(dim=1) == moved["target_position"])
                .sum()
                .item()
            )
            count += size
        scheduler.step()
        record = {
            "epoch": epoch + 1,
            **{name: value / count for name, value in sums.items()},
            "training_accuracy": correct / count,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        if epoch in {0, 9, 19, 29, 39, 49, 59}:
            print(json.dumps({"training": record}), flush=True)
    return model, {
        "config": asdict(config),
        **trainable_parameter_audit(model),
        "optimizer": "AdamW",
        "learning_rate": 3e-4,
        "weight_decay": 1e-3,
        "batch_size": 64,
        "epochs": epochs,
        "schedule": "cosine",
        "gradient_clip": 1.0,
        "candidate_pair_margin_weight": 0.25,
        "candidate_pair_margin": 0.5,
        "wrong_label_motion_correspondence_weight": 0.25,
        "wrong_label_motion_correspondence_margin": 0.5,
        "negative_motion_self_rows": negative_self,
        "history": history,
    }


@torch.no_grad()
def predict_model(
    model: P103MotionCandidateTeacher,
    local: P103LocalData,
    motion: P101Data,
    rows: np.ndarray,
    candidate_ids: np.ndarray,
    context: np.ndarray,
    labels: np.ndarray,
    a_probability: np.ndarray,
    device: torch.device,
    motion_source: np.ndarray | None = None,
    reverse_motion: bool = False,
    skeleton_scale: float = 1.0,
    imu_scale: float = 1.0,
    save_attention: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    dataset = MotionVisualDataset(
        local,
        motion,
        rows,
        candidate_ids,
        context,
        labels,
        motion_source=motion_source,
        reverse_motion=reverse_motion,
        skeleton_scale=skeleton_scale,
        imu_scale=imu_scale,
    )
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    probability = np.asarray(a_probability, dtype=np.float64).copy()
    attention = (
        np.full(
            (len(labels), MAX_CANDIDATES, len(model.ATTENTION_GROUP_NAMES)),
            np.nan,
            dtype=np.float32,
        )
        if save_attention
        else None
    )
    model.eval()
    for batch in loader:
        batch_rows = batch["row"].numpy()
        moved = move_batch(batch, device)
        output = model(moved, return_attention=save_attention)
        value = torch.softmax(output["candidate_scores"], dim=1).cpu().numpy()
        candidates = batch["candidate_ids"].numpy()
        for offset, row in enumerate(batch_rows):
            valid = candidates[offset] >= 0
            probability[row] = 0.0
            probability[row, candidates[offset, valid]] = value[offset, valid]
        if attention is not None:
            attention[batch_rows] = output["attention_groups"].cpu().numpy()
    return probability, attention


def attention_audit(
    attention: np.ndarray,
    candidate_ids: np.ndarray,
    labels: np.ndarray,
    b_probability: np.ndarray,
    selected: np.ndarray,
) -> dict[str, Any]:
    prediction = b_probability.argmax(axis=1)
    groups: dict[str, list[np.ndarray]] = {
        "rescue_true_query": [],
        "miss_true_query": [],
        "predicted_query": [],
    }
    for row in np.flatnonzero(selected):
        valid = candidate_ids[row][candidate_ids[row] >= 0]
        true_position = int(np.flatnonzero(valid == labels[row])[0])
        predicted_position = int(np.flatnonzero(valid == prediction[row])[0])
        key = "rescue_true_query" if prediction[row] == labels[row] else "miss_true_query"
        groups[key].append(attention[row, true_position])
        groups["predicted_query"].append(attention[row, predicted_position])
    return {
        key: {
            "rows": len(values),
            "mean": {
                name: float(np.mean(np.stack(values), axis=0)[index])
                for index, name in enumerate(P103MotionCandidateTeacher.ATTENTION_GROUP_NAMES)
            }
            if values
            else {},
        }
        for key, values in groups.items()
    } | {"warning": "attention is descriptive; motion counterfactuals are causal"}


def main() -> None:
    args = parse_args()
    folds_requested = tuple(sorted(set(map(int, args.folds))))
    if not set(folds_requested) <= {0, 1, 2, 3}:
        raise ValueError("folds must be a subset of 0..3")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.set_float32_matmul_precision("high")

    session = load_npz(args.session_oof.resolve())
    hard = load_npz(args.hard_set.resolve())
    session_summary = json.loads(args.session_summary.resolve().read_text(encoding="utf-8"))
    hard_summary = json.loads(args.hard_summary.resolve().read_text(encoding="utf-8"))
    sample_ids = session["sample_ids"].astype(str)
    users = session["users"].astype(str)
    labels = np.asarray(session["labels"], dtype=np.int64)
    fold_ids = np.asarray(session["fold_ids"], dtype=np.int64)
    a_probability = np.asarray(session["selected_probability"], dtype=np.float64)
    vs_raw_probability = np.asarray(session["vs_raw_probability"], dtype=np.float64)
    target_candidates = np.asarray(hard["candidate_ids"], dtype=np.int64)
    if set(users.tolist()) & set(H3_USERS):
        raise RuntimeError("H3 reached P103-B4")
    motion = load_p101_data()
    local = load_p103_local_data(sample_ids, users)
    if not np.array_equal(motion.sample_ids.astype(str), sample_ids):
        raise RuntimeError("P101 motion/P103-B4 order differs")
    if not np.array_equal(hard["sample_ids"].astype(str), sample_ids):
        raise RuntimeError("hard/P103-B4 order differs")
    metadata = align_metadata(args.metadata.resolve(), sample_ids)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    bases = ("full", "shuffle_motion", "zero_motion", "reverse_motion", "zero_skeleton", "zero_imu")
    variant_names = tuple(f"{base}_{stage}" for base in bases for stage in ("local", "session"))
    probabilities = {name: a_probability.copy() for name in variant_names}
    attention_groups = np.full(
        (len(labels), MAX_CANDIDATES, len(P103MotionCandidateTeacher.ATTENTION_GROUP_NAMES)),
        np.nan,
        dtype=np.float32,
    )
    fold_reports: list[dict[str, Any]] = []
    for outer_fold in folds_requested:
        source = fold_ids != outer_fold
        held = fold_ids == outer_fold
        selected_session = session_summary["folds"][outer_fold]["selected"]
        decoder = DecoderConfig(
            gap_seconds=float(selected_session["gap_seconds"]),
            transition_weight=float(selected_session["transition_weight"]),
            trigram_backoff=float(selected_session["trigram_backoff"]),
            beam_width=int(selected_session["beam_width"]),
        )
        source_probability, source_session_audit = source_crossfit_session_probability(
            outer_fold,
            args.nested_root.resolve(),
            sample_ids,
            users,
            fold_ids,
            labels,
            metadata,
            decoder,
        )
        recipe_values = hard_summary["folds"][outer_fold]["selected_recipe"]
        recipe = CandidateRecipe(**{key: int(value) for key, value in recipe_values.items()})
        source_prediction = source_probability.argmax(axis=1)
        source_candidates = np.full((len(labels), MAX_CANDIDATES), -1, dtype=np.int64)
        for user in sorted(set(users[source].tolist())):
            validation = source & (users == user)
            graph_fit = np.flatnonzero(source & (users != user)).astype(np.int64)
            neighbors, _ = confusion_neighbors(
                labels, source_prediction, users, graph_fit, recipe
            )
            source_candidates[validation, : recipe.max_size] = build_candidate_ids(
                source_probability[validation], neighbors, recipe
            )
        train_rows, train_weights, population_audit = build_hard_population(
            sample_ids, labels, source, source_probability, source_candidates
        )
        source_complete = a_probability.copy()
        source_complete[source] = source_probability[source]
        source_context = candidate_context(source_complete, source_candidates)
        model, training_audit = train_model(
            local,
            motion,
            train_rows,
            source_candidates,
            source_context,
            labels,
            train_weights,
            device,
            seed=20260823 + outer_fold,
        )
        torch.save(
            {
                "protocol": "P103-B4",
                "outer_fold": outer_fold,
                "config": training_audit["config"],
                "state_dict": model.state_dict(),
                "source_users": sorted(set(users[source].tolist())),
                "h3_users_loaded": [],
            },
            output / f"fold{outer_fold}_motion_candidate_teacher.pt",
        )
        held_rows = np.flatnonzero(held).astype(np.int64)
        held_context = candidate_context(a_probability, target_candidates)
        counterfactuals = {
            "full": (None, False, 1.0, 1.0),
            "shuffle_motion": (shuffle_feature_source(users, held), False, 1.0, 1.0),
            "zero_motion": (None, False, 0.0, 0.0),
            "reverse_motion": (None, True, 1.0, 1.0),
            "zero_skeleton": (None, False, 0.0, 1.0),
            "zero_imu": (None, False, 1.0, 0.0),
        }
        fold_variants: dict[str, Any] = {}
        session_audits: dict[str, Any] = {}
        for name, (motion_source, reverse, skeleton_scale, imu_scale) in counterfactuals.items():
            local_probability, fold_attention = predict_model(
                model,
                local,
                motion,
                held_rows,
                target_candidates,
                held_context,
                labels,
                a_probability,
                device,
                motion_source=motion_source,
                reverse_motion=reverse,
                skeleton_scale=skeleton_scale,
                imu_scale=imu_scale,
                save_attention=name == "full",
            )
            decoded, decode_audit = session_decode_candidate_probability(
                local_probability,
                target_candidates,
                labels,
                source,
                held,
                metadata,
                decoder,
            )
            probabilities[f"{name}_local"][held] = local_probability[held]
            probabilities[f"{name}_session"][held] = decoded[held]
            if fold_attention is not None:
                attention_groups[held] = fold_attention[held]
            fold_variants[f"{name}_local"] = evaluate_variant(
                labels, users, held, a_probability, local_probability
            )
            fold_variants[f"{name}_session"] = evaluate_variant(
                labels, users, held, a_probability, decoded
            )
            session_audits[name] = decode_audit
        fold_reports.append(
            {
                "fold": outer_fold,
                "held_users": list(FOLD_USERS[outer_fold]),
                "source_users": sorted(set(users[source].tolist())),
                "source_held_overlap": sorted(
                    set(users[source].tolist()) & set(FOLD_USERS[outer_fold])
                ),
                "candidate_recipe": asdict(recipe),
                "source_session": source_session_audit,
                "population": population_audit,
                "training": training_audit,
                "session_decode": session_audits,
                "variants": fold_variants,
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    evaluated = np.isin(fold_ids, np.asarray(folds_requested))
    variants = {
        name: evaluate_variant(labels, users, evaluated, a_probability, probability)
        for name, probability in probabilities.items()
    }
    a_prediction = a_probability.argmax(axis=1)
    attackable = evaluated & (a_prediction != labels) & np.asarray(hard["candidate_hit"], dtype=bool)
    conditional = {
        name: conditional_metrics(
            probability, target_candidates, labels, users, a_prediction, attackable
        )
        for name, probability in probabilities.items()
    }
    aligned_correct = variants["full_session"]["metrics"]["top1_correct"]
    summary = {
        "status": "complete" if folds_requested == (0, 1, 2, 3) else "smoke_complete",
        "protocol": (
            "P103-B4 staged candidate local-visual -> raw Skeleton/IMU interaction with "
            "same-subject wrong-label motion correspondence supervision; no PCA, prototype, "
            "40-class head, held-label tuning, routing scan or H3."
        ),
        "folds_requested": list(folds_requested),
        "data": {
            "rows": int(evaluated.sum()),
            "users": sorted(set(users[evaluated].tolist())),
            "h3_rows_selected": 0,
            "h3_users_loaded": [],
            "local_cache_contract": local.summary(),
            "motion_cache_contract": motion.summary(),
        },
        "a_baseline": classification_metrics(
            a_probability[evaluated], labels[evaluated], users[evaluated]
        ),
        "level1_conditional": conditional,
        "level2_full_system": variants,
        "counterfactual_gate": {
            name: int(aligned_correct - variants[f"{name}_session"]["metrics"]["top1_correct"])
            for name in ("shuffle_motion", "zero_motion", "reverse_motion", "zero_skeleton", "zero_imu")
        },
        "attention_audit": attention_audit(
            attention_groups,
            target_candidates,
            labels,
            probabilities["full_local"],
            attackable,
        ),
        "folds": fold_reports,
        "student_started": False,
    }
    np.savez_compressed(
        output / "b4_oof_predictions.npz",
        sample_ids=sample_ids,
        users=users,
        labels=labels,
        fold_ids=fold_ids,
        evaluated=evaluated,
        candidate_ids=target_candidates,
        vs_raw_probability=vs_raw_probability.astype(np.float32),
        a_session_probability=a_probability.astype(np.float32),
        attention_group_names=np.asarray(P103MotionCandidateTeacher.ATTENTION_GROUP_NAMES),
        aligned_attention_groups=attention_groups.astype(np.float16),
        **{
            f"{name}_probability": probability.astype(np.float32)
            for name, probability in probabilities.items()
        },
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
