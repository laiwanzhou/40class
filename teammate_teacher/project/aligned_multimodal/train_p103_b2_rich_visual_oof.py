"""Train P103-B2 rich visual candidate-query Teacher with subject-disjoint OOF."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
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
from p101_finegrained_teacher_data import P101Data, load_p101_data
from p103_rich_candidate_model import (
    P103RichCandidateTeacher,
    RichCandidateConfig,
    trainable_parameter_audit,
)
from train_p102_b1_visual_listwise_session_oof import session_decode_candidate_probability
from train_p102_candidate_reranker_oof import evaluate_variant


HERE = Path(__file__).resolve().parent
DEFAULT_SESSION = HERE / "runs/p102_session_closure_v1/session_oof_predictions.npz"
DEFAULT_SESSION_SUMMARY = HERE / "runs/p102_session_closure_v1/summary.json"
DEFAULT_HARD = HERE / "runs/p102_hard_set_v1/hard_set.npz"
DEFAULT_HARD_SUMMARY = HERE / "runs/p102_hard_set_v1/summary.json"
DEFAULT_NESTED = HERE / "runs/p101_f1_coarse_anchor_oof_v1/nested_coarse_vs"
DEFAULT_METADATA = HERE / "data/p85_recording_metadata/train_recording_metadata.csv"
DEFAULT_OUTPUT = HERE / "runs/p103_b2_rich_visual_oof_v1"
MAX_CANDIDATES = 8


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def entropy(probability: np.ndarray) -> np.ndarray:
    values = np.asarray(probability, dtype=np.float64)
    return -np.sum(values * np.log(np.maximum(values, 1e-12)), axis=1)


def margin(probability: np.ndarray) -> np.ndarray:
    values = np.sort(np.asarray(probability, dtype=np.float64), axis=1)
    return values[:, -1] - values[:, -2]


def candidate_context(probability: np.ndarray, candidate_ids: np.ndarray) -> np.ndarray:
    values = np.asarray(probability, dtype=np.float64)
    candidates = np.asarray(candidate_ids, dtype=np.int64)
    order = np.argsort(values, axis=1)[:, ::-1]
    ranks = np.empty_like(order)
    ranks[np.arange(len(order))[:, None], order] = np.arange(1, values.shape[1] + 1)
    prediction = values.argmax(axis=1)
    row_margin = margin(values)
    row_entropy = entropy(values) / math.log(values.shape[1])
    output = np.zeros((*candidates.shape, 7), dtype=np.float32)
    for row in range(len(values)):
        valid = candidates[row] >= 0
        ids = candidates[row, valid]
        count = int(valid.sum())
        output[row, valid, 0] = np.log(np.maximum(values[row, ids], 1e-12))
        output[row, valid, 1] = values[row, ids]
        output[row, valid, 2] = ranks[row, ids] / float(values.shape[1])
        output[row, valid, 3] = np.arange(count) / max(count - 1, 1)
        output[row, valid, 4] = ids == prediction[row]
        output[row, valid, 5] = row_margin[row]
        output[row, valid, 6] = row_entropy[row]
    return output


def stable_quarter(sample_id: str) -> bool:
    digest = hashlib.sha256(str(sample_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % 4 == 0


def build_hard_population(
    sample_ids: np.ndarray,
    labels: np.ndarray,
    source: np.ndarray,
    probability: np.ndarray,
    candidate_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    hit = np.any(candidate_ids == labels[:, None], axis=1)
    prediction = probability.argmax(axis=1)
    correct = prediction == labels
    source_correct = source & hit & correct
    source_margin = margin(probability)
    source_entropy = entropy(probability)
    margin_median = float(np.median(source_margin[source_correct]))
    entropy_median = float(np.median(source_entropy[source_correct]))
    core = source & hit & (~correct)
    protection = source_correct & (
        (source_margin <= margin_median) | (source_entropy >= entropy_median)
    )
    easy_pool = source_correct & (~protection)
    easy = easy_pool & np.asarray([stable_quarter(value) for value in sample_ids])
    selected = core | protection | easy
    weights = np.zeros(len(labels), dtype=np.float32)
    weights[core] = 4.0
    weights[protection] = 2.0
    weights[easy] = 0.5
    if not np.all(np.sum(candidate_ids[selected] == labels[selected, None], axis=1) == 1):
        raise RuntimeError("P103 training population has an invalid candidate target")
    rows = np.flatnonzero(selected).astype(np.int64)
    return rows, weights, {
        "source_rows": int(source.sum()),
        "source_candidate_hit_rows": int(np.sum(source & hit)),
        "core_hard_rows": int(core.sum()),
        "protection_rows": int(protection.sum()),
        "easy_pool_rows": int(easy_pool.sum()),
        "easy_replay_rows": int(easy.sum()),
        "selected_rows": int(selected.sum()),
        "source_correct_margin_median": margin_median,
        "source_correct_entropy_median": entropy_median,
        "row_weights": {"core_hard": 4.0, "protection": 2.0, "easy_replay": 0.5},
        "easy_replay_rule": "sha256(sample_id) first uint32 modulo 4 equals zero",
    }


class RichVisualDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        data: P101Data,
        rows: np.ndarray,
        candidate_ids: np.ndarray,
        context: np.ndarray,
        labels: np.ndarray,
        sample_weights: np.ndarray | None = None,
        feature_source: np.ndarray | None = None,
        visual_scale: float = 1.0,
    ) -> None:
        self.data = data
        self.rows = np.asarray(rows, dtype=np.int64)
        self.candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
        self.context = np.asarray(context, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.sample_weights = sample_weights
        self.feature_source = (
            np.arange(len(labels), dtype=np.int64)
            if feature_source is None
            else np.asarray(feature_source, dtype=np.int64)
        )
        self.visual_scale = float(visual_scale)

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def tensor(values: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.asarray(values, dtype=np.float32).copy())

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        row = int(self.rows[item])
        feature = int(self.feature_source[row])
        candidates = self.candidate_ids[row]
        hits = np.flatnonzero(candidates == self.labels[row])
        target = int(hits[0]) if len(hits) == 1 else -1
        weight = 1.0 if self.sample_weights is None else float(self.sample_weights[row])
        return {
            "row": torch.tensor(row, dtype=torch.long),
            "vmae_temporal": self.tensor(self.data.visual_vmae_temporal[feature]),
            "iv2_temporal": self.tensor(self.data.visual_iv2_temporal[feature]),
            "vmae_pooled": self.tensor(self.data.visual_vmae_pooled[feature]),
            "iv2_pooled": self.tensor(self.data.visual_iv2_pooled[feature]),
            "vmae_action": self.tensor(self.data.visual_vmae_action[feature]),
            "iv2_action": self.tensor(self.data.visual_iv2_action[feature]),
            "candidate_ids": torch.from_numpy(candidates.copy()).long(),
            "a_context": torch.from_numpy(self.context[row].copy()),
            "target_position": torch.tensor(target, dtype=torch.long),
            "sample_weight": torch.tensor(weight, dtype=torch.float32),
            "visual_scale": torch.tensor(self.visual_scale, dtype=torch.float32),
        }


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if key != "row"
    }


def candidate_list_loss(
    scores: torch.Tensor,
    candidate_ids: torch.Tensor,
    target_position: torch.Tensor,
    sample_weight: torch.Tensor,
    pair_weight: float = 0.25,
    pair_margin: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    if torch.any(target_position < 0):
        raise RuntimeError("training batch lacks a candidate target")
    row_ce = F.cross_entropy(scores, target_position, reduction="none")
    true_score = scores.gather(1, target_position[:, None])
    valid = candidate_ids >= 0
    negative = valid.clone()
    negative.scatter_(1, target_position[:, None], False)
    hinge = F.relu(pair_margin - true_score + scores) * negative.to(scores.dtype)
    row_pair = hinge.sum(dim=1) / negative.sum(dim=1).clamp_min(1)
    row_loss = row_ce + pair_weight * row_pair
    loss = torch.sum(sample_weight * row_loss) / sample_weight.sum().clamp_min(1e-6)
    return loss, {
        "ce": float(row_ce.detach().mean().item()),
        "pair": float(row_pair.detach().mean().item()),
    }


def train_model(
    data: P101Data,
    rows: np.ndarray,
    candidate_ids: np.ndarray,
    context: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    device: torch.device,
    seed: int,
) -> tuple[P103RichCandidateTeacher, dict[str, Any]]:
    set_seed(seed)
    config = RichCandidateConfig()
    model = P103RichCandidateTeacher(config).to(device)
    dataset = RichVisualDataset(
        data, rows, candidate_ids, context, labels, sample_weights=weights
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-3
    )
    epochs = 60
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history: list[dict[str, float | int]] = []
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        ce_sum = 0.0
        pair_sum = 0.0
        correct = 0
        count = 0
        for batch in loader:
            moved = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(moved)
            loss, components = candidate_list_loss(
                output["candidate_scores"],
                moved["candidate_ids"],
                moved["target_position"],
                moved["sample_weight"],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_rows = len(moved["target_position"])
            loss_sum += float(loss.detach().item()) * batch_rows
            ce_sum += components["ce"] * batch_rows
            pair_sum += components["pair"] * batch_rows
            correct += int(
                (output["candidate_scores"].argmax(dim=1) == moved["target_position"])
                .sum()
                .item()
            )
            count += batch_rows
        scheduler.step()
        record = {
            "epoch": epoch + 1,
            "loss": loss_sum / count,
            "ce": ce_sum / count,
            "pair": pair_sum / count,
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
        "list_ce_weight": 1.0,
        "pair_margin_weight": 0.25,
        "pair_margin": 0.5,
        "history": history,
    }


@torch.no_grad()
def predict_model(
    model: P103RichCandidateTeacher,
    data: P101Data,
    rows: np.ndarray,
    candidate_ids: np.ndarray,
    context: np.ndarray,
    labels: np.ndarray,
    a_probability: np.ndarray,
    device: torch.device,
    feature_source: np.ndarray | None = None,
    visual_scale: float = 1.0,
    save_attention: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    dataset = RichVisualDataset(
        data,
        rows,
        candidate_ids,
        context,
        labels,
        feature_source=feature_source,
        visual_scale=visual_scale,
    )
    loader = DataLoader(
        dataset,
        batch_size=96,
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
        score = output["candidate_scores"]
        value = torch.softmax(score, dim=1).cpu().numpy()
        candidates = batch["candidate_ids"].numpy()
        for offset, row in enumerate(batch_rows):
            valid = candidates[offset] >= 0
            probability[row] = 0.0
            probability[row, candidates[offset, valid]] = value[offset, valid]
        if attention is not None:
            attention[batch_rows] = output["attention_groups"].cpu().numpy()
    return probability, attention


def shuffle_feature_source(users: np.ndarray, selected: np.ndarray) -> np.ndarray:
    source = np.arange(len(users), dtype=np.int64)
    for user in sorted(set(users[selected].tolist())):
        rows = np.flatnonzero(selected & (users == user))
        if len(rows) > 1:
            source[rows] = np.roll(rows, 1)
    return source


def conditional_metrics(
    probability: np.ndarray,
    candidate_ids: np.ndarray,
    labels: np.ndarray,
    users: np.ndarray,
    a_prediction: np.ndarray,
    selected: np.ndarray,
) -> dict[str, Any]:
    rows = np.flatnonzero(selected).astype(np.int64)
    ranks: list[int] = []
    correct: list[bool] = []
    top2: list[bool] = []
    for row in rows:
        valid = candidate_ids[row][candidate_ids[row] >= 0]
        order = valid[np.argsort(probability[row, valid])[::-1]]
        position = int(np.flatnonzero(order == labels[row])[0]) + 1
        ranks.append(position)
        correct.append(position == 1)
        top2.append(position <= 2)
    ranks_array = np.asarray(ranks, dtype=np.int64)
    correct_array = np.asarray(correct, dtype=bool)
    top2_array = np.asarray(top2, dtype=bool)
    by_subject = {
        user: {
            "rows": int(np.sum(users[rows] == user)),
            "top1_correct": int(np.sum(correct_array[users[rows] == user])),
            "top1": float(np.mean(correct_array[users[rows] == user])),
            "mean_true_rank": float(np.mean(ranks_array[users[rows] == user])),
        }
        for user in sorted(set(users[rows].tolist()))
    }
    by_size = {
        str(size): {
            "rows": int(np.sum(np.sum(candidate_ids[rows] >= 0, axis=1) == size)),
            "top1": float(np.mean(correct_array[np.sum(candidate_ids[rows] >= 0, axis=1) == size])),
            "top2": float(np.mean(top2_array[np.sum(candidate_ids[rows] >= 0, axis=1) == size])),
            "mean_true_rank": float(np.mean(ranks_array[np.sum(candidate_ids[rows] >= 0, axis=1) == size])),
        }
        for size in sorted(set(np.sum(candidate_ids[rows] >= 0, axis=1).tolist()))
    }
    confusion_counts = Counter(
        (int(labels[row]), int(a_prediction[row])) for row in rows
    )
    by_confusion: list[dict[str, Any]] = []
    for (true, predicted), count in confusion_counts.most_common():
        mask = (labels[rows] == true) & (a_prediction[rows] == predicted)
        by_confusion.append(
            {
                "true": true,
                "a_prediction": predicted,
                "rows": count,
                "top1_correct": int(np.sum(correct_array[mask])),
                "top1": float(np.mean(correct_array[mask])),
                "mean_true_rank": float(np.mean(ranks_array[mask])),
            }
        )
    return {
        "rows": int(len(rows)),
        "top1_correct": int(correct_array.sum()),
        "top1": float(correct_array.mean()),
        "top2_correct": int(top2_array.sum()),
        "top2": float(top2_array.mean()),
        "mean_true_rank": float(ranks_array.mean()),
        "per_subject": by_subject,
        "candidate_size": by_size,
        "per_confusion": by_confusion,
    }


def attention_audit(
    attention: np.ndarray,
    candidate_ids: np.ndarray,
    labels: np.ndarray,
    a_probability: np.ndarray,
    b_probability: np.ndarray,
    selected: np.ndarray,
) -> dict[str, Any]:
    a_prediction = a_probability.argmax(axis=1)
    b_prediction = b_probability.argmax(axis=1)
    names = P103RichCandidateTeacher.ATTENTION_GROUP_NAMES
    groups: dict[str, list[np.ndarray]] = {"rescue_true_query": [], "miss_true_query": [], "predicted_query": []}
    for row in np.flatnonzero(selected):
        valid = candidate_ids[row][candidate_ids[row] >= 0]
        true_position = int(np.flatnonzero(valid == labels[row])[0])
        predicted_position = int(np.flatnonzero(valid == b_prediction[row])[0])
        key = "rescue_true_query" if b_prediction[row] == labels[row] else "miss_true_query"
        groups[key].append(attention[row, true_position])
        groups["predicted_query"].append(attention[row, predicted_position])
    return {
        key: {
            "rows": len(values),
            "mean": {
                name: float(np.mean(np.stack(values), axis=0)[index])
                for index, name in enumerate(names)
            } if values else {},
        }
        for key, values in groups.items()
    } | {
        "warning": "attention is descriptive only; aligned/shuffle/zero is the causal audit",
        "a_error_rows": int(np.sum(selected & (a_prediction != labels))),
    }


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
    data = load_p101_data()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    sample_ids = session["sample_ids"].astype(str)
    users = session["users"].astype(str)
    labels = np.asarray(session["labels"], dtype=np.int64)
    fold_ids = np.asarray(session["fold_ids"], dtype=np.int64)
    a_probability = np.asarray(session["selected_probability"], dtype=np.float64)
    vs_raw_probability = np.asarray(session["vs_raw_probability"], dtype=np.float64)
    target_candidates = np.asarray(hard["candidate_ids"], dtype=np.int64)
    if not np.array_equal(data.sample_ids.astype(str), sample_ids):
        raise RuntimeError("P101/P103 sample order differs")
    if not np.array_equal(hard["sample_ids"].astype(str), sample_ids):
        raise RuntimeError("hard/P103 sample order differs")
    if set(users.tolist()) & set(H3_USERS) or int(data.summary()["h3_rows"]) != 0:
        raise RuntimeError("H3 reached P103-B2")
    metadata = align_metadata(args.metadata.resolve(), sample_ids)

    variant_names = (
        "full_local", "full_session", "shuffle_local", "shuffle_session",
        "zero_local", "zero_session",
    )
    probabilities = {name: a_probability.copy() for name in variant_names}
    attention_groups = np.full(
        (len(labels), MAX_CANDIDATES, len(P103RichCandidateTeacher.ATTENTION_GROUP_NAMES)),
        np.nan,
        dtype=np.float32,
    )
    fold_reports: list[dict[str, Any]] = []
    for outer_fold in folds_requested:
        source = fold_ids != outer_fold
        held = fold_ids == outer_fold
        selected_session = session_summary["folds"][outer_fold]["selected"]
        config = DecoderConfig(
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
            config,
        )
        recipe_values = hard_summary["folds"][outer_fold]["selected_recipe"]
        recipe = CandidateRecipe(**{key: int(value) for key, value in recipe_values.items()})
        source_prediction = source_probability.argmax(axis=1)
        source_candidates = np.full((len(labels), MAX_CANDIDATES), -1, dtype=np.int64)
        for user in sorted(set(users[source].tolist())):
            validation = source & (users == user)
            graph_fit = np.flatnonzero(source & (users != user)).astype(np.int64)
            neighbors, _ = confusion_neighbors(labels, source_prediction, users, graph_fit, recipe)
            source_candidates[validation, : recipe.max_size] = build_candidate_ids(
                source_probability[validation], neighbors, recipe
            )
        train_rows, train_weights, population_audit = build_hard_population(
            sample_ids, labels, source, source_probability, source_candidates
        )
        source_probability_complete = a_probability.copy()
        source_probability_complete[source] = source_probability[source]
        source_context = candidate_context(source_probability_complete, source_candidates)
        model, training_audit = train_model(
            data,
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
                "protocol": "P103-B2",
                "outer_fold": outer_fold,
                "config": training_audit["config"],
                "state_dict": model.state_dict(),
                "source_users": sorted(set(users[source].tolist())),
                "h3_users_loaded": [],
            },
            output / f"fold{outer_fold}_rich_candidate_teacher.pt",
        )
        held_rows = np.flatnonzero(held).astype(np.int64)
        held_context = candidate_context(a_probability, target_candidates)
        feature_variants = {
            "full": (None, 1.0),
            "shuffle": (shuffle_feature_source(users, held), 1.0),
            "zero": (None, 0.0),
        }
        fold_variants: dict[str, Any] = {}
        session_audits: dict[str, Any] = {}
        for name, (feature_source, visual_scale) in feature_variants.items():
            local, fold_attention = predict_model(
                model,
                data,
                held_rows,
                target_candidates,
                held_context,
                labels,
                a_probability,
                device,
                feature_source=feature_source,
                visual_scale=visual_scale,
                save_attention=name == "full",
            )
            decoded, decode_audit = session_decode_candidate_probability(
                local, target_candidates, labels, source, held, metadata, config
            )
            probabilities[f"{name}_local"][held] = local[held]
            probabilities[f"{name}_session"][held] = decoded[held]
            if fold_attention is not None:
                attention_groups[held] = fold_attention[held]
            fold_variants[f"{name}_local"] = evaluate_variant(
                labels, users, held, a_probability, local
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
                "source_held_overlap": sorted(set(users[source].tolist()) & set(FOLD_USERS[outer_fold])),
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
    candidate_hit = np.asarray(hard["candidate_hit"], dtype=bool)
    attackable = evaluated & (a_prediction != labels) & candidate_hit
    conditional = {
        name: conditional_metrics(
            probability,
            target_candidates,
            labels,
            users,
            a_prediction,
            attackable,
        )
        for name, probability in probabilities.items()
    }
    aligned_correct = variants["full_session"]["metrics"]["top1_correct"]
    summary = {
        "status": "complete" if folds_requested == (0, 1, 2, 3) else "smoke_complete",
        "protocol": (
            "P103-B2 rich token-level visual candidate-query cross-attention; hard-centered "
            "candidate-list + pairwise supervision; no PCA, prototype, 40-class head, fixed "
            "A-logit residual, learned trigger, held-label tuning or H3."
        ),
        "folds_requested": list(folds_requested),
        "data": {
            "rows": int(evaluated.sum()),
            "users": sorted(set(users[evaluated].tolist())),
            "h3_rows_selected": 0,
            "h3_users_loaded": [],
            "rich_cache_contract": data.summary(),
        },
        "a_baseline": classification_metrics(
            a_probability[evaluated], labels[evaluated], users[evaluated]
        ),
        "level1_conditional": conditional,
        "level2_full_system": variants,
        "counterfactual_gate": {
            "aligned_correct_minus_shuffle": int(
                aligned_correct - variants["shuffle_session"]["metrics"]["top1_correct"]
            ),
            "aligned_correct_minus_zero": int(
                aligned_correct - variants["zero_session"]["metrics"]["top1_correct"]
            ),
        },
        "attention_audit": attention_audit(
            attention_groups,
            target_candidates,
            labels,
            a_probability,
            probabilities["full_local"],
            attackable,
        ),
        "folds": fold_reports,
        "student_started": False,
    }
    np.savez_compressed(
        output / "b2_oof_predictions.npz",
        sample_ids=sample_ids,
        users=users,
        labels=labels,
        fold_ids=fold_ids,
        evaluated=evaluated,
        candidate_ids=target_candidates,
        vs_raw_probability=vs_raw_probability.astype(np.float32),
        a_session_probability=a_probability.astype(np.float32),
        attention_group_names=np.asarray(P103RichCandidateTeacher.ATTENTION_GROUP_NAMES),
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
