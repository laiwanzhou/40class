"""Train P103-B3 explicit local-crop candidate-query Teacher with OOF."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
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
from p103_local_candidate_model import (
    LocalCandidateConfig,
    P103LocalCandidateTeacher,
    trainable_parameter_audit,
)
from p103_local_feature_data import P103LocalData, load_p103_local_data
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
DEFAULT_OUTPUT = HERE / "runs/p103_b3_local_visual_oof_v1"


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


class LocalVisualDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        data: P103LocalData,
        rows: np.ndarray,
        candidate_ids: np.ndarray,
        context: np.ndarray,
        labels: np.ndarray,
        sample_weights: np.ndarray | None = None,
        feature_source: np.ndarray | None = None,
        local_scale: float = 1.0,
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
        self.local_scale = float(local_scale)

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
            "vmae_features": self.tensor(self.data.vmae_features[feature]),
            "vmae_actions": self.tensor(self.data.vmae_actions[feature]),
            "vjepa_features": self.tensor(self.data.vjepa_features[feature]),
            "vjepa_actions": self.tensor(self.data.vjepa_actions[feature]),
            "candidate_ids": torch.from_numpy(candidates.copy()).long(),
            "a_context": torch.from_numpy(self.context[row].copy()),
            "target_position": torch.tensor(target, dtype=torch.long),
            "sample_weight": torch.tensor(weight, dtype=torch.float32),
            "local_scale": torch.tensor(self.local_scale, dtype=torch.float32),
        }


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if key != "row"
    }


def train_model(
    data: P103LocalData,
    rows: np.ndarray,
    candidate_ids: np.ndarray,
    context: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    device: torch.device,
    seed: int,
) -> tuple[P103LocalCandidateTeacher, dict[str, Any]]:
    set_seed(seed)
    config = LocalCandidateConfig()
    model = P103LocalCandidateTeacher(config).to(device)
    dataset = LocalVisualDataset(
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
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
    model: P103LocalCandidateTeacher,
    data: P103LocalData,
    rows: np.ndarray,
    candidate_ids: np.ndarray,
    context: np.ndarray,
    labels: np.ndarray,
    a_probability: np.ndarray,
    device: torch.device,
    feature_source: np.ndarray | None = None,
    local_scale: float = 1.0,
    save_attention: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    dataset = LocalVisualDataset(
        data,
        rows,
        candidate_ids,
        context,
        labels,
        feature_source=feature_source,
        local_scale=local_scale,
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
    a_probability: np.ndarray,
    b_probability: np.ndarray,
    selected: np.ndarray,
) -> dict[str, Any]:
    b_prediction = b_probability.argmax(axis=1)
    groups: dict[str, list[np.ndarray]] = {
        "rescue_true_query": [],
        "miss_true_query": [],
        "predicted_query": [],
    }
    for row in np.flatnonzero(selected):
        valid = candidate_ids[row][candidate_ids[row] >= 0]
        true_position = int(np.flatnonzero(valid == labels[row])[0])
        predicted_position = int(np.flatnonzero(valid == b_prediction[row])[0])
        key = "rescue_true_query" if b_prediction[row] == labels[row] else "miss_true_query"
        groups[key].append(attention[row, true_position])
        groups["predicted_query"].append(attention[row, predicted_position])
    names = P103LocalCandidateTeacher.ATTENTION_GROUP_NAMES
    return {
        key: {
            "rows": len(values),
            "mean": {
                name: float(np.mean(np.stack(values), axis=0)[index])
                for index, name in enumerate(names)
            }
            if values
            else {},
        }
        for key, values in groups.items()
    } | {
        "warning": "attention is descriptive only; aligned/shuffle/zero is causal",
        "a_error_rows": int(np.sum(selected & (a_probability.argmax(axis=1) != labels))),
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
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sample_ids = session["sample_ids"].astype(str)
    users = session["users"].astype(str)
    labels = np.asarray(session["labels"], dtype=np.int64)
    fold_ids = np.asarray(session["fold_ids"], dtype=np.int64)
    a_probability = np.asarray(session["selected_probability"], dtype=np.float64)
    vs_raw_probability = np.asarray(session["vs_raw_probability"], dtype=np.float64)
    target_candidates = np.asarray(hard["candidate_ids"], dtype=np.int64)
    if not np.array_equal(hard["sample_ids"].astype(str), sample_ids):
        raise RuntimeError("hard/P103-B3 sample order differs")
    if set(users.tolist()) & set(H3_USERS):
        raise RuntimeError("H3 reached P103-B3")
    data = load_p103_local_data(sample_ids, users)
    if data.summary()["h3_rows_selected"] != 0:
        raise RuntimeError("H3 reached P103-B3 feature tensors")
    metadata = align_metadata(args.metadata.resolve(), sample_ids)

    variant_names = (
        "full_local",
        "full_session",
        "shuffle_local",
        "shuffle_session",
        "zero_local",
        "zero_session",
    )
    probabilities = {name: a_probability.copy() for name in variant_names}
    attention_groups = np.full(
        (
            len(labels),
            MAX_CANDIDATES,
            len(P103LocalCandidateTeacher.ATTENTION_GROUP_NAMES),
        ),
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
            neighbors, _ = confusion_neighbors(
                labels, source_prediction, users, graph_fit, recipe
            )
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
                "protocol": "P103-B3",
                "outer_fold": outer_fold,
                "config": training_audit["config"],
                "state_dict": model.state_dict(),
                "source_users": sorted(set(users[source].tolist())),
                "h3_users_loaded": [],
            },
            output / f"fold{outer_fold}_local_candidate_teacher.pt",
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
        for name, (feature_source, local_scale) in feature_variants.items():
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
                local_scale=local_scale,
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
            "P103-B3 explicit hand/workspace local-crop candidate-query attention using "
            "frozen VideoMAEv2 and V-JEPA2 features; hard-centered candidate supervision; "
            "no PCA, prototype, 40-class head, fixed A residual, held-label tuning or H3."
        ),
        "folds_requested": list(folds_requested),
        "data": {
            "rows": int(evaluated.sum()),
            "users": sorted(set(users[evaluated].tolist())),
            "h3_rows_selected": 0,
            "h3_users_loaded": [],
            "local_cache_contract": data.summary(),
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
        output / "b3_oof_predictions.npz",
        sample_ids=sample_ids,
        users=users,
        labels=labels,
        fold_ids=fold_ids,
        evaluated=evaluated,
        candidate_ids=target_candidates,
        vs_raw_probability=vs_raw_probability.astype(np.float32),
        a_session_probability=a_probability.astype(np.float32),
        attention_group_names=np.asarray(P103LocalCandidateTeacher.ATTENTION_GROUP_NAMES),
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
