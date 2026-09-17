"""Run the single authorized P102-B1 structural correction.

B1 learns class-specific visual directions with a listwise loss over the frozen
deployment candidate set.  It removes B0's class prototypes and motion blocks.
Every local candidate distribution is passed through the already selected,
source-only Session decoder before it can become the main B1 output.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from analyze_p102_hard_set import (
    CandidateRecipe,
    build_candidate_ids,
    confusion_neighbors,
    source_crossfit_session_probability,
)
from audit_p102_session_closure import classification_metrics, comparison, load_npz, true_rank
from audit_p87_sequence_decoder import DecoderConfig, align_metadata
from build_p87s_structured_targets import backed_off_structured_probability, build_targets
from p100a_global_teacher_data import FOLD_USERS, H3_USERS
from p101_finegrained_teacher_data import load_p101_data
from train_p102_candidate_reranker_oof import evaluate_variant, visual_summary


HERE = Path(__file__).resolve().parent
DEFAULT_SESSION = HERE / "runs/p102_session_closure_v1/session_oof_predictions.npz"
DEFAULT_SESSION_SUMMARY = HERE / "runs/p102_session_closure_v1/summary.json"
DEFAULT_HARD = HERE / "runs/p102_hard_set_v1/hard_set.npz"
DEFAULT_HARD_SUMMARY = HERE / "runs/p102_hard_set_v1/summary.json"
DEFAULT_NESTED = HERE / "runs/p101_f1_coarse_anchor_oof_v1/nested_coarse_vs"
DEFAULT_METADATA = HERE / "data/p85_recording_metadata/train_recording_metadata.csv"
DEFAULT_OUTPUT = HERE / "runs/p102_b1_visual_listwise_session_oof_v1"
VISUAL_BLOCKS = ("videomae", "internvideo")


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
    return parser.parse_args()


def load_visual_blocks(sample_ids: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    data = load_p101_data()
    if not np.array_equal(data.sample_ids.astype(str), sample_ids.astype(str)):
        raise RuntimeError("P101 visual representation order differs from P102")
    contract = data.summary()
    if int(contract["h3_rows"]) != 0 or set(data.users.tolist()) & set(H3_USERS):
        raise RuntimeError("H3 reached B1 visual representation loader")
    blocks = {
        "videomae": visual_summary(data.visual_vmae_temporal, data.visual_vmae_action),
        "internvideo": visual_summary(data.visual_iv2_temporal, data.visual_iv2_action),
    }
    for name, values in blocks.items():
        if len(values) != 1941 or not np.isfinite(values).all():
            raise RuntimeError(f"invalid B1 visual block: {name}")
    return blocks, {
        "p101_contract": contract,
        "raw_shapes": {name: list(values.shape) for name, values in blocks.items()},
        "motion_blocks_loaded": [],
    }


def fit_visual_embeddings(
    raw_blocks: dict[str, np.ndarray], source: np.ndarray, seed: int
) -> tuple[np.ndarray, dict[str, Any]]:
    values: list[np.ndarray] = []
    audit: dict[str, Any] = {}
    for offset, name in enumerate(VISUAL_BLOCKS):
        raw = np.asarray(raw_blocks[name], dtype=np.float32)
        scaler = StandardScaler(copy=True)
        source_scaled = scaler.fit_transform(raw[source]).astype(np.float32)
        all_scaled = scaler.transform(raw).astype(np.float32)
        count = min(40, source_scaled.shape[0] - 1, source_scaled.shape[1])
        pca = PCA(n_components=count, svd_solver="randomized", random_state=seed + offset)
        pca.fit(source_scaled)
        transformed = pca.transform(all_scaled).astype(np.float32)
        component_scale = np.maximum(transformed[source].std(axis=0, keepdims=True), 1e-4)
        transformed /= component_scale
        values.append(transformed)
        audit[name] = {
            "raw_dim": int(raw.shape[1]),
            "components": int(count),
            "explained_variance": float(np.sum(pca.explained_variance_ratio_)),
            "fit_rows": int(source.sum()),
        }
    return np.concatenate(values, axis=1).astype(np.float32), audit


def candidate_targets(
    candidate_ids: np.ndarray, labels: np.ndarray, rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(candidate_ids[rows], dtype=np.int64)
    valid = selected >= 0
    hits = selected == np.asarray(labels[rows], dtype=np.int64)[:, None]
    if not np.all(hits.sum(axis=1) == 1):
        raise RuntimeError("listwise training rows must contain the true class exactly once")
    return valid, hits.argmax(axis=1).astype(np.int64)


def listwise_scores(
    embeddings: torch.Tensor,
    direction: torch.Tensor,
    candidate_ids: torch.Tensor,
    a_probability: torch.Tensor,
) -> torch.Tensor:
    residual = embeddings @ direction
    safe_ids = candidate_ids.clamp(min=0)
    score = torch.log(torch.clamp(a_probability, min=1e-12)) + residual
    score = torch.gather(score, 1, safe_ids)
    return score.masked_fill(candidate_ids < 0, -1e9)


def fit_listwise_directions(
    embeddings: np.ndarray,
    candidate_ids: np.ndarray,
    a_probability: np.ndarray,
    labels: np.ndarray,
    rows: np.ndarray,
    seed: int,
    l2: float = 0.05,
    max_iter: int = 100,
) -> tuple[np.ndarray, dict[str, Any]]:
    torch.manual_seed(seed)
    x = torch.as_tensor(embeddings[rows], dtype=torch.float64)
    candidates = torch.as_tensor(candidate_ids[rows], dtype=torch.long)
    probability = torch.as_tensor(a_probability[rows], dtype=torch.float64)
    valid, target_values = candidate_targets(candidate_ids, labels, rows)
    target = torch.as_tensor(target_values, dtype=torch.long)
    source_error = a_probability[rows].argmax(axis=1) != labels[rows]
    row_weight = torch.as_tensor(np.where(source_error, 2.0, 1.0), dtype=torch.float64)
    row_weight /= row_weight.mean()
    direction = torch.nn.Parameter(torch.zeros((embeddings.shape[1], 40), dtype=torch.float64))
    optimizer = torch.optim.LBFGS(
        [direction], lr=1.0, max_iter=max_iter, tolerance_grad=1e-7,
        tolerance_change=1e-10, line_search_fn="strong_wolfe"
    )

    def objective(backward: bool) -> torch.Tensor:
        score = listwise_scores(x, direction, candidates, probability)
        row_loss = F.cross_entropy(score, target, reduction="none")
        loss = torch.mean(row_weight * row_loss) + 0.5 * l2 * torch.sum(direction.square())
        if backward:
            loss.backward()
        return loss

    with torch.no_grad():
        initial_score = listwise_scores(x, direction, candidates, probability)
        initial_ce = float(F.cross_entropy(initial_score, target).item())

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        return objective(backward=True)

    optimizer.step(closure)
    with torch.no_grad():
        final_score = listwise_scores(x, direction, candidates, probability)
        final_ce = float(F.cross_entropy(final_score, target).item())
        final_accuracy = float((final_score.argmax(dim=1) == target).double().mean().item())
        weight_norm = float(torch.linalg.vector_norm(direction).item())
    result = direction.detach().cpu().numpy()
    if not np.isfinite(result).all():
        raise RuntimeError("B1 listwise direction is non-finite")
    return result, {
        "rows": int(len(rows)),
        "candidate_pairs": int(valid.sum()),
        "source_a_error_rows": int(source_error.sum()),
        "loss": "candidate-listwise cross entropy; source A-error rows x2",
        "optimizer": "full-batch LBFGS strong_wolfe",
        "max_iter": int(max_iter),
        "l2_sum": float(l2),
        "initial_unweighted_ce": initial_ce,
        "final_unweighted_ce": final_ce,
        "final_training_accuracy": final_accuracy,
        "direction_l2_norm": weight_norm,
    }


def candidate_local_probability(
    embeddings: np.ndarray,
    direction: np.ndarray,
    candidate_ids: np.ndarray,
    a_probability: np.ndarray,
    selected: np.ndarray,
) -> np.ndarray:
    output = np.asarray(a_probability, dtype=np.float64).copy()
    rows = np.flatnonzero(selected).astype(np.int64)
    residual = np.asarray(embeddings[rows], dtype=np.float64) @ np.asarray(direction, dtype=np.float64)
    for offset, row in enumerate(rows):
        valid = candidate_ids[row][candidate_ids[row] >= 0]
        score = np.log(np.maximum(a_probability[row, valid], 1e-12)) + residual[offset, valid]
        value = np.exp(score - score.max())
        value /= value.sum()
        output[row] = 0.0
        output[row, valid] = value
    return output


def session_decode_candidate_probability(
    local_probability: np.ndarray,
    candidate_ids: np.ndarray,
    labels: np.ndarray,
    source: np.ndarray,
    held: np.ndarray,
    metadata: Any,
    config: DecoderConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    masked_labels = np.full_like(labels, -10_000)
    masked_labels[source] = labels[source]
    log_probability = np.log(np.maximum(local_probability, 1e-12))
    target = build_targets(
        log_probability,
        masked_labels,
        np.flatnonzero(source).astype(np.int64),
        np.flatnonzero(held).astype(np.int64),
        metadata,
        config,
        posterior_temperature=1.0,
    )
    structured = np.asarray(target["structured_probability"], dtype=np.float64)
    candidate_mask = np.zeros_like(structured, dtype=bool)
    rows = np.flatnonzero(held).astype(np.int64)
    for row in rows:
        valid = candidate_ids[row][candidate_ids[row] >= 0]
        candidate_mask[row, valid] = True
    structured[held] *= candidate_mask[held]
    mass = structured[held].sum(axis=1, keepdims=True)
    empty = mass[:, 0] <= 0
    structured_rows = structured[held]
    structured_rows[~empty] /= mass[~empty]
    structured_rows[empty] = local_probability[held][empty]
    structured[held] = structured_rows
    probability, weight = backed_off_structured_probability(
        local_probability, structured, beam_width=config.beam_width
    )
    probability[held] *= candidate_mask[held]
    probability[held] /= probability[held].sum(axis=1, keepdims=True)
    return probability, {
        "transition_fit_rows": int(source.sum()),
        "transition_fit_labels_masked_outside_source": int((~source).sum()),
        "held_sessions": int(target["holdout_session_count"]),
        "decoded_held_rows": int(np.sum(np.asarray(target["session_id"])[held] >= 0)),
        "mean_structured_weight": float(weight[held].mean()),
        "hard_map_outside_candidate": int(
            np.sum(
                ~candidate_mask[held][
                    np.arange(int(held.sum())),
                    np.asarray(target["structured_map_prediction"])[held],
                ]
            )
        ),
        "final_mass_outside_candidate": float(np.sum(probability[held] * (~candidate_mask[held]))),
    }


def shuffle_within_subject(
    embeddings: np.ndarray, users: np.ndarray, selected: np.ndarray
) -> np.ndarray:
    output = embeddings.copy()
    for user in sorted(set(users[selected].tolist())):
        rows = np.flatnonzero(selected & (users == user))
        if len(rows) > 1:
            output[rows] = embeddings[np.roll(rows, 1)]
    return output


def main() -> None:
    args = parse_args()
    folds_requested = tuple(sorted(set(map(int, args.folds))))
    if not set(folds_requested) <= {0, 1, 2, 3}:
        raise ValueError("folds must be a subset of 0..3")
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
    target_candidates = np.asarray(hard["candidate_ids"], dtype=np.int64)
    if not np.array_equal(hard["sample_ids"].astype(str), sample_ids):
        raise RuntimeError("hard/session row order differs")
    if set(users.tolist()) & set(H3_USERS):
        raise RuntimeError("H3 subject reached B1")
    metadata = align_metadata(args.metadata.resolve(), sample_ids)
    raw_blocks, representation_audit = load_visual_blocks(sample_ids)

    variant_names = (
        "full_local", "full_session", "shuffle_local", "shuffle_session",
        "zero_local", "zero_session",
    )
    probabilities = {name: a_probability.copy() for name in variant_names}
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
            outer_fold, args.nested_root.resolve(), sample_ids, users, fold_ids,
            labels, metadata, config,
        )
        recipe_values = hard_summary["folds"][outer_fold]["selected_recipe"]
        recipe = CandidateRecipe(**{key: int(value) for key, value in recipe_values.items()})
        source_prediction = source_probability.argmax(axis=1)
        source_candidates = np.full((len(labels), 8), -1, dtype=np.int64)
        for user in sorted(set(users[source].tolist())):
            validation = source & (users == user)
            graph_fit = np.flatnonzero(source & (users != user)).astype(np.int64)
            neighbors, _ = confusion_neighbors(labels, source_prediction, users, graph_fit, recipe)
            source_candidates[validation, : recipe.max_size] = build_candidate_ids(
                source_probability[validation], neighbors, recipe
            )
        source_hit = np.any(source_candidates == labels[:, None], axis=1)
        train_rows = np.flatnonzero(source & source_hit).astype(np.int64)
        embeddings, embedding_audit = fit_visual_embeddings(
            raw_blocks, source, seed=20260823 + outer_fold * 100
        )
        direction, training_audit = fit_listwise_directions(
            embeddings, source_candidates, source_probability, labels, train_rows,
            seed=20260823 + outer_fold,
        )
        embedding_variants = {
            "full": embeddings,
            "shuffle": shuffle_within_subject(embeddings, users, held),
            "zero": np.where(held[:, None], 0.0, embeddings),
        }
        fold_variants: dict[str, Any] = {}
        session_audits: dict[str, Any] = {}
        for name, variant_embedding in embedding_variants.items():
            local = candidate_local_probability(
                variant_embedding, direction, target_candidates, a_probability, held
            )
            decoded, decode_audit = session_decode_candidate_probability(
                local, target_candidates, labels, source, held, metadata, config
            )
            probabilities[f"{name}_local"][held] = local[held]
            probabilities[f"{name}_session"][held] = decoded[held]
            fold_variants[f"{name}_local"] = evaluate_variant(
                labels, users, held, a_probability, local
            )
            fold_variants[f"{name}_session"] = evaluate_variant(
                labels, users, held, a_probability, decoded
            )
            session_audits[name] = decode_audit
        fold_reports.append({
            "fold": outer_fold,
            "held_users": list(FOLD_USERS[outer_fold]),
            "source_users": sorted(set(users[source].tolist())),
            "source_held_overlap": sorted(set(users[source].tolist()) & set(FOLD_USERS[outer_fold])),
            "candidate_recipe": asdict(recipe),
            "source_candidate": {
                "rows": int(source.sum()),
                "hits": int(np.sum(source & source_hit)),
                "recall": float(np.mean(source_hit[source])),
                "train_rows_with_positive_candidate": int(len(train_rows)),
            },
            "source_session": source_session_audit,
            "embedding": embedding_audit,
            "training": training_audit,
            "session_decode": session_audits,
            "variants": fold_variants,
        })

    evaluated = np.isin(fold_ids, np.asarray(folds_requested))
    variants = {
        name: evaluate_variant(labels, users, evaluated, a_probability, probability)
        for name, probability in probabilities.items()
    }
    main_probability = probabilities["full_session"]
    candidate_hit = np.asarray(hard["candidate_hit"], dtype=bool)
    a_error = a_probability.argmax(axis=1) != labels
    attackable = evaluated & a_error & candidate_hit
    main_prediction = main_probability.argmax(axis=1)
    raw_rank = true_rank(a_probability[evaluated], labels[evaluated])
    final_rank = true_rank(main_probability[evaluated], labels[evaluated])
    full_net = int(variants["full_session"]["vs_a"]["net"])
    aligned_superiority = {
        name: int(
            variants["full_session"]["metrics"]["top1_correct"]
            - variants[name]["metrics"]["top1_correct"]
        )
        for name in ("shuffle_session", "zero_session")
    }
    summary = {
        "status": "complete" if folds_requested == (0, 1, 2, 3) else "smoke_complete",
        "protocol": (
            "P102-B1 class-specific visual directions trained by candidate-listwise CE; "
            "no prototypes, motion, 40-class head, trigger, blend, held-label tuning or "
            "oracle family; final output reuses the source-only Session decoder."
        ),
        "folds_requested": list(folds_requested),
        "data": {
            "rows": int(evaluated.sum()),
            "users": sorted(set(users[evaluated].tolist())),
            "h3_rows_selected": 0,
            "h3_users_loaded": [],
        },
        "representation": representation_audit,
        "a_baseline": classification_metrics(a_probability[evaluated], labels[evaluated], users[evaluated]),
        "variants": variants,
        "candidate": {
            "attackable_a_error_rows": int(attackable.sum()),
            "conditional_rerank_correct": int(np.sum(attackable & (main_prediction == labels))),
            "conditional_rerank_accuracy": float(np.mean(main_prediction[attackable] == labels[attackable])),
            "candidate_miss_a_errors": int(np.sum(evaluated & a_error & (~candidate_hit))),
        },
        "true_rank_movement": {
            "improved": int(np.sum(final_rank < raw_rank)),
            "harmed": int(np.sum(final_rank > raw_rank)),
            "mean_delta": float(np.mean(final_rank - raw_rank)),
        },
        "gate": {
            "net_rescue_at_least_20": bool(full_net >= 20),
            "aligned_correct_minus_shuffle": aligned_superiority["shuffle_session"],
            "aligned_correct_minus_zero": aligned_superiority["zero_session"],
            "correspondence_supported": bool(min(aligned_superiority.values()) > 0),
        },
        "folds": fold_reports,
        "student_started": False,
    }
    np.savez_compressed(
        output / "b1_oof_predictions.npz",
        sample_ids=sample_ids,
        users=users,
        labels=labels,
        fold_ids=fold_ids,
        evaluated=evaluated,
        candidate_ids=target_candidates,
        a_probability=a_probability.astype(np.float32),
        **{f"{name}_probability": probability.astype(np.float32) for name, probability in probabilities.items()},
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
