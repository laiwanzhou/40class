"""Small frozen-token Transformer head with strict subject-fold OOF training."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from audit_p87_sequence_decoder import classification_metrics
from p90_teacher_common import load_protocol


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CACHE = REPO / "runs/p96_vjepa2_vitl_ssv2_dense24_fold0_v1/features.npy"
OUTPUT = HERE / "runs/p142_vjepa_token_transformer_single_seed_v1"


class TokenHead(nn.Module):
    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 192,
        heads: int = 6,
        layers: int = 2,
        dropout: float = 0.20,
        view_dropout: float = 0.15,
        num_tokens: int = 24,
    ) -> None:
        super().__init__()
        self.view_dropout = float(view_dropout)
        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.position = nn.Parameter(
            torch.zeros(1, int(num_tokens) + 1, hidden_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 40),
        )
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.position, std=0.02)

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        tokens = self.projection(values)
        padding = torch.zeros(
            (len(values), tokens.shape[1] + 1),
            dtype=torch.bool,
            device=values.device,
        )
        if self.training and self.view_dropout > 0:
            dropped = torch.rand(
                (len(values), tokens.shape[1]), device=values.device
            ) < self.view_dropout
            # Every row retains at least one observed view.
            all_dropped = dropped.all(dim=1)
            if all_dropped.any():
                dropped[all_dropped, 0] = False
            tokens = tokens.masked_fill(dropped[..., None], 0.0)
            padding[:, 1:] = dropped
        cls = self.cls_token.expand(len(values), -1, -1)
        encoded = self.encoder(
            torch.cat((cls, tokens), dim=1) + self.position,
            src_key_padding_mask=padding,
        )
        return encoded[:, 0]

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(values))


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values: torch.Tensor) -> torch.Tensor:
        return values.view_as(values)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor]:
        return (-gradient,)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def class_weights(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels, minlength=40).astype(np.float64)
    weights = np.sqrt(len(labels) / np.maximum(40.0 * counts, 1.0))
    weights /= weights.mean()
    return weights.astype(np.float32)


def soft_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    weighted_target = target * weights[None, :]
    loss = -(weighted_target * F.log_softmax(logits, dim=1)).sum(dim=1)
    normalizer = weighted_target.sum(dim=1).clamp_min(1e-6)
    return (loss / normalizer).mean()


def train_fold(
    features: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    held_indices: np.ndarray,
    domain_labels: np.ndarray,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
    repeat_pairs: np.ndarray | None = None,
    teacher_probability: np.ndarray | None = None,
) -> np.ndarray:
    seed_everything(seed)
    model = TokenHead(
        input_dim=features.shape[-1],
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        layers=args.layers,
        dropout=args.dropout,
        view_dropout=args.view_dropout,
        num_tokens=features.shape[1],
    ).to(device)
    domain_head = (
        nn.Sequential(
            nn.LayerNorm(args.hidden_dim),
            nn.Linear(args.hidden_dim, 3),
        ).to(device)
        if args.domain_adversarial_weight > 0
        else None
    )
    optimizer = torch.optim.AdamW(
        list(model.parameters())
        + (list(domain_head.parameters()) if domain_head is not None else []),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    total_steps = args.epochs * math.ceil(len(train_indices) / args.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps, 1), eta_min=args.learning_rate * 0.05
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_indices.astype(np.int64)),
            torch.from_numpy(labels[train_indices].astype(np.int64)),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )
    weights = torch.from_numpy(class_weights(labels[train_indices])).to(device)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda"
    )
    if repeat_pairs is None:
        fold_pairs = np.empty((0, 2), dtype=np.int64)
    else:
        train_members = np.zeros(len(labels), dtype=bool)
        train_members[train_indices] = True
        fold_pairs = repeat_pairs[
            train_members[repeat_pairs[:, 0]] & train_members[repeat_pairs[:, 1]]
        ]
        if args.repeat_same_label_only:
            fold_pairs = fold_pairs[
                labels[fold_pairs[:, 0]] == labels[fold_pairs[:, 1]]
            ]
    pair_rng = np.random.default_rng(seed + 77)
    class_rows = {
        class_id: train_indices[labels[train_indices] == class_id]
        for class_id in range(40)
    }
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        rows = 0
        for indices, batch_labels in loader:
            numpy_indices = indices.numpy()
            values = torch.from_numpy(
                np.asarray(features[numpy_indices], dtype=np.float32)
            ).to(device)
            batch_labels = batch_labels.to(device)
            target = F.one_hot(batch_labels, num_classes=40).float()
            if teacher_probability is not None and args.teacher_weight > 0:
                teacher_target = torch.from_numpy(
                    teacher_probability[numpy_indices]
                ).to(device)
                target = (
                    (1.0 - args.teacher_weight) * target
                    + args.teacher_weight * teacher_target
                )
            if args.mixup_alpha > 0 and len(values) > 1:
                lam = float(np.random.beta(args.mixup_alpha, args.mixup_alpha))
                order = torch.randperm(len(values), device=device)
                values = lam * values + (1.0 - lam) * values[order]
                target = lam * target + (1.0 - lam) * target[order]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                "cuda", enabled=device.type == "cuda", dtype=torch.float16
            ):
                if domain_head is None:
                    logits = model(values)
                    embedding = None
                else:
                    embedding = model.encode(values)
                    logits = model.head(embedding)
                loss = soft_cross_entropy(logits, target, weights)
                if domain_head is not None and embedding is not None:
                    domain_target = torch.from_numpy(
                        domain_labels[numpy_indices].astype(np.int64)
                    ).to(device)
                    domain_logits = domain_head(
                        GradientReverse.apply(embedding)
                    )
                    loss = loss + args.domain_adversarial_weight * F.cross_entropy(
                        domain_logits, domain_target
                    )
                if (
                    args.repeat_consistency_weight > 0
                    or args.repeat_embedding_weight > 0
                ) and len(fold_pairs):
                    pair_rows = fold_pairs[
                        pair_rng.integers(
                            0,
                            len(fold_pairs),
                            size=min(max(len(values) // 4, 1), len(fold_pairs)),
                        )
                    ]
                    left = torch.from_numpy(
                        np.asarray(features[pair_rows[:, 0]], dtype=np.float32)
                    ).to(device)
                    right = torch.from_numpy(
                        np.asarray(features[pair_rows[:, 1]], dtype=np.float32)
                    ).to(device)
                    left_embedding = model.encode(left)
                    right_embedding = model.encode(right)
                    if args.repeat_consistency_weight > 0:
                        left_logits = model.head(left_embedding)
                        right_logits = model.head(right_embedding)
                        mean_probability = 0.5 * (
                            F.softmax(left_logits, dim=1)
                            + F.softmax(right_logits, dim=1)
                        )
                        consistency = 0.5 * (
                            F.kl_div(
                                F.log_softmax(left_logits, dim=1),
                                mean_probability,
                                reduction="batchmean",
                            )
                            + F.kl_div(
                                F.log_softmax(right_logits, dim=1),
                                mean_probability,
                                reduction="batchmean",
                            )
                        )
                        loss = loss + args.repeat_consistency_weight * consistency
                    if args.repeat_embedding_weight > 0:
                        embedding_consistency = (
                            1.0
                            - F.cosine_similarity(
                                left_embedding, right_embedding, dim=1
                            )
                        ).mean()
                        loss = (
                            loss
                            + args.repeat_embedding_weight * embedding_consistency
                        )
                if args.class_triplet_weight > 0:
                    triplet_rows = min(max(len(values) // 4, 1), len(train_indices))
                    anchors = train_indices[
                        pair_rng.integers(0, len(train_indices), size=triplet_rows)
                    ]
                    positives = np.asarray(
                        [
                            class_rows[int(labels[index])][
                                pair_rng.integers(
                                    0, len(class_rows[int(labels[index])])
                                )
                            ]
                            for index in anchors
                        ],
                        dtype=np.int64,
                    )
                    negatives = train_indices[
                        pair_rng.integers(0, len(train_indices), size=triplet_rows)
                    ]
                    collision = labels[negatives] == labels[anchors]
                    while collision.any():
                        negatives[collision] = train_indices[
                            pair_rng.integers(
                                0, len(train_indices), size=int(collision.sum())
                            )
                        ]
                        collision = labels[negatives] == labels[anchors]
                    triplet_indices = np.concatenate(
                        (anchors, positives, negatives)
                    )
                    triplet_values = torch.from_numpy(
                        np.asarray(features[triplet_indices], dtype=np.float32)
                    ).to(device)
                    embedding = F.normalize(
                        model.encode(triplet_values), dim=1
                    )
                    anchor_embedding, positive_embedding, negative_embedding = (
                        embedding.chunk(3, dim=0)
                    )
                    positive_distance = 1.0 - (
                        anchor_embedding * positive_embedding
                    ).sum(dim=1)
                    negative_distance = 1.0 - (
                        anchor_embedding * negative_embedding
                    ).sum(dim=1)
                    triplet_loss = F.relu(
                        positive_distance
                        - negative_distance
                        + args.triplet_margin
                    ).mean()
                    loss = loss + args.class_triplet_weight * triplet_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_loss += float(loss.detach()) * len(values)
            rows += len(values)
        if epoch in (0, args.epochs - 1) or (epoch + 1) % 10 == 0:
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "epoch": epoch + 1,
                        "epochs": args.epochs,
                        "train_loss": epoch_loss / max(rows, 1),
                    }
                ),
                flush=True,
            )

    model.eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(held_indices), args.batch_size * 2):
            indices = held_indices[start : start + args.batch_size * 2]
            values = torch.from_numpy(
                np.asarray(features[indices], dtype=np.float32)
            ).to(device)
            with torch.amp.autocast(
                "cuda", enabled=device.type == "cuda", dtype=torch.float16
            ):
                output.append(model(values).float().cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--feature-cache", type=Path, default=CACHE)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--view-dropout", type=float, default=0.15)
    parser.add_argument("--mixup-alpha", type=float, default=0.20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seeds", type=int, nargs="+", default=(14201,))
    parser.add_argument("--repeat-consistency-weight", type=float, default=0.0)
    parser.add_argument("--repeat-embedding-weight", type=float, default=0.0)
    parser.add_argument("--repeat-same-label-only", action="store_true")
    parser.add_argument("--class-triplet-weight", type=float, default=0.0)
    parser.add_argument("--triplet-margin", type=float, default=0.20)
    parser.add_argument("--teacher-run", type=Path)
    parser.add_argument("--teacher-weight", type=float, default=0.0)
    parser.add_argument("--teacher-confidence", type=float, default=0.95)
    parser.add_argument("--domain-adversarial-weight", type=float, default=0.0)
    parser.add_argument(
        "--token-subset",
        choices=("all", "global", "hand", "workspace", "hand_interaction"),
        default="all",
    )
    return parser.parse_args()


def build_repeat_pairs(protocol) -> np.ndarray:
    """Frozen anonymous repeat alignment; labels and user IDs are never consulted."""
    from audit_p87_sequence_decoder import align_metadata
    from p117_transductive_multicandidate_router import load_candidate_splits
    from p88_aligned_repeat_holdout import align_probabilities
    from p89_global_repeat_decoder import cluster_sessions, date_session_lists
    from p134_frozen_repeat_consensus import CONFIG, METADATA, evidence, probability_lookup

    data = load_candidate_splits(
        full_visual_bank=True,
        structured_bank=True,
        legacy_visual_bank=True,
        hand_object_bank=True,
        vjepa_dense_bank=True,
        nonvisual_bank=True,
        hierarchical_bank=True,
        epic_bank=True,
    )
    lookup = probability_lookup(data)
    p128 = np.load(
        HERE / "runs/p128_base_hierarchical_meta_selector_v1/predictions.npz"
    )
    global_position = {
        value: row for row, value in enumerate(protocol.sample_ids.astype(str))
    }
    pairs: set[tuple[int, int]] = set()
    for split_name, value in data.items():
        sample_ids = value.split.sample_ids.astype(str)
        base = p128[f"{split_name}_prediction"].astype(np.int64)
        grouping_probability, _ = evidence(sample_ids, base, lookup, "vote")
        metadata = align_metadata(METADATA, sample_ids)
        for sessions in date_session_lists(
            np.arange(len(sample_ids), dtype=np.int64), metadata, 30.0
        ):
            for group in cluster_sessions(
                sessions, grouping_probability, base, metadata, CONFIG
            ):
                reference = max(group, key=len)
                for session in group:
                    if session is reference:
                        continue
                    aligned, _ = align_probabilities(
                        grouping_probability[reference],
                        grouping_probability[session],
                        CONFIG.alignment_gap_penalty,
                    )
                    for reference_position, other_position in aligned:
                        left = global_position[
                            sample_ids[int(reference[reference_position])]
                        ]
                        right = global_position[
                            sample_ids[int(session[other_position])]
                        ]
                        pairs.add(tuple(sorted((left, right))))
    return np.asarray(sorted(pairs), dtype=np.int64).reshape(-1, 2)


def build_teacher_probability(protocol, args: argparse.Namespace) -> np.ndarray | None:
    if args.teacher_run is None or args.teacher_weight <= 0:
        return None
    source = np.load(args.teacher_run.resolve())
    prediction = protocol.labels.copy()
    global_position = {
        value: row for row, value in enumerate(protocol.sample_ids.astype(str))
    }
    loaded = np.zeros(len(prediction), dtype=bool)
    for split_name in (
        "H1_selection",
        "H2_confirmation",
        "H3_independent_fold0",
    ):
        sample_key = f"{split_name}_sample_ids"
        prediction_key = f"{split_name}_prediction"
        if sample_key not in source.files or prediction_key not in source.files:
            raise RuntimeError(f"teacher run lacks {split_name} held predictions")
        for sample_id, class_id in zip(
            source[sample_key].astype(str),
            source[prediction_key].astype(np.int64),
        ):
            row = global_position[sample_id]
            prediction[row] = class_id
            loaded[row] = True
    confidence = float(args.teacher_confidence)
    probability = np.full(
        (len(prediction), 40),
        (1.0 - confidence) / 39.0,
        dtype=np.float32,
    )
    probability[np.arange(len(prediction)), prediction] = confidence
    # The 444 non-P89 rows have no P150 prediction; keep ordinary hard supervision.
    probability[~loaded] = 0.0
    probability[np.flatnonzero(~loaded), protocol.labels[~loaded]] = 1.0
    return probability


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = load_protocol()
    features = np.load(args.feature_cache.resolve(), mmap_mode="r")
    if features.ndim != 3 or len(features) != len(protocol.labels):
        raise RuntimeError(f"token feature shape changed: {features.shape}")
    subsets = {
        "all": np.arange(features.shape[1]),
        "global": np.arange(12),
        "hand": np.arange(12, 24),
        "workspace": np.asarray((2, 5, 8, 11)),
        "hand_interaction": np.asarray((14, 17, 20, 23)),
    }
    if args.token_subset != "all" and features.shape[1] != 24:
        raise RuntimeError(
            f"named P96 token subset requires 24 tokens, got {features.shape[1]}"
        )
    features = features[:, subsets[args.token_subset]]
    repeat_pairs = (
        build_repeat_pairs(protocol)
        if (
            args.repeat_consistency_weight > 0
            or args.repeat_embedding_weight > 0
        )
        else np.empty((0, 2), dtype=np.int64)
    )
    teacher_probability = build_teacher_probability(protocol, args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logits = np.zeros((len(protocol.labels), 40), dtype=np.float64)
    seed_reports = []
    for fold in range(3):
        train_indices = protocol.train_indices(fold)
        held_indices = protocol.val_indices(fold)
        members = []
        for seed in args.seeds:
            members.append(
                train_fold(
                    features,
                    protocol.labels,
                    train_indices,
                    held_indices,
                    protocol.fold_id,
                    int(seed + fold * 1000),
                    args,
                    device,
                    repeat_pairs,
                    teacher_probability,
                )
            )
        fold_logits = np.mean(np.stack(members), axis=0)
        logits[held_indices] = fold_logits
        prediction = fold_logits.argmax(axis=1)
        seed_reports.append(
            {
                "fold": fold,
                "train_rows": int(len(train_indices)),
                "held_rows": int(len(held_indices)),
                "metrics": classification_metrics(
                    protocol.labels[held_indices], prediction
                ),
            }
        )
    probability = np.exp(logits - logits.max(axis=1, keepdims=True))
    probability /= probability.sum(axis=1, keepdims=True)
    prediction = probability.argmax(axis=1)
    report = {
        "stage": "P142_frozen_VJEPA_token_transformer_OOF",
        "status": "complete_strict_subject_fold_oof",
        "protocol": {
            "backbone_frozen": True,
            "epochs_fixed": args.epochs,
            "seeds_fixed": list(args.seeds),
            "held_fold_used_for_selection": False,
            "user_id_used_as_feature": False,
            "test_rows_loaded": 0,
            "test_labels_loaded": 0,
            "submission_generated": False,
        },
        "architecture": {
            "hidden_dim": args.hidden_dim,
            "heads": args.heads,
            "layers": args.layers,
            "dropout": args.dropout,
            "view_dropout": args.view_dropout,
            "mixup_alpha": args.mixup_alpha,
            "token_subset": args.token_subset,
            "token_count": int(features.shape[1]),
            "feature_dim": int(features.shape[2]),
            "feature_cache": str(args.feature_cache.resolve()),
            "repeat_consistency_weight": args.repeat_consistency_weight,
            "repeat_embedding_weight": args.repeat_embedding_weight,
            "repeat_same_label_only": args.repeat_same_label_only,
            "class_triplet_weight": args.class_triplet_weight,
            "triplet_margin": args.triplet_margin,
            "teacher_weight": args.teacher_weight,
            "teacher_confidence": args.teacher_confidence,
            "teacher_run": (
                str(args.teacher_run.resolve()) if args.teacher_run else None
            ),
            "domain_adversarial_weight": args.domain_adversarial_weight,
            "domain_target": (
                "source subject-fold id" if args.domain_adversarial_weight > 0 else None
            ),
            "anonymous_repeat_pairs": int(len(repeat_pairs)),
        },
        "folds": seed_reports,
        "metrics": classification_metrics(protocol.labels, prediction),
    }
    np.savez_compressed(
        args.output_dir / "oof_predictions.npz",
        sample_ids=protocol.sample_ids,
        labels=protocol.labels,
        users=protocol.users,
        fold_id=protocol.fold_id,
        logits=logits.astype(np.float32),
        probability=probability.astype(np.float32),
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
