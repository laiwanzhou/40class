"""P111 label-free visual adaptation and source-safe pair-specialist tree.

The representation stage is deliberately label-free.  It adapts an ImageNet
ResNet18 to the CUHK-X scene/person/workspace composite with SimCLR, using two
augmentations of one frame from the same trial.  Labels are first read by the
pair-specialist audit after the encoder has been frozen.

For every P89 outer fold, candidate pairs and call/abstain decisions are made
from the other source users only.  A pair is deployable only when source-user
cross-fit replacement has positive net gain and no routed source user regresses.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2

from p90_crossuser_visual_router import load_splits


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_CACHE = HERE / "runs/p86_visual_pixel_cache_t16_r160_v12"
DEFAULT_OUTPUT = HERE / "runs/p111_transductive_simclr_tree_v1"

SEED = 20260824
OUTER_USERS = {
    "H1_selection": ("user6", "user8", "user17", "user23"),
    "H2_confirmation": ("user5", "user7", "user16", "user18", "user19"),
    "H3_independent_fold0": ("user20", "user22", "user24", "user3", "user4", "user9"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", choices=("all", "extract-control", "ssl", "extract", "route"), default="all")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--extract-batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    if not values:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


class TrialContrastiveDataset(Dataset):
    """No label field is accepted or returned by this dataset."""

    def __init__(self, images_path: Path, rows: int, image_size: int = 144) -> None:
        self.images = np.load(images_path, mmap_mode="r")
        if self.images.shape[0] != rows or self.images.ndim != 6:
            raise RuntimeError(f"unexpected P111 pixel cache shape: {self.images.shape}")
        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.RandomResizedCrop(
                    (image_size, image_size),
                    scale=(0.60, 1.0),
                    ratio=(0.85, 1.15),
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                # The three channels are scene/person/workspace, not RGB.  A
                # horizontal flip would erase useful left/right interaction.
                v2.ColorJitter(brightness=0.25, contrast=0.25),
                v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 1.5))], p=0.25),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        window = int(torch.randint(0, self.images.shape[1], ()).item())
        frame = int(torch.randint(0, self.images.shape[2], ()).item())
        # A copy avoids a non-writable memmap warning and makes augmentation
        # ownership explicit.
        image = torch.from_numpy(np.array(self.images[index, window, frame], copy=True))
        return self.transform(image), self.transform(image)


class SimCLREncoder(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
        )

    def features(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images).flatten(1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projector(self.features(images))


def ntxent_loss(first: torch.Tensor, second: torch.Tensor, temperature: float) -> torch.Tensor:
    first = F.normalize(first, dim=1)
    second = F.normalize(second, dim=1)
    values = torch.cat((first, second), dim=0)
    logits = values @ values.T / float(temperature)
    rows = logits.shape[0]
    logits.fill_diagonal_(-torch.inf)
    half = rows // 2
    targets = (torch.arange(rows, device=logits.device) + half) % rows
    return F.cross_entropy(logits, targets)


def train_ssl(args: argparse.Namespace, rows: list[dict[str, str]], output: Path) -> Path:
    dataset = TrialContrastiveDataset(args.cache / "images.npy", len(rows))
    if args.smoke:
        dataset = torch.utils.data.Subset(dataset, range(min(192, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=args.device.startswith("cuda"),
        drop_last=True,
        persistent_workers=args.workers > 0,
    )
    model = SimCLREncoder(pretrained=True).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    epochs = 1 if args.smoke else args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.device.startswith("cuda"))
    history: list[dict[str, Any]] = []
    for epoch in range(epochs):
        model.train()
        losses: list[float] = []
        for first, second in loader:
            first = first.to(args.device, non_blocking=True)
            second = second.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda" if args.device.startswith("cuda") else "cpu",
                dtype=torch.bfloat16,
                enabled=args.device.startswith("cuda"),
            ):
                loss = ntxent_loss(model(first), model(second), args.temperature)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        record = {
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        print(json.dumps({"p111_ssl": record}), flush=True)
    checkpoint = output / ("simclr_encoder_smoke.pt" if args.smoke else "simclr_encoder.pt")
    torch.save(
        {
            "encoder": model.encoder.state_dict(),
            "projector": model.projector.state_dict(),
            "protocol": {
                "label_free": True,
                "rows": len(dataset),
                "epochs": epochs,
                "seed": SEED,
                "temperature": args.temperature,
                "input_channels": ["scene", "person", "workspace"],
            },
        },
        checkpoint,
    )
    write_csv(output / ("ssl_history_smoke.csv" if args.smoke else "ssl_history.csv"), history)
    return checkpoint


class TrialExtractionDataset(Dataset):
    def __init__(self, images_path: Path) -> None:
        self.images = np.load(images_path, mmap_mode="r")

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(np.array(self.images[index], copy=True))


def normalize_images(values: torch.Tensor) -> torch.Tensor:
    values = values.float().div_(255.0)
    mean = torch.tensor((0.485, 0.456, 0.406), device=values.device)[None, :, None, None]
    std = torch.tensor((0.229, 0.224, 0.225), device=values.device)[None, :, None, None]
    return (values - mean) / std


def component_normalize(values: torch.Tensor) -> torch.Tensor:
    return F.normalize(values.float(), dim=-1)


def extract_descriptors(
    args: argparse.Namespace,
    rows: list[dict[str, str]],
    output_path: Path,
    checkpoint: Path | None,
) -> None:
    model = SimCLREncoder(pretrained=True)
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.encoder.load_state_dict(state["encoder"])
    model = model.to(args.device).eval()
    dataset: Dataset = TrialExtractionDataset(args.cache / "images.npy")
    if args.smoke:
        dataset = torch.utils.data.Subset(dataset, range(min(192, len(dataset))))
        rows = rows[: len(dataset)]
    loader = DataLoader(
        dataset,
        batch_size=args.extract_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.workers > 0,
    )
    all_descriptors: list[np.ndarray] = []
    with torch.inference_mode():
        for batch_index, images in enumerate(loader, start=1):
            batch, windows, frames, channels, height, width = images.shape
            flat = images.reshape(batch * windows * frames, channels, height, width)
            flat = normalize_images(flat.to(args.device, non_blocking=True))
            with torch.autocast(
                device_type="cuda" if args.device.startswith("cuda") else "cpu",
                dtype=torch.bfloat16,
                enabled=args.device.startswith("cuda"),
            ):
                token = model.features(flat).reshape(batch, windows, frames, 512)
            early = token[:, 0].mean(dim=1)
            late = token[:, 1].mean(dim=1)
            global_mean = token.mean(dim=(1, 2))
            global_std = token.float().std(dim=(1, 2), unbiased=False)
            maximum = token.float().amax(dim=(1, 2))
            delta = late - early
            descriptor = torch.cat(
                tuple(
                    component_normalize(value)
                    for value in (early, late, global_mean, global_std, maximum, delta)
                ),
                dim=1,
            )
            all_descriptors.append(descriptor.cpu().numpy().astype(np.float16))
            if batch_index % 25 == 0 or batch_index == len(loader):
                print(f"P111 extract {batch_index}/{len(loader)}", flush=True)
    matrix = np.concatenate(all_descriptors, axis=0)
    if matrix.shape != (len(rows), 3072):
        raise RuntimeError(f"P111 descriptor shape changed: {matrix.shape}")
    np.savez_compressed(
        output_path,
        sample_ids=np.asarray([row["sample_id"] for row in rows]),
        users=np.asarray([row["user_id"] for row in rows]),
        labels=np.asarray([int(row["class_id"]) for row in rows], dtype=np.int64),
        descriptors=matrix,
        adapted=np.asarray(checkpoint is not None),
    )


@dataclass
class PairProjection:
    scaler: StandardScaler
    pca: PCA | None

    @classmethod
    def fit(cls, values: np.ndarray) -> "PairProjection":
        scaler = StandardScaler().fit(np.asarray(values, dtype=np.float32))
        standardized = scaler.transform(np.asarray(values, dtype=np.float32))
        components = min(64, standardized.shape[0] - 2, standardized.shape[1])
        pca: PCA | None = None
        if components >= 4:
            pca = PCA(
                n_components=components,
                svd_solver="randomized",
                iterated_power=2,
                random_state=SEED,
            ).fit(standardized)
        return cls(scaler=scaler, pca=pca)

    def transform(self, values: np.ndarray) -> np.ndarray:
        result = self.scaler.transform(np.asarray(values, dtype=np.float32))
        if self.pca is not None:
            result = self.pca.transform(result)
        return np.asarray(result, dtype=np.float32)


@dataclass
class PairClassifier:
    projection: PairProjection
    classifier: LogisticRegression

    @classmethod
    def fit(
        cls, descriptors: np.ndarray, labels: np.ndarray, selected: np.ndarray
    ) -> "PairClassifier":
        projection = PairProjection.fit(descriptors[selected])
        classifier = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="liblinear",
            max_iter=2000,
            random_state=SEED,
        ).fit(projection.transform(descriptors[selected]), labels[selected])
        return cls(projection=projection, classifier=classifier)

    def predict(self, descriptors: np.ndarray) -> np.ndarray:
        return self.classifier.predict(self.projection.transform(descriptors)).astype(np.int64)


def unordered_pair(values: np.ndarray) -> np.ndarray:
    return np.sort(np.asarray(values, dtype=np.int64), axis=1)


def align_rows(source_ids: np.ndarray, target_ids: np.ndarray) -> np.ndarray:
    lookup = {value: index for index, value in enumerate(source_ids.astype(str))}
    missing = [value for value in target_ids.astype(str) if value not in lookup]
    if missing:
        raise RuntimeError(f"P111 descriptor misses {len(missing)} target rows")
    return np.asarray([lookup[value] for value in target_ids.astype(str)], dtype=np.int64)


def route_variant(descriptor_path: Path, output: Path, name: str) -> dict[str, Any]:
    with np.load(descriptor_path, allow_pickle=False) as archive:
        full_ids = archive["sample_ids"].astype(str)
        full_users = archive["users"].astype(str)
        full_labels = archive["labels"].astype(np.int64)
        full_descriptor = archive["descriptors"].astype(np.float32)
    splits = load_splits()
    split_names = tuple(OUTER_USERS)
    ids = np.concatenate([splits[key].sample_ids.astype(str) for key in split_names])
    users = np.concatenate([splits[key].users.astype(str) for key in split_names])
    labels = np.concatenate([splits[key].labels.astype(np.int64) for key in split_names])
    safe = np.concatenate([splits[key].safe_prediction.astype(np.int64) for key in split_names])
    probability = np.concatenate([splits[key].safe_probability.astype(np.float64) for key in split_names])
    fold_name = np.concatenate(
        [np.repeat(key, len(splits[key].sample_ids)) for key in split_names]
    ).astype(str)
    if int(np.sum(safe == labels)) != 2117 or len(ids) != 2470:
        raise RuntimeError("P111 P89 baseline changed")
    full_order = align_rows(full_ids, ids)
    descriptor = full_descriptor[full_order]
    top2 = np.argsort(-probability, axis=1, kind="stable")[:, :2]
    top2_pair = unordered_pair(top2)
    system = safe.copy()
    pair_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    fold_summaries: dict[str, Any] = {}

    for outer in split_names:
        held = fold_name == outer
        source = ~held
        held_users = set(OUTER_USERS[outer])
        if set(users[held].tolist()) != held_users:
            raise RuntimeError(f"P111 {outer} held users changed")
        opportunities: Counter[tuple[int, int]] = Counter()
        for row in np.flatnonzero(source & (safe != labels)):
            pair = tuple(map(int, top2_pair[row]))
            if int(labels[row]) in pair and int(safe[row]) in pair:
                opportunities[pair] += 1
        candidates = [pair for pair, count in opportunities.items() if count >= 2]
        candidates.sort(key=lambda pair: (-opportunities[pair], pair))
        activated: dict[tuple[int, int], PairClassifier] = {}

        for pair in candidates:
            crossfit_prediction: dict[int, int] = {}
            routed_user_net: dict[str, int] = {}
            for inner_user in sorted(set(users[source].tolist())):
                eval_rows = np.flatnonzero(
                    source
                    & (users == inner_user)
                    & np.all(top2_pair == np.asarray(pair)[None], axis=1)
                    & np.isin(safe, pair)
                )
                if not len(eval_rows):
                    continue
                train_full = (~np.isin(full_users, list(held_users) + [inner_user])) & np.isin(
                    full_labels, pair
                )
                if len(set(full_labels[train_full].tolist())) != 2:
                    continue
                classifier = PairClassifier.fit(full_descriptor, full_labels, train_full)
                prediction = classifier.predict(descriptor[eval_rows])
                for row, value in zip(eval_rows.tolist(), prediction.tolist()):
                    crossfit_prediction[int(row)] = int(value)
                base_correct = safe[eval_rows] == labels[eval_rows]
                candidate_correct = prediction == labels[eval_rows]
                routed_user_net[inner_user] = int(candidate_correct.sum() - base_correct.sum())

            evaluated = np.asarray(sorted(crossfit_prediction), dtype=np.int64)
            prediction = np.asarray([crossfit_prediction[int(row)] for row in evaluated], dtype=np.int64)
            base_correct = safe[evaluated] == labels[evaluated]
            candidate_correct = prediction == labels[evaluated]
            rescue = int(np.sum((~base_correct) & candidate_correct))
            harm = int(np.sum(base_correct & (~candidate_correct)))
            net = rescue - harm
            worst_user = min(routed_user_net.values()) if routed_user_net else -999
            deploy = rescue >= 2 and net > 0 and worst_user >= 0
            record = {
                "variant": name,
                "outer_fold": outer,
                "pair": f"{pair[0]}<->{pair[1]}",
                "source_opportunities": opportunities[pair],
                "source_crossfit_routes": len(evaluated),
                "source_crossfit_rescue": rescue,
                "source_crossfit_harm": harm,
                "source_crossfit_net": net,
                "source_crossfit_worst_user_net": worst_user,
                "deploy": int(deploy),
            }
            if deploy:
                train_full = (~np.isin(full_users, list(held_users))) & np.isin(full_labels, pair)
                activated[pair] = PairClassifier.fit(full_descriptor, full_labels, train_full)
            pair_rows.append(record)

        before = system.copy()
        for pair, classifier in activated.items():
            selected = np.flatnonzero(
                held
                & np.all(top2_pair == np.asarray(pair)[None], axis=1)
                & np.isin(safe, pair)
            )
            if len(selected):
                system[selected] = classifier.predict(descriptor[selected])
        changed = held & (system != before)
        base_correct = safe == labels
        system_correct = system == labels
        rescue = int(np.sum(held & (~base_correct) & system_correct))
        harm = int(np.sum(held & base_correct & (~system_correct)))
        fold_summaries[outer] = {
            "rows": int(held.sum()),
            "p89_correct": int(np.sum(held & base_correct)),
            "system_correct": int(np.sum(held & system_correct)),
            "accuracy": float(np.mean(system_correct[held])),
            "routes_changed": int(changed.sum()),
            "rescue": rescue,
            "harm": harm,
            "net": rescue - harm,
            "activated_pairs": [f"{pair[0]}<->{pair[1]}" for pair in activated],
        }

    base_correct = safe == labels
    system_correct = system == labels
    for row in range(len(ids)):
        sample_rows.append(
            {
                "variant": name,
                "sample_id": ids[row],
                "subject": users[row],
                "outer_fold": fold_name[row],
                "true_label": int(labels[row]),
                "p89_prediction": int(safe[row]),
                "adjusted_top2": f"{int(top2[row, 0])}|{int(top2[row, 1])}",
                "system_prediction": int(system[row]),
                "changed": int(system[row] != safe[row]),
                "rescued": int((not base_correct[row]) and system_correct[row]),
                "harmed": int(base_correct[row] and (not system_correct[row])),
            }
        )
    summary = {
        "variant": name,
        "descriptor": str(descriptor_path.resolve()),
        "p89": {"correct": 2117, "rows": 2470, "accuracy": 2117 / 2470},
        "system": {
            "correct": int(system_correct.sum()),
            "rows": len(labels),
            "accuracy": float(system_correct.mean()),
            "rescue": int(np.sum((~base_correct) & system_correct)),
            "harm": int(np.sum(base_correct & (~system_correct))),
            "net": int(system_correct.sum() - base_correct.sum()),
            "changed": int(np.sum(system != safe)),
        },
        "folds": fold_summaries,
        "router": {
            "candidate_structure": "P89 adjusted-probability Top-2 unordered pair",
            "source_pair_minimum_opportunities": 2,
            "deploy_rule": "source-user crossfit rescue>=2, net>0, worst routed user net>=0",
            "held_label_selection": False,
        },
    }
    write_csv(output / f"{name}_pair_audit.csv", pair_rows)
    write_csv(output / f"{name}_sample_predictions.csv", sample_rows)
    (output / f"{name}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    args = parse_args()
    seed_everything()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = read_csv((args.cache / "rows.csv").resolve())
    if len(rows) != 2914:
        raise RuntimeError(f"P111 cache rows changed: {len(rows)}")
    control_path = output / ("imagenet_descriptors_smoke.npz" if args.smoke else "imagenet_descriptors.npz")
    adapted_path = output / ("adapted_descriptors_smoke.npz" if args.smoke else "adapted_descriptors.npz")
    checkpoint = output / ("simclr_encoder_smoke.pt" if args.smoke else "simclr_encoder.pt")

    if args.stage in ("all", "extract-control") and not control_path.is_file():
        extract_descriptors(args, rows, control_path, checkpoint=None)
    if args.stage in ("all", "ssl"):
        checkpoint = train_ssl(args, rows, output)
    if args.stage in ("all", "extract"):
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        extract_descriptors(args, rows, adapted_path, checkpoint=checkpoint)

    summaries: dict[str, Any] = {}
    if args.stage in ("all", "route") and not args.smoke:
        if control_path.is_file():
            summaries["imagenet_control"] = route_variant(control_path, output, "imagenet_control")
        if adapted_path.is_file():
            summaries["simclr_primary"] = route_variant(adapted_path, output, "simclr_primary")
        aggregate = {
            "stage": "P111_transductive_simclr_pair_tree",
            "status": "complete",
            "ssl": {
                "uses_labels": False,
                "uses_all_source_trials_as_unlabelled": True,
                "method": "same-frame two-view SimCLR on scene/person/workspace composite",
                "test_time_analogue": "repeat label-free adaptation on source+anonymous test pixels",
            },
            "variants": summaries,
            "constraints": {
                "p89_baseline_frozen": True,
                "outer_held_labels_used_for_pair_or_route_selection": False,
                "source_user_crossfit_router": True,
                "model": "ImageNet ResNet18 plus small pair heads",
                "checkpoint_under_100mb": checkpoint.stat().st_size < 100_000_000,
            },
        }
        (output / "summary.json").write_text(
            json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(aggregate, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
