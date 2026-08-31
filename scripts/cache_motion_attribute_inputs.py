from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.motion_attribute_dataset import (
    MotionAttributeDataset,
    fit_apply_attribute_normalization,
)
from src.experiments.motion_attribute_config import (
    load_motion_attribute_config,
    project_path,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _atomic_write(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def cache_motion_attribute_inputs(
    config_path: Path, *, output_path: Path
) -> dict[str, object]:
    config = load_motion_attribute_config(config_path)
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    started = time.perf_counter()
    arrays: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "features",
            "mask",
            "segment_ids",
            "raw_attributes",
            "families",
            "available",
            "labels",
            "sample_ids",
            "user_ids",
            "partition",
        )
    }
    for partition in ("train", "validation"):
        dataset = MotionAttributeDataset(config, partition=partition)
        for index in range(len(dataset)):
            item = dataset[index]
            arrays["features"].append(item["features"].numpy()[None])
            arrays["mask"].append(item["mask"].numpy()[None])
            arrays["segment_ids"].append(item["segment_ids"].numpy()[None])
            arrays["raw_attributes"].append(item["attributes"].numpy()[None])
            arrays["families"].append(item["families"].numpy()[None])
            arrays["available"].append(np.asarray([bool(item["available"])]))
            arrays["labels"].append(np.asarray([int(item["label"])], dtype=np.int64))
            arrays["sample_ids"].append(np.asarray([str(item["sample_id"])]))
            arrays["user_ids"].append(np.asarray([str(item["user_id"])]))
            arrays["partition"].append(np.asarray([partition]))
            if (index + 1) % 500 == 0:
                print(
                    json.dumps(
                        {
                            "stage": "motion_attribute_cache",
                            "partition": partition,
                            "rows": index + 1,
                        }
                    ),
                    flush=True,
                )
    merged = {name: np.concatenate(parts, axis=0) for name, parts in arrays.items()}
    train_mask = merged["partition"].astype(str) == "train"
    normalized, mean, std = fit_apply_attribute_normalization(
        merged["raw_attributes"],
        train_mask=train_mask,
        available=merged["available"],
    )
    merged["attributes"] = normalized
    merged["attribute_mean"] = mean
    merged["attribute_std"] = std
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_npz(output_path, **merged)
    report = {
        "status": "completed",
        "path": str(output_path),
        "sha256": _sha256(output_path),
        "bytes": output_path.stat().st_size,
        "rows": len(merged["labels"]),
        "train_rows": int(train_mask.sum()),
        "validation_rows": int((~train_mask).sum()),
        "supported_train": int((train_mask & merged["available"]).sum()),
        "supported_validation": int(((~train_mask) & merged["available"]).sum()),
        "feature_shape": list(merged["features"].shape),
        "attribute_shape": list(merged["attributes"].shape),
        "seconds": time.perf_counter() - started,
        "config_sha256": _sha256(config_path),
        "clean_view_sha256": _sha256(project_path(str(config["data"]["clean_view"]))),
    }
    _atomic_write(output_path.with_suffix(".json"), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache Motion Attribute Expert inputs")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/experiments/motion_attribute_expert.yaml",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_motion_attribute_config(args.config.resolve())
    output = args.output or project_path(str(config["outputs"]["root"])) / "input_cache.npz"
    print(cache_motion_attribute_inputs(args.config.resolve(), output_path=output))


if __name__ == "__main__":
    main()
