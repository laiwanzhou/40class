from __future__ import annotations

import argparse
import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from aligned_data import AlignedSample, SKELETON_STRATEGIES, frame_map, load_skeleton_sequence


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PROJECT_DIR / "data" / "manifest.csv"
DEFAULT_SOURCE_CACHE = PROJECT_DIR / "cache" / "aligned_192x144"
DEFAULT_OUTPUT_CACHE = PROJECT_DIR / "cache" / "aligned_192x144_temporal_nearest"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="复用图像缓存，仅重建带时序选人的 Skeleton 缓存")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-cache", type=Path, default=DEFAULT_SOURCE_CACHE)
    parser.add_argument("--output-cache", type=Path, default=DEFAULT_OUTPUT_CACHE)
    parser.add_argument(
        "--skeleton-strategy",
        choices=SKELETON_STRATEGIES,
        default="temporal_nearest",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=64)
    return parser.parse_args()


def read_samples(path: Path) -> tuple[list[AlignedSample], list[int]]:
    samples: list[AlignedSample] = []
    lengths: list[int] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            samples.append(
                AlignedSample(
                    sample_id=row["sample_id"],
                    class_id=int(row["class_id"]),
                    user_id=row["user_id"],
                    trial_id=row["trial_id"],
                    depth_dir=Path(row["depth_dir"]),
                    ir_dir=Path(row["ir_dir"]),
                    skeleton_dir=Path(row["skeleton_dir"]),
                )
            )
            lengths.append(int(row["num_aligned_frames"]))
    return samples, lengths


def process_sample(work: tuple[AlignedSample, str]) -> np.ndarray:
    sample, strategy = work
    maps = {
        "depth": frame_map(sample.depth_dir, "depth"),
        "ir": frame_map(sample.ir_dir, "ir"),
        "skeleton": frame_map(sample.skeleton_dir, "skeleton"),
    }
    common_ids = sorted(set.intersection(*(set(mapping) for mapping in maps.values())))
    paths = [maps["skeleton"][frame_id] for frame_id in common_ids]
    return load_skeleton_sequence(paths, flip=False, strategy=strategy).numpy()


def main() -> None:
    args = parse_args()
    manifest = args.manifest.resolve()
    source_cache = args.source_cache.resolve()
    output_cache = args.output_cache.resolve()
    source_metadata_path = source_cache / "metadata.json"
    if not source_metadata_path.is_file():
        raise FileNotFoundError(f"源缓存元数据不存在：{source_metadata_path}")

    source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    samples, expected_lengths = read_samples(manifest)
    sample_ids = [sample.sample_id for sample in samples]
    if sample_ids != source_metadata["sample_ids"]:
        raise ValueError("manifest 的样本及顺序与源缓存不一致，不能安全复用图像数组")
    source_lengths = [int(length) for _, length in source_metadata["offsets"]]
    if expected_lengths != source_lengths:
        raise ValueError("manifest 的帧数与源缓存不一致")

    total_frames = int(source_metadata["total_frames"])
    output_cache.mkdir(parents=True, exist_ok=True)
    allowed_existing = {"metadata.json"}
    unexpected = [path for path in output_cache.iterdir() if path.name not in allowed_existing]
    if unexpected:
        raise FileExistsError(f"输出缓存目录不是空目录：{unexpected[0]}")

    temp_path = output_cache / "skeleton_float32.npy.building"
    final_path = output_cache / "skeleton_float32.npy"
    temp_path.unlink(missing_ok=True)
    final_path.unlink(missing_ok=True)
    skeleton_cache = np.lib.format.open_memmap(
        temp_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_frames, 17, 4),
    )

    workers = max(1, int(args.workers))
    chunk_size = max(workers, int(args.chunk_size))
    cursor = 0
    started = time.time()
    print(
        f"重建 {len(samples)} 个 trial、{total_frames:,} 帧 Skeleton，"
        f"strategy={args.skeleton_strategy}，线程数={workers}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for chunk_start in range(0, len(samples), chunk_size):
            chunk = samples[chunk_start : chunk_start + chunk_size]
            work = ((sample, args.skeleton_strategy) for sample in chunk)
            for local_index, skeleton in enumerate(executor.map(process_sample, work)):
                sample_index = chunk_start + local_index
                expected = expected_lengths[sample_index]
                if len(skeleton) != expected:
                    raise RuntimeError(
                        f"{samples[sample_index].sample_id} 帧数变化："
                        f"manifest={expected}，Skeleton={len(skeleton)}"
                    )
                end = cursor + expected
                skeleton_cache[cursor:end] = skeleton
                cursor = end
            done = min(len(samples), chunk_start + len(chunk))
            if done == len(samples) or done % 512 < chunk_size:
                elapsed = time.time() - started
                rate = done / max(elapsed, 1e-6)
                eta = (len(samples) - done) / max(rate, 1e-6)
                print(
                    f"已完成 {done}/{len(samples)} trial，"
                    f"耗时 {elapsed / 60:.1f} 分，预计剩余 {eta / 60:.1f} 分",
                    flush=True,
                )

    if cursor != total_frames:
        raise RuntimeError(f"最终帧数不一致：{cursor} != {total_frames}")
    skeleton_cache.flush()
    mmap_handle = getattr(skeleton_cache, "_mmap", None)
    if mmap_handle is not None:
        mmap_handle.close()
    del skeleton_cache
    os.replace(temp_path, final_path)

    for filename in ("depth_uint8.npy", "ir_uint8.npy"):
        source = source_cache / filename
        target = output_cache / filename
        if target.exists():
            target.unlink()
        os.link(source, target)

    metadata = dict(source_metadata)
    metadata.update(
        {
            "version": 2,
            "manifest": str(manifest),
            "skeleton_strategy": args.skeleton_strategy,
            "source_image_cache": str(source_cache),
            "build_seconds": round(time.time() - started, 2),
        }
    )
    (output_cache / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"Tracking 缓存完成：{output_cache}，耗时 {metadata['build_seconds'] / 60:.1f} 分",
        flush=True,
    )


if __name__ == "__main__":
    main()
