from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from aligned_data import AlignedSample, frame_map
from build_cache import read_all_samples
from depth_encoding import decode_jet_rgb, resize_decoded_depth


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PROJECT_DIR / "data" / "manifest.csv"
DEFAULT_SOURCE_CACHE = PROJECT_DIR / "cache" / "aligned_192x144"
DEFAULT_OUTPUT_CACHE = PROJECT_DIR / "cache" / "aligned_192x144_depth_decoded"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="先反解JET，再缩放并缓存Depth索引与valid mask")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-cache", type=Path, default=DEFAULT_SOURCE_CACHE)
    parser.add_argument("--output-cache", type=Path, default=DEFAULT_OUTPUT_CACHE)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--resume", action="store_true", help="从已写完的连续trial后继续")
    return parser.parse_args()


def decode_frame(path: Path, height: int, width: int) -> tuple[np.ndarray, np.ndarray, int]:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    depth, valid, repaired = decode_jet_rgb(rgb)
    depth, valid = resize_decoded_depth(depth, valid, height, width)
    return depth, valid, repaired


def process_sample(
    args: tuple[AlignedSample, int, int]
) -> tuple[np.ndarray, np.ndarray, int]:
    sample, height, width = args
    maps = {
        "depth": frame_map(sample.depth_dir, "depth"),
        "ir": frame_map(sample.ir_dir, "ir"),
        "skeleton": frame_map(sample.skeleton_dir, "skeleton"),
    }
    common_ids = sorted(set.intersection(*(set(mapping) for mapping in maps.values())))
    depth_frames: list[np.ndarray] = []
    valid_frames: list[np.ndarray] = []
    repaired = 0
    for key in common_ids:
        depth, valid, frame_repaired = decode_frame(maps["depth"][key], height, width)
        depth_frames.append(depth)
        valid_frames.append(valid)
        repaired += frame_repaired
    return np.stack(depth_frames), np.stack(valid_frames), repaired


def hardlink(source: Path, destination: Path) -> None:
    if destination.exists():
        destination.unlink()
    os.link(source, destination)


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    source_cache = args.source_cache.resolve()
    output_cache = args.output_cache.resolve()
    output_cache.mkdir(parents=True, exist_ok=True)

    source_metadata = json.loads((source_cache / "metadata.json").read_text(encoding="utf-8"))
    height = int(source_metadata["image_height"])
    width = int(source_metadata["image_width"])
    samples, expected_lengths = read_all_samples(manifest_path)
    if [sample.sample_id for sample in samples] != source_metadata["sample_ids"]:
        raise ValueError("manifest样本顺序与source cache不一致")
    total_frames = sum(expected_lengths)
    if total_frames != int(source_metadata["total_frames"]):
        raise ValueError("manifest总帧数与source cache不一致")

    shapes = {
        "depth_scalar_uint8.npy": (total_frames, height, width),
        "depth_valid_uint8.npy": (total_frames, height, width),
    }
    temp_paths = {name: output_cache / f"{name}.building" for name in shapes}
    can_resume = args.resume and all(path.is_file() for path in temp_paths.values())
    if can_resume:
        arrays = {
            "depth": np.load(temp_paths["depth_scalar_uint8.npy"], mmap_mode="r+"),
            "valid": np.load(temp_paths["depth_valid_uint8.npy"], mmap_mode="r+"),
        }
        for name, shape in (("depth", shapes["depth_scalar_uint8.npy"]), ("valid", shapes["depth_valid_uint8.npy"])):
            if tuple(arrays[name].shape) != shape:
                raise ValueError(f"断点缓存{name}形状错误：{arrays[name].shape} != {shape}")
    else:
        for path in temp_paths.values():
            path.unlink(missing_ok=True)
        arrays = {
            "depth": np.lib.format.open_memmap(
                temp_paths["depth_scalar_uint8.npy"],
                mode="w+",
                dtype=np.uint8,
                shape=shapes["depth_scalar_uint8.npy"],
            ),
            "valid": np.lib.format.open_memmap(
                temp_paths["depth_valid_uint8.npy"],
                mode="w+",
                dtype=np.uint8,
                shape=shapes["depth_valid_uint8.npy"],
            ),
        }

    workers = max(1, int(args.workers))
    chunk_size = max(workers, int(args.chunk_size))
    start_sample = 0
    cursor = 0
    if can_resume:
        for sample_index, length in enumerate(expected_lengths):
            end = cursor + length
            if not np.any(arrays["valid"][end - 1]):
                break
            start_sample = sample_index + 1
            cursor = end
        print(
            f"检测到断点：已完成连续 {start_sample}/{len(samples)} trial、{cursor:,}帧",
            flush=True,
        )
    repaired_pixels = 0
    started = time.time()
    print(
        f"准备反解 {len(samples)} 个trial、{total_frames:,}帧Depth，"
        f"输出={width}×{height}，线程数={workers}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for chunk_start in range(start_sample, len(samples), chunk_size):
            chunk = samples[chunk_start : chunk_start + chunk_size]
            work = ((sample, height, width) for sample in chunk)
            for local_index, (depth, valid, repaired) in enumerate(executor.map(process_sample, work)):
                sample_index = chunk_start + local_index
                expected = expected_lengths[sample_index]
                if len(depth) != expected or len(valid) != expected:
                    raise RuntimeError(
                        f"{samples[sample_index].sample_id}帧数变化："
                        f"manifest={expected}，实际={len(depth)}/{len(valid)}"
                    )
                end = cursor + expected
                arrays["depth"][cursor:end] = depth
                arrays["valid"][cursor:end] = valid
                cursor = end
                repaired_pixels += repaired

            done = min(len(samples), chunk_start + len(chunk))
            elapsed = time.time() - started
            rate = done / max(elapsed, 1e-6)
            eta = (len(samples) - done) / max(rate, 1e-6)
            print(
                f"已完成 {done:4d}/{len(samples)} trial（{cursor:,}/{total_frames:,}帧），"
                f"耗时 {elapsed / 60:.1f}分，预计剩余 {eta / 60:.1f}分",
                flush=True,
            )

    if cursor != total_frames:
        raise RuntimeError(f"最终帧数不一致：{cursor} != {total_frames}")
    for array in arrays.values():
        array.flush()
        mmap_handle = getattr(array, "_mmap", None)
        if mmap_handle is not None:
            mmap_handle.close()
    del array
    del arrays
    for name, temp_path in temp_paths.items():
        os.replace(temp_path, output_cache / name)

    linked_files = []
    for name in ("depth_uint8.npy", "ir_uint8.npy", "skeleton_float32.npy"):
        source = source_cache / name
        if source.is_file():
            hardlink(source, output_cache / name)
            linked_files.append(name)

    metadata = {
        "version": 3,
        "manifest": str(manifest_path),
        "source_cache": str(source_cache),
        "image_height": height,
        "image_width": width,
        "num_samples": len(samples),
        "total_frames": total_frames,
        "sample_ids": [sample.sample_id for sample in samples],
        "offsets": source_metadata["offsets"],
        "files": list(shapes) + linked_files,
        "depth_storage": "opencv_jet_index_with_valid_mask",
        "resize_order": "decode_original_then_mask_aware_bilinear_resize",
        "repaired_nonblack_source_pixels": repaired_pixels,
        "build_seconds": round(time.time() - started, 2),
    }
    (output_cache / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"Decoded Depth缓存完成：{output_cache}，"
        f"耗时{metadata['build_seconds'] / 60:.1f}分，修复像素={repaired_pixels}",
        flush=True,
    )


if __name__ == "__main__":
    main()
