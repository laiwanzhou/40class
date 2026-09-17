"""Resumable ranged downloader for the official EPIC-KITCHENS SlowFast checkpoint."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import os
import time
from pathlib import Path

import requests


URL = "https://www.dropbox.com/s/uxb6i2xkn91xqzi/SlowFast.pyth?dl=1"
TOTAL = 276_771_081


def download_part(
    part_dir: Path, index: int, start: int, end: int, retries: int
) -> tuple[int, int]:
    path = part_dir / f"part_{index:04d}.bin"
    expected = end - start + 1
    if path.exists() and path.stat().st_size == expected:
        return index, expected
    for attempt in range(retries):
        try:
            response = requests.get(
                URL,
                headers={
                    "Range": f"bytes={start}-{end}",
                    "Accept-Encoding": "identity",
                },
                timeout=(30, 300),
            )
            response.raise_for_status()
            if response.status_code != 206:
                raise RuntimeError(f"part {index}: expected 206, got {response.status_code}")
            expected_range = f"bytes {start}-{end}/{TOTAL}"
            if response.headers.get("Content-Range") != expected_range:
                raise RuntimeError(
                    f"part {index}: Content-Range={response.headers.get('Content-Range')}"
                )
            if len(response.content) != expected:
                raise RuntimeError(
                    f"part {index}: expected {expected}, received {len(response.content)}"
                )
            temporary = path.with_suffix(".building")
            temporary.write_bytes(response.content)
            temporary.replace(path)
            return index, expected
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(min(2**attempt, 15))
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--part-mib", type=int, default=4)
    parser.add_argument("--retries", type=int, default=8)
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    part_dir = output.parent / (output.name + ".parts")
    part_dir.mkdir(parents=True, exist_ok=True)
    part_size = args.part_mib * 1024 * 1024
    ranges = [
        (index, start, min(start + part_size - 1, TOTAL - 1))
        for index, start in enumerate(range(0, TOTAL, part_size))
    ]
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_part, part_dir, index, start, end, args.retries
            ): index
            for index, start, end in ranges
        }
        for future in concurrent.futures.as_completed(futures):
            index, size = future.result()
            completed += size
            print(
                f"completed part={index:04d} bytes={completed}/{TOTAL}", flush=True
            )
    temporary = output.with_suffix(output.suffix + ".building")
    digest = hashlib.sha256()
    with temporary.open("wb") as destination:
        for index, start, end in ranges:
            path = part_dir / f"part_{index:04d}.bin"
            with path.open("rb") as source:
                while True:
                    block = source.read(1024 * 1024)
                    if not block:
                        break
                    destination.write(block)
                    digest.update(block)
    if temporary.stat().st_size != TOTAL:
        raise RuntimeError(
            f"assembled size {temporary.stat().st_size} != official size {TOTAL}"
        )
    temporary.replace(output)
    print(f"output={output}")
    print(f"bytes={output.stat().st_size}")
    print(f"sha256={digest.hexdigest()}")


if __name__ == "__main__":
    main()
