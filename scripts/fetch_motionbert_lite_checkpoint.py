from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import urllib.request


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.motionbert_p6b_config import (
    load_motionbert_p6b_config,
    project_path,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_motionbert_checkpoint(
    path: Path, *, expected_bytes: int, expected_sha256: str
) -> None:
    if (
        not path.is_file()
        or path.stat().st_size != expected_bytes
        or file_sha256(path) != expected_sha256
    ):
        raise RuntimeError(f"MotionBERT checkpoint provenance mismatch: {path}")


def fetch_motionbert_lite_checkpoint(config_path: Path) -> dict[str, object]:
    config = load_motionbert_p6b_config(config_path)
    checkpoint = config["checkpoint"]
    target = project_path(str(checkpoint["path"])).resolve()
    expected_bytes = int(checkpoint["bytes"])
    expected_sha256 = str(checkpoint["sha256"])
    if target.exists():
        verify_motionbert_checkpoint(
            target,
            expected_bytes=expected_bytes,
            expected_sha256=expected_sha256,
        )
        return {"status": "reused", "path": str(target), "bytes": expected_bytes, "sha256": expected_sha256}
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    upstream = config["upstream"]
    url = (
        "https://huggingface.co/"
        f"{upstream['weight_repository']}/resolve/{upstream['weight_revision']}/"
        f"{upstream['weight_path']}"
    )
    try:
        urllib.request.urlretrieve(url, temporary)
        verify_motionbert_checkpoint(
            temporary,
            expected_bytes=expected_bytes,
            expected_sha256=expected_sha256,
        )
        temporary.replace(target)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return {"status": "downloaded", "path": str(target), "bytes": expected_bytes, "sha256": expected_sha256}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch pinned MotionBERT-Lite checkpoint")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/experiments/motionbert_lite_skeleton_expert_p6b.yaml",
    )
    args = parser.parse_args()
    print(fetch_motionbert_lite_checkpoint(args.config.resolve()))


if __name__ == "__main__":
    main()
