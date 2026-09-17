from __future__ import annotations

import argparse

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache the fixed P46 VideoMAE backbone.")
    parser.add_argument(
        "--model", default="MCG-NJU/videomae-base-finetuned-kinetics"
    )
    args = parser.parse_args()
    path = snapshot_download(
        args.model,
        allow_patterns=("*.json", "*.safetensors", "pytorch_model.bin", "*.txt"),
    )
    print(path, flush=True)


if __name__ == "__main__":
    main()
