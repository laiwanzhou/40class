from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUTS = {
    "v2": PROJECT_DIR / "runs" / "p46_unified_repair_v2",
    "v3": PROJECT_DIR / "runs" / "p46_unified_repair_v3_clean",
}
TRAIN_SCRIPTS = {
    "v2": PROJECT_DIR / "train_p46_unified_repair_v2.py",
    "v3": PROJECT_DIR / "train_p46_unified_repair_v3.py",
}


def parse_args(protocol: str) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            f"Run unified P46 {protocol} one Stage-A/Stage-B epoch per fresh Python process so "
            "Windows can reclaim variable-sized CPU/GPU allocations."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUTS[protocol])
    parser.add_argument("--stage-a-epochs", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    args, passthrough = parser.parse_known_args()
    forbidden = {
        "--resume",
        "--stage-a-resume",
        "--stop-after-stage-a-epoch",
        "--stop-after-epoch",
        "--output-dir",
        "--stage-a-epochs",
        "--epochs",
        "--smoke",
    }
    conflict = [value for value in passthrough if value.split("=", 1)[0] in forbidden]
    if conflict:
        raise ValueError(f"supervisor owns these arguments: {conflict}")
    if args.stage_a_epochs < 1 or args.epochs < 1:
        raise ValueError("stage-a-epochs and epochs must be positive")
    return args, passthrough


def load_resume_epoch(checkpoint_path: Path, epochs: int, protocol: str = "v2") -> int:
    checkpoint: dict[str, Any] = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    try:
        if checkpoint.get("stage") != f"P46_unified_repair_{protocol}_stageB":
            raise RuntimeError(f"incompatible {protocol} checkpoint: {checkpoint_path}")
        config = checkpoint.get("config", {})
        if int(config.get("stage_b_epochs", -1)) != epochs:
            raise RuntimeError(
                "existing checkpoint epoch protocol differs: "
                f"{config.get('stage_b_epochs')} != {epochs}"
            )
        return int(checkpoint["epoch"])
    finally:
        del checkpoint
        gc.collect()


def load_stage_a_epoch(
    checkpoint_path: Path, stage_a_epochs: int, epochs: int, protocol: str = "v2"
) -> int:
    checkpoint: dict[str, Any] = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    try:
        if checkpoint.get("stage") != f"P46_unified_repair_{protocol}_stageA":
            raise RuntimeError(
                f"incompatible {protocol} Stage-A checkpoint: {checkpoint_path}"
            )
        required = {
            "optimizer_state_dict",
            "scaler_state_dict",
            "rng_state",
            "history",
        }
        missing = required - set(checkpoint)
        if missing:
            raise RuntimeError(
                f"Stage-A checkpoint cannot resume exactly: missing {sorted(missing)}"
            )
        config = checkpoint.get("config", {})
        if (
            int(config.get("stage_a_epochs", -1)) != stage_a_epochs
            or int(config.get("stage_b_epochs", -1)) != epochs
        ):
            raise RuntimeError("existing Stage-A checkpoint protocol differs")
        return int(checkpoint["epoch"])
    finally:
        del checkpoint
        gc.collect()


def main(protocol: str = "v2") -> None:
    if protocol not in DEFAULT_OUTPUTS:
        raise ValueError(f"unsupported supervisor protocol: {protocol}")
    args, passthrough = parse_args(protocol)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint = output_dir / "last.pt"
    stage_a_checkpoint = output_dir / "stage_a_last.pt"
    completed_epoch = (
        load_resume_epoch(last_checkpoint, args.epochs, protocol)
        if last_checkpoint.is_file()
        else 0
    )
    if completed_epoch >= args.epochs:
        print(
            json.dumps(
                {"stage": "already_complete", "epoch": completed_epoch},
                ensure_ascii=False,
            ),
            flush=True,
        )
        return

    base_command = [
        sys.executable,
        str(TRAIN_SCRIPTS[protocol]),
        "--output-dir",
        str(output_dir),
        "--stage-a-epochs",
        str(args.stage_a_epochs),
        "--epochs",
        str(args.epochs),
        *passthrough,
    ]

    if not last_checkpoint.is_file():
        completed_stage_a = (
            load_stage_a_epoch(
                stage_a_checkpoint, args.stage_a_epochs, args.epochs, protocol
            )
            if stage_a_checkpoint.is_file()
            else 0
        )
        for target_stage_a in range(
            completed_stage_a + 1, args.stage_a_epochs + 1
        ):
            command = list(base_command)
            if target_stage_a > 1:
                command.extend(("--stage-a-resume", str(stage_a_checkpoint)))
            command.extend(
                ("--stop-after-stage-a-epoch", str(target_stage_a))
            )
            print(
                json.dumps(
                    {
                        "stage": "launch_stage_a_epoch_process",
                        "target_epoch": target_stage_a,
                        "resume": target_stage_a > 1,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            completed = subprocess.run(command, cwd=PROJECT_DIR, check=False)
            if completed.returncode != 0:
                raise SystemExit(completed.returncode)
            if not stage_a_checkpoint.is_file():
                raise RuntimeError(
                    "Stage-A epoch process exited without writing stage_a_last.pt"
                )

    for target_epoch in range(completed_epoch + 1, args.epochs + 1):
        command = list(base_command)
        if target_epoch > 1:
            command.extend(("--resume", str(last_checkpoint)))
        else:
            command.extend(("--stage-a-resume", str(stage_a_checkpoint)))
        if target_epoch < args.epochs:
            command.extend(("--stop-after-epoch", str(target_epoch)))
        print(
            json.dumps(
                {
                    "stage": "launch_epoch_process",
                    "target_epoch": target_epoch,
                    "resume": target_epoch > 1,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        completed = subprocess.run(command, cwd=PROJECT_DIR, check=False)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)
        if not last_checkpoint.is_file():
            raise RuntimeError("epoch process exited without writing last.pt")
        summary_path = output_dir / "summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("stop_reason") == "validation_plateau":
                print(
                    json.dumps(
                        {
                            "stage": "early_stopping_complete",
                            "epoch": summary.get("stage_b_epochs_completed"),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                return


if __name__ == "__main__":
    main()
