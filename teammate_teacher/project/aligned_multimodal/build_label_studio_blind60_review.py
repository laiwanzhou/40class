from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from urllib.parse import quote

from PIL import Image

from aligned_data import frame_map
from build_label_studio_roi216 import (
    RAW_HEIGHT,
    RAW_WIDTH,
    contact_sheet,
    prediction_result,
    save_jpeg,
)


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
ROI_DIR = PROJECT_DIR / "data" / "local_roi_annotation_v2"
DEFAULT_AUDIT = ROI_DIR / "roi_annotation_audit.csv"
DEFAULT_MANIFEST = PROJECT_DIR / "data" / "manifest.csv"
DEFAULT_PRIVATE_INDEX = ROI_DIR / "label_studio_roi216" / "task_index_private.csv"
DEFAULT_OUTPUT = ROI_DIR / "label_studio_blind60_review_v2"
NUM_FRAMES = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a second-pass review of the 60 formerly blind ROI tasks. "
            "The machine box and the first blind human box are both visible."
        )
    )
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--private-index", type=Path, default=DEFAULT_PRIVATE_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def local_file_url(path: Path) -> str:
    relative = path.resolve().relative_to(REPO_DIR.resolve()).as_posix()
    return f"/data/local-files/?d={quote(relative, safe='/')}"


def box(row: dict[str, str], prefix: str) -> list[float]:
    values = [
        float(row[f"{prefix}_x0"]),
        float(row[f"{prefix}_y0"]),
        float(row[f"{prefix}_x1"]),
        float(row[f"{prefix}_y1"]),
    ]
    if not (
        0 <= values[0] < values[2] < RAW_WIDTH
        and 0 <= values[1] < values[3] < RAW_HEIGHT
    ):
        raise ValueError(f"{row['sample_id']}: invalid {prefix} box {values}")
    return values


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    audit_rows = [
        row
        for row in read_csv(args.audit.resolve())
        if row["annotation_mode"] == "blind"
    ]
    audit_rows.sort(key=lambda row: int(row["selection_index"]))
    if len(audit_rows) != 60:
        raise ValueError("The second-pass review must contain exactly 60 blind tasks")
    manifest = {row["sample_id"]: row for row in read_csv(args.manifest.resolve())}
    private_by_sample = {
        row["sample_id"]: row
        for row in read_csv(args.private_index.resolve())
        if row["annotation_mode"] == "blind"
    }
    output_dir = args.output_dir.resolve()
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    tasks: list[dict[str, object]] = []
    index_rows: list[dict[str, object]] = []
    for review_index, row in enumerate(audit_rows, start=1):
        sample_id = row["sample_id"]
        private = private_by_sample[sample_id]
        positions = [int(value) for value in json.loads(private["inference_positions"])]
        frame_ids = [str(value) for value in json.loads(private["inference_frame_ids"])]
        if len(positions) != NUM_FRAMES or len(frame_ids) != NUM_FRAMES:
            raise ValueError(f"{sample_id}: expected 12 inference frames")
        depth = frame_map(Path(manifest[sample_id]["depth_dir"]), "depth")
        frames: list[Image.Image] = []
        for frame_id in frame_ids:
            with Image.open(depth[frame_id]) as image:
                frame = image.convert("RGB")
            if frame.size != (RAW_WIDTH, RAW_HEIGHT):
                raise ValueError(f"{sample_id}: unexpected Depth size {frame.size}")
            frames.append(frame)
        machine_box = box(row, "original")
        first_human_box = box(row, "human")
        stem = f"roi60_review_{review_index:03d}"
        raw_path = images_dir / f"{stem}_roi_frame.jpg"
        full12_path = images_dir / f"{stem}_depth12.jpg"
        machine_path = images_dir / f"{stem}_machine_overlay12.jpg"
        human_path = images_dir / f"{stem}_blind_human_overlay12.jpg"
        save_jpeg(frames[NUM_FRAMES // 2], raw_path, quality=94)
        save_jpeg(
            contact_sheet(
                frames,
                positions,
                "FULL DEPTH 12 | exact inference frames",
                (0, 190, 255),
            ),
            full12_path,
            quality=88,
        )
        save_jpeg(
            contact_sheet(
                frames,
                positions,
                "MACHINE ORIGINAL BOX 12 | review this box",
                (0, 255, 80),
                boxes=[machine_box] * NUM_FRAMES,
            ),
            machine_path,
            quality=88,
        )
        save_jpeg(
            contact_sheet(
                frames,
                positions,
                "FIRST BLIND HUMAN BOX 12 | reference, not automatic truth",
                (255, 0, 190),
                boxes=[first_human_box] * NUM_FRAMES,
            ),
            human_path,
            quality=88,
        )
        task_key = f"roi60_review_{review_index:03d}"
        tasks.append(
            {
                "data": {
                    "task_key": task_key,
                    "sample_info": (
                        f"机器原框复核 {review_index}/60 | "
                        "真实类别与模型分类预测仍然隐藏"
                    ),
                    "roi_frame_image": local_file_url(raw_path),
                    "depth12_image": local_file_url(full12_path),
                    "machine_overlay_image": local_file_url(machine_path),
                    "blind_human_overlay_image": local_file_url(human_path),
                },
                "predictions": [
                    {
                        "model_version": "motion_bbox_all_frames_v1_review",
                        "score": 0.5,
                        "result": [prediction_result(machine_box)],
                    }
                ],
            }
        )
        index_rows.append(
            {
                "review_index": review_index,
                "task_key": task_key,
                "sample_id": sample_id,
                "fold": int(row["fold"]),
                "class_id_private": int(row["class_id"]),
                "class_name_private": row["class_name"],
                "user_id": row["user_id"],
                "trial_id": row["trial_id"],
                "original_fallback": int(row["original_fallback"]),
                "machine_bbox_raw_private": json.dumps(machine_box),
                "first_blind_human_bbox_raw_private": json.dumps(first_human_box),
                "first_blind_machine_iou": float(row["auto_human_iou"]),
                "roi_frame_image": raw_path.relative_to(output_dir),
                "depth12_image": full12_path.relative_to(output_dir),
                "machine_overlay_image": machine_path.relative_to(output_dir),
                "blind_human_overlay_image": human_path.relative_to(output_dir),
            }
        )
        if review_index % 10 == 0:
            print(f"ROI60 review assets {review_index}/60", flush=True)
    (output_dir / "tasks.json").write_text(
        json.dumps(tasks, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(output_dir / "task_index_private.csv", index_rows)
    label_config = """<View>
  <Header value="$sample_info"/>
  <Text name="instruction" value="这次专门复核机器原框。上方绿色预框是机器原框，可以保留或调整；下面同时展示机器框和第一次盲画人工框在真实12帧上的覆盖。两种框没有天然唯一答案，请直接按动作证据覆盖情况判断。"/>

  <Header value="A｜在原始 Depth 上查看或调整机器原框"/>
  <Image name="roi_frame" value="$roi_frame_image" zoom="true"/>
  <RectangleLabels name="roi_box" toName="roi_frame" strokeWidth="4">
    <Label value="Action ROI" background="#00E676"/>
  </RectangleLabels>

  <Header value="B｜模型实际使用的12帧完整 Depth"/>
  <Image name="depth12" value="$depth12_image" zoom="true"/>
  <Header value="C｜机器原框在12帧上的覆盖"/>
  <Image name="machine_overlay" value="$machine_overlay_image" zoom="true"/>
  <Header value="D｜第一次盲画人工框在12帧上的覆盖（仅供对比）"/>
  <Image name="blind_human_overlay" value="$blind_human_overlay_image" zoom="true"/>

  <Header value="1. 机器原框本身应该如何评价？"/>
  <Choices name="machine_box_assessment" toName="roi_frame" choice="single-radio" required="true">
    <Choice value="machine_good（机器原框可直接使用）"/>
    <Choice value="machine_usable_human_better（机器原框基本可用，但第一次人工框或轻微调整更好）"/>
    <Choice value="machine_bad（机器原框明显漏掉关键区域、范围严重不当或框错位置）"/>
  </Choices>

  <Header value="2. 最终建议使用哪个框？"/>
  <Choices name="final_box_source" toName="roi_frame" choice="single-radio" required="true">
    <Choice value="keep_machine（保留机器原框）"/>
    <Choice value="use_first_blind_human（采用第一次盲画人工框，无需在上图重画）"/>
    <Choice value="use_current_adjustment（采用这次在上图调整后的框）"/>
  </Choices>

  <Header value="3. 备注（可选）"/>
  <TextArea name="free_note" toName="roi_frame" rows="4" placeholder="例如：两种框都合理、镜中人物、必须保留较大上下文等。"/>
</View>
"""
    (output_dir / "label_config.xml").write_text(label_config, encoding="utf-8")
    summary = {
        "tasks": 60,
        "predictions": 60,
        "classes_hidden": True,
        "classification_predictions_hidden": True,
        "machine_boxes_visible": True,
        "first_blind_human_boxes_visible": True,
        "source_blind_annotations_preserved": True,
    }
    (output_dir / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
