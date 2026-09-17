from __future__ import annotations

import argparse
import csv
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

from PIL import Image

from aligned_data import frame_map
from build_label_studio_roi216 import contact_sheet, save_jpeg
from local_roi_data import sample_positions, standardize_box


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
DEFAULT_PREDICTIONS = (
    PROJECT_DIR
    / "runs"
    / "p13_hard_roi_locator_v1"
    / "annotation500_locator_predictions.csv"
)
DEFAULT_MANIFEST = PROJECT_DIR / "data" / "manifest.csv"
DEFAULT_OUTPUT = (
    PROJECT_DIR / "data" / "hard_local_v1" / "label_studio_annotation500"
)
NUM_FRAMES = 12
DEPTH_WIDTH = 640
DEPTH_HEIGHT = 480
CONTEXT = 0.125


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build 500 joint Depth/Thermal ROI Label Studio tasks"
    )
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def local_file_url(path: Path) -> str:
    relative = path.resolve().relative_to(REPO_DIR.resolve()).as_posix()
    return f"/data/local-files/?d={quote(relative, safe='/')}"


def load_depth_frames(
    manifest_row: dict[str, str],
) -> tuple[list[Image.Image], list[int], list[str]]:
    maps = {
        "depth": frame_map(Path(manifest_row["depth_dir"]), "depth"),
        "ir": frame_map(Path(manifest_row["ir_dir"]), "ir"),
        "skeleton": frame_map(Path(manifest_row["skeleton_dir"]), "skeleton"),
    }
    common = sorted(set.intersection(*(set(mapping) for mapping in maps.values())))
    positions = sample_positions(len(common), NUM_FRAMES, augment=False)
    frame_ids = [common[position] for position in positions]
    frames: list[Image.Image] = []
    for frame_id in frame_ids:
        with Image.open(maps["depth"][frame_id]) as image:
            frame = image.convert("RGB")
        if frame.size != (DEPTH_WIDTH, DEPTH_HEIGHT):
            raise ValueError(f"Unexpected Depth size {frame.size}")
        frames.append(frame)
    return frames, positions, frame_ids


def load_thermal_frames(
    thermal_path: Path,
) -> tuple[list[Image.Image], list[int], list[str]]:
    paths = sorted(
        path
        for path in thermal_path.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not paths:
        raise ValueError(f"No Thermal frames in {thermal_path}")
    positions = sample_positions(len(paths), NUM_FRAMES, augment=False)
    selected = [paths[position] for position in positions]
    frames: list[Image.Image] = []
    size: tuple[int, int] | None = None
    for path in selected:
        with Image.open(path) as image:
            frame = image.convert("RGB")
        if size is None:
            size = frame.size
        elif frame.size != size:
            raise ValueError(f"Inconsistent Thermal size in {thermal_path}")
        frames.append(frame)
    return frames, positions, [path.name for path in selected]


def map_box(
    box: list[float],
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> list[float]:
    source_width, source_height = source_size
    target_width, target_height = target_size
    x0, y0, x1, y1 = box
    return [
        x0 * target_width / source_width,
        y0 * target_height / source_height,
        (x1 + 1) * target_width / source_width - 1,
        (y1 + 1) * target_height / source_height - 1,
    ]


def crop_frames(
    frames: list[Image.Image],
    raw_box: list[float],
) -> tuple[list[Image.Image], list[int]]:
    width, height = frames[0].size
    standardized = standardize_box(
        tuple(raw_box),
        width,
        height,
        context=CONTEXT,
        target_ratio=4.0 / 3.0,
    )
    cropped = [
        frame.crop(standardized).resize(
            (320, 240), Image.Resampling.BILINEAR
        )
        for frame in frames
    ]
    return cropped, list(standardized)


def rectangle_prediction(
    from_name: str,
    to_name: str,
    label: str,
    box: list[float],
    width: int,
    height: int,
    result_id: str,
) -> dict[str, object]:
    x0, y0, x1, y1 = box
    return {
        "id": result_id,
        "from_name": from_name,
        "to_name": to_name,
        "type": "rectanglelabels",
        "original_width": width,
        "original_height": height,
        "image_rotation": 0,
        "value": {
            "x": 100.0 * x0 / width,
            "y": 100.0 * y0 / height,
            "width": 100.0 * (x1 - x0 + 1) / width,
            "height": 100.0 * (y1 - y0 + 1) / height,
            "rotation": 0,
            "rectanglelabels": [label],
        },
    }


def make_task(
    index: int,
    row: dict[str, str],
    manifest_row: dict[str, str],
    images_dir: Path,
    output_dir: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    sample_id = row["sample_id"]
    depth_frames, depth_positions, depth_frame_ids = load_depth_frames(manifest_row)
    thermal_frames, thermal_positions, thermal_frame_ids = load_thermal_frames(
        Path(row["thermal_path"])
    )
    machine_depth = [
        float(row[field])
        for field in ("machine_x0", "machine_y0", "machine_x1", "machine_y1")
    ]
    thermal_size = thermal_frames[0].size
    mapped_thermal = map_box(
        machine_depth,
        (DEPTH_WIDTH, DEPTH_HEIGHT),
        thermal_size,
    )
    depth_local, depth_standardized = crop_frames(depth_frames, machine_depth)
    thermal_local, thermal_standardized = crop_frames(
        thermal_frames, mapped_thermal
    )
    stem = f"hard_local_{index:03d}"
    paths = {
        "depth_roi_frame": images_dir / f"{stem}_depth_roi_frame.jpg",
        "thermal_roi_frame": images_dir / f"{stem}_thermal_roi_frame.jpg",
        "depth_full12": images_dir / f"{stem}_depth_full12.jpg",
        "depth_box12": images_dir / f"{stem}_depth_box12.jpg",
        "depth_local12": images_dir / f"{stem}_depth_local12.jpg",
        "thermal_full12": images_dir / f"{stem}_thermal_full12.jpg",
        "thermal_box12": images_dir / f"{stem}_thermal_box12.jpg",
        "thermal_local12": images_dir / f"{stem}_thermal_local12.jpg",
    }
    save_jpeg(depth_frames[NUM_FRAMES // 2], paths["depth_roi_frame"], quality=94)
    save_jpeg(
        thermal_frames[NUM_FRAMES // 2],
        paths["thermal_roi_frame"],
        quality=94,
    )
    save_jpeg(
        contact_sheet(
            depth_frames,
            depth_positions,
            "DEPTH FULL 12 | exact current-model sampling",
            (0, 190, 255),
        ),
        paths["depth_full12"],
        quality=84,
    )
    save_jpeg(
        contact_sheet(
            depth_frames,
            depth_positions,
            "DEPTH MACHINE ROI 12 | green = locator v1",
            (0, 255, 80),
            boxes=[machine_depth] * NUM_FRAMES,
        ),
        paths["depth_box12"],
        quality=84,
    )
    save_jpeg(
        contact_sheet(
            depth_local,
            depth_positions,
            "DEPTH LOCAL 12 | 12.5% padding + 4:3",
            (255, 80, 210),
        ),
        paths["depth_local12"],
        quality=84,
    )
    save_jpeg(
        contact_sheet(
            thermal_frames,
            thermal_positions,
            "THERMAL FULL 12 | normalized trial progress",
            (255, 165, 0),
        ),
        paths["thermal_full12"],
        quality=84,
    )
    save_jpeg(
        contact_sheet(
            thermal_frames,
            thermal_positions,
            "THERMAL MAPPED ROI 12 | normalized Depth ROI",
            (0, 255, 80),
            boxes=[mapped_thermal] * NUM_FRAMES,
        ),
        paths["thermal_box12"],
        quality=84,
    )
    save_jpeg(
        contact_sheet(
            thermal_local,
            thermal_positions,
            "THERMAL LOCAL 12 | mapped ROI + 12.5% padding + 4:3",
            (255, 80, 210),
        ),
        paths["thermal_local12"],
        quality=84,
    )
    task_key = f"hard_local_500_{index:03d}"
    task = {
        "data": {
            "task_key": task_key,
            "sample_info": (
                f"{index}/500 | 真实类别：{row['true_class_name']} | "
                f"样本组：{row['selection_category']} | "
                f"fallback={row['motion_fallback']}"
            ),
            **{
                name: local_file_url(path)
                for name, path in paths.items()
            },
        },
        "predictions": [
            {
                "model_version": "hard_action_roi_locator_v1_all216",
                "score": max(
                    0.0,
                    min(1.0, 1.0 - float(row["locator_v1_uncertainty"])),
                ),
                "result": [
                    rectangle_prediction(
                        "depth_roi_box",
                        "depth_roi_frame",
                        "Depth Action ROI",
                        machine_depth,
                        DEPTH_WIDTH,
                        DEPTH_HEIGHT,
                        "depth_machine_roi",
                    ),
                    rectangle_prediction(
                        "thermal_roi_box",
                        "thermal_roi_frame",
                        "Thermal Action ROI",
                        mapped_thermal,
                        thermal_size[0],
                        thermal_size[1],
                        "thermal_mapped_roi",
                    ),
                ],
            }
        ],
    }
    private = {
        "annotation_index": index,
        "task_key": task_key,
        "sample_id": sample_id,
        "fold": int(row["fold"]),
        "class_id": int(row["true_class_id"]),
        "class_name": row["true_class_name"],
        "user_id": row["user_id"],
        "trial_id": row["trial_id"],
        "selection_category": row["selection_category"],
        "selection_reasons": row["selection_reasons"],
        "motion_fallback": int(row["motion_fallback"]),
        "locator_v1_machine_source": row["locator_v1_machine_source"],
        "locator_v1_uncertainty": float(row["locator_v1_uncertainty"]),
        "depth_width": DEPTH_WIDTH,
        "depth_height": DEPTH_HEIGHT,
        "depth_machine_bbox": json.dumps(machine_depth),
        "depth_standardized_bbox": json.dumps(depth_standardized),
        "depth_frame_positions": json.dumps(depth_positions),
        "depth_frame_ids": json.dumps(depth_frame_ids),
        "thermal_width": thermal_size[0],
        "thermal_height": thermal_size[1],
        "thermal_mapped_bbox": json.dumps(mapped_thermal),
        "thermal_standardized_bbox": json.dumps(thermal_standardized),
        "thermal_frame_positions": json.dumps(thermal_positions),
        "thermal_frame_ids": json.dumps(thermal_frame_ids),
        **{
            f"{name}_path": str(path.relative_to(output_dir))
            for name, path in paths.items()
        },
    }
    return task, private


LABEL_CONFIG = """<View>
  <Header value="$sample_info"/>
  <Text name="instruction" value="先检查Depth机器框。直接接受时不要动框；小幅调整或严重错误时，直接拖动/重画Depth框。Thermal框是Depth归一化映射结果；只有选择“需要单独调整”时才修改Thermal框。"/>

  <View style="display:flex; gap:18px; align-items:flex-start;">
    <View style="flex:1;">
      <Header value="A｜Depth中间帧：绿色预框可直接接受、调整或重画"/>
      <Image name="depth_roi_frame" value="$depth_roi_frame" zoom="true"/>
      <RectangleLabels name="depth_roi_box" toName="depth_roi_frame" strokeWidth="4">
        <Label value="Depth Action ROI" background="#00E676"/>
      </RectangleLabels>
    </View>
    <View style="flex:1;">
      <Header value="B｜Thermal中间帧：绿色预框来自Depth归一化映射"/>
      <Image name="thermal_roi_frame" value="$thermal_roi_frame" zoom="true"/>
      <RectangleLabels name="thermal_roi_box" toName="thermal_roi_frame" strokeWidth="4">
        <Label value="Thermal Action ROI" background="#FF9800"/>
      </RectangleLabels>
    </View>
  </View>

  <Header value="1｜Depth机器框结论"/>
  <Choices name="depth_box_assessment" toName="depth_roi_frame" choice="single-radio" required="true">
    <Choice value="direct_accept（直接接受：原框无需修改）"/>
    <Choice value="minor_adjustment（小幅调整：原框基本正确，仅微调边界）"/>
    <Choice value="severe_error_redraw（严重错误：漏关键区域、错位或范围严重不当，已重画）"/>
  </Choices>

  <Header value="2｜Thermal映射框结论"/>
  <Choices name="thermal_mapping_assessment" toName="thermal_roi_frame" choice="single-radio" required="true">
    <Choice value="direct_usable（直接可用：Depth框按比例映射后即可使用）"/>
    <Choice value="usable_with_padding（加padding后可用：中心基本正确，只需扩大上下文）"/>
    <Choice value="needs_manual_adjustment（需要单独调整：请修改上面的Thermal框）"/>
    <Choice value="unusable（无法使用：视角、缺失或画面质量导致不能可靠定位）"/>
  </Choices>

  <Header value="3｜一个trial级框能否覆盖12帧动作范围"/>
  <Choices name="single_box_temporally_valid" toName="depth_roi_frame" choice="single-radio" required="true">
    <Choice value="valid（可以覆盖12帧中的关键动作证据）"/>
    <Choice value="invalid（单框无法可靠覆盖，动作范围或主体变化过大）"/>
  </Choices>

  <View style="display:flex; gap:12px; align-items:flex-start;">
    <View style="flex:1;"><Header value="Depth Full 12"/><Image name="depth_full12_view" value="$depth_full12" zoom="true"/></View>
    <View style="flex:1;"><Header value="Depth机器框 12"/><Image name="depth_box12_view" value="$depth_box12" zoom="true"/></View>
    <View style="flex:1;"><Header value="Depth Local Crop 12"/><Image name="depth_local12_view" value="$depth_local12" zoom="true"/></View>
  </View>
  <View style="display:flex; gap:12px; align-items:flex-start;">
    <View style="flex:1;"><Header value="Thermal Full 12"/><Image name="thermal_full12_view" value="$thermal_full12" zoom="true"/></View>
    <View style="flex:1;"><Header value="Thermal映射框 12"/><Image name="thermal_box12_view" value="$thermal_box12" zoom="true"/></View>
    <View style="flex:1;"><Header value="Thermal Local Crop 12"/><Image name="thermal_local12_view" value="$thermal_local12" zoom="true"/></View>
  </View>

  <Header value="4｜自由备注（可选）"/>
  <TextArea name="free_note" toName="depth_roi_frame" rows="4" placeholder="多人、遮挡、镜中人影、关键物体不可见、Thermal特殊问题等。"/>
</View>
"""


def main() -> None:
    args = parse_args()
    rows = read_csv(args.predictions.resolve())
    if len(rows) != 500:
        raise ValueError(f"Expected 500 locator rows, got {len(rows)}")
    rows.sort(key=lambda row: int(row["annotation_index"]))
    manifest = {row["sample_id"]: row for row in read_csv(args.manifest.resolve())}
    output_dir = args.output_dir.resolve()
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    tasks: list[dict[str, object] | None] = [None] * len(rows)
    private_rows: list[dict[str, object] | None] = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=int(args.workers)) as executor:
        futures = {
            executor.submit(
                make_task,
                index,
                row,
                manifest[row["sample_id"]],
                images_dir,
                output_dir,
            ): index
            for index, row in enumerate(rows, start=1)
        }
        completed = 0
        for future in as_completed(futures):
            index = futures[future]
            task, private = future.result()
            tasks[index - 1] = task
            private_rows[index - 1] = private
            completed += 1
            if completed % 10 == 0 or completed == len(rows):
                print(f"hard-local Label Studio assets {completed}/500", flush=True)
    if any(task is None for task in tasks) or any(row is None for row in private_rows):
        raise AssertionError("Asset build did not fill all 500 rows")
    final_tasks = [task for task in tasks if task is not None]
    final_private = [row for row in private_rows if row is not None]
    (output_dir / "tasks.json").write_text(
        json.dumps(final_tasks, ensure_ascii=False), encoding="utf-8"
    )
    with (output_dir / "task_index_private.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(final_private[0]))
        writer.writeheader()
        writer.writerows(final_private)
    (output_dir / "label_config.xml").write_text(LABEL_CONFIG, encoding="utf-8")
    image_files = list(images_dir.glob("*.jpg"))
    report = {
        "tasks": len(final_tasks),
        "predictions": len(final_tasks),
        "images": len(image_files),
        "image_bytes": sum(path.stat().st_size for path in image_files),
        "depth_sampling": "Exact 12 validation/inference positions",
        "thermal_sampling": "12 normalized-progress positions within each Thermal trial",
        "depth_local": "raw 640x480 -> ROI -> 12.5% padding -> 4:3 -> 320x240 display",
        "thermal_mapping": (
            "Depth raw bbox normalized by 640x480 and directly mapped to Thermal size; "
            "manual Thermal adjustment is allowed in Label Studio"
        ),
        "model_predictions_hidden": True,
        "true_class_visible_for_action-ROI_annotation": True,
    }
    (output_dir / "build_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
