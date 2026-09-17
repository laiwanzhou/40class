from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from urllib.parse import quote

import numpy as np
from PIL import Image, ImageDraw

from aligned_data import AlignedMultimodalDataset, frame_map
from audit_motion_crop import analyse_trial


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
DEFAULT_DATA_DIR = PROJECT_DIR / "data" / "local_roi_annotation_v2"
DEFAULT_SELECTION = DEFAULT_DATA_DIR / "selection.csv"
DEFAULT_MANIFEST = PROJECT_DIR / "data" / "manifest.csv"
DEFAULT_PRIOR_ANNOTATIONS = (
    PROJECT_DIR
    / "data"
    / "local_action_audit_v1"
    / "annotations_temporal_pilot30.csv"
)
DEFAULT_PRIOR_SELECTION = (
    PROJECT_DIR
    / "data"
    / "local_action_audit_v1"
    / "label_studio_temporal_pilot30_v2"
    / "selection.csv"
)
DEFAULT_PRIOR_MANUAL = (
    PROJECT_DIR
    / "data"
    / "local_action_audit_v1"
    / "temporal_pilot30_manual_boxes.csv"
)
KAGGLE_LABEL_STUDIO_STATE = (
    PROJECT_DIR
    / "data"
    / "local_action_audit_v1"
    / "label_studio_pilot30"
    / "ls_state"
)
NUM_FRAMES = 12
RAW_WIDTH = 640
RAW_HEIGHT = 480
TILE_WIDTH = 320
TILE_HEIGHT = 240


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an independent Label Studio ROI project. It contains 60 blind "
            "tasks and 126 new correction tasks; the completed Pilot30 labels are "
            "ingested separately, so the frozen annotation set still totals 216."
        )
    )
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--prior-annotations", type=Path, default=DEFAULT_PRIOR_ANNOTATIONS
    )
    parser.add_argument("--prior-selection", type=Path, default=DEFAULT_PRIOR_SELECTION)
    parser.add_argument("--prior-manual", type=Path, default=DEFAULT_PRIOR_MANUAL)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DATA_DIR / "label_studio_roi216",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def inference_positions(length: int, count: int = NUM_FRAMES) -> list[int]:
    if length <= 0:
        raise ValueError("Frame sequence must not be empty")
    if length <= count:
        return (
            np.rint(np.linspace(0, length - 1, count, dtype=np.float64))
            .astype(np.int64)
            .tolist()
        )
    boundaries = np.floor(
        np.linspace(0, length, count + 1, dtype=np.float64)
    ).astype(np.int64)
    positions: list[int] = []
    for index in range(count):
        start = int(boundaries[index])
        end = max(start + 1, int(boundaries[index + 1]))
        positions.append(min(length - 1, (start + end - 1) // 2))
    return positions


def assert_sampling_matches_training(lengths: set[int]) -> None:
    dataset = object.__new__(AlignedMultimodalDataset)
    dataset.num_frames = NUM_FRAMES
    dataset.augment = False
    for length in sorted(lengths):
        expected = dataset._sample_positions(length)
        actual = inference_positions(length)
        if expected != actual:
            raise AssertionError(
                f"Sampling mismatch at length={length}: {actual} != {expected}"
            )


def local_file_url(path: Path) -> str:
    relative = path.resolve().relative_to(REPO_DIR.resolve()).as_posix()
    return f"/data/local-files/?d={quote(relative, safe='/')}"


def map_box(
    box: list[float],
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> list[float]:
    x0, y0, x1, y1 = box
    return [
        x0 * target_width / source_width,
        y0 * target_height / source_height,
        (x1 + 1) * target_width / source_width - 1,
        (y1 + 1) * target_height / source_height - 1,
    ]


def crop_and_resize(
    frame: Image.Image,
    box: list[float],
    output_size: tuple[int, int] = (TILE_WIDTH, TILE_HEIGHT),
) -> Image.Image:
    x0, y0, x1, y1 = box
    left = max(0, int(math.floor(x0)))
    top = max(0, int(math.floor(y0)))
    right = min(frame.width, int(math.ceil(x1 + 1)))
    bottom = min(frame.height, int(math.ceil(y1 + 1)))
    if right <= left or bottom <= top:
        raise ValueError(f"Invalid crop {box} for image {frame.size}")
    return frame.crop((left, top, right, bottom)).resize(
        output_size,
        Image.Resampling.BILINEAR,
    )


def contact_sheet(
    frames: list[Image.Image],
    positions: list[int],
    title: str,
    border: tuple[int, int, int],
    boxes: list[list[float] | None] | None = None,
) -> Image.Image:
    if len(frames) != NUM_FRAMES or len(positions) != NUM_FRAMES:
        raise ValueError("Contact sheet requires exactly 12 inference frames")
    columns, rows = 4, 3
    title_height, label_height, gap = 34, 20, 5
    width = columns * TILE_WIDTH + (columns + 1) * gap
    height = title_height + rows * (label_height + TILE_HEIGHT) + (rows + 1) * gap
    canvas = Image.new("RGB", (width, height), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), title, fill=(245, 245, 245))
    for frame_index, (frame, position) in enumerate(zip(frames, positions)):
        row, column = divmod(frame_index, columns)
        left = gap + column * (TILE_WIDTH + gap)
        top = title_height + gap + row * (label_height + TILE_HEIGHT + gap)
        draw.rectangle(
            (left, top, left + TILE_WIDTH - 1, top + label_height + TILE_HEIGHT - 1),
            outline=border,
            width=2,
        )
        draw.text(
            (left + 5, top + 4),
            f"t{frame_index + 1:02d} | aligned-index={position}",
            fill=(230, 230, 230),
        )
        tile = frame.convert("RGB").resize(
            (TILE_WIDTH, TILE_HEIGHT), Image.Resampling.BILINEAR
        )
        if boxes is not None and boxes[frame_index] is not None:
            box = map_box(
                boxes[frame_index],
                frame.width,
                frame.height,
                TILE_WIDTH,
                TILE_HEIGHT,
            )
            tile_draw = ImageDraw.Draw(tile)
            tile_draw.rectangle(box, outline=(0, 255, 80), width=4)
        canvas.paste(tile, (left, top + label_height))
    return canvas


def save_jpeg(image: Image.Image, path: Path, quality: int = 86) -> None:
    image.convert("RGB").save(path, quality=quality, optimize=True)


def prediction_result(box_raw: list[float]) -> dict[str, object]:
    x0, y0, x1, y1 = box_raw
    return {
        "id": "motion_bbox",
        "from_name": "roi_box",
        "to_name": "roi_frame",
        "type": "rectanglelabels",
        "original_width": RAW_WIDTH,
        "original_height": RAW_HEIGHT,
        "image_rotation": 0,
        "value": {
            "x": 100.0 * x0 / RAW_WIDTH,
            "y": 100.0 * y0 / RAW_HEIGHT,
            "width": 100.0 * (x1 - x0 + 1) / RAW_WIDTH,
            "height": 100.0 * (y1 - y0 + 1) / RAW_HEIGHT,
            "rotation": 0,
            "rectanglelabels": ["Action ROI"],
        },
    }


def build_prior_annotations(
    selection_rows: list[dict[str, str]],
    prior_annotations_path: Path,
    prior_selection_path: Path,
    prior_manual_path: Path,
    output_path: Path,
) -> None:
    prior_annotations = {
        row["sample_id"]: row for row in read_csv(prior_annotations_path)
    }
    prior_selection = {
        row["sample_id"]: row for row in read_csv(prior_selection_path)
    }
    manual = {row["sample_id"]: row for row in read_csv(prior_manual_path)}
    selected_prior = [
        row for row in selection_rows if int(row["prior_pilot30"]) == 1
    ]
    output_rows: list[dict[str, object]] = []
    for selected in selected_prior:
        sample_id = selected["sample_id"]
        annotation = prior_annotations[sample_id]
        if sample_id in manual:
            source = manual[sample_id]
            box_320 = [
                float(source["local_x_px"]),
                float(source["local_y_px"]),
                float(source["local_x_px"]) + float(source["local_width_px"]) - 1,
                float(source["local_y_px"]) + float(source["local_height_px"]) - 1,
            ]
            bbox_source = "pilot30_manual"
        else:
            box_320 = [
                float(value)
                for value in json.loads(prior_selection[sample_id]["bbox_320x240"])
            ]
            bbox_source = "pilot30_accepted_auto"
        box_raw = map_box(box_320, 320, 240, RAW_WIDTH, RAW_HEIGHT)
        output_rows.append(
            {
                "sample_id": sample_id,
                "fold": int(selected["fold"]),
                "class_id": int(selected["class_id"]),
                "class_name": selected["class_name"],
                "user_id": selected["user_id"],
                "trial_id": selected["trial_id"],
                "annotation_mode": "correction",
                "training_eligible": 1,
                "bbox_source": bbox_source,
                "region_status": annotation["bbox_quality"],
                "single_box_temporally_valid": "",
                "raw_width": RAW_WIDTH,
                "raw_height": RAW_HEIGHT,
                "x0": box_raw[0],
                "y0": box_raw[1],
                "x1": box_raw[2],
                "y1": box_raw[3],
                "free_note": annotation["free_note"],
            }
        )
    fieldnames = list(output_rows[0])
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)


def main() -> None:
    args = parse_args()
    selection_rows = read_csv(args.selection.resolve())
    manifest = {row["sample_id"]: row for row in read_csv(args.manifest.resolve())}
    if len(selection_rows) != 216:
        raise ValueError(f"Expected 216 frozen rows, got {len(selection_rows)}")
    if sum(int(row["prior_pilot30"]) for row in selection_rows) != 30:
        raise ValueError("Exactly 30 completed Pilot30 rows must be retained")
    assert_sampling_matches_training(
        {int(row["num_aligned_frames"]) for row in selection_rows}
    )

    output_dir = args.output_dir.resolve()
    images_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    build_prior_annotations(
        selection_rows,
        args.prior_annotations.resolve(),
        args.prior_selection.resolve(),
        args.prior_manual.resolve(),
        output_dir / "prior_pilot30_annotations.csv",
    )

    placeholder_path = images_dir / "blind_hidden.jpg"
    placeholder = Image.new("RGB", (1305, 819), (28, 28, 28))
    placeholder_draw = ImageDraw.Draw(placeholder)
    placeholder_draw.text(
        (40, 380),
        "BLIND ROI TASK: automatic box and crop preview are intentionally hidden.",
        fill=(230, 230, 230),
    )
    save_jpeg(placeholder, placeholder_path, quality=80)

    new_rows = [row for row in selection_rows if int(row["prior_pilot30"]) == 0]
    tasks: list[dict[str, object]] = []
    index_rows: list[dict[str, object]] = []
    image_bytes = 0
    for task_index, selected in enumerate(new_rows, 1):
        sample_id = selected["sample_id"]
        row = manifest[sample_id]
        maps = {
            "depth": frame_map(Path(row["depth_dir"]), "depth"),
            "ir": frame_map(Path(row["ir_dir"]), "ir"),
            "skeleton": frame_map(Path(row["skeleton_dir"]), "skeleton"),
        }
        common_ids = sorted(set.intersection(*(set(mapping) for mapping in maps.values())))
        if len(common_ids) != int(row["num_aligned_frames"]):
            raise ValueError(
                f"{sample_id}: manifest={row['num_aligned_frames']} common={len(common_ids)}"
            )
        positions = inference_positions(len(common_ids))
        frame_ids = [common_ids[position] for position in positions]
        depth_frames: list[Image.Image] = []
        for frame_id in frame_ids:
            with Image.open(maps["depth"][frame_id]) as image:
                frame = image.convert("RGB")
            if frame.size != (RAW_WIDTH, RAW_HEIGHT):
                raise ValueError(f"{sample_id}: unexpected Depth size {frame.size}")
            depth_frames.append(frame)

        middle_index = NUM_FRAMES // 2
        raw_frame = depth_frames[middle_index]
        mode = selected["annotation_mode"]
        auto_box_raw: list[float] | None = None
        auto_record: dict[str, object] | None = None
        if mode == "correction":
            auto_record, _ = analyse_trial(row, TILE_WIDTH, TILE_HEIGHT)
            auto_box_320 = [float(value) for value in auto_record["bbox"]]
            auto_box_raw = map_box(
                auto_box_320, TILE_WIDTH, TILE_HEIGHT, RAW_WIDTH, RAW_HEIGHT
            )

        task_key = (
            f"roi_blind_{task_index:03d}"
            if mode == "blind"
            else sample_id
        )
        stem = (
            task_key
            if mode == "blind"
            else f"{int(selected['selection_index']):03d}_{sample_id}"
        )
        raw_path = images_dir / f"{stem}_roi_frame.jpg"
        full12_path = images_dir / f"{stem}_depth12.jpg"
        overlay_path = (
            images_dir / f"{stem}_auto_overlay12.jpg"
            if auto_box_raw is not None
            else placeholder_path
        )
        crop_path = (
            images_dir / f"{stem}_auto_local12.jpg"
            if auto_box_raw is not None
            else placeholder_path
        )
        save_jpeg(raw_frame, raw_path, quality=94)
        save_jpeg(
            contact_sheet(
                depth_frames,
                positions,
                "DEPTH 12 | exact frames used by validation/test inference",
                (0, 190, 255),
            ),
            full12_path,
            quality=88,
        )
        if auto_box_raw is not None:
            save_jpeg(
                contact_sheet(
                    depth_frames,
                    positions,
                    "AUTO ROI OVERLAY 12 | same trial-level box on every frame",
                    (0, 255, 80),
                    boxes=[auto_box_raw] * NUM_FRAMES,
                ),
                overlay_path,
                quality=88,
            )
            local_frames = [
                crop_and_resize(frame, auto_box_raw) for frame in depth_frames
            ]
            save_jpeg(
                contact_sheet(
                    local_frames,
                    positions,
                    "AUTO LOCAL 12 | preview only; adjust the box above when needed",
                    (255, 0, 190),
                ),
                crop_path,
                quality=88,
            )

        unique_paths = {raw_path, full12_path, overlay_path, crop_path}
        image_bytes += sum(path.stat().st_size for path in unique_paths)
        correction = mode == "correction"
        task: dict[str, object] = {
            "data": {
                "task_key": task_key,
                "annotation_mode": mode,
                "sample_info": (
                    f"ROI {task_index}/{len(new_rows)} | "
                    + (
                        f"修正任务 | 真实类别：{selected['class_name']} | "
                        f"Subject：{selected['user_id']} | Trial：{selected['trial_id']}"
                        if correction
                        else "盲标任务 | 类别、模型预测和自动框均已隐藏"
                    )
                ),
                "roi_frame_image": local_file_url(raw_path),
                "depth12_image": local_file_url(full12_path),
                "auto_overlay_image": local_file_url(overlay_path),
                "auto_local_image": local_file_url(crop_path),
            }
        }
        if auto_box_raw is not None:
            task["predictions"] = [
                {
                    "model_version": "motion_bbox_all_frames_v1",
                    "score": 0.5,
                    "result": [prediction_result(auto_box_raw)],
                }
            ]
        tasks.append(task)
        index_rows.append(
            {
                "task_index": task_index,
                "selection_index": int(selected["selection_index"]),
                "task_key": task_key,
                "sample_id": sample_id,
                "annotation_mode": mode,
                "fold": int(selected["fold"]),
                "class_id_private": int(selected["class_id"]),
                "class_name_private": selected["class_name"],
                "user_id": selected["user_id"],
                "trial_id": selected["trial_id"],
                "common_frame_count": len(common_ids),
                "inference_positions": json.dumps(positions),
                "inference_frame_ids": json.dumps(frame_ids),
                "auto_fallback_private": (
                    int(bool(auto_record["fallback"])) if auto_record else ""
                ),
                "auto_bbox_raw_private": (
                    json.dumps(auto_box_raw) if auto_box_raw is not None else ""
                ),
                "roi_frame_width": RAW_WIDTH,
                "roi_frame_height": RAW_HEIGHT,
                "roi_frame_image": raw_path.relative_to(output_dir),
                "depth12_image": full12_path.relative_to(output_dir),
                "auto_overlay_image": overlay_path.relative_to(output_dir),
                "auto_local_image": crop_path.relative_to(output_dir),
            }
        )
        if task_index % 10 == 0 or task_index == len(new_rows):
            print(f"Label Studio assets {task_index}/{len(new_rows)}", flush=True)

    (output_dir / "tasks.json").write_text(
        json.dumps(tasks, ensure_ascii=False),
        encoding="utf-8",
    )
    with (output_dir / "task_index_private.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)

    config = """<View>
  <Header value="$sample_info"/>
  <Text name="instruction" value="请在下方单张原始 Depth 上保留一个最终动作区域框。盲标任务从零画框；修正任务可调整绿色预框。框应覆盖人、手、关键物体和必要上下文。"/>
  <Image name="roi_frame" value="$roi_frame_image" zoom="true"/>
  <RectangleLabels name="roi_box" toName="roi_frame" strokeWidth="4">
    <Label value="Action ROI" background="#00E676"/>
  </RectangleLabels>

  <Header value="模型真实使用的 12 帧完整 Depth（时间位置与验证/测试一致）"/>
  <Image name="depth12" value="$depth12_image" zoom="true"/>
  <Header value="自动框在 12 帧上的覆盖情况（盲标任务故意隐藏）"/>
  <Image name="auto_overlay" value="$auto_overlay_image" zoom="true"/>
  <Header value="自动框对应的 Local 12 帧（盲标任务故意隐藏）"/>
  <Image name="auto_local" value="$auto_local_image" zoom="true"/>

  <Header value="1. 最终局部框是否合适？"/>
  <Choices name="region_status" toName="roi_frame" choice="single-radio" required="true">
    <Choice value="suitable（合适：人、手、关键物体和必要上下文覆盖较好）"/>
    <Choice value="missing_key_region（缺关键区域：漏掉手、物体或必要上下文）"/>
    <Choice value="wrong_region（框错了：主要区域不是动作发生位置）"/>
  </Choices>

  <Header value="2. 同一个 trial 级框能否稳定覆盖这 12 帧？"/>
  <Choices name="single_box_temporally_valid" toName="depth12" choice="single-radio" required="true">
    <Choice value="stable（能：同一个框可稳定覆盖动作区域）"/>
    <Choice value="needs_wider_context（勉强：必须扩大范围才覆盖完整时序）"/>
    <Choice value="invalid（不能：人物位移过大或动作区域明显变化）"/>
  </Choices>

  <Header value="3. 备注（可选）"/>
  <TextArea name="free_note" toName="roi_frame" rows="4" placeholder="可写任何观察、疑问或框选原因。"/>
</View>
"""
    (output_dir / "label_config.xml").write_text(config, encoding="utf-8")

    start_script = f"""$ErrorActionPreference = "Stop"
$env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED = "true"
$env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT = "{REPO_DIR.resolve()}"
$labelStudioExe = "C:\\Users\\ncy\\AppData\\Roaming\\Python\\Python312\\Scripts\\label-studio.exe"
$dataDir = "{KAGGLE_LABEL_STUDIO_STATE.resolve()}"
$port = 8080
$hostUrl = "http://127.0.0.1:8080"
$listener = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue
if (-not $listener) {{
    $arguments = (
        'start --no-browser --data-dir "{{0}}" --port {{1}} ' +
        '--host http://127.0.0.1:{{1}} --enable-legacy-api-token --log-level INFO'
    ) -f $dataDir, $port
    Start-Process -FilePath $labelStudioExe -ArgumentList $arguments -WindowStyle Hidden
}}
Write-Host "Kaggle Label Studio: $hostUrl（独立 Kaggle 数据库）"
Write-Host "不要使用 localhost:8081；那是 CV/燃气表数据库。"
"""
    (output_dir / "start_local_file_server.ps1").write_text(
        start_script, encoding="utf-8-sig"
    )

    summary = {
        "frozen_annotation_set": 216,
        "new_label_studio_tasks": len(tasks),
        "blind_tasks": sum(
            row["annotation_mode"] == "blind" for row in index_rows
        ),
        "new_correction_tasks": sum(
            row["annotation_mode"] == "correction" for row in index_rows
        ),
        "completed_prior_pilot30": 30,
        "asset_bytes": image_bytes,
        "asset_megabytes": image_bytes / 1_000_000,
        "task_json_bytes": (output_dir / "tasks.json").stat().st_size,
        "image_contract": {
            "drawing_frame": [RAW_WIDTH, RAW_HEIGHT],
            "contact_sheet_tile": [TILE_WIDTH, TILE_HEIGHT],
            "frames": NUM_FRAMES,
            "sampling": "exact AlignedMultimodalDataset augment=False positions",
            "auto_box": "one trial-level motion bbox reused on all 12 frames",
        },
        "blind_contract": (
            "No true class, model prediction, automatic bbox, or automatic crop "
            "is present in the blind task data/UI. Private mappings remain only "
            "in task_index_private.csv and must not be shown during annotation."
        ),
    }
    (output_dir / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    readme = """# ROI 216 标注项目

该目录对应冻结的 216 条 ROI 集合：

- 60 条盲标任务：不展示类别、模型预测、自动框或自动 Local；
- 126 条新修正任务：可在 640×480 原始 Depth 中调整绿色预框；
- 30 条已完成的 Pilot30：已整理到 `prior_pilot30_annotations.csv`，无需重标。

Label Studio 中实际导入 `tasks.json`，共 186 条。项目必须独立于旧 CV
项目。标注配置使用 `label_config.xml`。静态图片通过 Label Studio 的本地文件
服务读取；文档根目录必须是仓库根目录，相关环境变量见
`start_local_file_server.ps1`。

注意：

1. 每条任务最终最多保留一个 `Action ROI` 框；
2. “合适”与“缺关键区域”应保留一个最终框；
3. 如果单一 trial 框完全不成立，可选择 `invalid` 并在备注中说明；
4. 盲标 60 条仅用于定位器评估，禁止用于训练；
5. 训练定位器时，每个 held fold 只能使用另外两折的 correction 标注。
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
