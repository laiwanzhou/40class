from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_DIR = Path(__file__).resolve().parent
ROI_DIR = PROJECT_DIR / "data" / "local_roi_annotation_v2"
REVIEW_DIR = ROI_DIR / "label_studio_blind60_review_v2"
RESULTS = ROI_DIR / "blind60_review_annotations.csv"
PRIVATE_INDEX = REVIEW_DIR / "task_index_private.csv"
OUTPUT_DIR = REVIEW_DIR / "result_montages"
RAW_WIDTH = 640
RAW_HEIGHT = 480


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def font(size: int) -> ImageFont.ImageFont:
    candidates = [
        Path(r"C:\Windows\Fonts\consola.ttf"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def stored_box(row: dict[str, str], field: str) -> tuple[float, float, float, float]:
    values = tuple(float(value) for value in json.loads(row[field]))
    if len(values) != 4:
        raise ValueError(f"{row['sample_id']}: invalid {field}")
    return values


def result_box(row: dict[str, str]) -> tuple[float, float, float, float]:
    return tuple(float(row[field]) for field in ("x0", "y0", "x1", "y1"))


def draw_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    color: tuple[int, int, int],
    label: str,
    width: int,
) -> None:
    draw.rectangle(box, outline=color, width=width)
    draw.rectangle((box[0], box[1], box[0] + 30, box[1] + 20), fill=color)
    draw.text((box[0] + 5, box[1] + 2), label, fill=(0, 0, 0), font=font(14))


def main() -> None:
    results = read_csv(RESULTS)
    private = {row["task_key"]: row for row in read_csv(PRIVATE_INDEX)}
    if len(results) != 60 or len(private) != 60:
        raise ValueError("Expected 60 ROI review results and 60 private rows")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tiles: list[Image.Image] = []
    for result in results:
        source = private[result["task_key"]]
        raw_path = REVIEW_DIR / source["roi_frame_image"]
        with Image.open(raw_path) as image:
            raw = image.convert("RGB")
        draw = ImageDraw.Draw(raw)
        machine = stored_box(source, "machine_bbox_raw_private")
        final = result_box(result)
        draw_box(draw, machine, (0, 255, 80), "M", 5)
        draw_box(draw, final, (255, 0, 190), "F", 3)
        footer_height = 76
        tile = Image.new("RGB", (RAW_WIDTH, RAW_HEIGHT + footer_height), (18, 18, 18))
        tile.paste(raw, (0, 0))
        tile_draw = ImageDraw.Draw(tile)
        assessment = result["machine_box_assessment"].replace("machine_", "")
        final_source = result["final_box_source"].replace("use_", "")
        tile_draw.text(
            (8, RAW_HEIGHT + 6),
            (
                f"#{int(result['review_index']):02d} c{int(result['class_id']):02d} "
                f"fold={result['fold']} fallback={result['original_fallback']}"
            ),
            fill=(245, 245, 245),
            font=font(18),
        )
        tile_draw.text(
            (8, RAW_HEIGHT + 36),
            f"assessment={assessment} | final={final_source}",
            fill=(255, 220, 80),
            font=font(16),
        )
        tiles.append(tile.resize((320, 278), Image.Resampling.LANCZOS))

    per_page = 15
    columns = 5
    rows_per_page = 3
    gap = 8
    for page_index, start in enumerate(range(0, len(tiles), per_page), start=1):
        subset = tiles[start : start + per_page]
        canvas = Image.new(
            "RGB",
            (
                columns * 320 + (columns + 1) * gap,
                rows_per_page * 278 + (rows_per_page + 1) * gap,
            ),
            (8, 8, 8),
        )
        for offset, tile in enumerate(subset):
            row_index, column_index = divmod(offset, columns)
            canvas.paste(
                tile,
                (
                    gap + column_index * (320 + gap),
                    gap + row_index * (278 + gap),
                ),
            )
        canvas.save(
            OUTPUT_DIR / f"overview_page_{page_index:02d}.jpg",
            quality=92,
            optimize=True,
        )
    print(
        json.dumps(
            {
                "samples": len(tiles),
                "overview_pages": len(list(OUTPUT_DIR.glob("overview_page_*.jpg"))),
                "legend": {
                    "M_green": "machine original box",
                    "F_magenta": "human-selected final box",
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
