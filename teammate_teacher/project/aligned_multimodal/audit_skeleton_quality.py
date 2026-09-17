from __future__ import annotations

import argparse
import csv
from pathlib import Path

from aligned_data import frame_map, load_skeleton_people, skeleton_distance


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PROJECT_DIR / "data" / "manifest.csv"
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "skeleton_quality.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统计每个 trial 的多人和 Skeleton 顺序切换风险")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = args.manifest.resolve()
    output = args.output.resolve()
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    results: list[dict[str, object]] = []
    for index, row in enumerate(rows, start=1):
        maps = {
            "depth": frame_map(Path(row["depth_dir"]), "depth"),
            "ir": frame_map(Path(row["ir_dir"]), "ir"),
            "skeleton": frame_map(Path(row["skeleton_dir"]), "skeleton"),
        }
        common_ids = sorted(set.intersection(*(set(mapping) for mapping in maps.values())))
        people_counts: list[int] = []
        previous_first = None
        previous_tracked = None
        switch_candidates = 0
        tracking_nonfirst_frames = 0
        for frame_id in common_ids:
            people = load_skeleton_people(maps["skeleton"][frame_id], flip=False)
            people_counts.append(len(people))
            if not people:
                continue

            if previous_first is not None and len(people) > 1:
                first_distance = skeleton_distance(previous_first, people[0])
                alternate_distance = min(skeleton_distance(previous_first, person) for person in people[1:])
                if first_distance > 0.20 and alternate_distance + 0.08 < first_distance:
                    switch_candidates += 1
            previous_first = people[0]

            if previous_tracked is None:
                tracked_index = 0
            else:
                tracked_index = min(
                    range(len(people)),
                    key=lambda person_index: skeleton_distance(previous_tracked, people[person_index]),
                )
            if tracked_index != 0:
                tracking_nonfirst_frames += 1
            previous_tracked = people[tracked_index]

        frames = len(people_counts)
        multi_frames = sum(count >= 2 for count in people_counts)
        results.append(
            {
                "sample_id": row["sample_id"],
                "split": row["split"],
                "class_id": int(row["class_id"]),
                "class_name": row["class_name"],
                "user_id": row["user_id"],
                "trial_id": row["trial_id"],
                "frames": frames,
                "multi_frames": multi_frames,
                "multi_ratio": multi_frames / frames if frames else 0.0,
                "majority_multi": int(frames > 0 and multi_frames > frames / 2),
                "max_people": max(people_counts, default=0),
                "switch_candidates": switch_candidates,
                "tracking_nonfirst_frames": tracking_nonfirst_frames,
                "tracking_changes_input": int(tracking_nonfirst_frames > 0),
            }
        )
        if index % 500 == 0 or index == len(rows):
            print(f"已审计 {index}/{len(rows)} trial", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"Skeleton 质量清单：{output}")


if __name__ == "__main__":
    main()
