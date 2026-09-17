"""Build a protocol-aware information map for the frozen small-action classes.

The output deliberately keeps measurements from different validation protocols
separate.  It is an audit aid, not a leaderboard of modalities.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def measured_band(value: float) -> str:
    if value >= 0.60:
        return "较强"
    if value >= 0.35:
        return "中等"
    return "较弱"


def expected_information(row: dict[str, str]) -> dict[str, str]:
    obj = row["object_dependency"]
    amplitude = row["motion_amplitude"]
    temporal = row["temporal_dependency"]
    body = row["primary_body_part"]

    return {
        "skeleton_expected": (
            "较强：姿态和大关节轨迹" if amplitude == "large" else "有限：缺少手指、小物体与接触语义"
        ),
        "depth_expected": (
            "中等：可补充人体轮廓、前后关系和局部运动；小物体仍偏弱"
            if obj == "high"
            else "较强：人体轮廓和三维运动通常可见"
        ),
        "ir_expected": (
            "中等：手部轮廓可能比伪彩 Depth 清楚；细小物体仍受分辨率限制"
            if obj == "high"
            else "中等：灰度外观和人体运动可见"
        ),
        "thermal_expected": (
            "有限：人体和接触位置可见，但笔、手机、遥控器等冷物体经常不清楚"
            if obj == "high"
            else "中等：人体热轮廓和粗动作可见"
        ),
        "imu_expected": (
            f"较强候选：{body}的局部动态与节律" if temporal == "high" else f"中等候选：{body}的运动幅度"
        ),
        "radar_expected": (
            "有限：可补充运动速度/方向，不擅长物体语义"
            if obj == "high"
            else "中等：粗粒度速度和空间运动"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=ROOT / "data" / "six_modality_audit" / "small_action_taxonomy_v1.csv",
    )
    parser.add_argument(
        "--sd-per-class",
        type=Path,
        default=ROOT / "runs" / "p0_six_modality_audit" / "sd_oof_per_class.csv",
    )
    parser.add_argument(
        "--complementarity",
        type=Path,
        default=ROOT / "runs" / "p0_six_modality_audit" / "remaining_modality_complementarity.json",
    )
    parser.add_argument(
        "--thermal-confusion",
        type=Path,
        default=ROOT.parent / "thermal_baseline" / "runs" / "resnet18_tsm_final" / "confusion_matrix.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "six_modality_audit" / "small_action_information_map.csv",
    )
    args = parser.parse_args()

    taxonomy = {
        int(row["class_id"]): row
        for row in read_csv(args.taxonomy)
        if int(row["include_fixed_small_action"]) == 1
    }
    sd = {int(row["class_id"]): row for row in read_csv(args.sd_per_class)}
    with args.complementarity.open("r", encoding="utf-8") as handle:
        complementarity = json.load(handle)
    ir = {int(row["class_id"]): row for row in complementarity["ir"]["per_class"]}
    radar = {int(row["class_id"]): row for row in complementarity["radar"]["per_class"]}

    thermal_cm = np.loadtxt(args.thermal_confusion, delimiter=",", dtype=np.int64)
    if thermal_cm.shape != (40, 40):
        raise ValueError(f"Expected a 40x40 Thermal confusion matrix, got {thermal_cm.shape}")

    raw_observations = {
        18: "实看样本：写字时笔和纸在 Depth/IR 中很小；Thermal 主要保留人体热轮廓。",
        25: "实看样本：电视不在画面内，手中物体像遥控器也像手机，原始观测本身存在语义歧义。",
        26: "实看样本：坐姿与玩手机/看电视接近，控制器细节在三种视觉图中都不明显。",
    }

    output_rows: list[dict[str, object]] = []
    for class_id in sorted(taxonomy):
        tax = taxonomy[class_id]
        sdr = sd[class_id]
        irr = ir[class_id]
        rr = radar[class_id]
        expected = expected_information(tax)
        thermal_support = int(thermal_cm[class_id].sum())
        thermal_recall = (
            float(thermal_cm[class_id, class_id] / thermal_support) if thermal_support else float("nan")
        )
        radar_accuracy = float(rr["radar_accuracy"])

        output_rows.append(
            {
                "class_id": class_id,
                "action_name": tax["action_name"],
                "semantic_group": tax["semantic_group"],
                "motion_amplitude": tax["motion_amplitude"],
                "object_dependency": tax["object_dependency"],
                "primary_body_part": tax["primary_body_part"],
                "temporal_dependency": tax["temporal_dependency"],
                "oof_support": int(sdr["oof_support"]),
                "skeleton_oof_recall": round(float(sdr["skeleton_recall"]), 6),
                "skeleton_measured_band": measured_band(float(sdr["skeleton_recall"])),
                "skeleton_expected_information": expected["skeleton_expected"],
                "depth_oof_recall": round(float(sdr["depth_recall"]), 6),
                "depth_measured_band": measured_band(float(sdr["depth_recall"])),
                "depth_expected_information": expected["depth_expected"],
                "sd_fusion_oof_recall": round(float(sdr["fusion_w040_recall"]), 6),
                "depth_conditional_gain_pp": round(float(sdr["fusion_minus_skeleton_recall_pp"]), 3),
                "ir_oof_recall": round(float(irr["ir_accuracy"]), 6),
                "ir_oracle_gain_over_sd_pp": round(float(irr["oracle_gain_pp"]), 3),
                "ir_expected_information": expected["ir_expected"],
                "thermal_fixed_split_support": thermal_support,
                "thermal_fixed_split_recall": round(thermal_recall, 6),
                "thermal_protocol_warning": "旧固定 subject split；只作线索，不能与三折 OOF 直接排序",
                "thermal_expected_information": expected["thermal_expected"],
                "imu_measured_result": "待室友在相同 subject-disjoint OOF 协议回填",
                "imu_expected_information": expected["imu_expected"],
                "radar_usable_oof_support": int(rr["samples"]),
                "radar_usable_oof_recall": round(radar_accuracy, 6),
                "radar_oracle_gain_over_sd_pp": round(float(rr["oracle_gain_pp"]), 3),
                "radar_protocol_warning": "仅 header 非空的可用子集；缺失严重，结果存在选择偏差",
                "radar_expected_information": expected["radar_expected"],
                "raw_visual_observation": raw_observations.get(class_id, "未做逐类人工精看；以定量 OOF 与后续错误样本复核为准"),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {len(output_rows)} small-action rows to {args.output}")


if __name__ == "__main__":
    main()
