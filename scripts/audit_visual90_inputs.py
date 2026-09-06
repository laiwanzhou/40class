"""Read-only source preflight; never marks geometry verified automatically."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.canonical_multimodal_index import build_canonical_trials
from src.data.pose_roi_dataset import paired_frame_key
from src.data.visual90_dataset import select_continuous_clips


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def inspect_frame_sequence(paths: list[Path], modality: str) -> dict:
    keys, unparsed = [], []
    for path in paths:
        try:
            key = paired_frame_key(path, modality)
            datetime.strptime(key[0], '%Y-%m-%d_%H-%M-%S.%f')
            keys.append(key)
        except ValueError:
            unparsed.append(path.name)
    result = {
        'frame_count': len(paths), 'unparsed_frames': len(unparsed),
        'unparsed_examples': unparsed[:3], 'continuity_status': 'continuity_unverified',
    }
    if not paths:
        result['continuity_status'] = 'not_present'
        return result
    if unparsed or len(set(keys)) != len(keys):
        return result
    # Integer millisecond offsets avoid platform/local-time timezone conversions.
    instants = [datetime.strptime(key[0], '%Y-%m-%d_%H-%M-%S.%f') for key in keys]
    times = np.array([(value - instants[0]).total_seconds() * 1000 for value in instants])
    selection = select_continuous_clips(times)
    result.update({
        'continuity_status': 'verified_timestamps',
        'counter_semantics': 'not_used_without_export_metadata',
        'source_segment_count': len(selection.source_breaks) - 1,
        'clip_bounds': selection.bounds.tolist(),
        'sampled_indices': selection.indices.tolist(),
        'key_digest': hashlib.sha256(json.dumps(keys).encode()).hexdigest(),
    })
    return result


def choose_smoke_rows(rows: list[dict]) -> list[dict]:
    if any(row['partition'] != 'train' for row in rows):
        raise ValueError('smoke selection requires training rows only')
    if len(rows) < 8:
        raise ValueError('at least eight training pairs required')
    ordered = sorted(rows, key=lambda row: (row['ir_frames'], row['sample_id']))
    chosen = []
    for k in range(4):
        group = ordered[k * len(rows) // 4:(k + 1) * len(rows) // 4]
        chosen.extend(sorted(group, key=lambda row: row['sample_id'])[:2])
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=Path('D:/work/2026.7.14_kaggle/datasets/Small-Model-Track/train'))
    parser.add_argument('--report', type=Path, default=ROOT / 'reports/visual90_appearance_temporal_input_preflight.json')
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    manifest = ROOT / 'metadata/manifest.csv'
    split = ROOT / 'metadata/splits/train12_val2_user6_user7_development.json'
    pose = ROOT.parent / '40class/outputs/depth_ir_person_crop_40class_fold0/person_crop_pose_tracks.npz'
    split_data = json.loads(split.read_text(encoding='utf-8'))
    train_users, val_users = set(split_data['train_user_ids']), set(split_data['validation_user_ids'])
    expected_train = {'user1','user2','user3','user5','user8','user9','user16','user18','user19','user20','user21','user22'}
    if train_users != expected_train or val_users != {'user6','user7'} or train_users & val_users:
        raise ValueError('fixed-user ownership changed')
    rows = []
    seen = set()
    for partition in ('train', 'validation'):
        for trial in build_canonical_trials(manifest, split, args.data_root, partition=partition):
            if trial.sample_id in seen:
                raise ValueError('overlapping sample IDs')
            seen.add(trial.sample_id)
            row = {'sample_id': trial.sample_id, 'partition': partition, 'user_id': trial.user_id, 'class_id': trial.class_id}
            seq = {}
            for modality, prefix in (('ir','IR'), ('depth_color','Depth')):
                folder = trial.paths[modality]
                paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in {'.png','.jpg','.jpeg'}) if folder and folder.is_dir() else []
                seq[modality] = inspect_frame_sequence(paths, prefix)
                row[modality + '_path'] = str(folder) if folder else None
            row.update({'ir_frames': seq['ir']['frame_count'], 'depth_frames': seq['depth_color']['frame_count'], 'sequences': seq})
            row['pairing_status'] = 'verified_keys' if (
                seq['ir'].get('key_digest') and seq['ir'].get('key_digest') == seq['depth_color'].get('key_digest')
            ) else 'unverified_or_unpaired'
            rows.append(row)
        print(f'indexed {partition}; cumulative rows={len(rows)}', flush=True)
    blockers = [
        {'sample_id': row['sample_id'], 'partition': row['partition'], 'modality': name,
         'examples': seq['unparsed_examples']}
        for row in rows for name, seq in row['sequences'].items()
        if seq['continuity_status'] == 'continuity_unverified'
    ]
    paired = [row for row in rows if row['partition'] == 'train' and row['pairing_status'] == 'verified_keys']
    smoke = choose_smoke_rows(paired)
    report = {
        'status': 'blocked_continuity' if blockers else 'awaiting_geometry',
        'population': dict(Counter(row['partition'] for row in rows)),
        'manifest_sha256': sha256(manifest), 'split_sha256': sha256(split),
        'pose_sha256': sha256(pose), 'pose_path': str(pose),
        'source_integrity': 'frame-key inventory only; full pixel hashes/readability not yet certified',
        'continuity_blockers': blockers,
        'pairing_counts': dict(Counter(row['pairing_status'] for row in rows)),
        'provisional_smoke_sample_ids': [row['sample_id'] for row in smoke],
        'smoke_selection_note': 'key-paired inventory; verify readability before final geometry selection',
        'geometry_status': 'geometry_unverified', 'rows': rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'rows'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
