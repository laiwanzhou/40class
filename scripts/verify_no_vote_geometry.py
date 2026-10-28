"""Task3: source parity on one deterministic train12 sample, no regeneration."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.experiments.artifact_record import ArtifactRegistry,VerificationContext
from src.experiments.no_vote_manifest import load_stage_inputs,read_public_rows
from src.experiments.no_vote_protocol import load_protocol
from src.experiments.no_vote_types import ArtifactRef,write_json
from src.experiments import pose_roi_adapter as adapter
from src.experiments.pixel_cache import _pixel_ops,_pixel_trial,load_pixels,acquisition_seconds
from src.experiments.teammate_source import sha256_file


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--submission-root',required=True,type=Path)
    args=parser.parse_args();started=time.perf_counter()
    p=load_protocol(args.config);context=VerificationContext(p);registry=ArtifactRegistry(p,context=context)
    receipt=json.loads((p.run_root/'protocol/task2_prepared.json').read_text(encoding='utf-8'))
    inputs=load_stage_inputs(p,'train12','raw',context=context)
    # Choice depends only on the fixed public ID order and IR availability.
    row=next(x for x in read_public_rows(inputs.public_manifest) if x['ir_available']=='1')
    import src.experiments.artifact_record as artifact_module
    import src.experiments.teammate_source as source_module
    raw_roots=[Path(row[k]).resolve() for k in ['ir_path','depth_path','skeleton_path'] if row[k]]
    raw_hash_attempts=[]
    def bounded_hash(path):
        path=Path(path).resolve()
        if any(path.is_relative_to(root) for root in raw_roots) or path.suffix in {'.npy','.npz','.pt','.pth','.safetensors'}:
            raw_hash_attempts.append(str(path))
            raise AssertionError('geometry parity attempted a raw/cache/weight full checksum')
        return sha256_file(path)
    artifact_module.sha256_file=bounded_hash
    source_module.sha256_file=bounded_hash
    adapter.sha256_file=bounded_hash
    def forbidden_regeneration(*a,**k):raise AssertionError('geometry parity attempted raw inventory/regeneration')
    adapter.snapshot_raw_files=forbidden_regeneration
    adapter._process_trial=forbidden_regeneration
    entries=[(ArtifactRef(Path(x['record_path']),x['sha256']),x) for x in receipt['imports']]
    roi_ref=next(ref for ref,_ in entries if registry.read(ref).stage=='p29'
        and row['sample_id'] in registry.read(ref).rows.sample_ids)
    roi_record=registry.verify(roi_ref,'p29','raw');pose_ref=roi_record.parents[0]
    pose,rois=adapter.load_pose_roi_sample(inputs,pose_ref,roi_ref,row['sample_id'],protocol=p,context=context)
    print(json.dumps({'event':'geometry_loaded','frames':len(pose['frame_ids']),
        'seconds':round(time.perf_counter()-started,3)}),flush=True)
    source_hashes={}
    for name in ['audit_yolo11_pose_skeleton.py','build_adaptive_yolo11_pose_skeleton_cache.py',
                 'build_multiscale_dir_rois.py','build_p46_videomae_cache.py','build_p30_shared_dir_roi_feature_cache.py']:
        snapshot=p.source_root/'aligned_multimodal'/name
        original=args.submission_root/'code/training/project/aligned_multimodal'/name
        a,b=sha256_file(snapshot),sha256_file(original)
        if a!=b:raise ValueError('training source differs from final submission: '+name)
        source_hashes[name]=a
    base,adaptive,roi=adapter._load_ops(p);settings=adapter._settings('cpu')
    rebuilt=roi.build_trial_rois(pose,settings)
    differences={}
    for name,expected in rebuilt.items():
        if name not in rois:raise ValueError('cached ROI missing source field: '+name)
        if expected.dtype.kind in 'fci':
            np.testing.assert_allclose(rois[name],expected,rtol=0,atol=0,equal_nan=True)
        else:np.testing.assert_array_equal(rois[name],expected)
        differences[name]=0
    reasons=adaptive.retry_reasons(pose['primary_ir_boxes_xyxy_conf'],pose['primary_ir_keypoints_xy_conf'],
        image_width=settings.width,image_height=settings.height,keypoint_conf=settings.keypoint_conf,
        box_conf=settings.retry_box_conf,min_height_ratio=settings.retry_min_height_ratio,
        min_upper_joints=settings.retry_min_upper_joints,min_wrists=settings.retry_min_wrists,
        center_jump_ratio=settings.retry_center_jump_ratio,area_ratio=settings.retry_area_ratio)
    np.testing.assert_array_equal(reasons,pose['retry_reason_bits'])
    actual=adapter.compatible_frame_map(Path(row['ir_path']),'ir')
    np.testing.assert_array_equal(pose['frame_ids'],np.asarray(sorted(actual)))
    # Check observed timestamp filenames, no skeleton payload/class/trial reads.
    known=0
    for key,stamp in zip(pose['frame_ids'].astype(str),pose['acquisition_ids'].astype(str)):
        if not stamp:continue
        if stamp.rsplit('_',1)[-1]!=key.rsplit('_',1)[-1]:raise ValueError('timestamp/frame counter mismatch')
        if key.isdigit() and not (Path(row['skeleton_path'])/'predictions'/('Color_'+stamp+'.json')).is_file():
            raise ValueError('acquisition timestamp not present in original skeleton filenames')
        known+=1
    pixel_ref=next(ref for ref,item in entries if item['stage']=='pixels' and item['partition']=='refit14')
    pixel_record,pixels=load_pixels(pixel_ref,protocol=p,context=context)
    pos=pixel_record.rows.sample_ids.index(row['sample_id'])
    roi_file=next(Path(f) for f in roi_record.files if Path(f).name==row['sample_id']+'.npz')
    reconstructed=_pixel_trial(row,roi_file,_pixel_ops(p),registry,roi_record)
    for name,values in reconstructed.items():
        np.testing.assert_array_equal(pixels[name][pos],values)
    with (p.run_root/'protocol/recording_metadata/train12.csv').open(encoding='utf-8',newline='') as f:
        sidecar=next(x for x in csv.DictReader(f) if x['sample_id']==row['sample_id'])
    expected_times=np.asarray([acquisition_seconds(s) for s in pose['acquisition_ids']])
    np.testing.assert_array_equal(pixels['source_time_seconds'][pos],expected_times[reconstructed['source_frame_indices']])
    first=next((s for s in pose['acquisition_ids'].astype(str) if np.isfinite(acquisition_seconds(s))),None)
    if first:
        stamp=first.rsplit('_',1)[0]
        parsed=datetime.strptime(stamp,'%Y-%m-%d_%H-%M-%S.%f' if '.' in stamp else '%Y-%m-%d_%H-%M-%S')
        seconds=parsed.hour*3600+parsed.minute*60+parsed.second+parsed.microsecond/1e6
        assert sidecar['recording_date']==parsed.date().isoformat() and float(sidecar['start_seconds'])==seconds
    else:assert sidecar['timestamp_available']=='0'
    body={'protocol_sha256':p.identity(),'partition':'train12','sample_id':row['sample_id'],
        'frames':len(pose['frame_ids']),'known_acquisition_frames':known,'source_files':source_hashes,
        'roi_fields_exact':differences,'retry_reason_bits_exact':True,'pixel_crop_quality_times_exact':True,
        'recording_day_seconds_exact':True,'full_geometry_regenerated':False,'detector_executed':False,
        'raw_sha_calls':len(raw_hash_attempts),'final_labels_read':False,'formal_generation_ready':False,
        'adapter_sha256':sha256_file(Path(adapter.__file__))}
    output=p.run_root/'protocol/task3_geometry_parity.json'
    if output.exists() and json.loads(output.read_text(encoding='utf-8'))!=body:
        raise ValueError('parity receipt changed; use a new version, never overwrite evidence')
    if not output.exists():write_json(output,body)
    print(json.dumps({'event':'geometry_parity_pass','frames':body['frames'],'roi_fields':len(differences),
        'seconds':round(time.perf_counter()-started,3),'report':str(output)}),flush=True)


if __name__=='__main__':main()
