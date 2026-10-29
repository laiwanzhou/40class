"""One public train12 trial: verify source math, pixel timestamps and reader fields."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.experiments.artifact_record import ArtifactRegistry,VerificationContext
from src.experiments.no_vote_protocol import load_protocol
from src.experiments.no_vote_types import canonical_data,write_json
from src.experiments.motion_cache import run_motion_pipeline,source_motion_ops,MOTION_FIELDS,BOOLEAN_MOTION_FIELDS,motion_item


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();started=time.perf_counter();p=load_protocol(args.config,inference_only=True)
    # The adapter is also guarded against all supervised label payload opens.
    original=Path.open;attempts=[]
    def guarded(path,*a,**k):
        if path.name.endswith('_labels.csv') or path.name=='final_labels.csv':
            attempts.append(str(path));raise AssertionError('motion acceptance attempted a label payload')
        return original(path,*a,**k)
    Path.open=guarded
    try:
        refs=run_motion_pipeline(p,partition='train12',max_trials=1)['train12']
        c=VerificationContext(p);registry=ArtifactRegistry(p,context=c);ops=source_motion_ops(p,c)
        source=registry.read(refs['source']);windows=registry.read(refs['windows'])
        file=next(Path(x) for x in source.files if Path(x).suffix=='.npz')
        with np.load(file,allow_pickle=False) as z:trial={k:z[k] for k in z.files}
        arrays={Path(x).stem:np.load(x,mmap_mode='r') for x in windows.files if Path(x).suffix=='.npy'}
        chosen=arrays['source_frame_indices'][0]
        rotated=ops['rotate_skeleton_cache'](trial['skeleton_features'],trial['skeleton_feature_mask'],trial['skeleton_joint_mask'],trial['skeleton_relations'],trial['skeleton_relation_mask'],trial['frame_time_seconds'])
        expected={}
        for k,v in [('skeleton_features','features'),('skeleton_feature_mask','feature_mask'),('skeleton_relations','relations'),('skeleton_relation_mask','relation_mask')]:expected[k]=rotated[v][chosen]
        for k in ['skeleton_joint_mask','skeleton_frame_quality']:expected[k]=trial[k][chosen]
        imu=ops['aggregate_imu'](trial['frame_time_seconds'],chosen,trial['imu_values'],trial['imu_time_seconds'],trial['imu_device_offsets'],4)
        for k,v in [('imu_sequences','sequences'),('imu_sequence_mask','sequence_mask'),('imu_bin_statistics','statistics'),('imu_bin_mask','bin_mask'),('imu_global_statistics','global_statistics'),('imu_global_mask','global_mask')]:expected[k]=imu[v]
        differences={}
        for name in MOTION_FIELDS:
            actual=arrays[name][0];reference=expected[name].astype(actual.dtype)
            np.testing.assert_array_equal(actual,reference)
            differences[name]=float(np.max(np.abs(actual.astype(np.float64)-reference.astype(np.float64)),initial=0))
        time_diff=float(np.max(np.abs(arrays['source_time_seconds'][0]-trial['frame_absolute_time_seconds'][chosen])))
        assert time_diff<=1e-6
        item=motion_item({k:arrays[k][0] for k in MOTION_FIELDS})
        for name in MOTION_FIELDS:
            assert item[name].dtype==np.dtype(bool if name in BOOLEAN_MOTION_FIELDS else np.float32)
        proof={'source':refs['source'],'windows':refs['windows'],'sample_id':str(trial['sample_id'].item()),
            'frames':len(trial['frame_ids']),'imu_points':len(trial['imu_values']),
            'source_math_max_differences':differences,'pixel_timestamp_max_difference':time_diff,
            'source_interval_point_accounting':int(trial['imu_interval_counts'].sum())==len(trial['imu_values']),
            'external_transform':'identity','imu_instance_normalization':False,'label_payload_open_attempts':len(attempts),
            'final_labels_read':False,'seconds':round(time.perf_counter()-started,2)}
        write_json(p.run_root/'protocol/task6_real_sample_acceptance.json',proof)
        print(json.dumps(canonical_data(proof),ensure_ascii=False),flush=True)
    finally:Path.open=original


if __name__=='__main__':main()
