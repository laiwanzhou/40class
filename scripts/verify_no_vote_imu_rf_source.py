"""One train12 raw IMU sample against source32 statistics, with no label reads."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.experiments.no_vote_protocol import load_protocol
from src.experiments.artifact_record import VerificationContext
from src.experiments.no_vote_manifest import read_public_rows
from src.experiments.no_vote_types import write_json
from src.experiments.imu_rf_teacher import source_rf_ops,statistics_for_row


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();p=load_protocol(args.config,inference_only=True);context=VerificationContext(p)
    started=time.perf_counter();ops=source_rf_ops(p,context)
    row=next(x for x in read_public_rows(p.public_manifests['train12']) if x['imu_available']=='1')
    original=Path.open;attempts=[]
    def guard(path,*a,**kw):
        if path.name.endswith('_labels.csv') or path.name=='final_labels.csv':
            attempts.append(str(path));raise AssertionError('RF stat acceptance must not read labels')
        return original(path,*a,**kw)
    Path.open=guard
    try:
        actual,mask,meta=statistics_for_row(row,ops)
        by_device=ops['load_trial'](Path(row['imu_path']))
        values,tm,dm,duration=ops['resample_trial'](by_device,32)
        expected,expected_mask=ops['feature_vector'](values,tm,dm)
        np.testing.assert_array_equal(actual,expected);np.testing.assert_array_equal(mask,expected_mask)
        proof={'protocol_sha256':p.identity(),'sample_id':row['sample_id'],'statistics_shape':list(actual.shape),
            'mask_shape':list(mask.shape),'source32_shape':list(values.shape),'usable':meta['usable'],
            'device_count':meta['device_count'],'feature_max_difference':float(np.max(np.abs(actual-expected))),
            'mask_max_difference':float(np.max(np.abs(mask-expected_mask))),'label_payload_open_attempts':len(attempts),
            'final_labels_read':False,'seconds':round(time.perf_counter()-started,2)}
        write_json(p.run_root/'protocol/task7_real_sample_acceptance.json',proof)
        print(json.dumps(proof,ensure_ascii=False),flush=True)
    finally:Path.open=original


if __name__=='__main__':main()
