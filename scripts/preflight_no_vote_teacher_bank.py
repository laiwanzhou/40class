"""Task11 source preflight/session adapter acceptance; never generates A8."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.experiments.no_vote_protocol import load_protocol
from src.experiments.no_vote_types import ArtifactRef,write_json
from src.experiments.artifact_record import VerificationContext
from src.experiments.source_teacher_bank import task11_preflight,require_task11_formal_ready
from src.experiments.session_metadata import load_recording_rows,session_blocks,repeat_session_lists,source_session_ops


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();p=load_protocol(args.config,inference_only=True);c=VerificationContext(p);start=time.perf_counter()
    marker=p.run_root/'protocol/task9_10_independent_review_P39.json';c.hash_file(marker)
    review=json.loads(marker.read_text(encoding='utf-8'))
    if (review.get('protocol_sha256')!=p.identity() or review.get('verdict')!='GO_TASK9_TASK10'
            or any(review.get(k)!='PASS' for k in ['engineering','methodology','artifacts'])
            or not review.get('task11_implementation_allowed') or review.get('unresolved_task9_task10_p1_p2')):
        raise ValueError('Task11 requires independent Task9/10 review GO')
    c.hash_file(Path(review['report']['path']),review['report']['sha256'])
    for item in review['reviewed_completions'].values():c.hash_file(Path(item['path']),item['sha256'])
    closure=task11_preflight(p,c);source=source_session_ops(p,c)
    prep=json.loads((p.run_root/'protocol/task2_prepared.json').read_text(encoding='utf-8'));sessions={}
    # Public sidecars only; do not open fit/final label tables or any predictions.
    original=Path.open;attempts=[]
    def guard(path,*a,**kw):
        if path.name.endswith('_labels.csv') or path.name=='final_labels.csv':
            attempts.append(str(path));raise AssertionError('label read in public Task11 preflight')
        return original(path,*a,**kw)
    Path.open=guard
    try:
        for partition in p.partitions:
            value=prep['recording_metadata'][partition];reference=ArtifactRef(Path(value['record_path']),value['sha256'])
            index,rows=load_recording_rows(p,partition,reference,c);blocks=session_blocks(index,rows,source)
            repeat=repeat_session_lists(index,rows,source)
            # Verify no source session escaped its explicit user/date/population block.
            for block in blocks.sessions:
                assert len({(rows[i]['partition'],rows[i]['user_id'],rows[i]['recording_date']) for i in block})==1
            covered=[int(i) for block in blocks.sessions for i in block]
            assert len(set(covered))==len(covered) and set(covered).isdisjoint(blocks.unknown.tolist())
            assert set(covered)|set(blocks.unknown.tolist())==set(range(len(index.sample_ids)))
            sessions[partition]={'rows':len(index.sample_ids),'known_time_rows':len(covered),'unknown_time_rows':len(blocks.unknown),
                'sessions':len(blocks.sessions),'repeat_session_lists':len(repeat),'cross_user_or_partition_edges':0}
    finally:Path.open=original
    try:require_task11_formal_ready(closure)
    except ValueError as error:reason=str(error)
    else:raise AssertionError('Task11 pending evidence unexpectedly opened formal bank')
    proof={'protocol_sha256':p.identity(),'session_adapter_acceptance':'PASS','partitions':sessions,
        'closure_path':str(p.run_root/'protocol/task11_source_closure_v1.json'),'formal_generation_ready':False,
        'task11_complete':False,'status':'BLOCKED_FORMAL_SOURCE_BANK','block_reason':reason,
        'a18_excluded_by_actual_roots':True,'original_roster_modified':False,'labels_open_attempts':len(attempts),
        'final_labels_read':False,'bank_generated':False,'a8_generated':False,'task12_started':False,
        'seconds':round(time.perf_counter()-start,2)}
    write_json(ROOT/'outputs/task5_audit/P40-task11-preflight.json',proof);print(json.dumps(proof),flush=True)
    return 2


if __name__=='__main__':raise SystemExit(main())
