"""Independent canonical and durable run acceptance for the P433 pilot."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
from .p90_teacher_common import load_protocol
from .p433_physical_cache import PhysicalTokenCache
from .p433_physical_provider import EXCLUDED_USERS,BASE_SEEDS,PhysicalProvider
from .p427_foundation_provider import array_hash
from .stable_routing_protocol import ProtocolError,experiment_sha256


def derive():
    from . import p433_rebuild_physical_bank as r
    p=load_protocol();cache=PhysicalTokenCache(manifest=r.CANONICAL)
    keep=~np.isin(p.users,list(EXCLUDED_USERS));ids,y,users,folds=[v[keep] for v in (p.sample_ids,p.labels,p.users,p.fold_id)]
    tr=np.flatnonzero(folds!=0);te=np.flatnonzero(folds==0)
    if (len(ids),len(tr),len(te))!=(2470,1497,973):raise ProtocolError('P433 canonical population differs')
    provider=PhysicalProvider(cache,ids,users);provider._revalidate_cache()
    expected={'context':'fold0.outer','outer_fold':0,'source_ids':ids[tr].tolist(),'source_users':users[tr].tolist(),
              'target_ids':ids[te].tolist(),'target_users':users[te].tolist(),'source_label_sha256':array_hash(y[tr]),
              'source_class_counts':np.bincount(y[tr],minlength=40).tolist(),'cache_provenance':cache.provenance}
    return r,cache,expected


def verify_run(out):
    torch.set_num_threads(4);out=Path(out);r,cache,expected=derive()
    process=json.loads(Path(str(out)+'.process.json').read_text())
    command=[sys.executable,'-u','-m','aligned_multimodal.p433_rebuild_physical_bank','--out-dir',str(out)]
    if (process.get('status')!='complete' or process.get('exit_code')!=0 or not 0<=process.get('seconds',-1)<=960
        or process.get('wall_limit_seconds')!=960 or process.get('command')!=command):raise ProtocolError('P433 process invalid')
    s=json.loads((out/'summary.json').read_text())
    if (s.get('mode')!='pilot' or s.get('contexts_completed')!=1 or not 0<=s.get('elapsed_seconds',-1)<=900
        or s.get('test_rows_loaded')!=0 or any(s.get(k) is not False for k in
        ('complete_p315','target_achieved','outer_accuracy_evaluated','promotion_allowed','submission_generated'))):raise ProtocolError('P433 summary scope invalid')
    sources=r._source_files();inputs=r._input_files(cache)
    spec={'name':'P433 original physical token expert','mode':'pilot','outer_fold':0,
          'source_sha256':{r._key(p):r.sha(p) for p in sources},'input_sha256':{r._key(p):r.sha(p) for p in inputs},
          'expert_names':['p238_physical_token'],'outer_accuracy_evaluated':False,'promotion_allowed':False,'excluded_users':sorted(EXCLUDED_USERS)}
    reg=json.loads((out/'experiment_registry.json').read_text())
    if reg.get('spec')!=spec or reg.get('sha256')!=experiment_sha256(spec):raise ProtocolError('P433 registration differs')
    snapshots={'__'.join(p.resolve().relative_to(r.ROOT).parts):p for p in sources}
    snap=out/'source_snapshot'
    if {p.name for p in snap.iterdir()}!=set(snapshots) or any(r.sha(snap/n)!=r.sha(p) for n,p in snapshots.items()):raise ProtocolError('P433 snapshots differ')
    fixed={'experiment_registry.json','expected.json','summary.json','fold0/outer/bank.npz','fold0/outer/provenance.json'}
    fixed.update(f'fold0/outer/members/seed{seed}/{file}' for seed in BASE_SEEDS for file in ('checkpoint.pt','outputs.npz','receipt.json'))
    actual={p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file() and 'source_snapshot' not in p.parts}
    if actual!=fixed or set(s.get('artifact_sha256',{}))!=fixed-{'summary.json'}:raise ProtocolError('P433 artifact inventory differs')
    if any(r.sha(out/k)!=v for k,v in s['artifact_sha256'].items()):raise ProtocolError('P433 artifact hash differs')
    if json.loads((out/'expected.json').read_text())!=expected:raise ProtocolError('P433 expected differs from canonical')
    from .p433_physical_verify import verify_context
    verify_context(out/'fold0/outer',expected)
    return s


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out-dir',required=True)
    verify_run(p.parse_args().out_dir);print('P433 run independently verified')
