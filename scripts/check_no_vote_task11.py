"""Bounded metadata/small-output acceptance and cached label-free A8 reuse."""
from pathlib import Path
import json
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT))
from src.experiments.no_vote_protocol import load_protocol
from src.experiments.artifact_record import ArtifactRegistry,VerificationContext,LEARNED
from src.experiments.no_vote_types import write_json,canonical_hash
from src.experiments.no_vote_manifest import row_index,read_public_rows
from src.experiments.p12_reconstruction import REVISION,ref
from src.experiments.source_postprocessing import apply_postprocessor

started=time.perf_counter();p=load_protocol(ROOT/'configs/experiments/teammate_single_teacher_fixed_split_v3.yaml',inference_only=True)
c=VerificationContext(p);r=ArtifactRegistry(p,context=c);protocol=p.run_root/'protocol'/REVISION
body=json.loads((protocol/'source_postprocessing_component.json').read_text(encoding='utf-8'));summary={'partitions':{},'budgets':{},'final_labels_read':False,'accuracy_computed':False}
def arrays(reference):
    rec=r.read(reference);path=next(Path(x) for x in rec.files if Path(x).suffix=='.npz');r.verify_file(rec,path)
    return rec,c.load_arrays(path)
seen={}
def ancestry(reference,phase):
    if reference.sha256 in seen:
        if seen[reference.sha256].kind in LEARNED:assert seen[reference.sha256].phase==phase
        return
    rec=r.read(reference);assert rec.complete and not rec.fixture and rec.protocol_sha256==p.identity() and rec.config_sha256==canonical_hash(rec.config)
    if rec.kind in LEARNED:
        assert rec.phase==phase and set(rec.fit_users)==set(p.partitions['train12' if phase=='select' else 'refit14'].users)
    seen[reference.sha256]=rec
    for parent in rec.parents:ancestry(parent,phase)
for pop in p.partitions:
    phase='select' if pop in {'train12','development2'} else 'refit';index=row_index(read_public_rows(p.public_manifests[pop]))
    reference=ref(body['outputs'][pop]);ancestry(reference,phase);rec,a=arrays(reference)
    assert rec.rows==index and tuple(a['sample_ids'].tolist())==index.sample_ids
    np.testing.assert_array_equal(a['class_ids'],np.arange(40))
    assert a['probabilities'].shape==(len(index.sample_ids),40) and np.isfinite(a['probabilities']).all()
    prior=np.asarray(r.read(ref(body['safe_models'][phase])).config.get('prior',[]))
    pr=next(r.read(x) for x in rec.parents if r.read(x).stage=='class_prior')
    prior=np.asarray(json.loads(Path(next(iter(pr.files))).read_text(encoding='utf-8'))['probabilities'],np.float32)
    valid=a['target_mask'];assert np.array_equal(valid,a['valid'])
    expected=np.full((len(valid),40),.0005,np.float32)
    prediction=np.where(a['new_group_prediction']!=a['old_group_prediction'],a['new_group_prediction'],a['sequence_prediction'])
    expected[np.arange(len(valid)),prediction]=.9805;expected[~valid]=prior
    np.testing.assert_array_equal(a['probabilities'],expected);np.testing.assert_array_equal(a['emission_probability'],expected)
    np.testing.assert_array_equal(a['structured_distillation_probability'],expected)
    np.testing.assert_array_equal(a['old_group_prediction'],a['new_group_prediction'])
    assert not a['precedence_route'].any()
    _,bank=arrays(ref(body['banks'][pop]));assert bank['probabilities'].shape==(len(valid),20,40)
    np.testing.assert_allclose(bank['probabilities'].sum(2),1.,atol=1e-5)
    assert np.array_equal(bank['availability'],~bank['fallback'])
    summary['partitions'][pop]={'rows':len(valid),'valid':int(valid.sum()),'prior':int((~valid).sum()),
        'group_changes':int(a['group_route'].sum()),'sequence_changes':int(a['sequence_route'].sum()),'precedence_changes':0,'same_phase_ancestry':True}
assert summary['partitions']['final4']['rows']==609 and summary['partitions']['final4']['valid']==591 and summary['partitions']['final4']['prior']==18
for role in ['skeleton','depth','thermal']:
    comp=json.loads((protocol/(role+'_fp16_component.json')).read_text(encoding='utf-8'))
    for phase,value in comp['models'].items():
        rec=r.read(ref(value));master=next(r.read(x) for x in rec.parents if r.read(x).stage==rec.stage and r.read(x).kind=='supervised_model')
        history=json.loads(next(Path(x) for x in master.files if Path(x).name=='history.json').read_text(encoding='utf-8'))
        assert [h['epoch'] for h in history]==list(range(1,16))
        summary['budgets'][role+'_'+phase]={'epochs':15,'fit_rows':master.config['fit_rows']}
base=json.loads((protocol/'source_base_adapt12_component.json').read_text(encoding='utf-8'))
for phase,value in base['models'].items():
    rec=r.read(ref(value));history=json.loads(next(Path(x) for x in rec.files if Path(x).name=='history.json').read_text(encoding='utf-8'))
    assert [h['epoch'] for h in history]==list(range(1,13)) and all(h['ground_truth_fields_seen']==0 for h in history)
    summary['budgets']['source_base_'+phase]={'epochs':12,'fit_rows':rec.config['fit_rows']}
# Cached application must leave published bytes/stamps untouched and must not
# open even the three available supervised-label CSVs during inference/reuse.
reference=ref(body['outputs']['final4']);rec=r.read(reference);file=Path(next(iter(rec.files)));before=c.stamp(file)
original=Path.open;blocked={Path(x).resolve() for x in p.supervised_labels.values()};attempts=[]
def guarded(path,*args,**kwargs):
    if path.resolve() in blocked:attempts.append(str(path));raise AssertionError('label payload opened during cached A8 application')
    return original(path,*args,**kwargs)
Path.open=guarded
try:again=apply_postprocessor(p,'final4',ref(body['banks']['final4']),{k:ref(v) for k,v in body['post_models']['refit'].items()},c)
finally:Path.open=original
assert again==reference and c.stamp(file)==before and not attempts
summary.update(unique_ancestor_metadata=len(seen),cached_A8_reference_and_payload_stamp_unchanged=True,supervised_label_open_attempts=len(attempts),seconds=time.perf_counter()-started)
write_json(ROOT/'outputs/task5_audit/P44-final-acceptance.json',summary);print(json.dumps(summary),flush=True)
