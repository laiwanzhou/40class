"""Bounded final endpoint acceptance: direct outputs and metadata, no truth."""
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.experiments.no_vote_protocol import load_protocol
from src.experiments.no_vote_types import Prediction,canonical_bytes,write_json
from src.experiments.artifact_record import ArtifactRegistry,VerificationContext
from src.experiments.target_adaptation import (target_inputs,target_args,teacher_student_agreement,
    allows_target_stage,require_prediction_binding,predict_target)
from src.experiments.p12_source_adapter import _read_prediction
from src.experiments.p12_reconstruction import ref
from src.experiments.visual_student_v3 import prior

started=time.perf_counter()
p=load_protocol(ROOT/'configs/experiments/teammate_single_teacher_fixed_split_v3.yaml',inference_only=True)
c=VerificationContext(p);r=ArtifactRegistry(p,context=c)
original_open=Path.open;blocked={Path(x).resolve() for x in p.supervised_labels.values()};attempts=[]
def guard(path,*args,**kw):
    if path.resolve() in blocked:attempts.append(str(path));raise RuntimeError('supervised label opened in Task12 acceptance')
    return original_open(path,*args,**kw)
Path.open=guard
try:
    inputs=target_inputs(p,c);body=json.loads((p.run_root/'protocol/task12_artifacts_ready.json').read_text(encoding='utf-8'))
    assert body['protocol_sha256']==p.identity() and body['task12_generation_complete'] and not body['task13_started']
    assert ref(body['base'])==inputs.base and ref(body['targets'])==inputs.targets
    _,ta=_read_prediction(p,inputs.targets,'A8','final4',c)
    target=Prediction(inputs.rows,ta['logits'],ta['probabilities'],ta['valid'],inputs.targets)
    old=json.loads((p.run_root/'protocol/task10_artifacts_ready.json').read_text(encoding='utf-8'))
    bref=ref(old['outputs']['final4']['A7']);br,ba=_read_prediction(p,bref,'A7','final4',c)
    assert inputs.base in br.parents
    before=Prediction(inputs.rows,ba['logits'],ba['probabilities'],ba['valid'],bref)
    _,pr=prior(p,'refit',c);summary={'endpoints':{},'final_labels_read':False,'raw_or_large_cache_full_scan':False}
    assert len(inputs.rows.sample_ids)==609 and int(ta['target_mask'].sum())==591
    for endpoint in [40,12]:
        value=body['endpoints'][str(endpoint)];model=ref(value['model']);rec=r.verify(model,f'A9-{endpoint}','adapt',inputs.rows)
        assert allows_target_stage(r,rec)
        assert not rec.fit_users and not rec.select_users and set(rec.adaptation_users)==set(p.partitions['final4'].users)
        assert inputs.base in rec.parents and inputs.targets in rec.parents
        for key in ['pixels','sequence','motion','normalization']:assert getattr(inputs,key) in rec.parents
        assert not any(r.read(x).kind=='adapted_model' for x in rec.parents)
        assert canonical_bytes(rec.config['adaptation_recipe'])==canonical_bytes(vars(target_args(p,endpoint)))
        for path,digest in rec.source_hashes.items():c.hash_file(Path(path),digest)
        history=json.loads(next(Path(x) for x in rec.files if Path(x).name=='history.json').read_text(encoding='utf-8'))
        assert [x['epoch'] for x in history]==list(range(1,endpoint+1))
        assert all(x['train_samples']==591 and x['ground_truth_fields_seen']==0 for x in history)
        cfg=vars(target_args(p,endpoint))
        assert history[0]['fusion_learning_rate']==cfg['fusion_learning_rate'] and history[-1]['fusion_learning_rate']==cfg['minimum_learning_rate']
        reference=ref(value['prediction']);pred_record,a=_read_prediction(p,reference,f'A9-{endpoint}','final4',c)
        require_prediction_binding(pred_record,model,rec)
        np.testing.assert_array_equal(a['valid'],ta['target_mask'])
        np.testing.assert_array_equal(a['probabilities'][~a['valid']],np.tile(pr,(18,1)))
        prediction=Prediction(inputs.rows,a['logits'],a['probabilities'],a['valid'],reference)
        expected=teacher_student_agreement(target,before,prediction,ta['target_mask'],phase='refit',endpoint=endpoint,base_model=inputs.base,adapted_model=model)
        stored=p.run_root/f'A9-{endpoint}/adapt/teacher_student_agreement.json'
        assert canonical_bytes(json.loads(stored.read_text(encoding='utf-8')))==canonical_bytes(expected)==canonical_bytes(value['agreement'])
        path=next(Path(x) for x in pred_record.files if Path(x).suffix=='.npz');stamp=c.stamp(path)
        reused=predict_target(p,model,inputs,c,device='cpu')  # Cache path only: no forward/model load.
        assert reused.artifact==reference and c.stamp(path)==stamp
        summary['endpoints'][str(endpoint)]=dict(epochs=len(history),rows=609,adapted=591,prior=18,
            scope=cfg['adaptation_scope'],same_base=True,raw_cached_reuse_unchanged=True,
            after_agreement=expected['after_adaptation'],before_agreement=expected['before_adaptation'])
    assert not attempts
    summary.update(seconds=time.perf_counter()-started,supervised_label_open_attempts=len(attempts),status='PASS')
    write_json(ROOT/'outputs/task5_audit/P47-target-acceptance.json',summary)
    print(json.dumps(summary),flush=True)
finally:
    Path.open=original_open
