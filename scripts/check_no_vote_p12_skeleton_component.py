"""Bounded check of completed Skeleton masters, FP16 copies and four small outputs."""
from pathlib import Path
import json
import sys
import hashlib
import subprocess
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.experiments.no_vote_protocol import load_protocol
from src.experiments.artifact_record import ArtifactRegistry,VerificationContext
from src.experiments.no_vote_types import write_json
from src.experiments.no_vote_manifest import read_public_rows,row_index
from src.experiments.p12_reconstruction import REVISION,ref

p=load_protocol(ROOT/'configs/experiments/teammate_single_teacher_fixed_split_v3.yaml',inference_only=True)
c=VerificationContext(p);registry=ArtifactRegistry(p,context=c);root=p.run_root/'protocol'/REVISION
master=json.loads((root/'skeleton_component.json').read_text(encoding='utf-8'));copy=json.loads((root/'skeleton_fp16_component.json').read_text(encoding='utf-8'))
historical=hashlib.sha256(subprocess.check_output(['git','show','69eed6d:src/experiments/p12_reconstruction.py'],cwd=ROOT)).hexdigest()
summary={'models':{},'outputs':{},'producer_commit':'69eed6d','master_source_sha256':historical,
    'full_raw_or_large_cache_scan':False,'final_labels_read':False,'task11_complete':False}
for phase in ['select','refit']:
    a=registry.verify(ref(master['models'][phase]),'source_expert/p12_skeleton',phase)
    b=registry.verify(ref(copy['models'][phase]),a.stage,phase)
    assert set(a.source_hashes.values())=={historical}
    checkpoint=next(Path(x) for x in a.files if Path(x).name=='checkpoint.pt')
    half=next(Path(x) for x in b.files if Path(x).name=='checkpoint.pt')
    state=torch.load(checkpoint,map_location='cpu',weights_only=True);half_state=torch.load(half,map_location='cpu',weights_only=True)
    assert state['epoch']==15 and half_state['epoch']==15
    history=json.loads(checkpoint.with_name('history.json').read_text(encoding='utf-8'))
    assert [x['epoch'] for x in history]==list(range(1,16))
    for key,value in state['model_state_dict'].items():
        expected=value.half() if value.is_floating_point() else value
        torch.testing.assert_close(expected,half_state['model_state_dict'][key],atol=0,rtol=0)
    summary['models'][phase]={'epochs':15,'fit_rows':a.config['fit_rows'],'sampled_rows_per_epoch':a.config['sampled_rows_per_epoch'],
        'training_seconds':sum(x['seconds'] for x in history),'fp16_exact_export':True,'master_source_matches_historical_commit':True}
for partition in p.partitions:
    index=row_index(read_public_rows(p.public_manifests[partition]));phase='select' if partition in {'train12','development2'} else 'refit'
    record=registry.verify(ref(copy['outputs'][partition]),'source_expert/p12_skeleton','predict',index)
    file=next(Path(x) for x in record.files if Path(x).suffix=='.npz');model=registry.read(ref(copy['models'][phase]))
    with np.load(file,allow_pickle=False) as z:
        assert tuple(z['sample_ids'].tolist())==index.sample_ids
        np.testing.assert_array_equal(z['class_ids'],np.arange(40))
        scores=z['logits'];prob=z['probabilities'];valid=z['valid']
        assert scores.shape==prob.shape==(len(index.sample_ids),40) and np.isfinite(scores).all() and np.isfinite(prob).all()
        np.testing.assert_allclose(prob.sum(1),1.,atol=1e-6)
        expected=np.tile(np.asarray(model.config['prior'],np.float32),(int((~valid).sum()),1))
        np.testing.assert_array_equal(prob[~valid],expected)
        summary['outputs'][partition]={'rows':len(valid),'valid':int(valid.sum()),'prior_rows':int((~valid).sum())}
write_json(ROOT/'outputs/task5_audit/P41-p12-skeleton-acceptance.json',summary)
print(json.dumps(summary),flush=True)
