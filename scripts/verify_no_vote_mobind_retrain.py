"""Bounded verification for corrected Task8, preserving all legacy evidence."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.experiments.no_vote_protocol import load_protocol
from src.experiments.artifact_record import ArtifactRegistry,VerificationContext
from src.experiments.no_vote_manifest import read_public_rows,row_index
from src.experiments.no_vote_types import ArtifactRef,RowIndex,canonical_hash,write_json
from src.experiments.motion_cache import load_motion,MOTION_FIELDS
from src.experiments.mobind_pretrain import (TASK8_REVISION,TRAINING_POPULATION_POLICY,motion_ready_path,motion_output_root,
    initialize_motion,NoVoteMotionDataset,motion_validity,motion_training_mask,source_motion_math,combine_motion_probabilities)


def ref(value):return ArtifactRef(Path(value['record_path']),value['sha256'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();p=load_protocol(args.config,inference_only=True);r=p.run_root;c=VerificationContext(p)
    registry=ArtifactRegistry(p,context=c);started=time.perf_counter();ready=json.loads(motion_ready_path(p).read_text(encoding='utf-8'))
    assert ready['complete'] and ready['task8_revision']==TASK8_REVISION and ready['training_population_policy']==TRAINING_POPULATION_POLICY
    m6=json.loads((r/'protocol/task6_artifacts_ready.json').read_text(encoding='utf-8'));cache={}
    for pop in ['refit14','final4']:cache[pop]=load_motion(ref(m6['partitions'][pop]['windows']),protocol=p,context=c)
    models={};model_proof={};metrics={};outputs={};total_bytes=0
    for phase,pop,expected in [('select','train12',1957),('refit','refit14',2342)]:
        mr=registry.verify(ref(ready['models'][phase]),'A5',phase);models[phase]=mr
        assert mr.fit_users==p.partitions[pop].users and not mr.select_users
        assert mr.config['task8_revision']==TASK8_REVISION and mr.config['training_population_policy']==TRAINING_POPULATION_POLICY
        assert mr.config['valid_fit_rows']==expected and mr.config['restored_visual_only_rows']==1
        root=motion_output_root(p)/phase
        assert all(Path(x).parent==root for x in mr.files)
        history=json.loads((root/'history.json').read_text(encoding='utf-8'))
        assert [x['epoch'] for x in history]==list(range(1,25)) and {x['train_samples'] for x in history}=={expected}
        assert all(not any(k.startswith(('val_','validation','development')) for k in row) for row in history)
        assert abs(history[0]['learning_rate']-4e-4)<1e-12 and abs(history[-1]['learning_rate']-2e-5)<1e-12
        index=row_index(read_public_rows(p.public_manifests[pop]));motion_record,arrays=cache['refit14']
        lookup={sid:i for i,sid in enumerate(motion_record.rows.sample_ids)};positions=[lookup[s] for s in index.sample_ids]
        sk,im=motion_validity(arrays)
        path=r/f'A1/{phase}/{pop}_targets.npz'
        # Only small valid/IDs are used from the directly consumed teacher target.
        with np.load(path,allow_pickle=False) as data:
            assert tuple(data['sample_ids'].astype(str))==index.sample_ids
            visual_valid=data['valid']
        valid=motion_training_mask(sk[positions],im[positions],visual_valid);restored=visual_valid & ~(sk|im)[positions]
        chosen=RowIndex(tuple(s for s,v in zip(index.sample_ids,valid) if v),tuple(u for u,v in zip(index.user_ids,valid) if v),index.class_ids)
        assert int(valid.sum())==expected and int(restored.sum())==1
        assert mr.config['fit_ids_sha256']==mr.config['class_weight_population_ids_sha256']==canonical_hash(chosen)
        assert mr.config['restored_visual_only_ids_sha256']==canonical_hash(tuple(s for s,v in zip(index.sample_ids,restored) if v))
        assert mr.source_hashes[str(ROOT/'src/experiments/mobind_pretrain.py')]==hashlib.sha256((ROOT/'src/experiments/mobind_pretrain.py').read_bytes()).hexdigest()
        model_proof[phase]={'epochs':24,'fit_rows':expected,'restored_visual_only_rows':1,'excluded_rows':int((~valid).sum()),
            'population_and_class_weights_digest_match':True,'train_seconds':round(sum(x['seconds'] for x in history),2)}
        total_bytes+=sum(Path(x).stat().st_size for x in mr.files)
    for pop,branches in ready['outputs'].items():
        index=row_index(read_public_rows(p.public_manifests[pop]));phase='select' if pop in {'train12','development2'} else 'refit'
        mr=models[phase];prior=np.asarray(mr.config['prior'],np.float32);record,arrays=cache['final4' if pop=='final4' else 'refit14']
        lookup={sid:i for i,sid in enumerate(record.rows.sample_ids)};positions=[lookup[s] for s in index.sample_ids]
        sk,im=motion_validity(arrays);valid={'skeleton':sk[positions],'imu':im[positions],'combined':(sk|im)[positions]}
        metrics[pop]={};outputs[pop]={}
        labels=None
        if pop!='final4':labels={x['sample_id']:int(x['class_id']) for x in csv.DictReader(p.supervised_labels[pop].open(encoding='utf-8-sig'))}
        for kind,item in branches.items():
            pr=registry.verify(ref(item),{'skeleton':'A5-S','imu':'A5-I','combined':'A5'}[kind],'predict',index)
            assert pr.config['task8_revision']==TASK8_REVISION and ref(ready['models'][phase]) in pr.parents
            path=next(Path(x) for x in pr.files);data=c.load_arrays(registry.verify_file(pr,path));outputs[pop][kind]=data;total_bytes+=path.stat().st_size
            assert tuple(data['sample_ids'].astype(str))==index.sample_ids and np.array_equal(data['class_ids'],np.arange(40))
            assert np.isfinite(data['logits']).all() and np.isfinite(data['probabilities']).all()
            assert np.allclose(data['probabilities'].sum(1),1,atol=1e-6);np.testing.assert_array_equal(data['valid'],valid[kind])
            if (~valid[kind]).any():np.testing.assert_array_equal(data['probabilities'][~valid[kind]],np.tile(prior,((~valid[kind]).sum(),1)))
            result={'rows':len(index.sample_ids),'valid':int(valid[kind].sum()),'prior_rows':int((~valid[kind]).sum()),'correct':None}
            if labels is not None:
                y=np.array([labels[s] for s in index.sample_ids]);correct=data['logits'].argmax(1)==y
                result.update(correct=int(correct.sum()),accuracy=float(correct.mean()))
            metrics[pop][kind]=result
        np.testing.assert_allclose(outputs[pop]['combined']['probabilities'],combine_motion_probabilities(outputs[pop]['skeleton']['probabilities'],outputs[pop]['imu']['probabilities'],valid['skeleton'],valid['imu'],prior),rtol=0,atol=1e-7)
    # Actual model inference without opening labels or any A1/A4 payloads.
    original=Path.open;attempts=[]
    def guard(path,*a,**kw):
        forbidden=path.name.endswith('_labels.csv') or path.name=='final_labels.csv' or (any(x in {'A1','A4'} for x in path.parts) and path.suffix.lower() in {'.npz','.npy','.pt','.joblib'})
        if forbidden:attempts.append(str(path));raise AssertionError('teacher/label reached motion inference')
        return original(path,*a,**kw)
    Path.open=guard
    try:
        mr=models['refit'];checkpoint=next(Path(x) for x in mr.files if Path(x).name=='checkpoint.pt');registry.verify_file(mr,checkpoint)
        state=torch.load(checkpoint,map_location='cpu',weights_only=True);network=initialize_motion(p,device='cpu',context=c)
        network.load_state_dict(state['model_state'],strict=True);network.eval().cuda()
        record,arrays=cache['final4'];dataset=NoVoteMotionDataset(arrays,record.rows)
        batch=torch.utils.data.default_collate([dataset[i] for i in range(64)]);ops=source_motion_math(p,c)
        assert 'label' not in batch and 'teacher_logits' not in batch
        batch=ops['to_device'](batch,torch.device('cuda'));sk=batch['skeleton_joint_mask'].flatten(1).any(1)
        im=batch['imu_global_mask'][:,:,0].any(1)|batch['imu_sequence_mask'].flatten(1).any(1);pos=torch.nonzero(sk|im,as_tuple=False).flatten()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):out=network({k:batch[k].index_select(0,pos) for k in MOTION_FIELDS},mask_ratio=0.)
        differences={}
        for kind,mask,key in [('skeleton',sk,'skeleton_logits'),('imu',im,'imu_logits')]:
            keep=mask.index_select(0,pos).cpu().numpy();rows=pos.cpu().numpy()[keep]
            actual=out[key].float().cpu().numpy()[keep];expected=outputs['final4'][kind]['logits'][rows]
            np.testing.assert_allclose(actual,expected,rtol=0,atol=.01)
            differences[kind]=float(np.max(np.abs(actual-expected),initial=0))
    finally:Path.open=original
    preserved=0
    for task in [6,7,8]:
        old=json.loads((r/f'protocol/task{task}_complete.json').read_text(encoding='utf-8'))
        for item in old['evidence'].values():
            assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256'];preserved+=1
    proof={'protocol_sha256':p.identity(),'task8_revision':TASK8_REVISION,'complete':True,'models':model_proof,'partitions':metrics,
        'legacy_evidence_bindings_preserved':preserved,'inference_teacher_label_open_attempts':len(attempts),
        'inference_max_logit_differences':differences,'models_and_predictions_bytes':total_bytes,'final_labels_read':False,
        'independent_review':'pending','task9_started':False,'seconds':round(time.perf_counter()-started,2)}
    write_json(ROOT/'outputs/task5_audit/P35-task8-final-acceptance.json',proof);print(json.dumps(proof),flush=True)


if __name__=='__main__':main()
