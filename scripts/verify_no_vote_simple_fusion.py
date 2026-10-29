"""Bounded Task9 acceptance: small scores only, no networks or final truth."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import logsumexp

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.experiments.artifact_record import ArtifactRegistry,VerificationContext
from src.experiments.no_vote_protocol import load_protocol
from src.experiments.no_vote_types import canonical_bytes,canonical_hash,write_json
from src.experiments.no_vote_manifest import read_public_rows,row_index
from src.experiments.simple_fusion import BRANCHES,CANDIDATES,ROLE,ref,current_inputs,load_prediction,read_labels,combine_available


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();p=load_protocol(args.config,inference_only=True);start=time.perf_counter()
    c=VerificationContext(p);r=ArtifactRegistry(p,context=c);root=p.run_root
    ready=json.loads((root/'protocol/task9_artifacts_ready.json').read_text(encoding='utf-8'))
    assert ready['complete'] and ready['role']==ROLE and ready['protocol_sha256']==p.identity()
    inputs,proofs=current_inputs(p,c);assert canonical_bytes(ready['input_refs'])==canonical_bytes(inputs)
    assert dict(ready['prerequisites'])==proofs
    selection_path=Path(ready['selection']);c.hash_file(selection_path,ready['selection_sha256'])
    selection=json.loads(selection_path.read_text(encoding='utf-8'));policy=selection['config']['policy']
    assert selection['stage']=='A6-VSI' and selection['config']['role']==ROLE and not selection['budget']
    assert {x['policy'] for x in selection['config']['candidates']}=={'identity','fitted'}
    best=min(selection['config']['candidates'],key=lambda x:(-x['accuracy'],-x['macro_f1'],-x['worst_user_accuracy'],x['policy']!='identity'))
    assert best['policy']==policy
    phase_configs={};populations={};nll={};maxdiff=0.;payload_bytes=0
    for phase,pop in [('select','train12'),('refit','refit14')]:
        models=[]
        for name in CANDIDATES:
            model=r.verify(ref(ready['models'][phase][name]),name,phase)
            assert model.fit_users==p.partitions[pop].users and not model.adaptation_users
            assert model.select_users==(p.partitions['development2'].users if phase=='select' else ())
            assert model.config['policy']==policy and model.config['role']==ROLE and model.config['fit_partition']==pop
            assert model.config['task8_revision']=='ir_supervision_v2'
            assert model.config['fit_ids_sha256']==canonical_hash(row_index(read_public_rows(p.public_manifests[pop])))
            assert model.config['selection_sha256']==ready['selection_sha256']
            assert model.source_hashes[str(ROOT/'src/experiments/simple_fusion.py')]==c.hash_file(ROOT/'src/experiments/simple_fusion.py')
            models.append(model)
        assert len({canonical_hash(x.config) for x in models})==1
        cfg=models[0].config;phase_configs[phase]=cfg
        index=row_index(read_public_rows(p.public_manifests[pop]));y=read_labels(p.supervised_labels[pop],index);nll[phase]={}
        for k in BRANCHES:
            pred=load_prediction(p,inputs[pop][k],{'visual':'A2','skeleton':'A5-S','imu':'A5-I'}[k],pop,c)
            mask=pred.valid;x=pred.logits[mask].astype(float);truth=y[mask];t=cfg['temperatures'][k]
            def loss(temp):
                z=x/temp;return float((logsumexp(z,axis=1)-z[np.arange(len(z)),truth]).mean())
            assert .25<=t<=4
            if policy=='fitted':assert loss(t)<=loss(1)+1e-7
            else:assert t==1
            nll[phase][k]={'fit_rows':int(mask.sum()),'temperature':t,'nll_identity':loss(1),'nll_selected':loss(t)}
    for pop in p.partitions:
        phase='select' if pop in {'train12','development2'} else 'refit';cfg=phase_configs[phase]
        index=row_index(read_public_rows(p.public_manifests[pop]));branches={k:load_prediction(p,inputs[pop][k],{'visual':'A2','skeleton':'A5-S','imu':'A5-I'}[k],pop,c) for k in BRANCHES}
        populations[pop]={}
        for name,keys in CANDIDATES.items():
            output=load_prediction(p,ref(ready['outputs'][pop][name]),name,pop,c);record=r.read(output.artifact)
            assert record.config['role']==ROLE and record.config['fit_phase']==phase and record.config['policy']==policy
            assert ref(ready['models'][phase][name]) in record.parents
            assert all(inputs[pop][k] in record.parents for k in keys)
            # Independent arithmetic, not the producer's combine function.
            count=sum(branches[k].valid.astype(int) for k in keys);expected=np.zeros_like(output.probabilities,dtype=float)
            for k in keys:
                z=branches[k].logits.astype(float)/cfg['temperatures'][k];z-=z.max(axis=1,keepdims=True)
                probs=np.exp(z);probs/=probs.sum(axis=1,keepdims=True)
                expected+=probs*branches[k].valid[:,None]
            valid=count>0;expected[valid]/=count[valid,None];expected[~valid]=cfg['prior']
            np.testing.assert_array_equal(output.valid,valid)
            # Producer preserves its FP32 source logits; this independent reference
            # divides/exp/reduces in FP64. Bound the rounding and check decisions too.
            np.testing.assert_allclose(output.probabilities,expected,rtol=0,atol=1e-6)
            np.testing.assert_array_equal(output.probabilities.argmax(1),expected.argmax(1))
            np.testing.assert_allclose(output.logits,np.log(np.maximum(output.probabilities,1e-12)),rtol=0,atol=1e-7)
            maxdiff=max(maxdiff,float(np.max(np.abs(output.probabilities-expected))))
            result={'rows':len(index.sample_ids),'valid':int(valid.sum()),'prior_rows':int((~valid).sum()),'correct':None}
            if pop!='final4':
                y=read_labels(p.supervised_labels[pop],index);result['correct']=int((expected.argmax(1)==y).sum())
            populations[pop][name]=result;payload_bytes+=sum(Path(x).stat().st_size for x in record.files)
        if pop=='final4':
            original=Path.open;attempts=[]
            def guard(path,*args,**kwargs):
                if path.name.endswith('_labels.csv') or path.suffix in {'.pt','.pth','.joblib','.safetensors'}:
                    attempts.append(str(path));raise AssertionError('labels/network payload entered mean inference')
                return original(path,*args,**kwargs)
            Path.open=guard
            try:
                for name,keys in CANDIDATES.items():
                    out,mask=combine_available([branches[k] for k in keys],cfg['prior'],index=index,temperatures=[cfg['temperatures'][k] for k in keys])
                    saved=load_prediction(p,ref(ready['outputs'][pop][name]),name,pop,c)
                    np.testing.assert_allclose(out,saved.probabilities,rtol=0,atol=1e-7)
            finally:Path.open=original
            assert not attempts
    if p.recipe['execution_kind']=='formal':
        assert all(x['rows']==609 and x['valid']==591 and x['prior_rows']==18 for x in populations['final4'].values())
    proof={'protocol_sha256':p.identity(),'complete':True,'role':ROLE,'policy':policy,'phase_fit_nll':nll,'partitions':populations,
        'outputs_checked':12,'independent_fp64_probability_tolerance':1e-6,'independent_arithmetic_max_difference':maxdiff,
        'independent_fp64_argmax_equal':True,'inference_forbidden_payload_open_attempts':len(attempts),
        'prerequisite_bindings_preserved':len(proofs),'target_payload_bytes':payload_bytes,'final_labels_read':False,
        'large_cache_scans':0,'network_training':False,'independent_subagent_review':'not_performed',
        'seconds':round(time.perf_counter()-start,2)}
    write_json(ROOT/'outputs/task5_audit/P37-task9-acceptance.json',proof);print(json.dumps(proof),flush=True)


if __name__=='__main__':main()
