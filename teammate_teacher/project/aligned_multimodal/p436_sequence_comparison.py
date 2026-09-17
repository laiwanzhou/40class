"""Fixed matched sequence comparison; reuse P435 weights without refitting."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='4'
import argparse,json,time
from pathlib import Path
import numpy as np
from .p435_measured_fusion import HERE,ROOT,load_banks,prior,describe,sha
from .p420_source_only_session_bridge import eligible_sessions
from .p418_nested_repeat_group_bridge import load_recording_metadata,_subset_metadata
from .audit_p87_sequence_decoder import fit_transition_model,decode_unique_beam
from .p90_teacher_common import load_protocol
from .stable_routing_protocol import register_experiment,ArtifactNode,assert_prediction_provenance
SOURCE=HERE/'runs/p435_measured_fusion_v1'
PREREG=ROOT/'docs/research/STABLE_093_P436_SEQUENCE_COMPARISON.md'
METADATA=HERE/'data/p85_recording_metadata/train_recording_metadata.csv'

def decode_fold(probability,labels,users,folds,fold,metadata):
    tr=np.flatnonzero(folds!=fold);te=np.flatnonzero(folds==fold)
    if set(users[tr])&set(users[te]):raise ValueError('subjectoverlap')
    p=np.asarray(probability[te],float)
    if p.shape!=(len(te),40) or not np.isfinite(p).all() or np.min(p)<0 or not np.allclose(p.sum(1),1):raise ValueError('emissions')
    src,sa=eligible_sessions(_subset_metadata(metadata,tr));dst,da=eligible_sessions(_subset_metadata(metadata,te))
    transition=fit_transition_model(labels[tr],src,40,1.,alpha=.25)
    pred=p.argmax(1);mix=.65*p+.35*np.eye(40)[pred];mix/=mix.sum(1,keepdims=True)
    logp=np.log(np.clip(mix,1e-9,1))
    for session in dst:pred[session]=decode_unique_beam(logp[session],transition,.45,50)
    nodes={'frozen_source_mixture':ArtifactNode('frozen_source_mixture',supervised_train_subjects=frozenset(users[tr]),provenance='supervised',has_task_labels=True),
           'source_transition':ArtifactNode('source_transition',supervised_train_subjects=frozenset(users[tr]),provenance='supervised',has_task_labels=True),
           'sequence':ArtifactNode('sequence',parents=('frozen_source_mixture','source_transition'))}
    for u in set(users[te]):assert_prediction_provenance(u,['sequence'],nodes)
    return te,pred,{'source_subjects':sorted(set(users[tr])),'held_subjects':sorted(set(users[te])),
                    'source_geometry':sa,'target_geometry':da,'target_sessions':[te[s].tolist() for s in dst],
                    'collapsed_upstream_scope':'frozen P435 registry and verified source-inner weights',
                    'artifact_dag':{k:{'parents':list(v.parents),'supervised_train_subjects':sorted(v.supervised_train_subjects),'provenance':v.provenance} for k,v in nodes.items()}}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out-dir',required=True);a=ap.parse_args();out=Path(a.out_dir)
    if out.exists():raise FileExistsError(out)
    start=time.monotonic();p=load_protocol();keep=~np.isin(p.users,['user1','user2','user21']);ids,y,users,folds=[v[keep] for v in (p.sample_ids,p.labels,p.users,p.fold_id)]
    frozen=json.loads((SOURCE/'registry.json').read_text())['spec']['input_sha256']
    if any(sha(f)!=h for f,h in frozen.items()):raise ValueError('P435inputchanged')
    weights=json.loads((SOURCE/'summary.json').read_text())['weights'];inputs={};banks,names=load_banks(ids,users,folds,inputs)
    metadata=load_recording_metadata(METADATA,ids)
    if any(m and m!=u for m,u in zip(metadata.users,users)):raise ValueError('metadatauserdiffers')
    sources=[Path(__file__),PREREG,*[HERE/n for n in ('p435_measured_fusion.py','p420_source_only_session_bridge.py','p418_nested_repeat_group_bridge.py','p416_nested_frozen_family_router.py','p419_vjepa_repeat_group_bridge.py','stable_routing_structure.py','stable_routing_protocol.py','audit_p87_sequence_decoder.py','p90_teacher_common.py')]]
    paths=[*sources,METADATA,SOURCE/'registry.json',SOURCE/'summary.json',SOURCE/'predictions.npz']
    inputs.update({str(q):sha(q) for q in paths});inputs.update(frozen)
    out.mkdir(parents=True);register_experiment(out/'registry.json',{'name':'P436 fixedsequence','input_sha256':inputs,'test_rows_loaded':0,'target_achieved':False})
    snap=out/'source_snapshot';snap.mkdir()
    for q in sources:(snap/q.name).write_bytes(q.read_bytes())
    probabilities={k:np.zeros((len(ids),40)) for k in ('family','learned')}
    for f,(tr,te,inner,outer) in enumerate(banks):
        if weights[f]['fold']!=f:raise ValueError('weightsfold')
        w=np.asarray(weights[f]['weights']);
        if w.shape!=(23,) or np.min(w)<0 or not np.isclose(w.sum(),1):raise ValueError('weights')
        probabilities['family'][te]=np.einsum('nec,e->nc',outer,prior());probabilities['learned'][te]=np.einsum('nec,e->nc',outer,w)
    with np.load(SOURCE/'predictions.npz') as z:
        if not np.array_equal(z['sample_ids'],ids):raise ValueError('P435IDs')
        for name,key in [('family','family_equal'),('learned','learned')]:
            if not np.array_equal(probabilities[name].argmax(1),z[key]):raise ValueError('rawpredictionchanged')
    predictions={name+'_raw':v.argmax(1) for name,v in probabilities.items()};logs={}
    for name,prob in probabilities.items():
        pred=np.full(len(ids),-1);logs[name]=[]
        for f in range(3):
            te,local,log=decode_fold(prob,y,users,folds,f,metadata);pred[te]=local;logs[name].append(log)
        predictions[name+'_sequence']=pred
    results={k:describe(v,y,users,folds) for k,v in predictions.items()}
    comparisons={};draw=np.random.default_rng(436).integers(0,15,(20000,15));us=sorted(set(users));counts=np.array([np.sum(users==u) for u in us])
    for candidate,base in [('learned_sequence','family_sequence'),('learned_sequence','learned_raw'),('family_sequence','family_raw')]:
        ca=predictions[candidate]==y;ba=predictions[base]==y;delta=ca.astype(int)-ba.astype(int);g=np.array([delta[users==u].sum() for u in us]);means=g/counts
        ci=np.quantile(g[draw].sum(1)/counts[draw].sum(1),[.005,.995]);sci=np.quantile(means[draw].mean(1),[.005,.995])
        comparisons[candidate+' minus '+base]={'rescue':int(np.sum(ca&~ba)),'harm':int(np.sum(ba&~ca)),'net':int(delta.sum()),'sample_delta_ci99':ci.tolist(),'subject_delta_ci99':sci.tolist(),
          'gate':bool(delta.sum()>0 and means.mean()>0 and results[candidate]['worst_subject_accuracy']>=results[base]['worst_subject_accuracy'] and ci[0]>0 and sci[0]>0)}
    report={'results':results,'comparisons':comparisons,'sequence_logs':logs,'beats_P420_benchmark':results['learned_sequence']['correct']>2164,
            'target_achieved':False,'test_rows_loaded':0,'official_best_reported':.91542,'limitation':'different23-expert system, reusedsubjects, notfullP315 or independenttest','seconds':time.monotonic()-start}
    if any(sha(q)!=h for q,h in inputs.items()):raise ValueError('inputs changed')
    if time.monotonic()-start>300:raise TimeoutError('P436budget')
    np.savez_compressed(out/'predictions.npz',sample_ids=ids,users=users,folds=folds,**predictions)
    (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in report.items() if k!='sequence_logs'}),flush=True)

if __name__=='__main__':main()
