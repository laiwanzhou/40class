"""Matched own/repeat group classifiers on frozen 23-expert nested banks."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='4'
import argparse,json,time,warnings
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.exceptions import ConvergenceWarning
from .p435_measured_fusion import HERE,ROOT,sha,prior,load_banks,describe
from .p436_sequence_comparison import decode_fold,METADATA
from .p418_nested_repeat_group_bridge import load_recording_metadata,_subset_metadata,fit_group_head,_align_scores
from .stable_routing_structure import build_group_features
from .p90_teacher_common import load_protocol
from .stable_routing_protocol import register_experiment
PREREG=ROOT/'docs/research/STABLE_093_P437_GROUP_COMPARISON.md'

def features(bank,meta,peers):
    geometry=np.einsum('nec,e->nc',bank,prior())[:,None,:]
    return build_group_features(bank,meta,include_peers=peers,grouping_bank=geometry)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out-dir',required=True);args=ap.parse_args();out=Path(args.out_dir)
    if out.exists():raise FileExistsError(out)
    started=time.monotonic();p=load_protocol();keep=~np.isin(p.users,['user1','user2','user21']);ids,y,u,f=[v[keep] for v in(p.sample_ids,p.labels,p.users,p.fold_id)]
    frozen=json.loads((HERE/'runs/p435_measured_fusion_v1/registry.json').read_text())['spec']['input_sha256']
    if any(sha(q)!=h for q,h in frozen.items()):raise ValueError('P435inputs changed')
    inputs={};banks,names=load_banks(ids,u,f,inputs);meta=load_recording_metadata(METADATA,ids)
    if any(a and a!=b for a,b in zip(meta.users,u)):raise ValueError('metadatauser')
    sources=[Path(__file__),PREREG,*[HERE/n for n in ('p435_measured_fusion.py','p436_sequence_comparison.py','p420_source_only_session_bridge.py','p418_nested_repeat_group_bridge.py','p419_vjepa_repeat_group_bridge.py','p416_nested_frozen_family_router.py','stable_routing_structure.py','stable_routing_protocol.py','audit_p87_sequence_decoder.py','p90_teacher_common.py')]]
    inputs.update(frozen);inputs.update({str(q):sha(q) for q in [*sources,METADATA]})
    out.mkdir(parents=True);register_experiment(out/'registry.json',{'name':'P437 fixedmatchedgroup','input_sha256':inputs,'test_rows_loaded':0,'target_achieved':False})
    snap=out/'source_snapshot';snap.mkdir()
    for q in sources:(snap/q.name).write_bytes(q.read_bytes())
    probability={k:np.zeros((len(ids),40)) for k in ('own','repeat')};audit={}
    warnings.filterwarnings('error',category=ConvergenceWarning)
    for fold,(tr,te,inner,outer) in enumerate(banks):
        audit[str(fold)]={};foldout=out/f'fold{fold}';foldout.mkdir()
        for kind,peers in [('own',False),('repeat',True)]:
            trainx=None;coverage=np.zeros(len(tr),int);innerlogs=[]
            for j,(src,val) in enumerate(GroupKFold(3).split(tr,groups=u[tr])):
                x,a=features(inner[val],_subset_metadata(meta,tr[val]),peers)
                if trainx is None:trainx=np.zeros((len(tr),x.shape[1]),np.float32)
                trainx[val]=x;coverage[val]+=1
                innerlogs.append({'inner':j,'source_subjects':sorted(set(u[tr[src]])),'validation_subjects':sorted(set(u[tr[val]])),'audit':a})
            if not np.all(coverage==1):raise ValueError('innercoverage')
            heldx,a=features(outer,_subset_metadata(meta,te),peers)
            with warnings.catch_warnings():
                scaler,head,_=fit_group_head(trainx,y[tr],np.arange(len(tr)),np.array([],int),u[tr])
            pred=_align_scores(head.decision_function(scaler.transform(heldx)),head.classes_,len(te));probability[kind][te]=pred
            np.savez_compressed(foldout/f'{kind}_model.npz',mean=scaler.mean_,scale=scaler.scale_,var=scaler.var_,coef=head.coef_,intercept=head.intercept_,classes=head.classes_,n_iter=head.n_iter_)
            np.savez_compressed(foldout/f'{kind}_held.npz',sample_ids=ids[te],features=heldx,probability=pred)
            audit[str(fold)][kind]={'source_subjects':sorted(set(u[tr])),'held_subjects':sorted(set(u[te])),'inner_partitions':innerlogs,'outer_geometry':a,'head_features':trainx.shape[1]}
            if time.monotonic()-started>600:raise TimeoutError('P437budget')
            print(json.dumps({'fold':fold,'head':kind,'elapsed':time.monotonic()-started}),flush=True)
    predictions={kind+'_raw':v.argmax(1) for kind,v in probability.items()}
    for kind,prob in probability.items():
        decoded=np.full(len(ids),-1)
        for fold in range(3):
            te,pred,log=decode_fold(prob,y,u,f,fold,meta);decoded[te]=pred
            log['collapsed_upstream_scope']='P437 source-inner fitted '+kind+' group head over frozen 23 expert banks'
            log['group_model_artifact']=f'fold{fold}/{kind}_model.npz'
            nodes=log['artifact_dag'];nodes['source_group_head']=nodes.pop('frozen_source_mixture')
            nodes['sequence']['parents']=['source_group_head','source_transition']
            audit[str(fold)][kind]['sequence']=log
        predictions[kind+'_sequence']=decoded
    results={k:describe(v,y,u,f) for k,v in predictions.items()};comparisons={};us=sorted(set(u));counts=np.array([np.sum(u==s) for s in us]);draw=np.random.default_rng(437).integers(0,len(us),(20000,len(us)))
    for cand,base in [('repeat_sequence','own_sequence'),('repeat_raw','own_raw')]:
        a=predictions[cand]==y;b=predictions[base]==y;d=a.astype(int)-b.astype(int);g=np.array([d[u==s].sum() for s in us]);means=g/counts
        ci=np.quantile(g[draw].sum(1)/counts[draw].sum(1),[.005,.995]);sci=np.quantile(means[draw].mean(1),[.005,.995])
        comparisons[cand+' minus '+base]={'rescue':int(np.sum(a&~b)),'harm':int(np.sum(b&~a)),'net':int(d.sum()),'sample_delta_ci99':ci.tolist(),'subject_delta_ci99':sci.tolist(),
            'gate':bool(d.sum()>0 and means.mean()>0 and results[cand]['worst_subject_accuracy']>=results[base]['worst_subject_accuracy'] and ci[0]>0 and sci[0]>0)}
    if any(sha(q)!=h for q,h in inputs.items()):raise ValueError('inputs changed')
    if time.monotonic()-started>600:raise TimeoutError('P437budget')
    report={'results':results,'comparisons':comparisons,'beats_P420':results['repeat_sequence']['correct']>2164,'target_achieved':False,'test_rows_loaded':0,'seconds':time.monotonic()-started,'limitation':'reused development subjects,not completeP315orofficialscore'}
    np.savez_compressed(out/'predictions.npz',sample_ids=ids,users=u,folds=f,**predictions)
    (out/'audit.json').write_text(json.dumps(audit,indent=2));(out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':main()
