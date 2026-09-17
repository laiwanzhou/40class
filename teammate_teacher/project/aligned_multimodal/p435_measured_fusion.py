"""Frozen, source-inner weight fitting and one outer evaluation of 23 experts."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='4'
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold
from .p90_teacher_common import load_protocol
from .stable_routing_protocol import register_experiment

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
RUNS=['p427_foundation_rebuild_v1','p428_skeleton_rebuild_v1','p429_thermal_rebuild_full_v1','p430_token_rebuild_full_v1','p431_irthermal_full_v2','p432_lavila_full_v1','p433_physical_full_v1']
PREREG=ROOT/'docs/research/STABLE_093_P435_COMPARISON.md'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def prior():return np.array([1/54]*6+[1/90]*10+[1/9]*7,dtype=float)

def fit_weights(bank,labels,users):
    q=bank[np.arange(len(labels)),:,labels].astype(float)
    unique,counts=np.unique(users,return_counts=True);lookup=dict(zip(unique,counts));a=np.array([1/lookup[u]/len(unique) for u in users])
    w0=prior()
    def objective(w):
        p=np.maximum(q@w,1e-12);delta=w-w0
        return float(-a@np.log(p)+.01*(delta@delta)), -(q.T@(a/p))+.02*delta
    r=minimize(objective,w0,jac=True,method='SLSQP',bounds=[(0.,1.)]*23,
               constraints={'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(23)},options={'maxiter':500,'ftol':1e-10})
    if not r.success or not np.isfinite(r.x).all() or np.min(r.x)<-1e-8 or not np.isclose(r.x.sum(),1):raise RuntimeError(str(r.message))
    w=np.maximum(r.x,0);w/=w.sum()
    return w,{'iterations':int(r.nit),'objective':float(r.fun),'source_subjects':unique.tolist()}

def load_banks(ids,users,folds,inputs):
    banks=[];roster=None
    for f in range(3):
        tr=np.flatnonzero(folds!=f);te=np.flatnonzero(folds==f);inners=[];outers=[];names=[]
        for n in RUNS:
            folder=HERE/'runs'/n;summary_path=folder/'summary.json';summary=json.loads(summary_path.read_text());inputs[str(summary_path)]=sha(summary_path)
            if summary.get('test_rows_loaded')!=0 or summary.get('target_achieved') is not False:raise ValueError('upstream scope')
            hashes={k.replace('\\','/'):v for k,v in summary.get('artifact_sha256',summary.get('artifacts',{})).items()}
            def read(path):
                rel=path.relative_to(folder).as_posix();h=sha(path)
                if hashes.get(rel)!=h:raise ValueError('upstream bank not verified: '+str(path))
                inputs[str(path)]=h;return np.load(path,allow_pickle=False)
            aggregate=folder/f'fold{f}_banks.npz'
            if aggregate.exists():
                with read(aggregate) as z:
                    for key,value in [('inner_sample_ids',ids[tr]),('outer_sample_ids',ids[te]),('inner_users',users[tr]),('outer_users',users[te])]:
                        if not np.array_equal(z[key],value):raise ValueError('aggregate identity mismatch')
                    inner=z['inner_probability_bank'];outer=z['outer_probability_bank'];expert=z['expert_names'].astype(str).tolist()
            else:
                with read(folder/f'fold{f}/outer/bank.npz') as z:
                    if not np.array_equal(z['sample_ids'],ids[te]) or not np.array_equal(z['users'],users[te]):raise ValueError('outer identity')
                    outer=z['probabilities'];expert=z['expert_names'].astype(str).tolist()
                inner=np.zeros((len(tr),len(expert),40));coverage=np.zeros(len(tr),int)
                for j,(_,val) in enumerate(GroupKFold(3).split(tr,groups=users[tr])):
                    with read(folder/f'fold{f}/inner{j}/bank.npz') as z:
                        if not np.array_equal(z['sample_ids'],ids[tr[val]]) or not np.array_equal(z['users'],users[tr[val]]) or z['expert_names'].astype(str).tolist()!=expert:raise ValueError('inner identity')
                        inner[val]=z['probabilities'];coverage[val]+=1
                if not np.all(coverage==1):raise ValueError('inner coverage')
            for b,size in ((inner,len(tr)),(outer,len(te))):
                if b.shape!=(size,len(expert),40) or not np.isfinite(b).all() or np.min(b)<0 or not np.allclose(b.sum(2),1,atol=2e-5):raise ValueError('invalid probability bank')
            inners.append(inner);outers.append(outer);names+=expert
        if len(names)!=23 or len(set(names))!=23 or (roster is not None and names!=roster):raise ValueError('expert roster')
        roster=names;banks.append((tr,te,np.concatenate(inners,1),np.concatenate(outers,1)))
    return banks,roster

def describe(pred,y,users,folds):
    by={u:{'rows':int(np.sum(users==u)),'correct':int(np.sum((pred==y)&(users==u))),'accuracy':float(np.mean(pred[users==u]==y[users==u]))} for u in sorted(set(users))}
    values=[v['accuracy'] for v in by.values()]
    return {'correct':int(np.sum(pred==y)),'rows':len(y),'accuracy':float(np.mean(pred==y)),'subject_mean':float(np.mean(values)),
            'subject_std':float(np.std(values,ddof=1)),'worst_subject_accuracy':min(values),'by_subject':by,
            'fold_correct':[int(np.sum((pred==y)&(folds==f))) for f in range(3)]}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out-dir',required=True);args=ap.parse_args();out=Path(args.out_dir)
    if out.exists():raise FileExistsError(out)
    started=time.monotonic();p=load_protocol();keep=~np.isin(p.users,['user1','user2','user21']);ids,y,users,folds=[v[keep] for v in (p.sample_ids,p.labels,p.users,p.fold_id)]
    if len(ids)!=2470:raise ValueError('population')
    sources=[Path(__file__),PREREG,HERE/'p90_teacher_common.py',HERE/'stable_routing_protocol.py']
    inputs={str(q):sha(q) for q in [*sources,HERE/'data/manifest.csv',*[HERE/f'data/subject_folds/fold_{f}.csv' for f in range(3)]]}
    banks,names=load_banks(ids,users,folds,inputs);out.mkdir(parents=True)
    register_experiment(out/'registry.json',{'name':'P435 fixed23expert comparison','input_sha256':inputs,'expert_names':names,'test_rows_loaded':0,'target_achieved':False})
    snap=out/'source_snapshot';snap.mkdir()
    for q in sources:(snap/q.name).write_bytes(q.read_bytes())
    predictions={k:np.full(len(y),-1,int) for k in ('flat_equal','family_equal','learned','source_selected_single')};weights=[];experts=np.zeros((len(y),23),int)
    for f,(tr,te,inner,outer) in enumerate(banks):
        w,record=fit_weights(inner,y[tr],users[tr]);best=int(np.argmax(np.mean(inner.argmax(2)==y[tr,None],axis=0)))
        for key,weight in [('flat_equal',np.ones(23)/23),('family_equal',prior()),('learned',w)]:predictions[key][te]=np.einsum('nec,e->nc',outer,weight).argmax(1)
        predictions['source_selected_single'][te]=outer[:,best].argmax(1);experts[te]=outer.argmax(2)
        weights.append({'fold':f,'weights':w.tolist(),'source_selected_expert':names[best],**record})
    # Outer labels first used for evaluation below, after all predictions frozen.
    results={k:describe(v,y,users,folds) for k,v in predictions.items()}
    baseline=predictions['family_equal']==y;candidate=predictions['learned']==y;difference=candidate.astype(int)-baseline.astype(int)
    us=sorted(set(users));counts=np.array([np.sum(users==u) for u in us]);gains=np.array([difference[users==u].sum() for u in us]);means=gains/counts
    draw=np.random.default_rng(435).integers(0,len(us),(20000,len(us)));ci=np.quantile(gains[draw].sum(1)/counts[draw].sum(1),[.005,.995]);sci=np.quantile(means[draw].mean(1),[.005,.995])
    report={'results':results,'weights':weights,'expert_names':names,'expert_accuracy':dict(zip(names,np.mean(experts==y[:,None],axis=0).tolist())),
            'rescue':int(np.sum(candidate&~baseline)),'harm':int(np.sum(~candidate&baseline)),'net':int(difference.sum()),'sample_delta_ci99':ci.tolist(),'subject_delta_ci99':sci.tolist(),
            'advance_gate':bool(difference.sum()>0 and means.mean()>0 and results['learned']['worst_subject_accuracy']>=results['family_equal']['worst_subject_accuracy'] and ci[0]>0 and sci[0]>0),
            'beats_source_selected_single':results['learned']['correct']>results['source_selected_single']['correct'],'official_best_reported':.91542,'target_achieved':False,'test_rows_loaded':0,
            'limitation':'23-expert diagnostic, not complete P315; reused development subjects, not independent Test','seconds':time.monotonic()-started}
    if any(sha(q)!=h for q,h in inputs.items()):raise ValueError('inputs changed')
    if time.monotonic()-started>300:raise TimeoutError('P435 budget')
    np.savez_compressed(out/'predictions.npz',sample_ids=ids,users=users,folds=folds,**predictions)
    (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
