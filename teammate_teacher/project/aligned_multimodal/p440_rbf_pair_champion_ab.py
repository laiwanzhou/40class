"""One fixed nonlinear readout for historical phone/game confusion24/26.
SVC C1,gamma scale,classbalanced; margin>=1 and group supports alternative.
No grid/search/Test inference; legacy embedding provenance limits retained.
"""
import json,hashlib
from pathlib import Path
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from .p90_teacher_common import load_protocol
H=Path(__file__).resolve().parent;S=('H1_selection','H2_confirmation','H3_independent_fold0')
def main():
    out=H/'runs/p440_rbf_pair_champion_ab_v1'
    if out.exists():raise FileExistsError(out)
    paths=[H/'runs/p333_p87s_outer_embedding_spaces_v1/embedding_spaces.npz',H/'runs/p310_union_repeat_precedence_teacher_v1/oof_predictions.npz',H/'runs/p307_union_repeat_group_sequence_audit_v1/oof_predictions.npz',H.parent/'runs/p90_crossuser_visual_router_v1/full_predictions.npz']
    e,a,b,z=[np.load(p,allow_pickle=False) for p in paths];p=load_protocol();outputs=[];idsout=[];logs=[]
    for name in S:
        ids=z[name+'_sample_ids'].astype(str);lookup={s:i for i,s in enumerate(p.sample_ids)};held=np.array([lookup[s] for s in ids]);heldusers=set(p.users[held]);source=np.isin(p.labels,[24,26])&~np.isin(p.users,list(heldusers|{'user1','user2','user21'}))
        assert not heldusers&set(p.users[source]);ei=e[name+'_sample_ids'].astype(str);lookup={s:i for i,s in enumerate(ei)};x=e[name+'_embedding'][[lookup[s] for s in p.sample_ids]].astype(np.float32);x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-6)
        model=make_pipeline(StandardScaler(),SVC(C=1.,gamma='scale',class_weight='balanced'))
        model.fit(x[source],p.labels[source]);margin=model.decision_function(x[held]);proposal=np.where(margin>=0,26,24)
        base=a[name+'_held_prediction'].astype(int);gp=b[name+'_group_probability'];pred=base.copy()
        route=np.isin(base,[24,26])&(proposal!=base)&(abs(margin)>=1.)&(gp[np.arange(len(base)),proposal]>=gp[np.arange(len(base)),base])
        pred[route]=proposal[route];outputs.append(pred);idsout.extend(ids)
        y=p.labels[held];logs.append({'cohort':name,'source_rows':int(source.sum()),'changed':int(route.sum()),'rescue':int(((pred==y)&(base!=y)).sum()),'harm':int(((pred!=y)&(base==y)).sum()),'net':int((pred==y).sum()-(base==y).sum())})
    pred=np.concatenate(outputs);base=a['prediction'];y=a['labels'];assert np.array_equal(y,np.concatenate([z[s+'_labels'] for s in S])) and int((base==y).sum())==2211
    result={'baseline_correct':2211,'correct':int((pred==y).sum()),'net':int((pred==y).sum()-2211),'cohorts':logs,'target_achieved':False,'test_rows_loaded':0,
            'limitation':'exploratory legacy embedding A/B on historically knownpair; notindependentconfirmation; no pretrainedencoderretraining',
            'input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    out.mkdir();np.savez_compressed(out/'predictions.npz',sample_ids=np.array(idsout),prediction=pred,base=base);(out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
