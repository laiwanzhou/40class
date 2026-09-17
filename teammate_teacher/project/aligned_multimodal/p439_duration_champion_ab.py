"""One fixed duration-outlier correction on P310 champion, legacy OOF A/B.
Source class medians/MAD only; no Test fields, hyperparameter search or submit.
Rule: base |z|>2.5, candidate |z|<1, duration likelihood ratio>20,
candidate in group top3 with probability>=base. Otherwise preserve champion.
"""
import csv,json,hashlib
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;ROOT=H.parent;S=('H1_selection','H2_confirmation','H3_independent_fold0')
def main():
    out=H/'runs/p439_duration_champion_ab_v1'
    if out.exists():raise FileExistsError(out)
    paths=[H/'runs/p310_union_repeat_precedence_teacher_v1/oof_predictions.npz',H/'runs/p307_union_repeat_group_sequence_audit_v1/oof_predictions.npz',ROOT/'runs/p90_crossuser_visual_router_v1/full_predictions.npz',H/'data/p85_recording_metadata/train_recording_metadata.csv']
    a,b,z=[np.load(p,allow_pickle=False) for p in paths[:3]];ids=np.concatenate([z[s+'_sample_ids'].astype(str) for s in S]);y=a['labels'].astype(int)
    assert np.array_equal(y,np.concatenate([z[s+'_labels'] for s in S])) and np.array_equal(y,b['labels'])
    rows={r['sample_id']:r for r in csv.DictReader(paths[3].open(encoding='utf-8-sig'))}
    users=np.array([rows[i]['user_id'] for i in ids]);assert not set(users)&{'user1','user2','user21'} and len(set(ids))==2470
    duration=np.array([float(rows[i]['duration_seconds']) if rows[i]['duration_seconds'] else np.nan for i in ids]);valid=np.isfinite(duration)&(duration>0)
    values=np.full(len(ids),np.nan);values[valid]=np.log(duration[valid])
    # Label-free within-user centering reduces different recording pace.
    for user in set(users):
        k=(users==user)&valid
        if k.any():values[k]-=np.median(values[k])
    folds=np.concatenate([np.full(len(z[s+'_sample_ids']),j) for j,s in enumerate(S)])
    base=a['prediction'].astype(int);assert int((base==y).sum())==2211
    gp=np.concatenate([b[s+'_group_probability'] for s in S]);pred=base.copy();records=[]
    for fold in range(3):
        source=(folds!=fold)&valid;held=np.flatnonzero((folds==fold)&valid)
        assert not set(users[source])&set(users[held]);centers=[];scales=[]
        for cls in range(40):
            x=values[source&(y==cls)]
            if len(x)<10:centers.append(np.nan);scales.append(np.nan);continue
            m=np.median(x);centers.append(m);scales.append(max(.15,1.4826*np.median(abs(x-m))))
        centers=np.array(centers);scales=np.array(scales)
        for i in held:
            standardized=(values[i]-centers)/scales;loglike=-.5*standardized**2-np.log(scales)
            top=np.argsort(gp[i])[-3:];best=int(top[np.argmax(np.nan_to_num(loglike[top],nan=-np.inf))]);old=base[i]
            if abs(standardized[old])>2.5 and abs(standardized[best])<1 and loglike[best]-loglike[old]>np.log(20) and gp[i,best]>=gp[i,old]:pred[i]=best
        records.append({'fold':fold,'source_rows':int(source.sum()),'centers':centers.tolist(),'scales':scales.tolist()})
    oldok=base==y;newok=pred==y
    report={'baseline_correct':2211,'correct':int(newok.sum()),'net':int(newok.sum()-2211),'changed':int((pred!=base).sum()),'rescue':int((newok&~oldok).sum()),'harm':int((oldok&~newok).sum()),'fold_net':[int((newok[k]-oldok[k]).sum()) for k in []],
            'valid_duration_rows':int(valid.sum()),'target_achieved':False,'test_rows_loaded':0,'limitation':'historical reused OOF; new duration fitting excludes held subjects; not independent confirmation',
            'input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    report['fold_net']=[int(newok[folds==j].sum()-oldok[folds==j].sum()) for j in range(3)]
    out.mkdir();np.savez_compressed(out/'predictions.npz',sample_ids=ids,cohort=folds,base=base,prediction=pred);(out/'summary.json').write_text(json.dumps(report,indent=2));(out/'duration_models.json').write_text(json.dumps(records,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
