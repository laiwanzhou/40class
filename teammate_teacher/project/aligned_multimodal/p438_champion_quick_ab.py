"""Three fixed low-cost changes to actual P310/P315 teacher precedence.
Legacy development A/B only; no Test fields read, no training/submission.
"""
import json,hashlib
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent
S=('H1_selection','H2_confirmation','H3_independent_fold0')
def main():
    out=H/'runs/p438_champion_quick_ab_v1'
    if out.exists():raise FileExistsError(out)
    paths=[H/'runs'/p for p in ('p310_union_repeat_precedence_teacher_v1/oof_predictions.npz','p307_union_repeat_group_sequence_audit_v1/oof_predictions.npz','p270_fixed_emission065_transition045_v1/predictions.npz','p255_repeat_augmented_physical_group_v1/predictions.npz')]
    # Rules fixed before evaluating labels. Do not tune thresholds after run.
    a,b,c,d=[np.load(p,allow_pickle=False) for p in paths]
    y=a['labels'];assert np.array_equal(y,b['labels']) and len(y)==2470
    outputs={k:[] for k in ('baseline','margin_precedence','confident_precedence','group_sequence_agreement')};folds=[]
    for name in S:
        base=a[name+'_held_prediction'].astype(int);seq=c[name+'_held_prediction'].astype(int);old=d[name+'_held_prediction'].astype(int)
        group=b[name+'_group_prediction'].astype(int);groupseq=b[name+'_sequence_prediction'].astype(int)
        gp=b[name+'_group_probability'];op=d[name+'_held_probability'];route=group!=old
        expected=seq.copy();expected[route]=group[route];assert np.array_equal(expected,base)
        margin=np.sort(gp,axis=1)[:,-1]-np.sort(gp,axis=1)[:,-2]
        oldmargin=np.sort(op,axis=1)[:,-1]-np.sort(op,axis=1)[:,-2]
        v1=seq.copy();mask=route&(margin>=oldmargin);v1[mask]=group[mask]
        v2=seq.copy();mask=route&(gp.max(1)>=.8)&(margin>=.5);v2[mask]=group[mask]
        v3=base.copy();mask=(group==groupseq)&(group!=base);v3[mask]=group[mask]
        for key,v in zip(outputs,(base,v1,v2,v3)):outputs[key].append(v)
        folds.extend([name]*len(base))
    outputs={k:np.concatenate(v) for k,v in outputs.items()};folds=np.array(folds);base=outputs['baseline'];assert int((base==y).sum())==2211
    report={}
    for key,v in outputs.items():
        oldok=base==y;newok=v==y
        report[key]={'correct':int(newok.sum()),'accuracy':float(newok.mean()),'changed':int((v!=base).sum()),'rescue':int((newok&~oldok).sum()),'harm':int((oldok&~newok).sum()),'net':int(newok.sum()-oldok.sum()),'fold_net':[int(newok[folds==s].sum()-oldok[folds==s].sum()) for s in S]}
    result={'results':report,'limitation':'Historical development A/B; reused/global fitted upstreams, not independent subject confirmation or official score','test_rows_loaded':0,'target_achieved':False,
            'input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'rules':['P307 precedence only if new group margin >= old group margin','P307 precedence only if confidence>=.8 and margin>=.5','correct unchanged rows only if P307 group and P307 sequence agree']}
    out.mkdir();np.savez_compressed(out/'predictions.npz',cohort=folds,**outputs);(out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
