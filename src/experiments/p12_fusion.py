"""P12 source calibration and routing, fit-only rather than historical OOF."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.special import log_softmax,softmax

from .teammate_source import verify_teammate_source,load_teammate_symbol


def p12_fusion_source_ops(protocol,context):
    key=(protocol.identity(),'reconstructed_p12_fusion_source')
    if key in context.public_rows:return context.public_rows[key]
    report=verify_teammate_source(protocol.source_root,protocol.source_root.parent/'source_manifest.json',
        expected_sha256=protocol.recipe['asset_bindings']['source_manifest_sha256'],verify_contents=False)
    symbols={'imu_temperature':('evaluate_imu_oof','fit_temperature'),
        'cross_fitted_imu':('evaluate_imu_oof','cross_fitted_fusion'),
        'thermal_temperature':('analyze_thermal_oof_fusion','fit_temperature'),
        'fused_logits':('analyze_thermal_oof_fusion','fused_logits'),
        'router_features':('evaluate_conditional_expert_routing','build_features'),
        'fit_simplex':('build_p85_multiexpert_submission_v1','fit_simplex'),
        'knn_unit_features':('build_p85_fullwindow_knn_teacher_v3','unit_features'),
        'knn_smooth':('build_p85_fullwindow_knn_teacher_v3','smooth')}
    ops={k:load_teammate_symbol(report,*value) for k,value in symbols.items()};context.public_rows[key]=ops;return ops


def _scores(*values):
    result=[np.asarray(v,dtype=np.float64) for v in values]
    if (not result or result[0].ndim!=2 or result[0].shape[1]!=40 or any(v.shape!=result[0].shape for v in result)
        or any(not np.isfinite(v).all() for v in result)):raise ValueError('finite matching [N,40] source scores required')
    return result


def _fit_arrays(a,b,labels,present):
    a,b=_scores(a,b);labels=np.asarray(labels);present=np.asarray(present)
    if (labels.shape!=(len(a),) or labels.dtype.kind not in 'iu' or np.any((labels<0)|(labels>=40))
        or present.shape!=(len(a),) or present.dtype!=np.bool_):raise ValueError('fit label/mask shape or class mismatch')
    if not present.any():raise ValueError('no source calibration rows available')
    return a,b,labels,present


def sd_scores(skeleton,depth):
    s,d=_scores(skeleton,depth);return .6*s+.4*d


def fit_imu(sd,imu,labels,present,ops):
    sd,imu,labels,present=_fit_arrays(sd,imu,labels,present);s=sd[present];i=imu[present];y=labels[present]
    st=ops['imu_temperature'](s,y);it=ops['imu_temperature'](i,y);weights=np.linspace(0.,.5,11)
    scores=[accuracy_score(y,((1-w)*s/st+w*i/it).argmax(1)) for w in weights]
    chosen=max(range(len(weights)),key=lambda n:(scores[n],-weights[n]))
    return {'sd_temperature':float(st),'imu_temperature':float(it),'selected_imu_weight':float(weights[chosen]),
        'fit_rows':int(present.sum()),'target_origin':'in_sample_non_oof'}


def apply_imu(sd,imu,present,device_counts,cfg):
    sd,imu=_scores(sd,imu);present=np.asarray(present);counts=np.asarray(device_counts)
    if (present.shape!=(len(sd),) or present.dtype!=np.bool_ or counts.shape!=(len(sd),)
        or not np.isfinite(counts).all() or np.any((counts<0)|(counts>5)) or np.any(counts[~present]!=0)):
        raise ValueError('source IMU presence/device-count mismatch')
    weight=cfg['selected_imu_weight']*np.clip(counts/5.,0.,1.)
    return (1-weight[:,None])*sd/cfg['sd_temperature']+weight[:,None]*imu/cfg['imu_temperature']


def fit_thermal(base,thermal,labels,present,ops):
    base,thermal,labels,present=_fit_arrays(base,thermal,labels,present)
    bt=ops['thermal_temperature'](base[present],labels[present]);tt=ops['thermal_temperature'](thermal[present],labels[present])
    best=None
    for weight in np.linspace(0.,1.,21):
        prediction=ops['fused_logits'](base,thermal,present,float(weight),bt,tt).argmax(1)
        score=(float(accuracy_score(labels,prediction)),float(f1_score(labels,prediction,average='macro',zero_division=0)),-float(weight))
        if best is None or score>best:best=score
    return {'base_temperature':float(bt),'thermal_temperature':float(tt),'selected_thermal_weight':-best[2],
        'fit_rows':len(labels),'calibration_present_rows':int(present.sum()),'target_origin':'in_sample_non_oof'}


def apply_thermal(base,thermal,present,cfg,ops):
    base,thermal=_scores(base,thermal);present=np.asarray(present)
    if present.dtype!=np.bool_ or present.shape!=(len(base),):raise ValueError('thermal mask shape mismatch')
    return ops['fused_logits'](base,thermal,present,cfg['selected_thermal_weight'],cfg['base_temperature'],cfg['thermal_temperature'])


def fit_thermal_router(base,candidate,labels,present,ops):
    base,candidate,labels,present=_fit_arrays(base,candidate,labels,present)
    base_correct=base.argmax(1)==labels;candidate_correct=candidate.argmax(1)==labels
    sensitive=present&(base_correct!=candidate_correct);target=candidate_correct[sensitive].astype(np.int64)
    info={'sensitive_rows':int(sensitive.sum()),'fit_rows':len(labels),'seed':20260724,
        'target_origin':'in_sample_non_oof','threshold':.5,'include_class':True}
    if len(np.unique(target))<2:
        return {**info,'kind':'constant','value':float(target[0]) if len(target) else 0.,
            'source_degenerate_rule':'evaluate_conditional_expert_routing.route_cross_fitted_single_class_fallback'}
    model=make_pipeline(StandardScaler(),LogisticRegression(C=.05,max_iter=2000,solver='lbfgs',random_state=20260724))
    model.fit(ops['router_features'](base,candidate,include_class=True)[sensitive],target)
    return {**info,'kind':'logistic','model':model}


def apply_thermal_router(base,candidate,present,gate,ops):
    base,candidate=_scores(base,candidate);present=np.asarray(present)
    if present.dtype!=np.bool_ or present.shape!=(len(base),):raise ValueError('thermal router mask shape mismatch')
    probability=np.zeros(len(base),np.float64)
    if gate['kind']=='constant':probability[present]=gate['value']
    elif gate['kind']=='logistic':
        if present.any():probability[present]=gate['model'].predict_proba(ops['router_features'](base,candidate,include_class=True)[present])[:,1]
    else:raise ValueError('unknown thermal gate')
    route=present&(probability>=.5);scores=base.copy();scores[route]=candidate[route]
    return {'logits':scores,'route':route,'route_probability':probability}


def apply_global(expert_logits,weights,visual_valid,fallback_probability):
    values=np.asarray(expert_logits,np.float64);weights=np.asarray(weights,np.float64);valid=np.asarray(visual_valid);fallback=np.asarray(fallback_probability,np.float64)
    if (values.ndim!=3 or values.shape[1:]!=(14,40) or weights.shape!=(14,) or valid.shape!=(len(values),) or valid.dtype!=np.bool_
        or fallback.shape!=(len(values),40) or not np.isfinite(values).all() or not np.isfinite(weights).all()
        or np.any(weights<0) or not np.isclose(weights.sum(),1) or np.any(fallback<0) or not np.isfinite(fallback).all()
        or not np.allclose(fallback.sum(1),1)):
        raise ValueError('global 14-expert simplex/mask/fallback mismatch')
    scores=np.sum(log_softmax(values,axis=2)*weights[None,:,None],axis=1);prob=softmax(scores,axis=1)
    prob[~valid]=fallback[~valid];scores[~valid]=np.log(np.maximum(fallback[~valid],1e-12))
    return scores,prob


def apply_knn(probability,features,valid,ops):
    probability=np.asarray(probability);features=np.asarray(features);valid=np.asarray(valid)
    if (probability.ndim!=2 or probability.shape[1]!=40 or len(features)!=len(probability)
        or valid.shape!=(len(probability),) or valid.dtype!=np.bool_ or not np.isfinite(probability).all()
        or np.any(probability<0) or not np.allclose(probability.sum(1),1)):
        raise ValueError('KNN probability/feature/mask mismatch')
    chosen=np.flatnonzero(valid)
    if len(chosen)<=3:raise ValueError('source three-neighbor smoothing requires at least four available rows')
    values=features[valid].astype(np.float32)
    if not np.isfinite(values).all():raise ValueError('non-finite consumed fullwindow features')
    smoothed,neighbors,similarities=ops['knn_smooth'](probability[valid],ops['knn_unit_features'](values))
    result=probability.copy();result[valid]=smoothed
    indices=np.full((len(probability),3),-1,np.int64);indices[valid]=chosen[neighbors]
    distances=np.zeros((len(probability),3),np.float32);distances[valid]=similarities
    return {'probability':result,'neighbor_indices':indices,'neighbor_similarities':distances}
