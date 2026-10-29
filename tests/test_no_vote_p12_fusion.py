"""P12 source arithmetic, calibration scope, routing and missing-input fallback."""
import importlib
import numpy as np
import pytest


def module():
    assert importlib.util.find_spec('src.experiments.p12_fusion'),'P12 fusion operators are missing'
    return importlib.import_module('src.experiments.p12_fusion')


def test_sd_imu_thermal_source_arithmetic_and_fallback(no_vote_fixture):
    m=module()
    from test_no_vote_source_ops import v3_protocol
    from src.experiments.artifact_record import VerificationContext
    p=v3_protocol(no_vote_fixture);ops=m.p12_fusion_source_ops(p,VerificationContext(p))
    s=np.arange(4*40,dtype=np.float64).reshape(4,40)/40;d=-s;imu=s*.3;thermal=s*2
    sd=m.sd_scores(s,d);np.testing.assert_allclose(sd,.2*s,rtol=0,atol=1e-14)
    valid=np.array([True,False,True,True]);counts=np.array([5,0,2,1])
    imu_cfg={'sd_temperature':.7,'imu_temperature':1.4,'selected_imu_weight':.35}
    result=m.apply_imu(sd,imu,valid,counts,imu_cfg)
    weight=.35*np.clip(counts/5,0,1);expected=(1-weight[:,None])*sd/.7+weight[:,None]*imu/1.4
    np.testing.assert_allclose(result,expected,atol=0,rtol=0)
    cfg={'base_temperature':.5,'thermal_temperature':2.,'selected_thermal_weight':.3}
    fused=m.apply_thermal(result,thermal,valid,cfg,ops)
    expected=ops['fused_logits'](result,thermal,valid,.3,.5,2.)
    np.testing.assert_array_equal(fused,expected);np.testing.assert_array_equal(fused[1],result[1])


def test_calibration_matches_source_once_without_oof_or_held_labels(no_vote_fixture):
    m=module()
    from test_no_vote_source_ops import v3_protocol
    from src.experiments.artifact_record import VerificationContext
    p=v3_protocol(no_vote_fixture);ops=m.p12_fusion_source_ops(p,VerificationContext(p));rng=np.random.default_rng(9)
    labels=np.arange(80)%40;sd=rng.normal(size=(80,40));imu=rng.normal(size=(80,40));present=np.ones(80,bool)
    actual=m.fit_imu(sd,imu,labels,present,ops)
    # Independent original cross-fit operator: empty held fold yields all-row
    # calibration at held_fold=2, with no new internally generated OOF predictions.
    _,rows=ops['cross_fitted_imu']({'labels':labels,'folds':np.arange(80)%2,'sd_logits':sd,'imu_logits':imu})
    reference=rows[2]
    for k in ['sd_temperature','imu_temperature','selected_imu_weight']:assert actual[k]==reference[k]
    with pytest.raises(ValueError,match='40|shape'):m.fit_imu(sd[:,:39],imu,labels,present,ops)


def test_one_class_router_uses_source_constant_fallback_without_fake_examples(no_vote_fixture):
    m=module()
    from test_no_vote_source_ops import v3_protocol
    from src.experiments.artifact_record import VerificationContext
    p=v3_protocol(no_vote_fixture);ops=m.p12_fusion_source_ops(p,VerificationContext(p))
    base=np.zeros((4,40));base[:,0]=3;candidate=base.copy();candidate[0]=0;candidate[0,1]=3
    labels=np.zeros(4,dtype=int);present=np.array([True,True,False,True])
    gate=m.fit_thermal_router(base,candidate,labels,present,ops)
    assert gate['kind']=='constant' and gate['value']==0. and gate['sensitive_rows']==1
    out=m.apply_thermal_router(base,candidate,present,gate,ops)
    np.testing.assert_array_equal(out['logits'],base);assert not out['route'].any()


def test_global_mixture_and_knn_keep_source_weights_self_exclusion_and_missing_fallback(no_vote_fixture):
    m=module()
    from test_no_vote_source_ops import v3_protocol
    from src.experiments.artifact_record import VerificationContext
    p=v3_protocol(no_vote_fixture);ops=m.p12_fusion_source_ops(p,VerificationContext(p));rng=np.random.default_rng(20)
    experts=rng.normal(size=(7,14,40));weights=np.arange(1,15)/105
    valid=np.array([True,True,False,True,True,True,False]);fallback=np.full((7,40),1/40)
    scores,prob=m.apply_global(experts,weights,valid,fallback)
    from scipy.special import log_softmax,softmax
    expected=np.sum(log_softmax(experts,axis=2)*weights[None,:,None],axis=1)
    np.testing.assert_allclose(scores[valid],expected[valid],atol=0,rtol=0)
    np.testing.assert_array_equal(prob[~valid],fallback[~valid])
    features=rng.normal(size=(7,3,8)).astype(np.float32)
    smooth=m.apply_knn(prob,features,valid,ops)
    original,indices,similarity=ops['knn_smooth'](prob[valid],ops['knn_unit_features'](features[valid]))
    np.testing.assert_array_equal(smooth['probability'][valid],original)
    np.testing.assert_array_equal(smooth['probability'][~valid],prob[~valid])
    mapped=np.flatnonzero(valid)[indices]
    np.testing.assert_array_equal(smooth['neighbor_indices'][valid],mapped)
    assert not np.any(mapped==np.flatnonzero(valid)[:,None])
