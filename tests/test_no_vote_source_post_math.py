import importlib
import numpy as np

def module():
    assert importlib.util.find_spec('src.experiments.source_post_math'),'postprocessing source math missing'
    return importlib.import_module('src.experiments.source_post_math')

def test_safe_probability_and_bank_fallback():
    m=module();from scipy.special import softmax
    raw=np.full((3,40),.025);raw[0]=.01;raw[0,0]=.61
    rf=np.arange(120,dtype=float).reshape(3,40)/10;prior=np.full(40,.025)
    valid=np.array([True,True,False]);p=m.safe_probability(raw,rf,valid,prior)
    expected=.95*raw+.05*softmax(rf/3,axis=1);expected/=expected.sum(1,keepdims=True);expected[2]=prior
    np.testing.assert_array_equal(p,expected)
    bank,av=m.bank_arrays(p,[raw,raw], [np.array([True,False,False]),np.array([True,True,False])],valid)
    np.testing.assert_array_equal(bank[1,1],bank[1,0]);assert not av[1,1]
    np.testing.assert_array_equal(bank[2],np.tile(prior.astype(np.float32),(3,1)))

def test_group_model_capture_and_empty_threshold_are_original(no_vote_fixture):
    m=module();from test_no_vote_source_ops import v3_protocol
    from src.experiments.artifact_record import VerificationContext
    p=v3_protocol(no_vote_fixture);ops=m.post_ops(p,VerificationContext(p));rng=np.random.default_rng(6)
    x=rng.normal(size=(60,9));x[:,-1]=np.arange(60)%2;y=np.arange(60)%3;tx=rng.normal(size=(7,9))
    model,a=m.fit_group(x,y,tx,ops);b=ops['fit_probability'](x,y,tx,.03,2.,False,0.)
    np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(m.group_probability(model,tx),a)
    base=a.argmax(1);assert m.group_threshold(base,base,a,np.arange(7)%3,ops)['threshold']==2.
    assert m.sequence_gate(base,base,np.zeros((7,5)),np.arange(7)%3,ops)['threshold']==2.

def test_joint_decode_is_original_inside_user_blocks(no_vote_fixture):
    m=module();from test_no_vote_source_ops import v3_protocol
    from src.experiments.artifact_record import VerificationContext
    from src.experiments.no_vote_types import RowIndex
    from src.experiments.no_vote_manifest import RECORDING_COLUMNS
    p=v3_protocol(no_vote_fixture);ops=m.post_ops(p,VerificationContext(p))
    idx=RowIndex(tuple('abcdefg'),('u1',)*3+('u2',)*3+('u1',),tuple(range(40)));rows=[]
    for i,s in enumerate(idx.sample_ids):
        row={k:'' for k in RECORDING_COLUMNS};row.update(sample_id=s,user_id=idx.user_ids[i],partition='train12',recording_date='2026-10-30' if i<6 else '',
            start_seconds=str(i%3) if i<6 else 'nan',timestamp_available='1' if i<6 else '0',time_source='acquisition_ids' if i<6 else 'unknown');rows.append(row)
    state=ops['fit_transition_model'](np.arange(3),[np.arange(3)],40,1.);cfg=ops['DecoderConfig'](30.,.3,1.,50)
    prob=np.full((7,40),.025);base=prob.argmax(1);actual=m.safe_decision(idx,rows,prob,prob,state,cfg,ops)
    for start in [0,3]:
        ids=np.asarray(idx.sample_ids)[start:start+3];meta=ops['metadata_class'](sample_ids=ids,users=np.asarray(idx.user_ids)[start:start+3],dates=np.full(3,'2026-10-30'),starts=np.arange(3,dtype=float))
        sessions=ops['build_sessions'](np.arange(3),meta,30.,'anonymous_date')
        decoded=ops['decode_sessions'](np.log(np.maximum(prob[start:start+3],1e-12)),sessions,state,cfg)
        protocol=(None,None,None,None,meta,np.arange(3),sessions,state,cfg)
        expected,_=ops['joint_decode'](prob[start:start+3],decoded,protocol,ops['safe_repeat'],.25,1.)
        np.testing.assert_array_equal(actual[start:start+3],expected)
    assert actual[-1]==base[-1]
