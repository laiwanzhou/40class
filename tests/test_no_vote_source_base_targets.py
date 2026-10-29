"""Intermediate source-base decoder: original posterior/backoff, fixed fit pools."""
import importlib
import numpy as np
import pytest


def module():
    assert importlib.util.find_spec('src.experiments.source_base_targets'),'source-base target adapter is missing'
    return importlib.import_module('src.experiments.source_base_targets')


def test_source_posterior_backoff_and_unknown_identity(no_vote_fixture):
    m=module()
    from src.experiments.artifact_record import VerificationContext
    from src.experiments.no_vote_types import RowIndex
    from src.experiments.no_vote_manifest import RECORDING_COLUMNS
    from test_no_vote_source_ops import v3_protocol
    p=v3_protocol(no_vote_fixture);ops=m.source_base_target_ops(p,VerificationContext(p))
    index=RowIndex(tuple('abcd'),('fit1',)*4,tuple(range(40)));rows=[]
    for i,sid in enumerate(index.sample_ids):
        row={k:'' for k in RECORDING_COLUMNS};row.update(sample_id=sid,user_id='fit1',partition='train12',
            recording_date='2026-10-29' if i<3 else '',start_seconds=str(i) if i<3 else 'nan',
            timestamp_available='1' if i<3 else '0',time_source='acquisition_ids' if i<3 else 'unknown');rows.append(row)
    cfg=dict(gap_seconds=30.,transition_weight=.3,trigram_backoff=1.,beam_width=50,posterior_temperature=1.)
    labels=np.array([0,1,2,3]);state=ops['fit_transition_model'](labels,[np.arange(3)],40,1.)
    prob=np.full((4,40),.025);prob[0]=.01;prob[0,0]=.61
    log=np.log(np.clip(prob,1e-8,1.));valid=np.array([True,True,True,False])
    out=m.structured_source_base(index,rows,prob,log,valid,state,cfg,ops)
    original=ops['decode_unique_beam_posterior'](log[:3],state,.3,50,1.)
    expected,weight=ops['backed_off_structured_probability'](prob[:3],original.marginals,50)
    np.testing.assert_array_equal(out['structured_probability'][:3],original.marginals)
    np.testing.assert_array_equal(out['structured_distillation_probability'][:3],expected)
    np.testing.assert_array_equal(out['structured_distillation_probability'][3],prob[3])
    assert out['sequence_session_id'][3]==-1 and not out['target_mask'][3]
    assert np.all(out['structured_distillation_probability'][:3]>0)


def test_intermediate_targets_refuse_development_and_final_before_reading_inputs(no_vote_fixture):
    m=module()
    from src.experiments.no_vote_protocol import load_protocol
    from src.experiments.artifact_record import VerificationContext
    p=load_protocol(no_vote_fixture[0]);c=VerificationContext(p)
    for pop in ['development2','final4']:
        with pytest.raises(ValueError,match='fit pool'):m.make_source_base_targets(p,None,None,pop,c)


def test_unbound_transition_does_not_open_legacy_global_gate(no_vote_fixture):
    m=module()
    from test_no_vote_source_ops import v3_protocol
    from src.experiments.artifact_record import ArtifactRegistry,VerificationContext
    p=v3_protocol(no_vote_fixture);r=ArtifactRegistry(p,context=VerificationContext(p));file=p.run_root/'unbound_transition.json';file.write_text('{}')
    with pytest.raises(ValueError,match='not ready'):
        r.register(stage=m.STAGE,kind='transition',phase='select',files=[file],fit_users=p.partitions['train12'].users)
