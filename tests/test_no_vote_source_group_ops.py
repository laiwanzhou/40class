"""Retained-bank group geometry stays source-exact inside each known-user block."""
import importlib
import types
import numpy as np


def module():
    assert importlib.util.find_spec('src.experiments.source_group_ops'),'source group adapter is missing'
    return importlib.import_module('src.experiments.source_group_ops')


def test_group_features_match_original_per_user_and_leave_unknown_unlinked(no_vote_fixture):
    m=module()
    from test_no_vote_source_ops import v3_protocol
    from src.experiments.artifact_record import VerificationContext
    from src.experiments.no_vote_types import RowIndex
    from src.experiments.no_vote_manifest import RECORDING_COLUMNS
    p=v3_protocol(no_vote_fixture);ops=m.source_group_ops(p,VerificationContext(p));n=13
    index=RowIndex(tuple(f'id{i}' for i in range(n)),('fit1',)*6+('fit2',)*6+('fit1',),tuple(range(40)))
    base=np.arange(n)%3;bank=np.full((n,20,40),.2/39,np.float32)
    bank[np.arange(n)[:,None],np.arange(20)[None,:],base[:,None]]=.8
    rows=[]
    for i,sid in enumerate(index.sample_ids):
        row={k:'' for k in RECORDING_COLUMNS};known=i<12
        row.update(sample_id=sid,user_id=index.user_ids[i],partition='train12',recording_date='2026-10-29' if known else '',
            start_seconds=str((i%6)//3*70+i%3) if known else 'nan',timestamp_available='1' if known else '0',time_source='acquisition_ids' if known else 'unknown')
        rows.append(row)
    actual=m.group_feature_blocks(index,rows,base,bank,ops)
    assert actual.shape==(13,1641) and actual[-1,-1]==0
    for positions in [np.arange(6),np.arange(6,12)]:
        ids=np.asarray(index.sample_ids)[positions];lookup=dict(zip(ids,bank[positions]))
        meta=ops['metadata_class'](sample_ids=ids,users=np.asarray(index.user_ids)[positions],dates=np.full(6,'2026-10-29'),starts=np.array([0,1,2,70,71,72],float))
        original=ops['group_features'];namespace={**original.__globals__,'align_metadata':lambda path,requested:meta}
        direct=types.FunctionType(original.__code__,namespace,argdefs=original.__defaults__)
        expected=direct(ids,base[positions],lookup,posterior_feature_mode='sqrt',group_config=ops['group_config'],
            group_feature_layout='full',teacher_subset='full',grouping_lookup=lookup,include_quality_features=False)
        np.testing.assert_array_equal(actual[positions],expected)
        assert expected[:,-1].max()==1


def test_pruned_old_new_banks_are_proven_equivalent_from_source(no_vote_fixture):
    m=module()
    from test_no_vote_source_ops import v3_protocol
    from src.experiments.artifact_record import VerificationContext
    p=v3_protocol(no_vote_fixture);c=VerificationContext(p)
    recipe={'roster_view':{'columns':['p89_safe_probability','p85_teacher'],
        'slots':[{'name':'p306_union_repeat_physical','status':'excluded'}]}}
    proof=m.pruned_group_equivalence(p,c,recipe)
    assert proof['source_added_columns']==['p306_union_repeat_physical']
    assert proof['retained_added_columns']==[] and proof['same_group_inputs_and_recipe'] is True
    assert proof['expected_precedence_route']=='all_false_if_same_full_coverage'
