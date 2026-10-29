"""Task11 source sessions must not cross users, populations or unknown time."""
import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from src.experiments.no_vote_types import RowIndex


def module():
    assert importlib.util.find_spec('src.experiments.session_metadata'), 'Task11 session adapter is missing'
    return importlib.import_module('src.experiments.session_metadata')


def entry(sid,user,time,*,partition='train12',known=True):
    return dict(sample_id=sid,user_id=user,partition=partition,recording_date='2026-01-01' if known else '',
        start_seconds=str(time) if known else 'nan',timestamp_available='1' if known else '0',
        time_source='acquisition_ids' if known else 'unknown')


def test_known_user_session_adapter(no_vote_fixture):
    m=module()
    from test_no_vote_source_ops import v3_protocol
    from src.experiments.artifact_record import VerificationContext
    p=v3_protocol(no_vote_fixture);ops=m.source_session_ops(p,VerificationContext(p))
    ids=('a','b','c','d','e','f');users=('fit1','fit1','dev1','dev1','fit1','fit1')
    index=RowIndex(ids,users,tuple(range(40)))
    rows=[entry('a','fit1',10),entry('b','fit1',20),entry('c','dev1',11,partition='development2'),
        entry('d','dev1',70,partition='development2'),entry('e','fit1',15,partition='refit14'),entry('f','fit1',0,known=False)]
    blocks=m.session_blocks(index,list(reversed(rows)),ops,gap_seconds=30)
    assert [x.tolist() for x in blocks.sessions]==[[0,1],[2],[3],[4]]
    assert blocks.unknown.tolist()==[5]
    # A single known-user population matches the unmodified source anonymous-date function.
    meta=ops['metadata_class'](np.array(['a','b']),np.array(['fit1','fit1']),np.array(['2026-01-01']*2),np.array([10.,20.]))
    expected=ops['build_sessions'](np.arange(2),meta,30,'anonymous_date')
    assert [x.tolist() for x in expected]==[x.tolist() for x in blocks.sessions[:1]]


def test_session_metadata_rejects_missing_duplicate_user_and_fake_unknown(no_vote_fixture):
    m=module();idx=RowIndex(('a','b'),('fit1','fit1'),tuple(range(40)))
    rows=[entry('a','fit1',10),entry('b','fit1',20)]
    for bad in [rows[:1],[rows[0],rows[0]],[rows[0],entry('b','dev1',20)],
                [rows[0],{**rows[1],'timestamp_available':'0'}]]:
        with pytest.raises(ValueError):m.validate_recording_rows(idx,bad)


def test_repeat_session_groups_keep_global_indices_and_unknown_rows(no_vote_fixture):
    m=module()
    from test_no_vote_source_ops import v3_protocol
    from src.experiments.artifact_record import VerificationContext
    p=v3_protocol(no_vote_fixture);ops=m.source_session_ops(p,VerificationContext(p))
    index=RowIndex(('a','b','c','d','e'),('fit1','fit1','fit1','fit1','dev1'),tuple(range(40)))
    rows=[entry('a','fit1',0),entry('b','fit1',10),entry('c','fit1',60),entry('d','fit1',70),entry('e','dev1',0,known=False)]
    groups=m.repeat_session_lists(index,rows,ops,short_gap=30)
    assert [[x.tolist() for x in group] for group in groups]==[[[0,1],[2,3]]]
