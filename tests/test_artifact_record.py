from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest


def context(no_vote_fixture):
    from src.experiments.no_vote_protocol import load_protocol
    from src.experiments.artifact_record import ArtifactRegistry
    return ArtifactRegistry(load_protocol(no_vote_fixture[0]))


def model(registry,tmp_path,stage='A1',phase='select',parents=()):
    file=tmp_path/f'{stage}-{phase}-{len(parents)}.json';file.write_text('{}')
    return registry.register(stage=stage,kind='supervised_model',phase=phase,files=[file],
        parents=parents,fit_users=registry.protocol.partitions['train12' if phase=='select' else 'refit14'].users,
        config={'budget':1})


def test_ancestor_roles_reject_refit_teacher_in_development(no_vote_fixture,tmp_path):
    r=context(no_vote_fixture); teacher=model(r,r.root,'A1','refit')
    with pytest.raises(ValueError,match='ancestor'):
        model(r,r.root,'A2','select',parents=(teacher,))


def test_refit_rejects_select_initializer(no_vote_fixture,tmp_path):
    r=context(no_vote_fixture); teacher=model(r,r.root,'A2','select')
    with pytest.raises(ValueError,match='ancestor'):
        model(r,r.root,'A7','refit',parents=(teacher,))


def test_registry_detects_file_and_parent_drift(no_vote_fixture,tmp_path):
    from src.experiments.no_vote_types import RowIndex
    r=context(no_vote_fixture); parent=model(r,r.root)
    child=model(r,r.root,'A2',parents=(parent,))
    record=r.read(parent);Path(next(iter(record.files))).write_text('changed')
    with pytest.raises(ValueError,match='hash'):
        r.verify(child,'A2','select',None)


def test_unprovenanced_bank_and_wrong_final_ids_rejected(no_vote_fixture,tmp_path):
    from src.experiments.no_vote_types import RowIndex
    r=context(no_vote_fixture); file=r.root/'renamed.json';file.write_text('{}')
    with pytest.raises(ValueError,match='kind'):
        r.register(stage='A8',kind='expert_bank',phase='predict',files=[file])
    ref=r.register(stage='A0',kind='raw_cache',phase='raw',files=[file],
        rows=RowIndex(('a',),('target1',),tuple(range(40))),config={})
    with pytest.raises(ValueError,match='IDs'):
        r.verify(ref,'A0','raw',RowIndex(('b',),('target1',),tuple(range(40))))


def test_target_adaptation_requires_own_refit_a7_and_a8(no_vote_fixture,tmp_path):
    r=context(no_vote_fixture); base=model(r,r.root,'A7','refit')
    file=r.root/'targets.json';file.write_text('{}')
    other=model(r,r.root,'A2','refit')
    rows=prediction_rows(r,'final4')
    wrong=r.register(stage='A2',kind='predictions',phase='predict',files=[file],parents=(other,),rows=rows,config={'partition':'final4'})
    with pytest.raises(ValueError,match='A8'):
        r.register(stage='A9-12',kind='adapted_model',phase='adapt',files=[file],parents=(base,wrong),
            adaptation_users=r.protocol.partitions['final4'].users,config={})


def prediction_rows(r,partition):
    import csv
    from src.experiments.no_vote_types import RowIndex
    from src.experiments.no_vote_manifest import PUBLIC_COLUMNS
    p=r.protocol.partitions[partition];ids=tuple(f'{i:064x}' for i in range(p.expected_rows))
    users=tuple(p.users[i%len(p.users)] for i in range(len(ids)))
    path=r.protocol.public_manifests[partition];path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=PUBLIC_COLUMNS);w.writeheader()
        for sid,user in zip(ids,users):
            row={name:('0' if name.endswith('_available') else '') for name in PUBLIC_COLUMNS}
            row.update(sample_id=sid,user_id=user);w.writerow(row)
    return RowIndex(ids,users,tuple(range(40)))


def test_final_predictions_require_model_ancestry(no_vote_fixture):
    r=context(no_vote_fixture);rows=prediction_rows(r,'final4')
    file=r.root/'no-model.json';file.write_text('{}')
    with pytest.raises(ValueError,match='ancestor'):
        r.register(stage='A1',kind='predictions',phase='predict',files=[file],rows=rows,config={'partition':'final4'})


@pytest.mark.parametrize('bad',['wrong_ids','partial'])
def test_correct_stage_a8_requires_exact_final_ids_and_complete(no_vote_fixture,bad):
    from src.experiments.no_vote_types import RowIndex
    r=context(no_vote_fixture);rows=prediction_rows(r,'final4');base=model(r,r.root,'A7','refit')
    file=r.root/'targets.json';file.write_text('{}')
    if bad=='wrong_ids':rows=RowIndex(('wrong-final-id',),('target1',),tuple(range(40)))
    with pytest.raises(ValueError):
        target=r.register(stage='A8',kind='predictions',phase='predict',files=[file],parents=(base,),
            rows=rows,complete=bad!='partial',config={'partition':'final4'})
        r.register(stage='A9-12',kind='adapted_model',phase='adapt',files=[file],parents=(base,target),
            adaptation_users=r.protocol.partitions['final4'].users,config={})


def test_refit_predictions_can_include_development_users_as_refit_population(no_vote_fixture):
    r=context(no_vote_fixture);rows=prediction_rows(r,'refit14');base=model(r,r.root,'A1','refit')
    file=r.root/'refit-pred.json';file.write_text('{}')
    ref=r.register(stage='A1',kind='predictions',phase='predict',files=[file],parents=(base,),
        rows=rows,config={'partition':'refit14'})
    assert r.verify(ref).rows==rows


def test_valid_complete_a8_target_and_same_a7_are_accepted(no_vote_fixture):
    r=context(no_vote_fixture);rows=prediction_rows(r,'final4');base=model(r,r.root,'A7','refit')
    file=r.root/'good-target.json';file.write_text('{}')
    target=r.register(stage='A8',kind='predictions',phase='predict',files=[file],parents=(base,),
        rows=rows,config={'partition':'final4'})
    adapted=r.register(stage='A9-12',kind='adapted_model',phase='adapt',files=[file],parents=(base,target),
        adaptation_users=r.protocol.partitions['final4'].users,config={})
    assert r.verify(adapted).stage=='A9-12'


def test_public_caches_reject_label_fields_in_json_and_npz(no_vote_fixture,tmp_path):
    import numpy as np
    r=context(no_vote_fixture)
    for suffix in ('json','npz'):
        file=r.root/f'bad.{suffix}'
        if suffix=='json':file.write_text('{"nested":{"action_name":"secret"}}')
        else:np.savez(file,class_id=np.array([0]))
        with pytest.raises(ValueError,match='label'):
            r.register(stage='A0',kind='raw_cache',phase='raw',files=[file],config={})
