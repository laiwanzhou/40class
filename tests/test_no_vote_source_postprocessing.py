import importlib
from types import SimpleNamespace
import pytest

def module():
    assert importlib.util.find_spec('src.experiments.source_postprocessing'),'formal postprocessing producer missing'
    return importlib.import_module('src.experiments.source_postprocessing')

def test_unbound_post_models_cannot_unlock_generation(no_vote_fixture):
    m=module();from src.experiments.artifact_record import ArtifactRegistry,VerificationContext
    from src.experiments.no_vote_protocol import load_protocol
    p=load_protocol(no_vote_fixture[0]);r=ArtifactRegistry(p,context=VerificationContext(p))
    for stage,kind in [('safe_base','statistics'),('group_old','supervised_model'),('group_new','supervised_model'),('sequence','transition'),('gate','supervised_model')]:
        x=SimpleNamespace(stage=stage,kind=kind,phase='select',parents=(),config={})
        assert m.allows_post_stage(r,x) is False

def test_source_bank_contract_has_no_removed_or_duplicate_columns(no_vote_fixture):
    m=module()
    columns=['p89_safe_probability','p85_teacher','p85_head_early_logits','p86_mechanism_drop_scene_logits','p12_skeleton','p12_thermal']
    assert [m.column_stage(x) for x in columns]==['safe_base','global_mixture','visual_head/early','visual_mechanism','source_expert/p12_skeleton','source_expert/p12_thermal']
    for name in ['a18_best_session','p306_union_repeat_physical','A7-mask','made_up']:
        with pytest.raises(ValueError):m.column_stage(name)

def test_a8_rejects_other_prediction_partition_even_in_same_refit_phase(monkeypatch):
    m=module();from src.experiments.no_vote_types import canonical_hash
    recipe={'fixture':'P44 metadata boundary'};rr='recipe';rows=('final-row',)
    bank=SimpleNamespace(stage='teacher_bank',kind='teacher_bank',phase='predict',rows=('refit-row',),config={'partition':'refit14'},parents=('safe',))
    safe=SimpleNamespace(stage='safe_base',kind='predictions',phase='predict',rows=('refit-row',),config={'partition':'refit14'})
    records={'bank':bank,'safe':safe}
    for s,k in [('group_old','supervised_model'),('group_new','supervised_model'),('sequence','transition'),('gate','supervised_model'),('class_prior','statistics')]:
        records[s]=SimpleNamespace(stage=s,kind=k,phase='refit',rows=('refit-row',),config={})
    records[rr]=SimpleNamespace(stage='reconstruction_recipe',kind='raw_cache',phase='raw')
    registry=SimpleNamespace(protocol=object(),context=object(),read=lambda x:records[x])
    monkeypatch.setattr(m,'load_reconstruction',lambda p,c:(recipe,rr,{}))
    record=SimpleNamespace(config={'partition':'final4','post_contract':m.CONTRACT,'recipe_sha256':canonical_hash(recipe)},
        parents=tuple(records),rows=rows)
    with pytest.raises(ValueError,match='partition|rows'):m.validate_reconstructed_a8(registry,record)
    bank.rows=safe.rows=rows;bank.config=safe.config={'partition':'final4'};bank.parents=('different_safe',)
    records['different_safe']=SimpleNamespace(stage='safe_base',kind='predictions',phase='predict',rows=rows,config={'partition':'final4'})
    with pytest.raises(ValueError,match='safe'):m.validate_reconstructed_a8(registry,record)
