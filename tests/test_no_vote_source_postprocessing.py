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
