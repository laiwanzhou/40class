"""Approximate reconstruction still needs exact public inputs, not guessed masks."""
import importlib
from pathlib import Path

import pytest


def module():
    assert importlib.util.find_spec('src.experiments.reconstruction_inputs'), 'reconstruction public-input adapter is missing'
    return importlib.import_module('src.experiments.reconstruction_inputs')


def test_thermal_sidecar_uses_exact_trial_sibling_and_never_copies_labels(tmp_path):
    m=module();root=tmp_path/'train';ir=root/'IR'/'opaque-action'/'user6'/'take';ir.mkdir(parents=True)
    thermal=root/'Thermal'/'opaque-action'/'user6'/'take';thermal.mkdir(parents=True);(thermal/'frame_0001.jpg').write_bytes(b'not decoded during metadata discovery')
    row={'sample_id':'a'*64,'user_id':'user6','ir_path':str(ir),'depth_path':'','skeleton_path':'','imu_path':''}
    result=m.thermal_public_rows([row],root)
    assert result==[{'sample_id':'a'*64,'user_id':'user6','thermal_path':str(thermal.resolve()),'thermal_available':'1'}]
    missing={**row,'sample_id':'b'*64,'ir_path':str(root/'IR'/'opaque-action'/'user6'/'other')}
    assert m.thermal_public_rows([missing],root)[0]['thermal_available']=='0'


def test_disagreeing_trial_paths_and_outside_root_are_not_guessed(tmp_path):
    m=module();root=tmp_path/'train';row=dict(sample_id='a'*64,user_id='user1',ir_path=str(root/'IR'/'action'/'user1'/'take'),
        depth_path=str(root/'Depth_Color'/'action'/'user1'/'different'),skeleton_path='',imu_path='')
    with pytest.raises(ValueError,match='disagree'):m.thermal_public_rows([row],root)
    with pytest.raises(ValueError,match='root'):m.thermal_public_rows([{**row,'ir_path':str(tmp_path/'elsewhere'),'depth_path':''}],root)
    unknown=m.thermal_public_rows([{**row,'ir_path':'','depth_path':''}],root)[0]
    assert unknown['thermal_available']=='unknown' and unknown['thermal_path']==''
