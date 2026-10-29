"""Explicit v3 artifact contracts; v2 identity and role behavior stay unchanged."""
import csv
import json
from pathlib import Path

import numpy as np

FAMILIES=frozenset({'early','late','window_mean','early_late','temporal_delta','kinetics'})
FIXED_MODELS=frozenset({'A1','A2','A5','A6-VS','A6-VI','A6-VSI','A7','A7-mask',
    'A7-shuffle-S','A7-shuffle-I','visual_mechanism'})
FITTED_OPERATORS={'p12_fusion':'statistics','global_mixture':'statistics','safe_base':'statistics','group_old':'supervised_model',
    'group_new':'supervised_model','gate':'supervised_model','sequence':'transition','fullwindow_knn':'statistics'}
P12_EXPERTS=frozenset({'p12_skeleton','p12_depth','p12_skeleton_depth','p12_thermal','imu_rf'})
DIAGNOSTIC_STAGES=frozenset({'A6-VS','A6-VI','A6-VSI','A7-mask','A7-shuffle-S','A7-shuffle-I','A7-zero-S','A7-zero-I'})


def expert_names(registry):
    bindings=registry.protocol.recipe.get('source_ops_bindings',{})
    if 'roster_path' not in bindings:return P12_EXPERTS
    path=Path(bindings['roster_path'])
    if registry.context:registry.context.hash_file(path,bindings['roster_sha256'])
    body=json.loads(path.read_text(encoding='utf-8'))
    return P12_EXPERTS | {s['name'] for s in body['slots'] if s.get('status')=='retained' and 'name' in s}


def model_kind(registry,stage):
    if stage in FIXED_MODELS:return 'supervised_model'
    if stage.startswith('visual_head/') and stage.split('/',1)[1] in FAMILIES:return 'supervised_model'
    if stage.startswith('source_expert/') and stage.split('/',1)[1] in expert_names(registry):return 'supervised_model'
    return FITTED_OPERATORS.get(stage)


def validate_v3_role(registry,r):
    if r.stage in {'A6-VS','A6-VI','A6-VSI'} and r.config.get('role')!='diagnostic_not_source_main':
        raise ValueError('A6 requires explicit diagnostic role')
    if r.kind=='public_weights' and r.stage.startswith('source_initializer/'):
        role=r.stage.split('/',1)[1]
        asset=registry.protocol.recipe.get('teacher_only_assets',{}).get(role)
        if (not asset or not asset.get('origin') or r.phase!='public' or r.parents
                or dict(r.files)!=dict(asset.get('files',{}))):
            raise ValueError('teacher-only public asset needs explicit recipe origin/file binding')
    if r.kind in {'supervised_model','statistics','calibration','transition'}:
        if r.stage=='class_prior':expected='statistics'
        else:expected=model_kind(registry,r.stage)
        # A visual head can have its own fit-only temperature record.
        if r.kind=='calibration' and (r.stage=='A1' or r.stage.startswith('visual_head/')):expected='calibration'
        if r.kind!=expected:raise ValueError('unknown v3 learned stage/kind')
    if r.kind=='adapted_model' and r.stage not in {'A9-12','A9-40'}:
        raise ValueError('unknown v3 adapted stage')
    if r.kind=='predictions' and (r.phase!='predict' or (model_kind(registry,r.stage) is None
            and r.stage not in {'A5-S','A5-I','A7-zero-S','A7-zero-I','A8','A9-12','A9-40'})):
        raise ValueError('unknown v3 prediction stage/phase')
    if r.kind=='teacher_bank':
        if r.stage!='teacher_bank' or r.phase!='predict' or r.rows is None or not r.complete:
            raise ValueError('invalid teacher_bank role')
        validate_bank(registry,r)


def verify_prediction_ancestry(registry,r,lineage):
    aliases={'A5-S':'A5','A5-I':'A5','A7-zero-S':'A7','A7-zero-I':'A7','A8':'A7'}
    stage=aliases.get(r.stage,r.stage)
    if stage in {'A9-12','A9-40'}:kind='adapted_model'
    else:kind=model_kind(registry,stage)
    pop=r.config['partition'];phase='select' if pop in {'train12','development2'} else 'refit'
    matches=[x for x in lineage if x.stage==stage and x.kind==kind
        and x.phase==('adapt' if kind=='adapted_model' else phase)]
    if not matches:raise ValueError('prediction missing own stage/phase model ancestor')
    # All supervised ancestors must belong to the prediction's fit phase.
    if any(x.kind in {'supervised_model','statistics','calibration','transition'} and x.phase!=phase for x in lineage):
        raise ValueError('prediction supervised ancestor phase mismatch')


def validate_descriptor(registry,r):
    if len(r.files)!=1:raise ValueError('descriptor needs exactly one payload')
    path=Path(next(iter(r.files)))
    if not path.is_relative_to(registry.root/'protocol'):raise ValueError('descriptor outside protocol')
    parents=[registry.read(x) for x in r.parents]
    if r.stage=='source_transform':
        body=json.loads(path.read_text(encoding='utf-8'))
        if (path.suffix!='.json' or body.get('schema_version')!=3
                or body.get('cross_sample_fit') is not False or not body.get('operators')
                or len(parents)!=1 or parents[0].stage!='source_ops' or parents[0].kind!='descriptor'):
            raise ValueError('invalid deterministic source transform descriptor')
    else:
        if (path.suffix!='.csv' or len(parents)!=2 or not any(p.stage.startswith('manifest_') for p in parents)
                or not any(p.stage in {'p28','p29'} and p.kind=='raw_cache' for p in parents)):
            raise ValueError('recording descriptor requires manifest and acquisition')
        from .no_vote_manifest import RECORDING_COLUMNS,read_public_rows
        partition=r.config.get('partition')
        if partition not in registry.protocol.partitions:raise ValueError('recording partition missing')
        with path.open(encoding='utf-8',newline='') as f:
            reader=csv.DictReader(f)
            if tuple(reader.fieldnames or ())!=RECORDING_COLUMNS:raise ValueError('recording sidecar whitelist mismatch')
            data=list(reader)
        expected=read_public_rows(registry.protocol.public_manifests[partition])
        if [(x['sample_id'],x['user_id']) for x in data]!=[(x['sample_id'],x['user_id']) for x in expected]:
            raise ValueError('recording sidecar IDs/user mismatch')
        from datetime import date
        for x in data:
            if x['partition']!=partition:raise ValueError('recording partition mismatch')
            seconds=float(x['start_seconds'])
            if x['timestamp_available']=='1':
                date.fromisoformat(x['recording_date'])
                if not 0<=seconds<86400 or x['time_source']!='acquisition_ids':raise ValueError('recording time basis mismatch')
            elif x['timestamp_available']!='0' or x['recording_date'] or not np.isnan(seconds) or x['time_source']!='unknown':
                raise ValueError('unknown recording time must remain unknown')


def validate_bank(registry,r):
    if any(x in DIAGNOSTIC_STAGES for x in r.config.get('experts',[])):
        raise ValueError('diagnostic controls cannot enter teacher bank')
    from .no_vote_manifest import read_public_rows,row_index
    partition=r.config.get('partition');parts=registry.protocol.partitions
    if partition not in parts or r.rows!=row_index(read_public_rows(registry.protocol.public_manifests[partition])):
        raise ValueError('teacher bank partition IDs mismatch')
    parents=[registry.read(x) for x in r.parents]
    rosters=[x for x in parents if x.stage=='teacher_roster' and x.kind=='descriptor']
    predictions=[x for x in parents if x.kind=='predictions']
    experts=list(r.config.get('experts',[]))
    if (len(rosters)!=1 or not experts or len(set(experts))!=len(experts)
            or len(predictions)!=len(experts) or [x.stage for x in predictions]!=experts
            or any(x.rows!=r.rows or x.phase!='predict' or x.config.get('partition')!=partition for x in predictions)):
        raise ValueError('teacher bank needs exact ordered same-phase expert predictions and roster')
    if len(r.files)!=1:raise ValueError('teacher bank needs one NPZ')
    path=Path(next(iter(r.files)))
    arrays=registry.context.load_arrays(path) if registry.context else None
    if arrays is None:
        with np.load(path,allow_pickle=False) as data:arrays={k:data[k] for k in data.files}
    if set(arrays)!={'sample_ids','class_ids','probabilities','availability','fallback'}:
        raise ValueError('teacher bank schema mismatch')
    n,e=len(r.rows.sample_ids),len(experts)
    if (tuple(arrays['sample_ids'].astype(str))!=r.rows.sample_ids
            or not np.array_equal(arrays['class_ids'],np.arange(40))
            or arrays['probabilities'].shape!=(n,e,40)
            or any(arrays[k].shape!=(n,e) or arrays[k].dtype!=np.bool_ for k in ['availability','fallback'])):
        raise ValueError('teacher bank axes/IDs mismatch')
    probs=arrays['probabilities']
    if not np.isfinite(probs).all() or np.any(probs<0) or not np.allclose(probs.sum(-1),1):
        raise ValueError('teacher bank probability mismatch')
    if not np.array_equal(arrays['fallback'],~arrays['availability']):raise ValueError('teacher bank fallback mask mismatch')
