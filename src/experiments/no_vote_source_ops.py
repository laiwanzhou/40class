"""Read source recipes without importing legacy programs or loading their arrays.

Descriptors separate facts recovered from source from unavailable historical
configuration evidence. A complete export is not permission to run a blocked
teacher graph.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
import re
import shlex

from .artifact_record import ArtifactRegistry
from .no_vote_types import ArtifactRef, canonical_bytes, write_json
from .teammate_source import SOURCE_FILE_COUNT, SOURCE_MANIFEST_SHA256, verify_teammate_source


def _scalar(node):
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'log':
            return math.log(_scalar(node.args[0]))
        raise ValueError(f'non-literal source parameter: {ast.unparse(node)}')


def _name(node):
    return node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else ''


class _Reader:
    def __init__(self, root):
        self.root = Path(root).resolve()
        report = verify_teammate_source(self.root, self.root.parent/'source_manifest.json',
            expected_count=SOURCE_FILE_COUNT, expected_sha256=SOURCE_MANIFEST_SHA256, verify_contents=False)
        self.manifest_sha = report.manifest_sha256
        self.entries = {x['path']: x for x in json.loads(report.manifest_path.read_text(encoding='utf-8'))['files']}
        self.files = {}
        self.trees = {}
        self.evidence = {}

    def read(self, relative):
        if relative in self.trees:
            return self.trees[relative]
        path = _consumed_path(self.root, relative)
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if not relative.startswith('@package_docs/'):
            entry = self.entries.get(relative)
            if entry is None or len(data) != entry['bytes'] or digest != entry['sha256']:
                raise ValueError(f'source file mismatch: {relative}')
        self.files[relative] = {'bytes':len(data), 'sha256':digest}
        text = data.decode('utf-8-sig')
        tree = ast.parse(text) if path.suffix == '.py' else text
        self.trees[relative] = tree
        return tree

    def py(self, name):
        return self.read('aligned_multimodal/'+name)

    def note(self, profile, name, symbol, node, origin):
        relative = name if name.startswith('@') else 'aligned_multimodal/'+name
        self.evidence.setdefault(profile, []).append(dict(path=relative, symbol=symbol,
            line=getattr(node, 'lineno', 1), sha256=self.files[relative]['sha256'], origin=origin))

    def function(self, name, symbol):
        tree = self.py(name)
        parts = symbol.split('.')
        nodes = tree.body
        for part in parts:
            found = next((n for n in nodes if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name == part), None)
            if found is None:
                raise ValueError(f'missing source symbol: {name}:{symbol}')
            nodes = found.body
        return found

    def assignment(self, name, variable, profile=None):
        tree = self.py(name)
        node = next(n for n in tree.body if isinstance(n,ast.Assign) and any(_name(t)==variable for t in n.targets))
        if profile:
            self.note(profile,name,variable,node,'source_constant')
        return node.value

    def defaults(self, name, profile):
        func = self.function(name,'parse_args')
        out = {}
        for node in ast.walk(func):
            if not isinstance(node,ast.Call) or _name(node.func) != 'add_argument' or not node.args:
                continue
            key = _scalar(node.args[0])
            default = next((x.value for x in node.keywords if x.arg == 'default'), None)
            if default is not None:
                try:
                    out[key] = _scalar(default)
                except ValueError:
                    continue  # historical paths are not configuration bindings
                self.note(profile,name,'parse_args',node,'source_default')
            elif any(x.arg == 'action' and isinstance(x.value,ast.Constant) and x.value.value=='store_true' for x in node.keywords):
                out[key] = False
                self.note(profile,name,'parse_args',node,'source_default')
        return out

    def scheduled(self, filename, defaults, profile):
        name = 'run_p87s_final_pipeline.py'
        tree = self.py(name)
        lists = [n for n in ast.walk(tree) if isinstance(n,ast.List) and any(filename in ast.unparse(x) for x in n.elts)]
        if len(lists) != 1:
            raise ValueError(f'ambiguous final scheduled command: {filename}')
        node = lists[0]
        command = [str(x.value) if isinstance(x,ast.Constant) else ast.unparse(x) for x in node.elts]
        out = dict(defaults)
        for key,value in _flags(command).items():
            out[key] = _convert(value, defaults.get(key))
        self.note(profile,name,'main',node,'actual_override')
        return out

    def bind(self, profile, name, symbols):
        refs=[]
        for symbol in symbols:
            node=self.function(name,symbol)
            self.note(profile,name,symbol,node,'source_expression')
            refs.append(dict(path='aligned_multimodal/'+name,symbol=symbol,line=node.lineno,
                sha256=self.files['aligned_multimodal/'+name]['sha256']))
        return refs


def _flags(tokens):
    out = {}; i = 0
    while i < len(tokens):
        key = tokens[i]
        if key.startswith('--'):
            if i+1 < len(tokens) and not tokens[i+1].startswith('--'):
                out[key] = tokens[i+1]; i += 2
            else:
                out[key] = True; i += 1
        else:
            i += 1
    return out


def _convert(value, default):
    if value is True:
        return True
    if isinstance(default,(int,float)) and not isinstance(default,bool):
        return type(default)(value)
    return value


def _kwargs(profile, flags, mapping):
    return {dest: flags[src] for dest,src in mapping.items()}


def _consumed_path(root, name):
    if name == '@package_docs/TRAINING_PIPELINE.md':
        return root.parent/'package_docs/TRAINING_PIPELINE.md'
    path = (root/name).resolve()
    if not path.is_relative_to(root) or path.suffix != '.py':
        raise ValueError('unapproved consumed source path')
    return path


def compile_source_ops(source_root: Path) -> dict:
    """Inspect only source text; never import a producer or touch legacy runs."""
    r = _Reader(source_root)
    student_defaults = r.defaults('train_p86_visual_pixel_oof.py','visual_student')
    student = r.scheduled('train_p86_visual_pixel_oof.py',student_defaults,'visual_student')
    student_recipe = _kwargs('visual_student',student,{
        'backbone':'--backbone','frames':'--frames','resolution':'--input-resolution',
        'freeze_through':'--freeze-through','mode':'--mode','head_learning_rate':'--head-learning-rate',
        'backbone_learning_rate':'--backbone-learning-rate','minimum_learning_rate':'--minimum-learning-rate',
        'weight_decay':'--weight-decay','batch_size':'--batch-size','gradient_accumulation':'--gradient-accumulation',
        'workers':'--workers','fusion_mode':'--fusion-mode','feature_weight_configured':'--feature-weight',
        'class_weight_power':'--class-weight-power','label_smoothing':'--label-smoothing',
        'distillation_temperature':'--distillation-temperature','distillation_weight':'--distillation-weight',
        'relation_weight':'--relation-weight','augmentation_mode':'--augmentation-mode','seed':'--seed'})
    student_recipe['fixed_epochs'] = int(student['--all-label-fixed-epochs'])
    ctor = r.function('p86_mc3_visual_model.py','P86MC3VisualStudent.__init__')
    ctor_defaults = dict(zip([x.arg for x in ctor.args.args][-len(ctor.args.defaults):],map(_scalar,ctor.args.defaults)))
    student_recipe.update(width=ctor_defaults['width'],dropout=ctor_defaults['dropout'],
        enable_distillation_projection=False,feature_weight_effective=0.,stage_distillation_weight=student['--stage-distillation-weight'])
    student_recipe['temporal_modeling']=student['--backbone']=='mc3_18_temporal'
    for field,flag in [('exact_time_modeling','--exact-time-position'),('cross_view_time_modeling','--same-time-cross-view'),
            ('spatial_region_modeling','--spatial-region-modeling'),('region_temporal_modeling','--region-local-temporal'),
            ('structured_region_modeling','--structured-spatial-region')]:
        student_recipe[field]=student[flag]
    r.note('visual_student','p86_mc3_visual_model.py','P86MC3VisualStudent.__init__',ctor,'source_default')
    motion_defaults = r.defaults('train_p86_mobind_pretrain.py','motion_pretrain')
    motion = r.scheduled('train_p86_mobind_pretrain.py',motion_defaults,'motion_pretrain')
    motion_recipe = {key.removeprefix('--').replace('-','_'):value for key,value in motion.items()
        if isinstance(value,(int,float,bool)) or isinstance(value,str) and not key.endswith(('cache','checkpoint','features','logits','dir','users'))}
    motion_recipe['fixed_epochs'] = int(motion['--epochs'])
    fusion_defaults = r.defaults('train_p86_mobind_fusion_proxy.py','fusion')
    fusion = r.scheduled('train_p86_mobind_fusion_proxy.py',fusion_defaults,'fusion')
    fusion_recipe = {key.removeprefix('--').replace('-','_'):value for key,value in fusion.items()
        if isinstance(value,(int,float,bool)) or isinstance(value,str) and not key.endswith(('cache','checkpoint','features','logits','dir','users'))}
    base12 = r.defaults('adapt_p87s_test_student.py','adaptation')
    doc_name = '@package_docs/TRAINING_PIPELINE.md'
    doc = r.read(doc_name)
    blocks = [x for x in re.findall(r'```(?:bash|sh)\s*(.*?)```',doc,re.S) if 'adapt_p87s_test_student.py' in x]
    if len(blocks) != 1:
        raise ValueError('missing actual final adaptation command')
    command = shlex.split(blocks[0].replace('\\\n',' '))
    final40 = dict(base12)
    for key,value in _flags(command).items():
        final40[key] = _convert(value,base12.get(key))
    command_line=doc[:doc.index(blocks[0])].count('\n')+1
    r.evidence.setdefault('adaptation',[]).append(dict(path=doc_name,symbol='final_adaptation_command',
        line=command_line,sha256=r.files[doc_name]['sha256'],origin='actual_override'))
    adapt_map = {'epochs':'--epochs','scope':'--adaptation-scope','seed':'--seed',
        'fusion_learning_rate':'--fusion-learning-rate','visual_head_learning_rate':'--visual-head-learning-rate',
        'minimum_learning_rate':'--minimum-learning-rate','batch_size':'--batch-size','workers':'--workers',
        'weight_decay':'--weight-decay','temperature':'--temperature','confidence_power':'--confidence-power'}
    mechanism = dict(alpha=_scalar(r.assignment('audit_p86_teacher_mechanisms.py','FROZEN_ALPHA','mechanism')),
        class_weight_power=_scalar(r.assignment('audit_p86_teacher_mechanisms.py','FROZEN_CLASS_WEIGHT_POWER','mechanism')),
        head_temperature=1,backbone_root='videomae_large')
    ridge_function=r.function('train_p46_videomae_head.py','make_model')
    ridge_call=next(n for n in ast.walk(ridge_function) if isinstance(n,ast.Call) and _name(n.func)=='RidgeClassifier')
    ridge_options={k.arg:_scalar(k.value) for k in ridge_call.keywords if k.arg in {'solver','tol','max_iter'}}
    mechanism.update(ridge_solver=ridge_options['solver'],ridge_tol=ridge_options['tol'],
        ridge_max_iter=ridge_options['max_iter'],standardizer='StandardScaler')
    r.note('mechanism','train_p46_videomae_head.py','make_model',ridge_function,'source_expression')
    for name,symbol in [('audit_p86_teacher_mechanisms.py','fixed_model_perturbations'),('p89_build_multiexpert_test_logits.py','build_p86')]:
        r.note('mechanism',name,symbol,r.function(name,symbol),'source_expression')
    bank_func = r.function('p165_deployable_group_teacher.py','build_test_bank')
    temp_call = next(n for n in ast.walk(bank_func) if isinstance(n,ast.Call) and _name(n.func)=='softmax')
    mix = next(n for n in ast.walk(bank_func) if isinstance(n,ast.BinOp) and isinstance(n.op,ast.Add)
        and isinstance(n.left,ast.BinOp) and isinstance(n.left.op,ast.Mult)
        and isinstance(n.left.left,ast.Constant) and isinstance(n.right,ast.BinOp) and isinstance(n.right.left,ast.Constant))
    bank = dict(rf_temperature=_scalar(temp_call.args[1]),base_weight=_scalar(mix.left.left),rf_weight=_scalar(mix.right.left))
    r.note('source_bank','p165_deployable_group_teacher.py','build_test_bank',bank_func,'source_expression')
    cfg = _scalar(r.assignment('p270_fixed_emission065_transition045.py','CFG','postprocessing'))
    target_func = r.function('p310_union_repeat_precedence_teacher.py','main')
    chosen = next(n for n in ast.walk(target_func) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Subscript) and _name(t.value)=='prob' for t in n.targets))
    filled = next(n for n in ast.walk(target_func) if isinstance(n,ast.Assign) and any(_name(t)=='prob' for t in n.targets))
    post = dict(emission_weight=cfg[0],transition_weight=cfg[1],chosen_probability=_scalar(chosen.value),
        other_probability=_scalar(filled.value.args[1]))
    r.note('postprocessing','p310_union_repeat_precedence_teacher.py','main',target_func,'source_expression')
    rf = r.defaults('train_final_imu_rf.py','imu_teacher')
    rf_func = r.function('train_final_imu_rf.py','main')
    rf_ctor = next(n for n in ast.walk(rf_func) if isinstance(n,ast.Call) and _name(n.func)=='RandomForestClassifier')
    rf_recipe = {k.arg:_scalar(k.value) for k in rf_ctor.keywords if isinstance(k.value,(ast.Constant,ast.UnaryOp))}
    rf_recipe.update(seed=rf['--seed'],dropout_rng='source_seed_no_fold',fit_missing_score=-1e6,
        terminal_missing_score=math.log(1e-12))
    r.note('imu_teacher','train_final_imu_rf.py','main',rf_ctor,'source_expression')
    recipe = dict(visual_student=student_recipe,motion_pretrain=motion_recipe,fusion=fusion_recipe,
        adaptation={'base12':_kwargs('adaptation',base12,adapt_map),'final40':_kwargs('adaptation',final40,adapt_map)},
        mechanism=mechanism,source_bank=bank,postprocessing=post,imu_teacher=rf_recipe)
    features_func=r.function('train_p85_videomae_full40_head.py','feature_sets')
    family_dict=next(n.value for n in ast.walk(features_func) if isinstance(n,ast.Return) and isinstance(n.value,ast.Dict))
    head_main=r.function('train_p85_videomae_full40_head.py','main')
    grids={}
    for node in ast.walk(head_main):
        if isinstance(node,ast.For) and isinstance(node.target,ast.Name) and node.target.id in {'power','alpha'}:
            try:grids[node.target.id]=list(_scalar(node.iter))
            except ValueError:pass
    temperature_func=r.function('train_p46_videomae_head.py','fit_temperature')
    temperature_call=next(n for n in ast.walk(temperature_func) if isinstance(n,ast.Call) and _name(n.func)=='minimize_scalar')
    bounds=next(x.value for x in temperature_call.keywords if x.arg=='bounds')
    recipe['visual_teacher']=dict(families=[_scalar(x) for x in family_dict.keys],
        class_weight_powers=grids['power'],alphas=grids['alpha'],log_temperature_bounds=list(_scalar(bounds)),
        primary_family='early_late',target_origin='fixed_split_fit_predictions_not_oof')
    r.note('visual_teacher','train_p85_videomae_full40_head.py','main',head_main,'source_expression')
    r.note('visual_teacher','train_p46_videomae_head.py','fit_temperature',temperature_func,'source_expression')
    recipe['fullwindow_knn']=dict(neighbors=_scalar(r.assignment('build_p85_fullwindow_knn_teacher_v3.py','NEIGHBORS','fullwindow_knn')),
        neighbor_weight=_scalar(r.assignment('build_p85_fullwindow_knn_teacher_v3.py','NEIGHBOR_WEIGHT','fullwindow_knn')))
    baseline_func=r.function('run_imu_stat_baseline.py','main')
    baseline_fill=next(n for n in ast.walk(baseline_func) if isinstance(n,ast.Assign) and any(_name(t)=='logits' for t in n.targets))
    terminal_fill=next(n for n in ast.walk(rf_func) if isinstance(n,ast.Assign) and any(_name(t)=='imu_logits' for t in n.targets))
    rf_recipe['fit_missing_score']=_scalar(baseline_fill.value.args[1])
    rf_recipe['terminal_missing_score']=_scalar(terminal_fill.value.args[1])
    r.note('imu_teacher','run_imu_stat_baseline.py','main',baseline_fill,'source_constant')
    operator_specs={
        'geometry':[('audit_yolo11_pose_skeleton.py',['select_track']),('build_multiscale_dir_rois.py',['main'])],
        'pixels':[('build_p86_visual_pixel_cache.py',['window_indices','main']),('p86_visual_pixel_data.py',['P86VisualPixelDataset._subject_robust_augment'])],
        'motion_processing':[('build_p86_motion_window_cache.py',['aggregate_imu','main']),('p86_mobind_lite_data.py',['P86MoBindMotionDataset.__getitem__']),('p86_cached_motion_data.py',['P86CachedSequenceMotionDataset.__getitem__'])],
        'p12':[('build_complete_p11_oof.py',['main']),('analyze_thermal_oof_fusion.py',['cross_fitted_thermal_residual']),('evaluate_conditional_expert_routing.py',['route_cross_fitted'])],
        'global_mixture':[('build_p85_multiexpert_submission_v1.py',['fit_simplex']),('build_p85_fullwindow_knn_teacher_v3.py',['unit_features','top_neighbors','smooth'])],
        'repeat_group':[('p134_frozen_repeat_consensus.py',['evidence']),('p137_group_classifier_selector.py',['group_features','fit_probability','choose_threshold'])],
        'sequence_gate':[('p139_soft_sequence_gate.py',['gate_features','select_gate']),('p257_adaptive_physical_sequence.py',['emission','dec']),('p310_union_repeat_precedence_teacher.py',['main'])],
        'safe_base':[('p165_deployable_group_teacher.py',['build_test_bank']),('p89_build_dual_consensus_submission.py',['merge'])],
        'student_training':[('train_p86_visual_pixel_oof.py',['build_model','optimizer_for','train_epochs','learning_rate_scale']),
            ('train_p86_visual_student_oof.py',['class_weights','relation_loss'])],
        'motion_training':[('train_p86_mobind_pretrain.py',['losses','train'])],
        'fusion_training':[('train_p86_mobind_fusion_proxy.py',['build_model','set_stage','train_stage','losses','visual_head_parameters'])],
        'adaptation_training':[('adapt_p87s_structured_student.py',['configure_adaptation_parameters','train_label_free'])],
    }
    operators={key:[ref for name,symbols in definitions for ref in r.bind(key,name,symbols)] for key,definitions in operator_specs.items()}
    # Recover the original column registry and lineage while source is verified.
    slots = _original_slots(r)
    return dict(schema_version=3,source_manifest_sha256=r.manifest_sha,training_executed=False,
        fixed_split=True,oof_enabled=False,recipe=recipe,evidence=r.evidence,
        original_slots=slots,operators=operators,consumed_files=r.files,
        unresolved_producer_evidence=_blocking_requirements())


def _original_slots(r):
    node = r.assignment('p88_oof_candidate_ensemble.py','CANDIDATE_SOURCES')
    def evaluate(n,env=None):
        env = {} if env is None else env
        if isinstance(n,ast.Name) and n.id in env:return env[n.id]
        if isinstance(n,ast.JoinedStr):
            return ''.join(str(evaluate(x.value,env)) if isinstance(x,ast.FormattedValue) else x.value for x in n.values)
        if isinstance(n,(ast.Tuple,ast.List)):return [evaluate(x,env) for x in n.elts]
        if isinstance(n,ast.Dict):
            out={}
            for k,v in zip(n.keys,n.values):
                if k is None:out.update(evaluate(v,env))
                else:out[evaluate(k,env)]=evaluate(v,env)
            return out
        if isinstance(n,ast.DictComp) and len(n.generators)==1 and not n.generators[0].ifs:
            g=n.generators[0];return {evaluate(n.key,{**env,g.target.id:x}):evaluate(n.value,{**env,g.target.id:x}) for x in evaluate(g.iter,env)}
        return _scalar(n)
    candidates = evaluate(node)
    slots=[dict(name='p89_safe_probability',category='safe_base',roots=[],status='unresolved',evidence=[]) ]
    for name,source in candidates.items():
        category = 'visual_head' if name.startswith('p85_head_') else 'mechanism' if name.startswith('p86_mechanism_') else 'knn' if name=='p85_teacher' else 'independent_small_expert'
        slots.append(dict(name=name,category=category,historical_reference=source[0],
            roots=['videomae_large'] if category in {'visual_head','mechanism','knn'} else [],status='retained',evidence=[]))
    slots.append(dict(name='a18_best_session',category='independent_small_expert',roots=[],status='unresolved',evidence=[]))
    for name in ('p142_all_token','p144_hand_interaction','p149_repeat_consistency'):
        slots.append(dict(name=name,category='extra_visual_expert',roots=['vjepa2'],status='excluded',evidence=[]))
    extra = r.assignment('p309_union_repeat_group_test.py','SOURCES')
    for row in extra.elts:
        name = _scalar(row.elts[-1])
        slots.append(dict(name=name,category='derived_expert',roots=[],status='unresolved',evidence=[]))
    registry_names=['p88_oof_candidate_ensemble.py','p165_deployable_group_teacher.py','p173_vjepa_augmented_group_teacher.py',
        'p255_repeat_augmented_physical_group.py','p307_union_repeat_group_sequence_audit.py','p309_union_repeat_group_test.py']
    for filename in registry_names:r.py(filename)
    producer_names={
        'p128_hierarchical_multimodal':'p128_hierarchical_multimodal_full_oof.py',
        'p238_physical_token':'p238_physical_token_transformer_oof.py',
        'p231_ir_thermal':'p231_depth_thermal_ir_oof.py',
        'p253_repeat_physical':'p253_repeat_physical_transformer_oof.py',
        'p306_union_repeat_physical':'p306_union_full_repeat_physical_oof.py'}
    for slot in slots:
        proofs=[dict(path='aligned_multimodal/'+x,line=1,sha256=r.files['aligned_multimodal/'+x]['sha256']) for x in registry_names]
        if slot['name']=='p158_lavila_frame_token':
            # P158 uses the shared token trainer; no standalone P158 trainer
            # exists. Its actual registered input paths name the LaViLa cache.
            row=next(x for x in extra.elts if _scalar(x.elts[-1])==slot['name'])
            paths=[n.value for n in ast.walk(row) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
            if not any('lavila' in x.lower() and 'runs/' in x for x in paths):
                raise ValueError('LaViLa registry input lineage missing')
            slot['roots']=['lavila'];slot['status']='excluded'
            proofs.append(dict(path='aligned_multimodal/p309_union_repeat_group_test.py',line=row.lineno,
                sha256=r.files['aligned_multimodal/p309_union_repeat_group_test.py']['sha256'],expression=ast.unparse(row)))
        if slot['name'] in producer_names:
            filename=producer_names[slot['name']]
            tree=r.py(filename)
            # Inspect actual input-path constants, not prose or the word physical.
            constants=[(n.value,n.lineno) for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
            roots=[]
            for marker,root in [('videomaev2','videomaev2'),('internvideo','internvideo2'),('lavila','lavila'),('vjepa','vjepa2'),('dinov2','dinov2')]:
                matches=[(s,line) for s,line in constants if marker in s.lower() and ('runs/' in s.replace('\\','/') or '/' in s)]
                if matches:
                    roots.append(root)
                    proofs.extend(dict(path='aligned_multimodal/'+filename,line=line,sha256=r.files['aligned_multimodal/'+filename]['sha256'],expression=s) for s,line in matches)
            if slot['name']=='p306_union_repeat_physical':
                parent=next(x for x in slots if x['name']=='p253_repeat_physical')
                roots=sorted(set(roots+parent['roots']));proofs.extend(parent['evidence'])
            if slot['name']=='p128_hierarchical_multimodal':
                # The P128 wrapper imports a hierarchy builder, which imports
                # the pretrained fusion cohort builder. Follow this explicit
                # source-only chain instead of classifying from wrapper strings.
                for dependency,symbol in [('p91_hierarchical_multimodal_teacher.py','build_data'),
                        ('p91_unrestricted_fusion_teacher.py','build_cohorts')]:
                    func=r.function(dependency,symbol)
                    for n in ast.walk(r.py(dependency)):
                        if isinstance(n,ast.Constant) and isinstance(n.value,str) and 'runs/' in n.value:
                            for marker,large in [('videomaev2','videomaev2'),('internvideo','internvideo2')]:
                                if marker in n.value.lower():
                                    roots.append(large)
                                    proofs.append(dict(path='aligned_multimodal/'+dependency,line=n.lineno,
                                        symbol=symbol,sha256=r.files['aligned_multimodal/'+dependency]['sha256'],expression=n.value))
                roots=sorted(set(roots))
            slot['roots']=roots
            if roots:slot['status']='excluded'
        slot['evidence']=proofs
    return slots


def _blocking_requirements():
    return [dict(code=code,reason=reason,required_evidence=evidence) for code,reason,evidence in [
        ('p12_sd_final_configuration','Snapshot refers to p11_fp16_oof outputs but does not include their selected configuration.',
         'Selected S+D training config_used.json or recorded invocation, public initializer and architecture binding.'),
        ('p12_thermal_final_configuration','Thermal trainer requires a config; the p11_thermal_imagenet_fp16 selected config is not packaged.',
         'Selected thermal config_used.json or recorded invocation and ImageNet initialization binding.'),
        ('p89_safe_base_producer_closure','The safe-base and a18 output references do not constitute a regenerated fixed-split producer.',
         'Closed P89/A18 producer graph with actual configs, public assets and phase-generic inputs.')]]


def compile_teacher_roster(compiled, retained_large_roots):
    if tuple(retained_large_roots) != ('videomae_large',):
        raise ValueError('exactly one large root, videomae_large, is permitted')
    slots=[]
    for original in compiled['original_slots']:
        slot=dict(original)
        removed=sorted(set(slot['roots'])-set(retained_large_roots))
        slot['removed_large_roots']=removed
        if removed:
            slot['status']='excluded'
            slot['reason']='Consumer depends on removed large visual roots; no feature substitution is permitted.'
        elif slot['status']=='unresolved':
            slot['reason']='Actual root/configuration/producer lineage is not closed; not silently classified as small.'
        else:
            slot['reason']='Retained original expert or head; requires a fresh phase-correct producer.'
        slots.append(slot)
    blockers=list(compiled['unresolved_producer_evidence'])
    unresolved=[x['name'] for x in slots if x['status']=='unresolved']
    if unresolved:
        blockers.append(dict(code='unresolved_teacher_lineage',reason='Some original slots have unclosed root/producer lineage.',
            required_evidence='Original producer and actual root/config evidence for: '+', '.join(unresolved),slots=unresolved))
    return dict(schema_version=3,retained_large_roots=list(retained_large_roots),
        source_ops_sha256=hashlib.sha256(canonical_bytes(compiled)+b'\n').hexdigest(),slots=slots,
        formal_generation_ready=False,blocking_requirements=blockers,
        oof_enabled=False,training_executed=False)


def verify_consumed_source(body, source_root):
    root=Path(source_root).resolve()
    for name,expected in body['consumed_files'].items():
        data=_consumed_path(root,name).read_bytes()
        if len(data)!=expected['bytes'] or hashlib.sha256(data).hexdigest()!=expected['sha256']:
            raise ValueError(f'source file mismatch: {name}')


def runtime_recipe_values(compiled):
    """Canonical projection consumed by the fixed-split runtime, not new defaults."""
    source=json.loads(json.dumps(compiled['recipe']))
    student=dict(source['visual_student'])
    student['epochs']={'min':student['fixed_epochs'],'max':student['fixed_epochs'],'select_on':'fixed_source_budget'}
    teacher=dict(source['visual_teacher'])
    teacher['tie_break']=['accuracy','balanced_accuracy','macro_f1','smaller_class_weight_power','smaller_alpha']
    motion=dict(source['motion_pretrain'])
    motion['contrastive_temperature']=motion['temperature']
    motion['epochs']={'min':motion['fixed_epochs'],'max':motion['fixed_epochs'],'select_on':'fixed_source_budget'}
    fusion=dict(source['fusion'])
    fusion['visual_head_learning_rate']=fusion['visual_learning_rate']
    fusion.update(motion_aux_mask='source_batch_reduction',reliability_mask='source_loss_semantics',
        teacher_loss_mask='source_loss_semantics',anchor_source='frozen_a2_same_phase',
        freeze_visual_backbone_and_bn=True)
    rf=dict(source['imu_teacher'])
    rf['consumer_profiles']={'fit_teacher_target':{'missing_score':rf['fit_missing_score']},
        'terminal_export':{'missing_score':rf['terminal_missing_score']}}
    return dict(source_recipe=source,retained_large_roots=['videomae_large'],
        visual_teacher=teacher,visual_student=student,mechanism=source['mechanism'],
        motion={'external_transform':'identity_source_descriptor','imu_instance_normalization':False},
        imu_teacher=rf,motion_pretrain=motion,fusion=fusion,source_bank=source['source_bank'],
        fullwindow_knn=source['fullwindow_knn'],
        session_repeat={'operator_binding':'source_ops_manifest.operators.repeat_group',
            'parameters':source['postprocessing'],'metadata_start_seconds':'seconds_within_day',
            'empty_disagreement_policy':'identity'},
        adaptation={'endpoints':{'primary40':source['adaptation']['final40'],
            'diagnostic12':source['adaptation']['base12']}},primary_contrast='A9-40_minus_A1')


def validate_runtime_recipe(recipe, compiled):
    def compare(expected,actual,path):
        if isinstance(expected,dict):
            if not isinstance(actual,dict):raise ValueError(f'source recipe mismatch: {path}')
            for key,value in expected.items():
                if key not in actual:raise ValueError(f'source recipe missing: {path}.{key}')
                compare(value,actual[key],path+'.'+key)
        elif actual!=expected:
            raise ValueError(f'source recipe mismatch: {path}')
    compare(runtime_recipe_values(compiled),recipe,'recipe')
    if 'normalization_fit' in recipe.get('motion',{}) or 'normalization_std_floor' in recipe.get('motion',{}):
        raise ValueError('source recipe forbids extra motion standardization')


def _write_immutable(path, body):
    path=Path(path)
    data=canonical_bytes(body)+b'\n'
    if path.exists() and path.read_bytes()!=data:
        raise ValueError('source descriptor identity drift; use a new protocol/run')
    if not path.exists():write_json(path,body)


def export_source_ops(source_root: Path, output: Path, *, protocol) -> ArtifactRef:
    if Path(source_root).resolve()!=protocol.source_root.resolve():
        raise ValueError('source root differs from protocol')
    body=compile_source_ops(source_root)
    _write_immutable(output,body)
    reg=ArtifactRegistry(protocol)
    return reg.register(stage='source_ops',kind='descriptor',phase='public',files=[output],
        config={'source_contract_version':3},source_files=[Path(__file__)])


def resolve_teacher_roster(source_ops, retained_large_roots, *, protocol, output=None):
    reg=ArtifactRegistry(protocol)
    record=reg.verify(source_ops,'source_ops','public')
    if record.kind!='descriptor':raise ValueError('source_ops must be a descriptor')
    body=json.loads(next(Path(p) for p in record.files).read_text(encoding='utf-8'))
    verify_consumed_source(body,protocol.source_root)
    roster=compile_teacher_roster(body,retained_large_roots)
    path=protocol.run_root/'protocol/teacher_roster.json' if output is None else Path(output)
    _write_immutable(path,roster)
    return reg.register(stage='teacher_roster',kind='descriptor',phase='public',files=[path],parents=[source_ops],
        config={'source_contract_version':3},source_files=[Path(__file__)])
