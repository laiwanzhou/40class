"""Freeze the authorized approximate Task11 recipe and discover public Thermal inputs."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.experiments.no_vote_protocol import load_protocol
from src.experiments.artifact_record import VerificationContext
from src.experiments.source_teacher_bank import pinned_source,a18_lineage
from src.experiments.no_vote_types import canonical_bytes
from src.experiments.no_vote_cache_import import immutable_bytes
from src.experiments.reconstruction_inputs import build_thermal_public_sidecars


def prepare(protocol,train_root):
    c=VerificationContext(protocol);start=time.perf_counter();root=protocol.run_root;revision='task11_reconstruction_v2'
    choices={
        'skeleton':'aligned_multimodal/configs/skeleton_only.json',
        'depth':'aligned_multimodal/configs/depth_layer3_spatial.json',
        'thermal':'thermal_baseline/configs/resnet18_tsm_imagenet_12f.json'}
    models={};pins=[]
    for role,path in choices.items():
        text,pin=pinned_source(protocol,path,c);cfg=json.loads(text);pins.append(pin)
        models[role]={'source_config':pin,'configuration':cfg,'budget_epochs':15,'checkpoint':'fixed_last_epoch_15',
            'checkpoint_selection_uses_development':False,'seed':20260720,
            'imagenet_pretrained':bool(cfg.get('imagenet_pretrained',False)),
            'initializer':'ResNet18_Weights.IMAGENET1K_V1' if cfg.get('imagenet_pretrained') else 'source_random_initialization',
            'training_precision':'source_cuda_amp_fp16_with_fp32_master_parameters',
            'export_precision':'float16_versioned_copy_preserve_fp32_master',
            'historical_selected_config_verified':False}
    original_roster_path=Path(protocol.recipe['source_ops_bindings']['roster_path'])
    roster_sha=c.hash_file(original_roster_path,protocol.recipe['source_ops_bindings']['roster_sha256'])
    roster=json.loads(original_roster_path.read_text(encoding='utf-8'));a18=a18_lineage(protocol,c)
    p91,p91pin=pinned_source(protocol,'aligned_multimodal/p91_hierarchical_multimodal_teacher.py',c);pins.append(p91pin)
    tree=ast.parse(p91)
    required={'DEPTH':'p91_videomaev2_depth_fold0_v1','THERMAL':'p91_videomaev2_thermal_fold0_v1','HAND':'p91_videomaev2_hand_fold0_v1'}
    for name,marker in required.items():
        matches=[x for x in tree.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in x.targets)]
        if len(matches)!=1 or not any(isinstance(x,ast.Constant) and isinstance(x.value,str) and marker in x.value for x in ast.walk(matches[0])):
            raise ValueError('P128/P91 removed-root input evidence changed')
    slots=[]
    for slot in roster['slots']:
        current=dict(slot)
        if slot['name']=='a18_best_session':current.update(status='excluded',roots=a18['roots'],reason='A18 actual removed-root consumption; see P39')
        if slot['name']=='p128_hierarchical_multimodal':current.update(status='excluded',roots=['videomaev2'],reason='P128 imports P91 teacher with actual VideoMAEv2 depth/thermal/hand inputs')
        if slot['name']=='p89_safe_probability':current.update(status='retained_reconstructed',roots=['videomae_large'],reason='fresh own-phase intermediate adapt12 source-base, explicitly reconstructed decoder')
        slots.append(current)
    repeat={'maximum_session_rank_distance':3,'maximum_start_gap_seconds':300.,'minimum_probability_similarity':.84,
        'minimum_path_overlap':.2,'minimum_length_ratio':.8,'consensus_weight':.75,'alignment_gap_penalty':.2,'maximum_group_size':3}
    body={'schema_version':1,'revision':revision,'base_protocol_sha256':protocol.identity(),
        'authorization':{'date':'2026-10-29','user_request':'可以改为近似复刻','mode':'source_constrained_approximate_reconstruction'},
        'historical_exact_reproduction':False,'task1_to_10_reused':True,'legacy_protocol_modified':False,
        'fixed_split':{'select_fit':'train12','development':'development2','refit_fit':'refit14','sealed_evaluation':'final4',
            'oof_enabled':False,'development_used_for_new_epoch_or_recipe_selection':False,'final_labels_read':False},
        'source_models':models,'source_evidence':pins,
        'thermal_coverage':{'unknown_observation_preserved':True,'unknown_has_no_usable_input':True,
            'never_reverse_opaque_ids_using_labels':True,'all_original_rows_retained':True},
        'intermediate_source_base':{'base_model':'own_phase_A7_full','teacher':'own_phase_reconstructed_P85_global_mixture_then_KNN',
            'distinct_stage':'source_base_adapt12_reconstructed','fit_pool':{'select':'train12','refit':'refit14'},
            'development_target_adaptation':False,'final_target_adaptation_in_this_stage':False,
            'initialization':'fresh_own_phase_A7_not_A9_12','recipe':dict(protocol.recipe['adaptation']['endpoints']['diagnostic12']),
            'decoder':{'gap_seconds':30.,'transition_weight':.30,'trigram_backoff':1.,'beam_width':50,'posterior_temperature':1.},
            'decoder_parameter_origin':'fixed_source_P139_values_replacing_unavailable_outer_fold_median_summary',
            'transition_fit':'own_phase_fit_labels_only','entropy_backoff':'source_backed_off_structured_probability'},
        'safe_base':{'probability':'normalise(0.95 * reconstructed_adapt12_raw_probability + 0.05 * softmax(own_phase_RF_terminal_logits / 3))',
            'safe_decision':'source_joint_decode_on_same_safe_probability_with_explicit_session_blocks',
            'joint_evidence_weight':.25,'joint_transition_scale':1.,'repeat_config':repeat,
            'repeat_config_origin':'fixed_P89_build_final_GLOBAL_REPEAT_instead_of_unavailable_H1_selected_summary',
            'forbidden_substitutes':['raw_A7_as_adapted_source_base','final_A9_12_as_upstream_base','historical_predictions']},
        'target_final_adaptation':{'task':12,'pool':'final4','expected_loss_rows':591,'expected_prior_rows':18,
            'primary_epochs':40,'diagnostic_epochs':12,'unchanged':True},
        'roster_view':{'parent':{'path':str(original_roster_path),'sha256':roster_sha},'slots':slots,
            'columns':[x['name'] for x in slots if x['status'] in {'retained','retained_reconstructed'}],
            'excluded_roots_not_reintroduced':True},
        'source_postprocessing':'retain_known_P12_global_KNN_group_sequence_P310_operators_in_source_order',
        'missing_history_resolution':'declared_fixed_reconstruction_not_claimed_recovered_record',
        'generation_gate':'real_same_phase_producers_public_inputs_assets_and_bounded_tests_still_required',
        'legacy_formal_generation_ready_unchanged':False,'new_training_started':False}
    recipe_path=ROOT/'configs/experiments'/f'{revision}.json';immutable_bytes(recipe_path,canonical_bytes(body)+b'\n')
    recipe_sha=hashlib.sha256(recipe_path.read_bytes()).hexdigest()
    sidecars=build_thermal_public_sidecars(protocol,train_root,root/'protocol'/revision/'thermal_inputs')
    receipt={'revision':revision,'base_protocol_sha256':protocol.identity(),'recipe':{'path':str(recipe_path),'sha256':recipe_sha},
        'thermal_public_sidecars':sidecars,'historical_missing_records_accepted_as_reconstruction':True,
        'formal_bank_ready':False,'a8_generated':False,'task11_complete':False,'task12_started':False,
        'remaining_dependencies':['public_Thermal_ImageNet_initializer_receipt','fresh_same_phase_P12_models',
            'fresh_global_mixture_and_KNN','intermediate_source_base_adapt12_and_safe_decision','fresh_group_sequence_gate_and_A8'],
        'final_labels_read':False,'raw_images_decoded':False}
    # Runtime duration is diagnostic, excluded from the immutable configuration identity.
    receipt_path=root/'protocol'/revision/'preparation.json'
    immutable_bytes(receipt_path,canonical_bytes(receipt)+b'\n')
    print(json.dumps({**receipt,'seconds':round(time.perf_counter()-start,2)}),flush=True);return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--train-root',type=Path,required=True);args=parser.parse_args()
    prepare(load_protocol(args.config,inference_only=True),args.train_root)
