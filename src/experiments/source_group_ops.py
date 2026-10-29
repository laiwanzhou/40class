"""Original P134/P137 group math with explicit public known-user time blocks."""
import ast
from types import FunctionType
import numpy as np

from .teammate_source import verify_teammate_source,load_teammate_symbol
from .session_metadata import source_session_ops,_local_blocks
from .source_teacher_bank import pinned_source


def source_group_ops(protocol,context):
    key=(protocol.identity(),'source_group_ops')
    if key not in context.public_rows:
        report=verify_teammate_source(protocol.source_root,protocol.source_root.parent/'source_manifest.json',
            expected_sha256=protocol.recipe['asset_bindings']['source_manifest_sha256'],verify_contents=False)
        features=load_teammate_symbol(report,'p137_group_classifier_selector','group_features')
        context.public_rows[key]={**source_session_ops(protocol,context),'group_features':features,
            'group_config':features.__globals__['CONFIG'],
            'fit_probability':load_teammate_symbol(report,'p137_group_classifier_selector','fit_probability'),
            'choose_threshold':load_teammate_symbol(report,'p137_group_classifier_selector','choose_threshold')}
    return context.public_rows[key]


def group_feature_blocks(index,rows,base,bank,ops):
    base=np.asarray(base);bank=np.asarray(bank)
    if (base.shape!=(len(index.sample_ids),) or base.dtype.kind not in 'iu' or np.any((base<0)|(base>=40))
        or bank.ndim!=3 or bank.shape[0]!=len(base) or bank.shape[2]!=40 or bank.shape[1]<1
        or not np.isfinite(bank).all() or np.any(bank<0) or not np.allclose(bank.sum(2),1.,atol=1e-5)):
        raise ValueError('group bank/base/40-class normalization mismatch')
    blocks,unknown=_local_blocks(index,rows,ops);own=np.sqrt(np.clip(bank,0.,1.)).reshape(len(base),-1).astype(np.float32)
    one_hot=np.eye(40,dtype=np.float32)[base]
    # Exact no-peer limit of the source full/sqrt feature layout. Unknown times
    # do not acquire fabricated dates or peers.
    result=np.concatenate([own,own.copy(),one_hot,np.zeros((len(base),1),np.float32)],axis=1)
    for positions,metadata in blocks:
        ids=np.asarray(index.sample_ids)[positions];lookup=dict(zip(ids,bank[positions]))
        def aligned(public_path,requested):
            if tuple(requested.astype(str))!=tuple(metadata.sample_ids):raise ValueError('source group local metadata IDs mismatch')
            return metadata
        original=ops['group_features'];namespace={**original.__globals__,'align_metadata':aligned}
        source=FunctionType(original.__code__,namespace,argdefs=original.__defaults__,closure=original.__closure__)
        result[positions]=source(ids,base[positions],lookup,posterior_feature_mode='sqrt',group_config=ops['group_config'],
            group_feature_layout='full',teacher_subset='full',grouping_lookup=lookup,embedding_lookup=None,include_quality_features=False)
    return result


def pruned_group_equivalence(protocol,context,recipe):
    old,old_pin=pinned_source(protocol,'aligned_multimodal/p255_repeat_augmented_physical_group.py',context)
    new,new_pin=pinned_source(protocol,'aligned_multimodal/p309_union_repeat_group_test.py',context)
    a=ast.parse(old);b=ast.parse(new)
    imported=[]
    for tree in [a,b]:
        sources=[n.module for n in tree.body if isinstance(n,ast.ImportFrom) and any(x.name=='build_train_bank' for x in n.names)]
        if sources!=['p173_vjepa_augmented_group_teacher']:raise ValueError('old/new source base bank origins changed')
        imported.append(sources[0])
    appends=[n.value for n in ast.walk(a) if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='names' for x in n.targets)
        and isinstance(n.value,ast.List) and n.value.elts and isinstance(n.value.elts[0],ast.Starred)]
    source=[n.value for n in b.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='SOURCES' for x in n.targets)]
    if len(appends)!=1 or len(source)!=1:raise ValueError('old/new added expert schema changed')
    old_names=[ast.literal_eval(x) for x in appends[0].elts[1:]]
    new_names=[ast.literal_eval(x.elts[-1]) for x in source[0].elts]
    added=[x for x in new_names if x not in old_names]
    if new_names[:-1]!=old_names or added!=['p306_union_repeat_physical']:raise ValueError('new group is no longer the pinned P306-only addition')
    fit_args=[]
    for tree in [a,b]:
        calls=[x for x in ast.walk(tree) if isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='fit_probability']
        if len(calls)!=1:raise ValueError('source group fit call changed')
        fit_args.append([ast.literal_eval(x) for x in calls[0].args[-4:]])
    if fit_args!=[[.03,2.,False,0.]]*2:raise ValueError('source old/new group recipes differ')
    columns=recipe['roster_view']['columns'];status={x['name']:x['status'] for x in recipe['roster_view']['slots']}
    if any(status.get(x)!='excluded' for x in added):raise ValueError('P306 removal is not explicitly justified by current roster')
    retained_old=[x for x in old_names if x in columns];retained_new=[x for x in new_names if x in columns]
    return {'source_added_columns':added,'retained_added_columns':[x for x in added if x in columns],
        'same_group_inputs_and_recipe':retained_old==retained_new,'source_base_bank_origin':imported[0],
        'source_fit_arguments':fit_args[0],'retained_columns':list(columns),'source_evidence':[old_pin,new_pin],
        'scope':'lineage_and_frozen_recipe_only_numeric_banks_not_yet_generated',
        'expected_precedence_route':'all_false_if_same_full_coverage','actual_precedence_evaluated':False,'task11_complete':False}
