"""Frozen P89/P137/P270/P139 math; no historical prediction readers."""
import ast
import inspect
import textwrap
import numpy as np
from scipy.special import softmax
from threadpoolctl import threadpool_limits
from .source_group_ops import source_group_ops
from .session_metadata import _local_blocks,session_blocks
from .source_base_targets import source_base_target_ops
from .teammate_source import verify_teammate_source,load_teammate_symbol

def post_ops(protocol,context):
    key=(protocol.identity(),'task11_post_math')
    if key not in context.public_rows:
        report=verify_teammate_source(protocol.source_root,protocol.source_root.parent/'source_manifest.json',
            expected_sha256=protocol.recipe['asset_bindings']['source_manifest_sha256'],verify_contents=False)
        names={'joint_decode':('p89_global_joint_repeat_decoder','joint_decode'),
            'safe_repeat':('p89_build_final_test_submissions','GLOBAL_REPEAT'),
            'DecoderConfig':('audit_p87_sequence_decoder','DecoderConfig'),
            'decode_sessions':('audit_p87_sequence_decoder','decode_sessions'),
            'sequence_emission':('p257_adaptive_physical_sequence','emission'),
            'sequence_config':('p257_adaptive_physical_sequence','dec'),
            'gate_features':('p139_soft_sequence_gate','gate_features'),
            'select_gate':('p139_soft_sequence_gate','select_gate')}
        context.public_rows[key]={**source_group_ops(protocol,context),**source_base_target_ops(protocol,context),
            **{k:load_teammate_symbol(report,*v) for k,v in names.items()}}
    return context.public_rows[key]

def safe_probability(raw,rf_logits,valid,prior):
    raw=np.asarray(raw,np.float64);rf=np.asarray(rf_logits,np.float64);valid=np.asarray(valid);prior=np.asarray(prior,np.float64)
    if raw.ndim!=2 or raw.shape[1]!=40 or rf.shape!=raw.shape or valid.shape!=(len(raw),) or valid.dtype!=np.bool_:raise ValueError('safe input schema')
    p=.95*raw+.05*softmax(rf/3.,axis=1);p/=p.sum(1,keepdims=True);p[~valid]=prior
    return p

def bank_arrays(safe,columns,available,safe_valid):
    safe=np.asarray(safe,np.float32);bank=[safe];masks=[np.asarray(safe_valid,bool)]
    for probability,valid in zip(columns,available,strict=True):
        values=np.asarray(probability,np.float32).copy();valid=np.asarray(valid)
        if values.shape!=safe.shape or valid.shape!=(len(safe),) or valid.dtype!=np.bool_:raise ValueError('bank source axes')
        values[~valid]=safe[~valid];bank.append(values);masks.append(valid)
    result=np.stack(bank,axis=1)
    if not np.isfinite(result).all() or np.any(result<0) or not np.allclose(result.sum(2),1.,atol=1e-5):raise ValueError('bank probability normalization')
    return result,np.column_stack(masks)

def safe_decision(index,rows,raw,safe,transition,decoder,ops):
    output=np.asarray(raw).argmax(1).astype(np.int64);blocks,_=_local_blocks(index,rows,ops)
    for positions,metadata in blocks:
        sessions=ops['build_sessions'](np.arange(len(positions)),metadata,decoder.gap_seconds,'anonymous_date')
        base=ops['decode_sessions'](np.log(np.maximum(raw[positions],1e-12)),sessions,transition,decoder)
        protocol=(None,None,None,None,metadata,np.arange(len(positions)),sessions,transition,decoder)
        output[positions],_=ops['joint_decode'](safe[positions],base,protocol,ops['safe_repeat'],evidence_weight=.25,transition_scale=1.)
    return output

def fit_group(x,y,predict_x,ops):
    original=ops['fit_probability'];tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
    returns=[n for n in ast.walk(tree) if isinstance(n,ast.Return)]
    if len(returns)!=1 or ast.unparse(returns[0].value)!='output':raise ValueError('source group model capture structure changed')
    returns[0].value=ast.Tuple(elts=[ast.Name(id='model',ctx=ast.Load()),ast.Name(id='output',ctx=ast.Load())],ctx=ast.Load())
    namespace=dict(original.__globals__);exec(compile(ast.fix_missing_locations(tree),'<source_group_capture>','exec'),namespace)
    with threadpool_limits(limits=4):return namespace['fit_probability'](x,y,predict_x,.03,2.,False,0.)

def group_probability(model,x):
    p=np.zeros((len(x),40),np.float64)
    with threadpool_limits(limits=4):p[:,model.named_steps['logisticregression'].classes_.astype(int)]=model.predict_proba(x)
    return p

def group_threshold(base,proposal,probability,labels,ops):
    if not np.any(proposal!=base):return {'threshold':2.,'changed':0,'rescue':0,'harm':0,'net':0}
    return ops['choose_threshold'](base,proposal,probability,labels)

def sequence_gate(base,sequence,features,labels,ops):
    if not np.any(sequence!=base):return {'score_index':0,'score_name':ops['select_gate'].__globals__['SCORE_NAMES'][0],
        'threshold':2.,'changed':0,'rescue':0,'harm':0,'net':0}
    return ops['select_gate'](base,sequence,features,labels)

def sequence_prediction(index,rows,probability,base,transition,ops):
    blocks=session_blocks(index,rows,ops,gap_seconds=30.)
    prediction=ops['decode_sessions'](ops['sequence_emission'](probability,base,.65),list(blocks.sessions),transition,ops['sequence_config'](.45))
    prediction[blocks.unknown]=base[blocks.unknown]
    return prediction
