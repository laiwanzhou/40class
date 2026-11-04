"""Task13 fixed-candidate sealing and once-only fixed-split truth evaluation."""
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
from filelock import FileLock
from sklearn.metrics import confusion_matrix, f1_score

from .artifact_record import ArtifactRegistry, VerificationContext, LEARNED
from .no_vote_types import ArtifactRef, RowIndex, Prediction, canonical_hash, canonical_bytes, write_json
from .no_vote_manifest import read_public_rows, row_index, NAMESPACE
from .no_vote_cache_import import immutable_bytes
from .p12_reconstruction import ref

CANDIDATES=('A1','A2','A5-S','A5-I','A6-VS','A6-VI','A6-VSI','A7','A7-mask','A7-shuffle-S',
            'A7-shuffle-I','A7-zero-S','A7-zero-I','A8','A9-12','A9-40')
CONTRASTS=(('A1','A9-40'),('A1','A8'),('A8','A9-40'),('A7','A9-40'))


def _read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def _now():return datetime.now().astimezone().isoformat()


def _bound_read(path, context, expected=None):
    digest=context.hash_file(Path(path),expected)
    return _read(path),dict(path=Path(path),sha256=digest)


def _source_and_completions(p,c):
    if p.recipe.get('source_contract_version')!=3:raise ValueError('freeze requires v3 source operations')
    b=p.recipe.get('source_ops_bindings',{})
    if not all(b.get(x) for x in ['manifest_path','manifest_sha256','roster_path','roster_sha256']):
        raise ValueError('freeze requires bound source operations and roster')
    ops,opref=_bound_read(b['manifest_path'],c,b['manifest_sha256'])
    roster,rr=_bound_read(b['roster_path'],c,b['roster_sha256'])
    if (ops.get('schema_version')!=3 or roster.get('schema_version')!=3 or roster.get('source_ops_sha256')!=opref['sha256']
            or roster.get('retained_large_roots')!=['videomae_large']):raise ValueError('source roster/operations mismatch')
    root=p.run_root/'protocol';proofs={'source_ops':opref,'roster':rr};bodies={}
    for n in [5,8,9,10,11,12]:
        name='task8_current.json' if n==8 else f'task{n}_complete.json'
        body,binding=_bound_read(root/name,c)
        if body.get('protocol_sha256')!=p.identity():raise ValueError('completion protocol mismatch')
        if n!=8 and body.get('complete') is not True:raise ValueError('incomplete prerequisite')
        if n==8 and (body.get('verdict')!='GO' or body.get('task8_revision')!='ir_supervision_v2'):
            raise ValueError('current motion revision is not GO')
        bodies[n]=body;proofs[f'task{n}']=binding
    for n in [11,12]:
        review=bodies[n].get('independent_review',{})
        if review.get('status')!='GO':raise ValueError('completion independent review not GO')
        rb,binding=_bound_read(review['path'],c,review['sha256'])
        if rb.get('protocol_sha256')!=p.identity() or rb.get('verdict')!='GO' or rb.get('unresolved_p1_p2'):raise ValueError('unresolved independent review')
        proofs[f'task{n}_review']=binding
        if n==12:
            if rb.get('generation')!=bodies[n].get('generation'):raise ValueError('Task12 review generation binding mismatch')
            generated,gref=_bound_read(bodies[n]['generation']['path'],c,bodies[n]['generation']['sha256'])
            if (generated.get('protocol_sha256')!=p.identity() or generated.get('base')!=bodies[n]['base']
                    or generated.get('targets')!=bodies[n]['targets']
                    or any(generated['endpoints'][str(e)]['prediction']!=bodies[n]['outputs']['final4'][f'A9-{e}'] for e in [12,40])):
                raise ValueError('Task12 reviewed outputs mismatch')
            proofs['task12_generation']=gref
    review,binding=_bound_read(root/'task9_10_independent_review_P39.json',c)
    if review.get('verdict')!='GO_TASK9_TASK10' or review.get('unresolved_task9_task10_p1_p2'):
        raise ValueError('Task9/10 review not closed')
    for n in [9,10]:
        approved=review.get('reviewed_completions',{}).get(f'task{n}',{})
        if (Path(approved.get('path','')).resolve()!=Path(proofs[f'task{n}']['path']).resolve()
                or approved.get('sha256')!=proofs[f'task{n}']['sha256']):raise ValueError('Task9/10 completion differs from reviewed bytes')
    proofs['task9_10_review']=binding
    return proofs,bodies


def collect_candidates(p):
    root=p.run_root;pr=root/'protocol'
    c=VerificationContext(p,metadata_only=True);_,b=_source_and_completions(p,c)
    out={'A1':ref(_read(root/'A1/refit/final4_targets.artifact.json')),
         'A2':ref(b[5]['partitions']['final4']['prediction']),
         'A5-S':ref(b[8]['outputs']['final4']['skeleton']),'A5-I':ref(b[8]['outputs']['final4']['imu'])}
    for n in [9,10]:out.update({k:ref(v) for k,v in b[n]['outputs']['final4'].items()})
    out['A8']=ref(b[11]['outputs']['final4'])
    out.update({k:ref(v) for k,v in b[12]['outputs']['final4'].items()})
    if set(out)!=set(CANDIDATES):raise ValueError('exact sixteen candidate outputs required')
    return {name:out[name] for name in CANDIDATES}


def validate_final_prediction(prediction,index,observed,prior):
    if prediction.index!=index:raise ValueError('final prediction exact ID/class order mismatch')
    mask=np.asarray(observed)
    if mask.dtype!=np.bool_ or mask.shape!=(len(index.sample_ids),) or np.any(prediction.valid&~mask):
        raise ValueError('final observed mask mismatch')
    expected=np.tile(np.asarray(prior,np.float32),((~mask).sum(),1))
    if not np.array_equal(prediction.probabilities[~mask],expected):raise ValueError('final all-missing prior mismatch')


def freeze_generation(protocol,registry,candidates,*,publish=True):
    p=protocol;c=VerificationContext(p,metadata_only=True);r=ArtifactRegistry(p,context=c)
    proofs,bodies=_source_and_completions(p,c)
    if tuple(p.recipe.get('candidates',()))!=CANDIDATES or tuple(candidates)!=CANDIDATES:
        raise ValueError('frozen sixteen candidate order/list mismatch')
    expected=collect_candidates(p)
    if any(candidates[k]!=expected[k] for k in CANDIDATES):raise ValueError('candidate references differ from completed stage outputs')
    root=p.run_root;state_path=root/'run_state.json';out=root/'protocol/task13_frozen_generation.json'
    with FileLock(root/'protocol/task13.lock',timeout=0):
        state=_read(state_path)
        if state.get('protocol_sha256')!=p.identity():raise ValueError('run state protocol mismatch')
        if state['state']=='revealed':raise ValueError('cannot freeze after revealed')
        if state['state']=='frozen':
            c.hash_file(Path(state['generation']['path']),state['generation']['sha256'])
            body=_read(out)
            if body['candidates']!={k:json.loads(canonical_bytes(v)) for k,v in candidates.items()}:
                raise ValueError('cannot change frozen candidates')
            load_frozen_predictions(out)
            return out
        if state['state'] not in {'prepared','generating'}:raise ValueError('invalid generation state')
        rows=read_public_rows(p.public_manifests['final4']);index=row_index(rows)
        observed=np.array([any(x[m+'_available']=='1' for m in ['ir','depth','skeleton','imu']) for x in rows],bool)
        if len(rows)!=p.partitions['final4'].expected_rows or set(index.user_ids)!=set(p.partitions['final4'].users):raise ValueError('final population mismatch')
        if p.recipe['execution_kind']=='formal' and (len(rows)!=609 or int((~observed).sum())!=18):raise ValueError('final 609/18 boundary mismatch')
        prior_ref=ref(_read(root/'A1/refit/prior_artifact.json'));prior_record=r.verify(prior_ref,'class_prior','refit')
        prior_file=Path(next(iter(prior_record.files)));r.verify_file(prior_record,prior_file)
        prior=np.asarray(_read(prior_file)['probabilities'],np.float32)
        payloads={};predictions={}
        for name in CANDIDATES:
            record=r.verify(candidates[name],name,'predict',index)
            if record.config.get('partition')!='final4':raise ValueError('candidate wrong final partition')
            paths=[Path(x) for x in record.files if Path(x).suffix=='.npz']
            if len(paths)!=1:raise ValueError('one candidate NPZ required')
            path=r.verify_file(record,paths[0]);a=c.load_arrays(path)
            if tuple(a['sample_ids'].tolist())!=index.sample_ids or not np.array_equal(a['class_ids'],np.arange(40)):
                raise ValueError('candidate payload ID/class mismatch')
            pred=Prediction(index,a['logits'],a['probabilities'],a['valid'],candidates[name])
            validate_final_prediction(pred,index,observed,prior)
            if name=='A8':
                if not np.array_equal(a['target_mask'],observed):raise ValueError('A8 final target mask mismatch')
                if not any(r.read(x).kind=='teacher_bank' for x in record.parents):raise ValueError('A8 requires closed source bank')
            predictions[name]=pred;payloads[name]=dict(path=path,sha256=record.files[str(path)])
        for record in list(c.validated_records.values()):
            if record.kind=='teacher_bank':
                for file in record.files:r.verify_file(record,Path(file))
        # Recheck small metadata/stat trust before publishing; no ancestor payload scan.
        for (_,path),(stamp,digest) in list(c.direct_files.items()):
            if c.stamp(Path(path))!=stamp:raise ValueError('freeze input changed during validation')
        metadata=[dict(path=Path(key[2]),sha256=key[1]) for key in c.validated_records]
        body=dict(schema_version=1,protocol_sha256=p.identity(),run_id=p.run_id,state_path=state_path,
            execution_kind=p.recipe['execution_kind'],
            candidate_order=list(CANDIDATES),candidates=candidates,payloads=payloads,rows=index,
            evaluator_source=dict(path=Path(__file__),sha256=c.hash_file(Path(__file__))),
            target_mask=observed.tolist(),prior=prior.tolist(),modalities={m:[x[m+'_available']=='1' for x in rows] for m in ['ir','depth','skeleton','imu']},
            proofs=proofs,metadata_records=metadata,
            consumed_small_files=[dict(path=Path(path),sha256=digest) for (_,path),(_,digest) in c.direct_files.items()],
            primary_contrast='A9-40_minus_A1',
            target_inputs_adapted=True,final_labels_read=False,accuracy_computed=False,
            bounded_verification=dict(unique_metadata=len(metadata),raw_payloads=0,large_payloads=0))
        if not publish:
            preview=root/'protocol/task13_freeze_preflight.json'
            write_json(preview,{**body,'unpublished_preview':True})
            return preview
        immutable_bytes(out,canonical_bytes(body)+b'\n')
        state.update(state='frozen',generation=dict(path=out,sha256=c.hash_file(out)),frozen_at=_now())
        write_json(state_path,state)
        return out


def load_frozen_predictions(generation):
    body=_read(generation)
    if tuple(body.get('candidate_order',()))!=CANDIDATES:raise ValueError('frozen candidate list mismatch')
    binding=body.get('evaluator_source')
    if binding and hashlib.sha256(Path(binding['path']).read_bytes()).hexdigest()!=binding['sha256']:
        raise ValueError('frozen evaluator source drift')
    for binding in [*body.get('proofs',{}).values(),*body.get('metadata_records',[]),*body.get('consumed_small_files',[])]:
        if hashlib.sha256(Path(binding['path']).read_bytes()).hexdigest()!=binding['sha256']:
            raise ValueError('frozen metadata/proof drift')
    index=RowIndex(tuple(body['rows']['sample_ids']),tuple(body['rows']['user_ids']),tuple(body['rows']['class_ids']))
    mask=np.asarray(body['target_mask'],bool);out={}
    for name in CANDIDATES:
        binding=body['payloads'][name];path=Path(binding['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=binding['sha256']:raise ValueError('frozen candidate payload drift')
        reference=ref(body['candidates'][name])
        if hashlib.sha256(reference.record_path.read_bytes()).hexdigest()!=reference.sha256:raise ValueError('frozen record drift')
        with np.load(path,allow_pickle=False) as a:
            if tuple(a['sample_ids'].tolist())!=index.sample_ids or not np.array_equal(a['class_ids'],np.arange(40)):raise ValueError('frozen prediction ID/class mismatch')
            out[name]=Prediction(index,a['logits'],a['probabilities'],a['valid'],reference)
        validate_final_prediction(out[name],index,mask,body['prior'])
    return body,out,index,mask,body['modalities']


def classification_metrics(prediction,labels):
    y=np.asarray(labels);n=len(prediction.index.sample_ids)
    if y.shape!=(n,) or not np.issubdtype(y.dtype,np.integer) or np.any((y<0)|(y>=40)):raise ValueError('evaluation label shape/range invalid')
    guesses=prediction.probabilities.argmax(1);correct=guesses==y
    support=np.bincount(y,minlength=40)
    return dict(total=n,correct=int(correct.sum()),accuracy=float(correct.mean()) if n else None,
        macro_f1_fixed40=float(f1_score(y,guesses,labels=np.arange(40),average='macro',zero_division=0)) if n else None,
        class_support=support.tolist(),observed_class_count=int((support>0).sum()),
        confusion_fixed40=confusion_matrix(y,guesses,labels=np.arange(40)).tolist())


def transfer_contrast(before,after,labels,mask,*,before_phase,after_phase):
    if before_phase!=after_phase or before_phase!='refit':raise ValueError('contrast requires same refit phase')
    if before.index!=after.index:raise ValueError('contrast ID/user/class population mismatch')
    y=np.asarray(labels);classification_metrics(before,y);classification_metrics(after,y)
    mask=np.asarray(mask)
    if mask.dtype!=np.bool_ or mask.shape!=y.shape:raise ValueError('contrast target mask mismatch')
    a=before.probabilities.argmax(1);b=after.probabilities.argmax(1);good_a=a==y;good_b=b==y
    result={}
    for name,part in [('all',np.ones(len(y),bool)),('adapted',mask),('excluded_prior',~mask)]:
        total=int(part.sum());rescue=int((~good_a&good_b&part).sum());harm=int((good_a&~good_b&part).sum())
        result[name]=dict(total=total,before_correct=int(good_a[part].sum()),after_correct=int(good_b[part].sum()),
            rescue=rescue,harm=harm,delta_correct=rescue-harm,delta_accuracy_points=100*(rescue-harm)/total if total else None,
            disagreement=int(((a!=b)&part).sum()))
    return result


def read_final_labels(path,index,*,label_format='opaque'):
    if label_format not in {'opaque','canonical_manifest'}:raise ValueError('explicit supported label format required')
    values={};seen=set();expected=dict(zip(index.sample_ids,index.user_ids))
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if not {'sample_id','class_id'}<=set(reader.fieldnames or []):raise ValueError('label columns missing')
        if label_format=='opaque' and set(reader.fieldnames)!= {'sample_id','class_id'}:raise ValueError('opaque label whitelist mismatch')
        for row in reader:
            original=row['sample_id'];sid=hashlib.sha256((NAMESPACE+original).encode()).hexdigest() if label_format=='canonical_manifest' else original
            if sid in seen:raise ValueError('duplicate label ID')
            seen.add(sid)
            if sid not in expected:
                if label_format=='opaque':raise ValueError('unexpected final label ID')
                continue
            if label_format=='canonical_manifest' and row.get('user_id')!=expected[sid]:raise ValueError('label user ownership mismatch')
            try:label=int(row['class_id'])
            except (TypeError,ValueError) as error:raise ValueError('invalid label class') from error
            if label not in range(40):raise ValueError('label class outside 0..39')
            values[sid]=label
    if set(values)!=set(index.sample_ids):raise ValueError('final label IDs incomplete')
    return np.asarray([values[s] for s in index.sample_ids],np.int64)


def approve_frozen_generation(generation,review_path):
    generation=Path(generation).resolve();body=_read(generation);state_path=Path(body['state_path'])
    with FileLock(state_path.parent/'protocol/task13.lock',timeout=0):
        state=_read(state_path);review_path=Path(review_path).resolve();review=_read(review_path)
        digest=hashlib.sha256(generation.read_bytes()).hexdigest()
        if (state.get('state')!='frozen' or state.get('generation',{}).get('sha256')!=digest
                or not review_path.is_relative_to(state_path.parent/'protocol')
                or review.get('generation_sha256')!=digest or review.get('protocol_sha256')!=body['protocol_sha256']
                or review.get('verdict')!='GO' or review.get('unresolved_p1_p2')):
            raise ValueError('review must approve this exact frozen generation/protocol')
        state['freeze_review']=dict(path=review_path,sha256=hashlib.sha256(review_path.read_bytes()).hexdigest())
        write_json(state_path,state)


def evaluate_frozen(generation,private_labels,output,*,label_format='opaque'):
    generation=Path(generation).resolve();body=_read(generation);state_path=Path(body['state_path']);output=Path(output).resolve()
    with FileLock(state_path.parent/'protocol/task13.lock',timeout=0):
        state=_read(state_path)
        if state.get('state')!='frozen':raise ValueError('final evaluation is once-only; state must be frozen, not revealed')
        digest=hashlib.sha256(generation.read_bytes()).hexdigest()
        if (state.get('protocol_sha256')!=body['protocol_sha256'] or state.get('generation',{}).get('sha256')!=digest
                or Path(state['generation']['path']).resolve()!=generation):raise ValueError('frozen generation/state binding mismatch')
        if body.get('execution_kind')=='formal':
            binding=state.get('freeze_review',{});path=Path(binding.get('path',''))
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=binding.get('sha256'):
                raise ValueError('formal reveal requires frozen-generation independent review')
            review=_read(path)
            if review.get('verdict')!='GO' or review.get('unresolved_p1_p2') or review.get('generation_sha256')!=digest or review.get('protocol_sha256')!=body['protocol_sha256']:
                raise ValueError('frozen generation independent review mismatch')
        body,predictions,index,mask,modalities=load_frozen_predictions(generation)
        # Burn access before opening truth: even read/validation failure cannot reset it.
        state.update(state='revealed',revealed_at=_now(),evaluation_output=str(output),label_format=label_format)
        write_json(state_path,state)
        labels=read_final_labels(private_labels,index,label_format=label_format)
        report=dict(protocol_sha256=body['protocol_sha256'],generation_sha256=digest,state='revealed',
            label_format=label_format,label_source=str(Path(private_labels).resolve()),candidates={},contrasts={},
            primary_contrast=body.get('primary_contrast','A9-40_minus_A1'),final_inputs_were_adapted=True)
        users=np.asarray(index.user_ids)
        for name,prediction in predictions.items():
            metrics=classification_metrics(prediction,labels);guesses=prediction.probabilities.argmax(1)
            metrics['by_user']={u:dict(total=int((users==u).sum()),correct=int((guesses[users==u]==labels[users==u]).sum()),accuracy=float((guesses[users==u]==labels[users==u]).mean())) for u in sorted(set(users))}
            metrics['by_modality']={}
            for modality,available in modalities.items():
                m=np.asarray(available,bool)
                metrics['by_modality'][modality]={str(v):dict(total=int((m==v).sum()),correct=int((guesses[m==v]==labels[m==v]).sum()),accuracy=float((guesses[m==v]==labels[m==v]).mean()) if (m==v).any() else None) for v in [True,False]}
            metrics['a8_target_mask']={str(v):dict(total=int((mask==v).sum()),correct=int((guesses[mask==v]==labels[mask==v]).sum()),accuracy=float((guesses[mask==v]==labels[mask==v]).mean()) if (mask==v).any() else None) for v in [True,False]}
            report['candidates'][name]=metrics
        for before,after in CONTRASTS:
            report['contrasts'][after+'_minus_'+before]=transfer_contrast(predictions[before],predictions[after],labels,mask,before_phase='refit',after_phase='refit')
        output.mkdir(parents=True,exist_ok=True)
        immutable_bytes(output/'metrics.json',canonical_bytes(report)+b'\n')
        return output/'metrics.json'
