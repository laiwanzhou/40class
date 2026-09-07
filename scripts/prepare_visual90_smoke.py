"""Prepare only the prescribed training smoke examples and geometric QA panels."""
from __future__ import annotations
from pathlib import Path
import json
import sys
from datetime import datetime
import numpy as np
from PIL import Image,ImageDraw
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.data.pose_roi_dataset import PoseTrackCache,paired_frame_paths,paired_frame_key,depth_frame_key
from src.data.visual90_dataset import select_continuous_clips
from src.data.visual90_evidence import build_view_boxes
from src.models.visual90_encoders import file_hash


def main():
    output=ROOT/'outputs/visual90_appearance_temporal/smoke'
    if (output/'evidence.json').exists(): raise FileExistsError('smoke evidence already exists')
    output.mkdir(parents=True,exist_ok=True)
    preflight_path=ROOT/'reports/visual90_appearance_temporal_input_preflight_v3.json'
    preflight=json.loads(preflight_path.read_text(encoding='utf-8'))
    if preflight['continuity_blockers']: raise ValueError('unresolved source continuity')
    selected=preflight['provisional_smoke_sample_ids']
    rows={r['sample_id']:r for r in preflight['rows']}
    pose_path=Path(preflight['pose_path'])
    if file_hash(pose_path)!=preflight['pose_sha256']: raise ValueError('pose source changed')
    pose=PoseTrackCache(pose_path)
    config=yaml.safe_load((ROOT/'configs/experiments/ir_depth_videomaev2_vit_b_p0.yaml').read_text())
    trials=[]; records=[]; observations=[]
    for index,sid in enumerate(selected):
        row=rows[sid]
        if row['partition']!='train' or row['training_disposition']!='eligible_pending_geometry':
            raise ValueError('nontraining/ineligible smoke member')
        dp,ip=paired_frame_paths(Path(row['depth_color_path']),Path(row['ir_path']))
        source_hashes=[]
        for pair in zip(dp,ip):
            for path in pair:
                with Image.open(path) as im: im.load(); size=im.size
                if size!=(640,480): raise ValueError('unreviewed source dimensions')
                source_hashes.append((str(path),file_hash(path)))
        for path in dp:
            if (sid,depth_frame_key(path)) not in pose.metadata_lookup: raise ValueError('pose metadata missing')
            pose.validate_frame(sid,path,640,480)
        p,k,c=pose.trial_arrays(sid,[depth_frame_key(path) for path in dp])
        moments=[datetime.strptime(paired_frame_key(path,'IR')[0],'%Y-%m-%d_%H-%M-%S.%f') for path in ip]
        times=np.array([(t-moments[0]).total_seconds()*1000 for t in moments])
        selection=select_continuous_clips(times)
        boxes,mask,diagnostics=build_view_boxes(p,k,c,selection,640,480,config['roi'])
        metadata={'sample_id':sid,'user_id':row['user_id'],'class_id':row['class_id'],
                  'source_hashes':source_hashes,'boxes':boxes.tolist(),'mask':mask.tolist(),
                  'indices':selection.indices.tolist(),'diagnostics':diagnostics,
                  'frame_times':[[float(times[j]) if j>=0 else 0 for j in js] for js in selection.indices]}
        trials.append(metadata)
        canvas=Image.new('RGB',(1280,4*510),(28,28,28)); draw=ImageDraw.Draw(canvas)
        for clip in range(4):
            if not selection.valid[clip]: continue
            frame_index=int(selection.indices[clip,7])
            for modality,paths in enumerate((ip,dp)):
                with Image.open(paths[frame_index]) as raw: image=raw.convert('RGB')
                d=ImageDraw.Draw(image)
                for view,color in ((1,'cyan'),(2,'red'),(3,'yellow')):
                    if mask[clip,view]:
                        d.rectangle(tuple(boxes[clip,view,7]),outline=color,width=3)
                        d.text(tuple(boxes[clip,view,7,:2]),['','person','left','right'][view],fill=color)
                canvas.paste(image,(modality*640,clip*510+30))
                draw.text((modality*640+5,clip*510+5),f'sample {index+1} clip {clip} '+('IR' if modality==0 else 'Depth'),fill='white')
                for view in range(4):
                    if not mask[clip,view]: continue
                    records.append({'trial':index,'clip_index':clip,'view':view,'modality':modality,
                        'paths':[str(paths[j]) for j in selection.indices[clip]],'boxes':boxes[clip,view].tolist()})
            observations.append({'trial':index,'clip':clip,'available_views':mask[clip].tolist(),
                                 'verdict':'pending_visual_review'})
        canvas.save(output/f'geometry_{index+1}.jpg',quality=94)
    evidence={'geometry_status':'unverified','preflight_sha256':file_hash(preflight_path),
              'pose_sha256':preflight['pose_sha256'],'trials':trials,'records':records,
              'observations':observations,'formal_training_authorized':False}
    (output/'evidence.json').write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'trials':len(trials),'encodable_video_records':len(records),
        'view_clip_support':np.asarray([t['mask'] for t in trials]).sum(axis=(0,1)).tolist(),
        'output':str(output),'geometry_status':'unverified'}),flush=True)


if __name__=='__main__': main()
