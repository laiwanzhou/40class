"""Prepare train-only scene coverage panels and inventory source dimensions."""
from pathlib import Path
import json
import sys
from collections import Counter
from PIL import Image,ImageDraw
import yaml
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.data.visual90_corpus import prepare_trial,atomic_json
from src.data.pose_roi_dataset import PoseTrackCache
from src.models.visual90_encoders import file_hash


def main():
    out=ROOT/'outputs/visual90_appearance_temporal/corpus_qualification'
    if (out/'inventory.json').exists():raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    pfpath=ROOT/'reports/visual90_appearance_temporal_input_preflight_v3.json'
    pf=json.loads(pfpath.read_text(encoding='utf-8'));rows=pf['rows']
    if pf['continuity_blockers']:raise ValueError('continuity blocked')
    sizes=Counter();missing=[]
    for row in rows:
        if row['pairing_status']!='verified_keys':continue
        for name in ('ir','depth_color'):
            folder=Path(row[name+'_path']);paths=sorted(p for p in folder.iterdir() if p.suffix.lower() in {'.png','.jpg','.jpeg'})
            if not paths:raise ValueError('source disappeared')
            with Image.open(paths[0]) as im:size=im.size
            sizes[f'{name}:{size}']+=1
            if size!=(640,480):missing.append({'sample_id':row['sample_id'],'modality':name,'size':size})
    if missing:raise ValueError(f'unreviewed dimensions: {missing[:3]}')
    posepath=Path(pf['pose_path'])
    if file_hash(posepath)!=pf['pose_sha256']:raise ValueError('pose changed')
    pose=PoseTrackCache(posepath)
    roi=yaml.safe_load((ROOT/'configs/experiments/ir_depth_videomaev2_vit_b_p0.yaml').read_text())['roi']
    selected=[min((r for r in rows if r['partition']=='train' and r['class_id']==c and r['training_disposition']=='eligible_pending_geometry'),key=lambda r:r['sample_id']) for c in range(40)]
    reviews=[]
    for page in range(8):
        canvas=Image.new('RGB',(1280,2550),(25,25,25));cd=ImageDraw.Draw(canvas)
        for offset,row in enumerate(selected[page*5:(page+1)*5]):
            meta=prepare_trial(row,pose,roi,verify_pixels=False)
            active=[i for i in range(4) if any(meta['mask'][i])]
            clip=active[0];frame=int(meta['indices'][clip][7]);panels=[]
            for modality in range(2):
                source=Path(meta['paths'][modality][frame])
                with Image.open(source) as im:image=im.convert('RGB')
                draw=ImageDraw.Draw(image)
                for view,color in ((1,'cyan'),(2,'red'),(3,'yellow')):
                    if meta['mask'][clip][view]:draw.rectangle(tuple(meta['boxes'][clip][view][7]),outline=color,width=3)
                canvas.paste(image,(modality*640,offset*510+30))
                cd.text((modality*640+5,offset*510+5),f'QA {page*5+offset+1} '+('IR' if modality==0 else 'Depth'),fill='white')
                panels.append({'path':str(source),'sha256':file_hash(source)})
            reviews.append({'qa_index':page*5+offset+1,'sample_id':row['sample_id'],'class_id':row['class_id'],
                'partition':'train','clip':clip,'available_views':meta['mask'][clip],'sources':panels,'verdict':'pending'})
        canvas.save(out/f'page_{page+1}.jpg',quality=94)
    inventory={'preflight_sha256':file_hash(pfpath),'pose_sha256':pf['pose_sha256'],
        'dimension_inventory':dict(sizes),'selection_rule':'lexicographically first eligible TRAIN trial per class; first nonempty clip; position7',
        'samples':reviews,'status':'awaiting_visual_review'}
    atomic_json(out/'inventory.json',inventory)
    print(json.dumps({'dimension_inventory':dict(sizes),'training_panels':len(reviews),'path':str(out)}),flush=True)


if __name__=='__main__':main()
