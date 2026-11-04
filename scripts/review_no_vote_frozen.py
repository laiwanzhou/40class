"""Register an independent freeze review using physical relocation-safe paths.

This separate administrative step preserves the already sealed evaluator source
and generation bytes. It never opens labels or changes candidate predictions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from filelock import FileLock
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.experiments.no_vote_types import write_json


def register_frozen_review(generation,review_path):
    generation=Path(generation).resolve();review_path=Path(review_path).resolve()
    body=json.loads(generation.read_text(encoding='utf-8'))
    state_path=Path(body['state_path']).resolve()
    protocol_root=state_path.parent/'protocol'
    with FileLock(protocol_root/'task13.lock',timeout=0):
        state=json.loads(state_path.read_text(encoding='utf-8'))
        review=json.loads(review_path.read_text(encoding='utf-8'))
        digest=hashlib.sha256(generation.read_bytes()).hexdigest()
        if (not generation.is_relative_to(protocol_root) and body.get('execution_kind')=='formal'
                or not review_path.is_relative_to(protocol_root)
                or state.get('state')!='frozen'
                or state.get('protocol_sha256')!=body['protocol_sha256']
                or Path(state.get('generation',{}).get('path','')).resolve()!=generation
                or state['generation'].get('sha256')!=digest
                or review.get('generation_sha256')!=digest
                or review.get('protocol_sha256')!=body['protocol_sha256']
                or review.get('verdict')!='GO' or review.get('unresolved_p1_p2')):
            raise ValueError('review must approve this exact frozen generation/protocol')
        state['freeze_review']=dict(path=review_path,sha256=hashlib.sha256(review_path.read_bytes()).hexdigest())
        write_json(state_path,state)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generation',type=Path,required=True)
    parser.add_argument('--review',type=Path,required=True)
    args=parser.parse_args();register_frozen_review(args.generation,args.review)
