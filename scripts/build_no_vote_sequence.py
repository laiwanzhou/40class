"""Extract a frozen same-phase A2 native sequence and anchor."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.experiments.no_vote_protocol import load_protocol
from src.experiments.no_vote_manifest import load_stage_inputs,read_public_rows,row_index
from src.experiments.visual_student import read_ref,build_sequence


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--partition',choices=['train12','development2','refit14','final4'],required=True)
    parser.add_argument('--model-phase',choices=['select','refit'],required=True)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args();p=load_protocol(args.config)
    model=read_ref(p.run_root/'A2'/args.model_phase/'artifact.json')
    pixel_part='final4' if args.partition=='final4' else 'refit14'
    pixels=read_ref(p.run_root/'pixels'/pixel_part/'full/artifact.json')
    rows=row_index(read_public_rows(load_stage_inputs(p,args.partition,'predict').public_manifest))
    ref=build_sequence(model,pixels,rows,p.run_root/'sequence'/args.model_phase/args.partition,protocol=p,device=args.device)
    print(json.dumps({'record_path':str(ref.record_path),'sha256':ref.sha256}),flush=True)


if __name__=='__main__':main()
