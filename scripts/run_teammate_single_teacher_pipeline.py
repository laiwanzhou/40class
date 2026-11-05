"""Preflight/synthetic smoke/source-stage generate/freeze; never evaluates truth."""
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.experiments.no_vote_orchestrator import run_pipeline
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['preflight','smoke','generate','freeze'])
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--smoke-output',type=Path)
    args=parser.parse_args();result=run_pipeline(args.mode,args.config,smoke_output=args.smoke_output)
    print(json.dumps(result,ensure_ascii=False) if isinstance(result,dict) else result,flush=True)
