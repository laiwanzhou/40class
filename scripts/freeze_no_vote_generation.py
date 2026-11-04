"""Freeze the completed sixteen outputs without opening private truth."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.experiments.no_vote_protocol import load_protocol
from src.experiments.artifact_record import ArtifactRegistry
from src.experiments.no_vote_evaluation import freeze_generation,collect_candidates
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args();p=load_protocol(args.config,inference_only=True)
    path=freeze_generation(p,ArtifactRegistry(p),collect_candidates(p),publish=not args.preflight_only)
    print(path,flush=True)
