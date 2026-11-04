"""Once-only final4 fixed-split evaluation; generation must already be frozen."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.experiments.no_vote_evaluation import evaluate_frozen
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generation',required=True,type=Path)
    parser.add_argument('--private-labels',required=True,type=Path)
    parser.add_argument('--label-format',choices=['opaque','canonical_manifest'],default='opaque')
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    print(evaluate_frozen(args.generation,args.private_labels,args.output,label_format=args.label_format),flush=True)
