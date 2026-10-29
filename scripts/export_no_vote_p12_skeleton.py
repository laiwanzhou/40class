"""Export fresh reconstructed P12 Skeleton teacher in source FP16 storage format."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.experiments.no_vote_protocol import load_protocol
from src.experiments.p12_skeleton_export import export_skeleton_component
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--device',default='cuda')
    a=p.parse_args();export_skeleton_component(load_protocol(a.config,inference_only=True),device=a.device)
