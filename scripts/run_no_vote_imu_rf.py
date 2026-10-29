"""Fit source IMU RF per phase and export both consumer-specific logits."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.experiments.no_vote_protocol import load_protocol
from src.experiments.imu_rf_teacher import run_rf_pipeline
from src.experiments.visual_student import phase_lock


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();protocol=load_protocol(args.config,inference_only=True)
    with phase_lock(protocol.run_root/'A4'):run_rf_pipeline(protocol)


if __name__=='__main__':main()
