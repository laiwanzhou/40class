"""Top-3-only strict outer-cross-fit reranker ablation.

This is deliberately a source-side protocol restriction, not a held-fold
exception: every outer fold uses only the P244 posterior Top-3 candidate set.
"""
from pathlib import Path

import p311_p245_topk_pair_reranker_oof as p311


def main() -> None:
    p311.KS = (3,)
    p311.O = Path(__file__).resolve().parent / "runs/p351_p245_top3_pair_reranker_oof_v1"
    p311.main()


if __name__ == "__main__":
    main()
