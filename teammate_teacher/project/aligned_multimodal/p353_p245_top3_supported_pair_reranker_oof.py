"""Strict Top-3 pair reranker with a minimum seven-error source support.

The support rule is uniform across outer folds and uses only the two source
cohorts.  It suppresses tiny pair specialists whose apparent perfect source
precision is based on only a handful of examples.
"""
from pathlib import Path

import numpy as np

import p311_p245_topk_pair_reranker_oof as p311


MIN_SOURCE_ERRORS = 7


def supported_source_pairs(part, k):
    pairs = {}
    labels = part["labels"]
    base = part["base"]
    recovered = (base != labels) & np.any(part["order"][:, :k] == labels[:, None], axis=1)
    for i in np.flatnonzero(recovered):
        pair = tuple(sorted((int(base[i]), int(labels[i]))))
        pairs[pair] = pairs.get(pair, 0) + 1
    return {pair for pair, count in pairs.items() if count >= MIN_SOURCE_ERRORS}


def main() -> None:
    p311.KS = (3,)
    p311.source_pairs = supported_source_pairs
    p311.O = Path(__file__).resolve().parent / "runs/p353_p245_top3_supported_pair_reranker_oof_v1"
    p311.main()


if __name__ == "__main__":
    main()
