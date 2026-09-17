from __future__ import annotations

import sys
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from cache_p101_large_temporal_tokens import (  # noqa: E402
    DEV_USERS,
    H3_USERS,
    P101TemporalClipDataset,
    load_dev_rows,
    user_from_sample_id,
)


def test_p101_large_cache_allow_list_excludes_h3() -> None:
    rows = load_dev_rows("videomaev2", limit=None)
    users = {user_from_sample_id(row["sample_id"]) for row in rows}
    assert len(rows) == 1941
    assert users == DEV_USERS
    assert not users & H3_USERS
    assert len({row["sample_id"] for row in rows}) == len(rows)


def test_p101_clip_contract_retains_six_local_temporal_views() -> None:
    rows = load_dev_rows("internvideo2", limit=1)
    item = P101TemporalClipDataset(rows)[0]
    assert user_from_sample_id(item["sample_id"]) in DEV_USERS
    assert len(item["clips"]) == 6
    assert all(len(clip) == 16 for clip in item["clips"])
