# Hard-subset Depth_Color / IR frame pairing audit

Pairing key is the parsed `(absolute timestamp, frame ID)` from each filename. Sorted array indices are not used.

- Hard-subset samples audited: 744.
- Completely frame-aligned samples: 743.
- Samples with missing, duplicate, unparsed, or misaligned frames: 1.
- Final usable train samples: 595.
- Final usable validation samples: 148.
- Exact paired frames: 21069.

## Exceptions

| sample_id | split | reason |
| --- | --- | --- |
| train__c10__user1__2-1-1 | train | depth_unparsed=37; ir_unparsed=37 |

The usable set is 595 train / 148 validation, not E1's 596/148. A Depth-only pose-ROI baseline must therefore be retrained on this exact common subset.
