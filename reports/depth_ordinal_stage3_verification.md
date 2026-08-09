# Stage 3: inverse-JET ordinal Depth verification

- Status: **passed**
- Real train/val trials checked: 2,910
- Valid source pixels decoded: 723,756,503
- Black invalid source pixels: 170,195,497
- Exact JET round-trip mismatches: 0
- Nonzero pixels behind the resized invalid mask: 0
- Output-size smoke check: `256x256`
- Competition test read: no
- Depth assets exported: no
- Training run: no

This stage implements and verifies the codec and mask-aware resize only. The Depth-only exporter and combined manifest belong to the next stage.
