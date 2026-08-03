# Depth/IR pose localization feasibility probe

## Probe design

- Training split only: 14 users, 12 actions, 48 trials, 288 paired frames.
- Four trials per action and six uniformly sampled paired frames per trial.
- Model: Ultralytics YOLO11n-pose, detection threshold 0.25, keypoint threshold 0.25, image size 640.
- Inputs tested independently: original Depth_Color pseudo-color and IR copied to three channels by the image loader.

## Overall metrics

| Input | Person | Both shoulders | Both wrists | >=1 wrist | Head/nose | Mean conf | Mean bbox area | Raw/EMA center jitter | Raw/EMA size jitter |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| depth | 57.29% | 56.60% | 54.51% | 56.25% | 56.60% | 0.693 | 0.145 | 0.0391/0.0218 | 0.0473/0.0246 |
| ir | 96.18% | 96.18% | 93.40% | 94.79% | 96.18% | 0.801 | 0.121 | 0.0141/0.0094 | 0.0208/0.0121 |

## Per-action metrics

| Input | Group | Person | Both shoulders | Both wrists | >=1 wrist | Head/nose | Mean conf | Bbox area |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| depth | Drink_water | 66.67% | 66.67% | 66.67% | 66.67% | 66.67% | 0.663 | 0.126 |
| depth | Eat_food | 66.67% | 66.67% | 66.67% | 66.67% | 66.67% | 0.755 | 0.126 |
| depth | Make_a_phone_call | 37.50% | 37.50% | 37.50% | 37.50% | 37.50% | 0.399 | 0.059 |
| depth | Play_games | 45.83% | 45.83% | 45.83% | 45.83% | 45.83% | 0.548 | 0.092 |
| depth | Stir_drinks | 75.00% | 75.00% | 75.00% | 75.00% | 75.00% | 0.825 | 0.153 |
| depth | Take_a_selfie | 58.33% | 58.33% | 58.33% | 58.33% | 58.33% | 0.855 | 0.198 |
| depth | Take_body_temperature | 62.50% | 62.50% | 62.50% | 62.50% | 62.50% | 0.806 | 0.187 |
| depth | Take_medicine | 50.00% | 50.00% | 50.00% | 50.00% | 50.00% | 0.701 | 0.223 |
| depth | Turn_pages | 25.00% | 25.00% | 25.00% | 25.00% | 25.00% | 0.475 | 0.164 |
| depth | Walk | 58.33% | 54.17% | 29.17% | 50.00% | 54.17% | 0.739 | 0.172 |
| depth | Watch_TV | 50.00% | 45.83% | 45.83% | 45.83% | 45.83% | 0.593 | 0.087 |
| depth | Wipe_bowls | 91.67% | 91.67% | 91.67% | 91.67% | 91.67% | 0.653 | 0.132 |
| ir | Drink_water | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 0.783 | 0.101 |
| ir | Eat_food | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 0.895 | 0.120 |
| ir | Make_a_phone_call | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 0.685 | 0.046 |
| ir | Play_games | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 0.804 | 0.080 |
| ir | Stir_drinks | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 0.888 | 0.146 |
| ir | Take_a_selfie | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 0.884 | 0.172 |
| ir | Take_body_temperature | 100.00% | 100.00% | 91.67% | 100.00% | 100.00% | 0.885 | 0.178 |
| ir | Take_medicine | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 0.819 | 0.126 |
| ir | Turn_pages | 58.33% | 58.33% | 58.33% | 58.33% | 58.33% | 0.599 | 0.114 |
| ir | Walk | 100.00% | 100.00% | 75.00% | 83.33% | 100.00% | 0.864 | 0.193 |
| ir | Watch_TV | 95.83% | 95.83% | 95.83% | 95.83% | 95.83% | 0.562 | 0.058 |
| ir | Wipe_bowls | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 0.853 | 0.108 |

## Per-user metrics

| Input | Group | Person | Both shoulders | Both wrists | >=1 wrist | Head/nose | Mean conf | Bbox area |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| depth | user1 | 26.67% | 23.33% | 20.00% | 20.00% | 23.33% | 0.483 | 0.106 |
| depth | user16 | 86.67% | 86.67% | 73.33% | 83.33% | 86.67% | 0.670 | 0.116 |
| depth | user18 | 66.67% | 66.67% | 66.67% | 66.67% | 66.67% | 0.687 | 0.199 |
| depth | user19 | 45.83% | 45.83% | 45.83% | 45.83% | 45.83% | 0.668 | 0.120 |
| depth | user2 | 44.44% | 44.44% | 44.44% | 44.44% | 44.44% | 0.621 | 0.136 |
| depth | user20 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 0.746 | 0.139 |
| depth | user21 | 50.00% | 50.00% | 50.00% | 50.00% | 50.00% | 0.905 | 0.208 |
| depth | user22 | 91.67% | 91.67% | 91.67% | 91.67% | 91.67% | 0.664 | 0.148 |
| depth | user3 | 75.00% | 75.00% | 75.00% | 75.00% | 75.00% | 0.735 | 0.143 |
| depth | user5 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | nan | nan |
| depth | user6 | 37.50% | 37.50% | 37.50% | 37.50% | 37.50% | 0.689 | 0.145 |
| depth | user7 | 54.17% | 54.17% | 54.17% | 54.17% | 54.17% | 0.789 | 0.153 |
| depth | user8 | 79.17% | 75.00% | 70.83% | 79.17% | 75.00% | 0.676 | 0.152 |
| depth | user9 | 33.33% | 33.33% | 33.33% | 33.33% | 33.33% | 0.805 | 0.141 |
| ir | user1 | 83.33% | 83.33% | 66.67% | 70.00% | 83.33% | 0.531 | 0.099 |
| ir | user16 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 0.823 | 0.100 |
| ir | user18 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 0.777 | 0.137 |
| ir | user19 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 0.851 | 0.080 |
| ir | user2 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 0.815 | 0.099 |
| ir | user20 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 0.883 | 0.129 |
| ir | user21 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 0.711 | 0.130 |
| ir | user22 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 0.858 | 0.134 |
| ir | user3 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 0.804 | 0.108 |
| ir | user5 | 58.33% | 58.33% | 58.33% | 58.33% | 58.33% | 0.649 | 0.050 |
| ir | user6 | 95.83% | 95.83% | 87.50% | 95.83% | 95.83% | 0.778 | 0.147 |
| ir | user7 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 0.886 | 0.155 |
| ir | user8 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 0.879 | 0.136 |
| ir | user9 | 100.00% | 100.00% | 94.44% | 100.00% | 100.00% | 0.896 | 0.144 |

## Gate status

The numeric gate is met by ir; `ir` is the provisional locator input. Training remains blocked until representative overlays are manually checked for true-person coverage, background false detections, small-person behavior, low-motion actions, and unacceptable temporal jumps.

## Manual overlay review and decision

The 12-action contact sheet and targeted failure overlays were inspected after
the numeric probe. IR is selected as the ROI localization input.

- In the representative IR overlays, the detected boxes and shoulder/elbow/wrist
  skeletons cover the actual person. No persistent detection of a door, sink,
  table, sofa, or other furniture as a person was observed.
- Depth_Color is not suitable as the locator source at the fixed threshold. In
  addition to its low 57.29% detection rate, at least one `Watch_TV` overlay
  placed a small pose box around the hands/table region instead of the full
  person.
- Small and distant people are often localized successfully in IR, but not
  reliably in every scene. The main failure is `Turn_pages` (58.33% person and
  wrist success): five of six sampled frames fail for `user5`, four fail for
  `user1`, and one fails for `user6`. These scenes show a small seated person
  partially occluded by the table/sofa arrangement.
- The low-motion targets remain detectable when the person is sufficiently
  visible: IR reaches 95.83% person success for `Watch_TV` and 100% for
  `Play_games`. `Turn_pages` is the specific low-motion exception.
- IR wrist localization is sufficient for guarded local cropping overall:
  at least one wrist is available in 94.79% of all frames and both wrists in
  93.40%. Missing frames must still use temporal interpolation and documented
  fallbacks rather than unchecked coordinates.
- Raw normalized center/size jitter is 0.0141/0.0208 and falls to
  0.0094/0.0121 with the probe EMA. This does not show severe persistent jumps;
  the ROI implementation must use interpolation plus sequence smoothing.
- The weakest IR users are `user5` (58.33% person and wrist success, driven by
  `Turn_pages`) and `user1` (83.33% person, 70.00% at least-one-wrist success).

All five training gates are therefore considered passed for IR with an explicit
small-person/`Turn_pages` risk. The next stage may train the matched E0/E1 hard
action experts, using IR coordinates only for ROI localization and Depth_Color
for every classifier view.
