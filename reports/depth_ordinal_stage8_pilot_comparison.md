# Stage 8 Depth representation pilot comparison

All three pilots used the same split, seed, model, optimizer, frame budget, and fixed epoch budget.

Selection rule fixed before pilot results: retain representations within `0.020` absolute Accuracy of the best pilot, then maximize validation Macro-F1, small-action Macro-F1, smaller absolute train/validation gap, worst-user Macro-F1, and Accuracy in that order.

| depth_representation   |   best_macro_epoch |   accuracy |   macro_f1 |   small_action_macro_f1 |   generalization_gap |   worst_user_macro_f1 |   zero_f1_class_count |   predicted_class_count | accuracy_eligible   |
|:-----------------------|-------------------:|-----------:|-----------:|------------------------:|---------------------:|----------------------:|----------------------:|------------------------:|:--------------------|
| raw                    |                  8 |   0.252542 |  0.0975415 |               0.0483631 |           0.0323714  |             0.0522807 |                    23 |                      20 | False               |
| relative               |                  6 |   0.274576 |  0.12065   |               0.0596118 |           0.00645821 |             0.0864621 |                    23 |                      18 | True                |
| raw+relative           |                  7 |   0.262712 |  0.101712  |               0.025847  |           0.0174605  |             0.0584516 |                    27 |                      17 | True                |

Selected representation: **`relative`**.

Competition test was not read. Skeleton/IMU/Radar were not read or connected.
