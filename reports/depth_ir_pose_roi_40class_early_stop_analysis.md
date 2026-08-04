# Early-stop analysis of the 40-class Depth+IR pose-ROI run

## 1. Safe stop status

The run planned 30 epochs and has 19 complete epochs. The process was terminated while epoch 20 had no formal rows. The termination itself was not graceful, but all epoch-19 artifacts are complete and stable. No test data was read. `last_model.pt` is absent because the original trainer writes it only after the full loop.

## 2. Best checkpoints

| Checkpoint | Epoch | Accuracy | Macro-F1 | Weighted F1 | Loss | Top-3 | Top-5 | Predicted classes | Zero recall | Zero F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| best_accuracy | 9 | 0.450847 | 0.329802 | 0.420376 | 2.306281 | 0.644068 | 0.720339 | 37 | 8 | 8 |
| best_macro_f1 | 8 | 0.444068 | 0.339342 | 0.427365 | 2.205297 | 0.642373 | 0.733898 | 37 | 5 | 5 |

## 3. Overall tradeoff

Best Accuracy gains 0.006780 Accuracy while losing 0.009540 Macro-F1. It predicts 37 classes versus 37 for best Macro-F1.
The checkpoints disagree on 266 of 590 samples; only Accuracy is correct on 53, only Macro is correct on 49, both are correct on 213, and both are wrong on 275.

## 4. Complete action groups

### stable_easy

- Jog_in_place: F1 0.667/0.643, recall 0.545/0.818, predicted 7/17, true rank 3.00/1.27, top confusion Walk/Walk.
- Do_jumping_jacks: F1 0.667/0.667, recall 1.000/0.833, predicted 12/9, true rank 1.00/1.17, top confusion /Do_stretching_exercises.
- Sit_down: F1 0.725/0.762, recall 0.935/0.774, predicted 49/32, true rank 1.06/1.32, top confusion Do_squats/Do_squats.
- Stand_up: F1 0.814/0.711, recall 1.000/0.667, predicted 35/21, true rank 1.00/1.71, top confusion /Walk.
- Wash_face: F1 0.846/0.857, recall 0.917/1.000, predicted 14/16, true rank 1.17/1.00, top confusion Take_off_clothes/.
- Walk: F1 0.909/0.933, recall 0.986/0.986, predicted 83/79, true rank 1.01/1.01, top confusion Stand_up/Stand_up.

### usable

- Drink_water: F1 0.000/0.412, recall 0.000/0.538, predicted 3/21, true rank 9.31/4.54, top confusion Take_medicine/Sweep_the_floor.
- Eat_food: F1 0.350/0.415, recall 0.219/0.344, predicted 8/21, true rank 9.12/5.09, top confusion Peel_fruits/Peel_fruits.
- Do_stretching_exercises: F1 0.375/0.429, recall 0.333/0.500, predicted 14/24, true rank 5.56/6.28, top confusion Do_jumping_jacks/Check_the_time.
- Sweep_the_floor: F1 0.400/0.154, recall 0.444/0.111, predicted 11/4, true rank 6.67/4.89, top confusion Mop_the_floor/Mop_the_floor.
- Read_documents: F1 0.400/0.333, recall 0.533/0.200, predicted 25/3, true rank 3.67/8.40, top confusion Peel_fruits/Turn_pages.
- Listen_to_music_with_headphones: F1 0.421/0.231, recall 0.471/0.176, predicted 21/9, true rank 5.18/8.24, top confusion Sweep_the_floor/Drink_water.
- Wipe_windows_and_tables: F1 0.429/0.364, recall 0.333/0.667, predicted 5/24, true rank 4.89/6.56, top confusion Walk/Do_stretching_exercises.
- Brush_teeth: F1 0.444/0.571, recall 0.333/0.500, predicted 6/9, true rank 5.92/2.33, top confusion Massage_oneself/Comb_hair.
- Pour_drinks: F1 0.545/0.471, recall 0.562/0.500, predicted 17/18, true rank 3.44/2.88, top confusion Wipe_bowls/Stir_drinks.
- Tap_the_keyboard: F1 0.545/0.474, recall 0.706/0.529, predicted 27/21, true rank 3.53/5.29, top confusion Wipe_windows_and_tables/Pour_drinks.
- Wipe_hands: F1 0.552/0.552, recall 0.533/0.533, predicted 14/14, true rank 2.13/2.40, top confusion Wash_face/Wash_face.
- Do_squats: F1 0.609/0.560, recall 0.467/0.467, predicted 8/10, true rank 5.80/4.53, top confusion Take_off_clothes/Take_off_clothes.
- Lie_down: F1 0.667/0.545, recall 0.500/0.500, predicted 3/5, true rank 6.83/4.50, top confusion Read_documents/Wipe_windows_and_tables.

### difficult

- Comb_hair: F1 0.000/0.300, recall 0.000/0.333, predicted 6/11, true rank 6.00/2.67, top confusion Put_on_clothes/Put_on_clothes.
- Wipe_bowls: F1 0.000/0.167, recall 0.000/0.167, predicted 7/6, true rank 6.67/6.67, top confusion Wipe_hands/Take_medicine.
- Take_medicine: F1 0.074/0.069, recall 0.059/0.059, predicted 10/12, true rank 12.00/6.18, top confusion Take_off_clothes/Take_a_selfie.
- Take_body_temperature: F1 0.077/0.194, recall 0.067/0.200, predicted 11/16, true rank 4.93/3.27, top confusion Check_the_time/Massage_oneself.
- Use_a_mobile_phone: F1 0.080/0.083, recall 0.062/0.062, predicted 9/8, true rank 7.44/5.75, top confusion Peel_fruits/Peel_fruits.
- Massage_oneself: F1 0.133/0.261, recall 0.167/0.250, predicted 18/11, true rank 6.67/9.33, top confusion Listen_to_music_with_headphones/Take_a_selfie.
- Take_off_clothes: F1 0.154/0.100, recall 0.273/0.091, predicted 28/9, true rank 3.91/5.82, top confusion Put_on_clothes/Put_on_clothes.
- Check_the_time: F1 0.182/0.111, recall 0.333/0.167, predicted 32/24, true rank 3.42/5.50, top confusion Take_and_use_tableware/Take_and_use_tableware.
- Turn_pages: F1 0.182/0.316, recall 0.111/0.333, predicted 2/10, true rank 2.67/6.33, top confusion Read_documents/Peel_fruits.
- Stir_drinks: F1 0.211/0.242, recall 0.125/0.250, predicted 3/17, true rank 6.06/4.25, top confusion Pour_drinks/Pour_drinks.
- Mop_the_floor: F1 0.222/0.333, recall 0.167/0.500, predicted 3/12, true rank 9.00/8.83, top confusion Comb_hair/Take_body_temperature.
- Fold_clothes: F1 0.222/0.200, recall 0.222/0.111, predicted 9/1, true rank 5.22/8.11, top confusion Take_off_clothes/Jog_in_place.
- Put_on_clothes: F1 0.270/0.333, recall 0.333/0.333, predicted 22/15, true rank 2.33/4.87, top confusion Take_off_clothes/Do_stretching_exercises.
- Take_a_selfie: F1 0.312/0.294, recall 0.357/0.357, predicted 18/20, true rank 8.14/8.29, top confusion Check_the_time/Take_body_temperature.
- Play_games: F1 0.333/0.261, recall 0.333/0.250, predicted 12/11, true rank 9.67/8.17, top confusion Check_the_time/Stir_drinks.
- Peel_fruits: F1 0.375/0.226, recall 0.500/0.333, predicted 30/35, true rank 4.78/4.94, top confusion Drink_water/Stir_drinks.

### unrecognized

- Take_and_use_tableware: F1 0.000/0.000, recall 0.000/0.000, predicted 7/14, true rank 10.93/6.86, top confusion Use_a_mobile_phone/Wipe_bowls.
- Write: F1 0.000/0.000, recall 0.000/0.000, predicted 1/0, true rank 3.91/7.18, top confusion Tap_the_keyboard/Play_games.
- Make_a_phone_call: F1 0.000/0.000, recall 0.000/0.000, predicted 0/0, true rank 11.25/13.25, top confusion Put_on_clothes/Put_on_clothes.
- Watch_TV: F1 0.000/0.000, recall 0.000/0.000, predicted 0/0, true rank 16.17/13.83, top confusion Take_off_clothes/Drink_water.
- Do_lunges: F1 0.000/0.000, recall 0.000/0.000, predicted 0/1, true rank 18.00/15.20, top confusion Stand_up/Sit_down.

## 5. Checkpoint-sensitive actions

Comb_hair, Drink_water, Take_and_use_tableware, Wipe_bowls, Take_body_temperature, Massage_oneself, Take_off_clothes, Check_the_time, Turn_pages, Stir_drinks, Mop_the_floor, Fold_clothes, Put_on_clothes, Eat_food, Peel_fruits, Do_stretching_exercises, Sweep_the_floor, Read_documents, Listen_to_music_with_headphones, Wipe_windows_and_tables, Brush_teeth, Tap_the_keyboard, Jog_in_place, Do_jumping_jacks, Sit_down, Stand_up

## 6. Major confusion directions

### best_accuracy

- Do_squats -> Take_off_clothes: 6.
- Write -> Tap_the_keyboard: 5.
- Turn_pages -> Read_documents: 5.
- Take_body_temperature -> Check_the_time: 5.
- Eat_food -> Peel_fruits: 5.
- Do_stretching_exercises -> Do_jumping_jacks: 5.
- Comb_hair -> Put_on_clothes: 5.
- Write -> Read_documents: 4.
- Use_a_mobile_phone -> Peel_fruits: 4.
- Put_on_clothes -> Take_off_clothes: 4.

### best_macro_f1

- Eat_food -> Peel_fruits: 10.
- Use_a_mobile_phone -> Peel_fruits: 6.
- Take_medicine -> Take_a_selfie: 6.
- Write -> Play_games: 5.
- Take_body_temperature -> Massage_oneself: 5.
- Sweep_the_floor -> Mop_the_floor: 5.
- Stir_drinks -> Pour_drinks: 5.
- Peel_fruits -> Stir_drinks: 5.
- Listen_to_music_with_headphones -> Drink_water: 5.
- Wipe_hands -> Wash_face: 4.

## 7. Data-supported conclusions

Stable easy actions: Jog_in_place, Do_jumping_jacks, Sit_down, Stand_up, Wash_face, Walk.
Unrecognized by both checkpoints: Take_and_use_tableware, Write, Make_a_phone_call, Watch_TV, Do_lunges.
Actions most harmed when selecting Accuracy: Drink_water, Comb_hair, Wipe_bowls, Turn_pages, Massage_oneself.
Actions retained more strongly by the Macro-F1 checkpoint: Drink_water, Comb_hair, Wipe_bowls, Turn_pages, Massage_oneself, Brush_teeth, Take_body_temperature, Mop_the_floor, Eat_food, Put_on_clothes.
This report describes observed results only; it does not select experts, routers, class weights, modalities, or a next training configuration.
