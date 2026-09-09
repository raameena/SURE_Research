# Object in Road Data Dictionary

Blank CSV fields mean that a value was unavailable or not applicable. Boolean values are written as `True` or `False`.

## Inference tables

`inference_log.csv` contains one row for every inference attempt in one run. `all_inferences.csv` contains the same rows appended across all Object in Road runs.

| Column | Datatype | Units | Description |
| --- | --- | --- | --- |
| experiment | string | — | Experiment family; `Object in Road`. |
| scenario | string | — | Scenario variant: Chair, Pedestrian, or Vehicle. |
| run_id | string | — | Shared experiment run identifier, such as `run_8`. |
| tick | integer | tick | Scenario-local control-loop tick, beginning at zero. |
| carla_frame | integer | frame | CARLA frame used by the model camera and LiDAR inputs. |
| simulation_time_s | float | s | Scenario-local simulated time, computed from tick and fixed simulation step. |
| ego_speed_mps | float | m/s | Actual CARLA ego vehicle speed, separate from model target speed. |
| ego_acceleration_mps2 | float | m/s² | Magnitude of CARLA's ego acceleration vector at inference time. |
| ground_truth_object_type | string | — | Actual obstacle variant placed in the road. |
| distance_to_object_m | float | m | Ground-truth distance from ego to the obstacle's near surface. |
| camera_vehicle_pixels | integer | pixels | Semantic-image pixels assigned to vehicle; this is not an object count. |
| camera_pedestrian_pixels | integer | pixels | Semantic-image pixels assigned to pedestrian; this is not an object count. |
| camera_traffic_light_pixels | integer | pixels | Semantic-image pixels assigned to traffic light; this is not an object count. |
| bev_vehicle_pixels | integer | BEV cells/pixels | Semantic BEV cells assigned to vehicle; this is not an object count. |
| bev_walker_pixels | integer | BEV cells/pixels | Semantic BEV cells assigned to walker/pedestrian. |
| bev_stop_sign_pixels | integer | BEV cells/pixels | Semantic BEV cells assigned to stop sign. |
| bev_light_green_pixels | integer | BEV cells/pixels | Semantic BEV cells assigned to green traffic light. |
| bev_light_yellow_pixels | integer | BEV cells/pixels | Semantic BEV cells assigned to yellow traffic light. |
| bev_light_red_pixels | integer | BEV cells/pixels | Semantic BEV cells assigned to red traffic light. |
| prob_speed_0 | float | probability [0,1] | Probability assigned to the 0 m/s target-speed bin. |
| prob_speed_2 | float | probability [0,1] | Probability assigned to the 2 m/s target-speed bin. |
| prob_speed_3 | float | probability [0,1] | Probability assigned to the 3 m/s target-speed bin. |
| prob_speed_6 | float | probability [0,1] | Probability assigned to the 6 m/s target-speed bin. |
| pred_target_speed_mps | float | m/s | PCLA-processed target speed: brake-bin threshold or probability-weighted expected speed. |
| decided_action | string | — | High-level action decoded from the model target speed. |
| previous_action | string | — | Action held before this inference result. |
| decision_changed | boolean | — | Whether `decided_action` differs from `previous_action`. |
| throttle | float | [0,1] | Newly issued PCLA-controller throttle command for this inference row. |
| brake | float | [0,1] | Newly issued PCLA-controller service-brake command for this inference row. |
| steer | float | [-1,1] | Newly issued PCLA-controller steering command for this inference row. |
| target_point_x | float | m | Active world-route target in ego-local coordinates; positive is forward. |
| target_point_y | float | m | Active world-route target in ego-local coordinates; positive is right. |
| road_command | string | command | Lagged PCLA RoadOption command supplied to the model. |
| expected_perception | string | — | Ground-truth perception label expected at the measured distance. |
| actual_perception | string | — | Perception label resolved from the model's own output. |
| perception_correct | boolean | — | Whether actual and expected perception labels match. |
| expected_safe_action | string | — | Safe action expected from scenario ground truth. |
| action_correct | boolean | — | Whether the decoded action matches the expected safe action. |
| timely_action | boolean | — | Whether a correct action was issued with sufficient physical stopping distance; blank when not applicable. |
| collision | boolean | — | Whether a collision event occurred since the preceding inference row. |
| time_to_collision_s | float | s | Estimated center-to-center time to collision from current relative motion; blank when actors are not closing. |

The `camera_*_pixels` fields count semantic-image pixels assigned to a class. They are **not object counts**. The `bev_*_pixels` fields count semantic BEV cells/pixels assigned to a class. Raw counts are preserved even when below the model's detection threshold.

## Completed-run table

`all_runs.csv` contains one row per successfully completed run.

| Column | Datatype | Units | Description |
| --- | --- | --- | --- |
| experiment | string | — | Experiment family. |
| scenario | string | — | Scenario variant. |
| run_id | string | — | Run folder identifier. |
| initial_distance_m | float | m | Configured starting object distance. |
| minimum_distance_m | float | m | Smallest object distance observed at an inference check. |
| first_slowdown_tick | integer | tick | First tick whose decided action was `slow`; blank if absent. |
| first_slowdown_distance_m | float | m | Object distance at the first `slow` decision. |
| first_stop_tick | integer | tick | First tick whose action was `stop` or `hard_brake`; blank if absent. |
| first_stop_distance_m | float | m | Object distance at the first stop/hard-brake decision. |
| minimum_ego_speed_mps | float | m/s | Smallest actual ego speed observed at inference checks. |
| final_action | string | — | Last action recorded by an inference row. |
| decision_change_count | integer | changes | Number of inference rows that changed the held action. |
| collision_occurred | boolean | — | Whether the collision sensor recorded any event during the run. |
