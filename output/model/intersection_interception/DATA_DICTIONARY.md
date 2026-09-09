# Intersection Interception Data Dictionary

Blank fields mean a value was unavailable or not physically applicable. Pixel fields are class-assigned pixel/cell counts, not object counts.

## Inference tables

`inference_log.csv` is one run; `all_inferences.csv` contains the same rows across runs.

| Column | Datatype | Units | Meaning |
| --- | --- | --- | --- |
| experiment | string | — | Experiment family; Intersection Interception. |
| scenario | string | — | Early, Medium, or Late Interception variant. |
| run_id | string | — | Run folder identifier, such as run_3. |
| tick | integer | tick | Scenario-local control-loop tick. |
| carla_frame | integer | frame | CARLA frame shared with RGB/LiDAR filenames. |
| simulation_time_s | float | s | Scenario-local simulated time. |
| ego_speed_mps | float | m/s | Actual ego speed at inference. |
| ego_acceleration_mps2 | float | m/s² | Magnitude of ego acceleration. |
| camera_vehicle_pixels | integer | pixels | Camera semantic pixels classified as vehicle. |
| camera_pedestrian_pixels | integer | pixels | Camera semantic pixels classified as pedestrian. |
| camera_traffic_light_pixels | integer | pixels | Camera semantic pixels classified as traffic light. |
| bev_vehicle_pixels | integer | BEV cells | BEV cells classified as vehicle. |
| bev_walker_pixels | integer | BEV cells | BEV cells classified as walker. |
| bev_stop_sign_pixels | integer | BEV cells | BEV cells classified as stop sign. |
| bev_light_green_pixels | integer | BEV cells | BEV cells classified as green traffic light. |
| bev_light_yellow_pixels | integer | BEV cells | BEV cells classified as yellow traffic light. |
| bev_light_red_pixels | integer | BEV cells | BEV cells classified as red traffic light. |
| prob_speed_0 | float | probability | Probability of target speed 0 m/s. |
| prob_speed_2 | float | probability | Probability of target speed 2 m/s. |
| prob_speed_3 | float | probability | Probability of target speed 3 m/s. |
| prob_speed_6 | float | probability | Probability of target speed 6 m/s. |
| pred_target_speed_mps | float | m/s | Model-selected target speed. |
| decided_action | string | — | Decoded go, slow, stop, or hard_brake action. |
| previous_action | string | — | Action held before this inference. |
| decision_changed | boolean | — | Whether the decoded action changed. |
| throttle | float | [0,1] | Applied CARLA throttle. |
| brake | float | [0,1] | Applied CARLA brake. |
| steer | float | [-1,1] | Applied CARLA steering. |
| expected_perception | string | — | Ground-truth perception label. |
| actual_perception | string | — | Perception resolved from model BEV output. |
| perception_correct | boolean | — | Whether actual matches expected perception. |
| expected_safe_action | string | — | Scenario ground-truth safe action. |
| action_correct | boolean | — | Whether model action matches expected action. |
| timely_action | boolean | — | Stopping-distance timing result when applicable. |
| collision | boolean | — | Collision sensor event since the previous inference row. |
| distance_to_crossing_vehicle_m | float | m | Signed longitudinal gap from ego's front bumper to the interception vehicle's rear bumper, using interception_car.py logic; it may be negative while the vehicle is beside or behind ego. |
| crossing_vehicle_speed_mps | float | m/s | Actual interception-vehicle speed. |
| time_to_collision_s | float | s | Relative-motion TTC; blank when actors are not closing. |

## Completed-run table

`all_runs.csv` contains one row per completed run.

| Column | Datatype | Units | Meaning |
| --- | --- | --- | --- |
| experiment | string | — | Experiment family. |
| scenario | string | — | Scenario variant. |
| run_id | string | — | Run folder identifier. |
| interception_severity | string | — | EARLY, MEDIUM, or LATE. |
| trigger_distance_m | float | m | Configured ego travel distance before release. |
| trigger_max_seconds | float | s | Shared timeout fallback for crossing-car release. |
| actual_trigger_tick | integer | tick | Tick on which the crossing car was released. |
| actual_trigger_distance_m | float | m | Actual ego travel at crossing-car release. |
| initial_ego_speed_mps | float | m/s | Ego speed at first inference. |
| first_vehicle_detection_tick | integer | tick | First inference resolving vehicle presence. |
| first_vehicle_detection_distance_m | float | m | Vehicle distance at first detection. |
| first_slowdown_tick | integer | tick | First slow action. |
| first_slowdown_distance_m | float | m | Vehicle distance at first slow action. |
| first_stop_tick | integer | tick | First stop action, excluding hard_brake. |
| first_stop_distance_m | float | m | Vehicle distance at first stop action. |
| first_hard_brake_tick | integer | tick | First hard_brake action. |
| first_hard_brake_distance_m | float | m | Vehicle distance at first hard brake. |
| minimum_distance_to_crossing_vehicle_m | float | m | Smallest recorded signed longitudinal bumper gap. |
| minimum_time_to_collision_s | float | s | Smallest valid relative-motion TTC. |
| minimum_ego_speed_mps | float | m/s | Smallest recorded actual ego speed. |
| final_action | string | — | Last recorded decoded action. |
| decision_change_count | integer | changes | Number of action changes. |
| collision_occurred | boolean | — | Whether any collision event was recorded. |
