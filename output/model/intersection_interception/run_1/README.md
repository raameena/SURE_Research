# Run 1 — Early Interception

## Experiment

Intersection Interception

## Scenario

Cut-in vehicle is released immediately, providing the largest reaction margin.

## Setup

- Run ID: run_1
- CARLA Map: Town10HD_Opt
- Model: tfv4_l6_0
- Ego Spawn Point: 11
- Ego Traffic Light State: GREEN (fixed during experiment)
- Other Intersection Signals: RED (fixed during experiment)
- Interception Severity: EARLY
- Trigger Configuration: ego travel >= 0.000 m or 8.0 s timeout
- Interception Vehicle: vehicle.dodge.charger_2020
- Interception Vehicle Target Speed: 14.0 m/s
- Simulation Frequency: 20 Hz
- Model Inference Interval: Every 5 ticks
- Weather: WeatherParameters(cloudiness=20.000000, precipitation=0.000000, precipitation_deposits=0.000000, wind_intensity=10.000000, sun_azimuth_angle=300.000000, sun_altitude_angle=45.000000, fog_density=2.000000, fog_distance=0.750000, fog_falloff=0.100000, wetness=0.000000, scattering_intensity=1.000000, mie_scattering_scale=0.030000, rayleigh_scattering_scale=0.033100, dust_storm=0.000000)

## Relevant notes

interception_car.py cut-in/settling geometry and Early/Medium/Late trigger values are initial tuning parameters requiring a live CARLA run.
