import carla
import logging
import math
import os


# -------------------------
# RGB camera sensor
# -------------------------

def attach_rgb_camera(world, bp_lib, vehicle, images_folder, camera_transform=None):
    """
    Attaches an RGB camera to the ego vehicle and saves image frames
    into the images folder.

    camera_transform, if given, overrides the default mount position/
    rotation below -- e.g. so a scenario's recording camera can match its
    model-input camera's mount (see scenarios/model/intersection/
    simple_stop_go.py's CAMERA_MOUNT). Defaults to the original mount used
    by every other scenario, unaffected by this parameter.

    Returns:
        camera actor
    """
    camera_bp = bp_lib.find("sensor.camera.rgb")

    camera_bp.set_attribute("image_size_x", "800")
    camera_bp.set_attribute("image_size_y", "600")

    if camera_transform is None:
        camera_transform = carla.Transform(
            carla.Location(x=1.5, z=2.4)
        )

    camera = world.spawn_actor(
        camera_bp,
        camera_transform,
        attach_to=vehicle
    )

    camera.listen(
        lambda image: image.save_to_disk(
            os.path.join(images_folder, f"frame_{image.frame:06d}.png")
        )
    )

    logging.debug("RGB camera attached and recording.")

    return camera


# -------------------------
# LiDAR sensor
# -------------------------

def attach_lidar(world, bp_lib, vehicle, lidar_folder):
    """
    Attaches a LiDAR sensor to the ego vehicle and saves point cloud files
    into the lidar_pointclouds folder.

    Returns:
        lidar actor
    """
    lidar_bp = bp_lib.find("sensor.lidar.ray_cast")

    lidar_bp.set_attribute("range", "50")
    lidar_bp.set_attribute("rotation_frequency", "10")
    lidar_bp.set_attribute("channels", "32")
    lidar_bp.set_attribute("points_per_second", "56000")

    lidar_transform = carla.Transform(
        carla.Location(x=0.0, z=2.5)
    )

    lidar = world.spawn_actor(
        lidar_bp,
        lidar_transform,
        attach_to=vehicle
    )

    lidar.listen(
        lambda point_cloud: point_cloud.save_to_disk(
            os.path.join(lidar_folder, f"frame_{point_cloud.frame:06d}.ply")
        )
    )

    logging.debug("LiDAR sensor attached and recording.")

    return lidar


# -------------------------
# GPS / GNSS sensor
# -------------------------

def attach_gps(world, bp_lib, vehicle, gps_data):
    """
    Attaches a GPS/GNSS sensor to the ego vehicle.

    The GPS readings are appended to the gps_data list.

    Returns:
        gps actor
    """
    gps_bp = bp_lib.find("sensor.other.gnss")

    gps_transform = carla.Transform(
        carla.Location(x=0.0, z=2.8)
    )

    gps = world.spawn_actor(
        gps_bp,
        gps_transform,
        attach_to=vehicle
    )

    def gps_callback(data):
        gps_data.append(
            {
                "frame": data.frame,
                "timestamp": data.timestamp,
                "latitude": data.latitude,
                "longitude": data.longitude,
                "altitude": data.altitude
            }
        )

    gps.listen(gps_callback)

    logging.debug("GPS/GNSS sensor attached and recording.")

    return gps


# -------------------------
# Collision sensor
# -------------------------

def attach_collision_sensor(world, bp_lib, vehicle, collision_data):
    """
    Attaches a collision sensor to the ego vehicle.

    Each collision event is appended to the collision_data list, recording
    the simulation frame, timestamp, the actor that was struck, and the
    collision impulse.

    Returns:
        collision sensor actor
    """
    collision_bp = bp_lib.find("sensor.other.collision")

    collision_transform = carla.Transform(carla.Location(x=0.0, z=0.0))

    collision_sensor = world.spawn_actor(
        collision_bp,
        collision_transform,
        attach_to=vehicle
    )

    def collision_callback(event):
        impulse = event.normal_impulse

        impulse_magnitude = math.sqrt(
            impulse.x ** 2 +
            impulse.y ** 2 +
            impulse.z ** 2
        )

        other_actor = event.other_actor

        collision_data.append(
            {
                "frame": event.frame,
                "timestamp": event.timestamp,
                "impact_actor_id": other_actor.id if other_actor else None,
                "impact_actor_type": other_actor.type_id if other_actor else None,
                "impulse": {
                    "x": impulse.x,
                    "y": impulse.y,
                    "z": impulse.z,
                    "magnitude": impulse_magnitude
                }
            }
        )

        logging.warning(
            f"Collision detected at frame {event.frame} with "
            f"{other_actor.type_id if other_actor else 'unknown actor'} "
            f"(impulse magnitude={impulse_magnitude:.1f})."
        )

    collision_sensor.listen(collision_callback)

    logging.debug("Collision sensor attached and recording.")

    return collision_sensor


# -------------------------
# Standard sensor package
# -------------------------

def attach_standard_sensors(world, bp_lib, vehicle, images_folder, lidar_folder, gps_data, camera_transform=None):
    """
    Attaches the standard sensor package to the ego vehicle:

        RGB camera
        LiDAR
        GPS/GNSS

    camera_transform is forwarded to attach_rgb_camera() -- see its
    docstring.

    Returns:
        list of sensor actors
    """
    sensors = []

    camera = attach_rgb_camera(
        world,
        bp_lib,
        vehicle,
        images_folder,
        camera_transform=camera_transform
    )
    sensors.append(camera)

    lidar = attach_lidar(
        world,
        bp_lib,
        vehicle,
        lidar_folder
    )
    sensors.append(lidar)

    gps = attach_gps(
        world,
        bp_lib,
        vehicle,
        gps_data
    )
    sensors.append(gps)

    logging.debug("Standard sensor package attached.")

    return sensors


# -------------------------
# Spectator camera
# -------------------------

def follow_vehicle_with_spectator(
    world,
    vehicle,
    distance=12,
    height=5,
    pitch=-8
):
    """
    Moves the CARLA spectator camera behind the ego vehicle.

    This affects the CARLA simulator window view only.
    It does not affect the RGB camera sensor attached to the vehicle.
    """
    spectator = world.get_spectator()
    vehicle_transform = vehicle.get_transform()

    forward_vector = vehicle_transform.get_forward_vector()

    spectator_location = vehicle_transform.location - (
        forward_vector * distance
    ) + carla.Location(z=height)

    spectator_transform = carla.Transform(
        spectator_location,
        carla.Rotation(
            pitch=pitch,
            yaw=vehicle_transform.rotation.yaw
        )
    )

    spectator.set_transform(spectator_transform)
