#!/usr/bin/env python3

import argparse
import json
import logging
import time
import math

import numpy as np
import carla
import cv2

import rclpy
from rclpy.node import Node
from rclpy.time import Time
# [수정] NavSatFix, Imu 메시지 추가
from sensor_msgs.msg import Image as RosImage, PointCloud2, PointField, NavSatFix, Imu
from rosgraph_msgs.msg import Clock
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist

# ==============================================================================
# -- Sensor Publishers ---------------------------------------------------------
# ==============================================================================

class LidarPublisher:
    def __init__(self, node: Node, sensor_id: str):
        self.node = node
        self.topic_name = f"/carla/hero/{sensor_id}"
        self.pub = node.create_publisher(PointCloud2, self.topic_name, 10)
        self.node.get_logger().info(f"[LiDAR] Ready to publish -> {self.topic_name}")

    def handle(self, carla_lidar_measurement):
        header = self.node.get_clock().now().to_msg()
        lidar_bytes = carla_lidar_measurement.raw_data
        num_points = len(lidar_bytes) // 16
        
        msg = PointCloud2()
        msg.header.stamp = header
        msg.header.frame_id = "lidar"
        msg.height = 1
        msg.width = num_points
        
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        
        msg.is_bigendian = False
        msg.point_step = 16 
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = False
        msg.data = lidar_bytes
        
        self.pub.publish(msg)

# [NEW] GNSS Publisher 클래스 추가
class GnssPublisher:
    def __init__(self, node: Node, sensor_id: str):
        self.node = node
        self.topic_name = f"/carla/hero/{sensor_id}"
        self.pub = node.create_publisher(NavSatFix, self.topic_name, 10)
        self.node.get_logger().info(f"[GNSS] Ready to publish -> {self.topic_name}")

    def handle(self, event):
        msg = NavSatFix()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = "gnss"
        msg.latitude = event.latitude
        msg.longitude = event.longitude
        msg.altitude = event.altitude
        self.pub.publish(msg)

# [NEW] IMU Publisher 클래스 추가
class ImuPublisher:
    def __init__(self, node: Node, sensor_id: str):
        self.node = node
        self.topic_name = f"/carla/hero/{sensor_id}"
        self.pub = node.create_publisher(Imu, self.topic_name, 10)
        self.node.get_logger().info(f"[IMU] Ready to publish -> {self.topic_name}")

    def handle(self, event):
        msg = Imu()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = "imu"
        # 좌표계 변환 (CARLA -> ROS)
        msg.linear_acceleration.x = event.accelerometer.x
        msg.linear_acceleration.y = -event.accelerometer.y
        msg.linear_acceleration.z = event.accelerometer.z
        msg.angular_velocity.x = event.gyroscope.x
        msg.angular_velocity.y = -event.gyroscope.y
        msg.angular_velocity.z = event.gyroscope.z
        self.pub.publish(msg)

class RgbPublisher:
    def __init__(self, node: Node, sensor_id: str):
        self.node = node
        self.bridge = CvBridge()
        self.topic_name = f"/carla/hero/{sensor_id}/image_color"
        self.pub = node.create_publisher(RosImage, self.topic_name, 10)
        self.node.get_logger().info(f"[RGB] Ready to publish -> {self.topic_name}")

    def handle(self, image: carla.Image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3]
        msg = self.bridge.cv2_to_imgmsg(array, encoding="bgr8")
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = "camera_front"
        self.pub.publish(msg)

class DepthColorizer:
    def __init__(self, node: Node, sensor_id: str):
        self.node = node
        self.bridge = CvBridge()
        self.topic_name = f"/carla/hero/{sensor_id}/image_depth"
        self.pub = node.create_publisher(RosImage, self.topic_name, 10)

    def handle(self, image: carla.Image):
        image.convert(carla.ColorConverter.LogarithmicDepth)
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3]
        msg = self.bridge.cv2_to_imgmsg(array, encoding="bgr8")
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = "camera_depth"
        self.pub.publish(msg)

class SemanticColorizer:
    def __init__(self, node: Node, sensor_id: str):
        self.node = node
        self.bridge = CvBridge()
        self.topic_name = f"/carla/hero/{sensor_id}/image_semantic"
        self.pub = node.create_publisher(RosImage, self.topic_name, 10)

    def handle(self, image: carla.Image):
        image.convert(carla.ColorConverter.CityScapesPalette)
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3]
        msg = self.bridge.cv2_to_imgmsg(array, encoding="bgr8")
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = "camera_semantic"
        self.pub.publish(msg)

# ==============================================================================
# -- Setup Functions -----------------------------------------------------------
# ==============================================================================

def _setup_vehicle(world, config):
    logging.info(f"Spawning vehicle: {config.get('type')}")
    bp_library = world.get_blueprint_library()
    map_ = world.get_map()
    bp = bp_library.filter(config.get("type"))[0]
    bp.set_attribute("role_name", config.get("id"))
    bp.set_attribute("ros_name", config.get("id"))
    return world.spawn_actor(bp, map_.get_spawn_points()[0])

def _setup_sensors(world, vehicle, sensors_config, node):
    bp_library = world.get_blueprint_library()
    sensor_actors = []
    handlers = []

    for sensor_conf in sensors_config:
        s_type = sensor_conf.get("type")
        s_id = sensor_conf.get("id")
        
        logging.debug(f"Setting up sensor: {s_id} ({s_type})")

        bp = bp_library.filter(s_type)[0]
        bp.set_attribute("role_name", s_id)
        bp.set_attribute("ros_name", s_id)
        for key, value in sensor_conf.get("attributes", {}).items():
            bp.set_attribute(str(key), str(value))

        sp = sensor_conf["spawn_point"]
        tf = carla.Transform(
            carla.Location(x=sp["x"], y=-sp["y"], z=sp["z"]),
            carla.Rotation(roll=sp["roll"], pitch=-sp["pitch"], yaw=-sp["yaw"])
        )

        actor = world.spawn_actor(bp, tf, attach_to=vehicle)
        sensor_actors.append(actor)

        if s_type.startswith("sensor.camera.rgb"):
            h = RgbPublisher(node, s_id)
            actor.listen(h.handle)
            handlers.append(h)
        elif s_type.startswith("sensor.camera.depth"):
            h = DepthColorizer(node, s_id)
            actor.listen(h.handle)
            handlers.append(h)
        elif s_type.startswith("sensor.camera.semantic"):
            h = SemanticColorizer(node, s_id)
            actor.listen(h.handle)
            handlers.append(h)
        elif s_type.startswith("sensor.lidar.ray_cast"):
            h = LidarPublisher(node, s_id)
            actor.listen(h.handle)
            handlers.append(h)
        # [NEW] GNSS 및 IMU 핸들러 연결
        elif s_type.startswith("sensor.other.gnss"):
            h = GnssPublisher(node, s_id)
            actor.listen(h.handle)
            handlers.append(h)
        elif s_type.startswith("sensor.other.imu"):
            h = ImuPublisher(node, s_id)
            actor.listen(h.handle)
            handlers.append(h)

    return sensor_actors, handlers

# ==============================================================================
# -- Main Loop -----------------------------------------------------------------
# ==============================================================================

def main(args):
    rclpy.init(args=None)
    node = rclpy.create_node("koo_bridge_node")
    clock_pub = node.create_publisher(Clock, "/clock", 10)
    
    client = None
    world = None
    vehicle = None
    sensor_actors = []
    
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)
        world = client.get_world()

        tm = client.get_trafficmanager(8000)

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        tm.set_synchronous_mode(True)

        with open(args.file) as f:
            config = json.load(f)

        vehicle = _setup_vehicle(world, config)
        vehicle.set_autopilot(False, tm.get_port()) 

        sensor_actors, handlers = _setup_sensors(world, vehicle, config.get("sensors", []), node)

        def _on_cmd(msg: Twist):
            if not vehicle: return
            
            vel = vehicle.get_velocity()
            spd = (vel.x**2 + vel.y**2 + vel.z**2) ** 0.5
            tgt = float(msg.linear.x)
            
            # 조향 각도 범위 보정 (기존 35.0 -> 70.0으로 조정, 더 잘 꺾이게)
            steer = max(-1.0, min(float(msg.angular.z)/70.0, 1.0))
            
            KP_THR, KP_BRK = 0.5, 1.0
            err = tgt - spd
            thr = max(0.0, min(err*KP_THR, 1.0)) if err > 0 else 0.0
            brk = max(0.0, min(-err*KP_BRK, 1.0)) if err <= 0 else 0.0
            
            if msg.linear.y > 0.0: brk = msg.linear.y; thr = 0.0
            vehicle.apply_control(carla.VehicleControl(throttle=thr, brake=brk, steer=steer))

        node.create_subscription(Twist, '/carla/hero/cmd_vel', _on_cmd, 10)
        logging.info("Koo Bridge Running (GNSS/IMU Added)...")
        
        spectator = world.get_spectator()
        
        while rclpy.ok():
            world.tick()
            
            snapshot = world.get_snapshot()
            sim_time = Time(seconds=snapshot.timestamp.elapsed_seconds)
            clock_msg = Clock()
            clock_msg.clock = sim_time.to_msg()
            clock_pub.publish(clock_msg)

            rclpy.spin_once(node, timeout_sec=0.001)

            if vehicle:
                tf = vehicle.get_transform()
                loc = tf.location - (tf.get_forward_vector() * 5.0)
                loc.z += 2.5
                rot = carla.Rotation(pitch=-15.0, yaw=tf.rotation.yaw, roll=0.0)
                spectator.set_transform(carla.Transform(loc, rot))

    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.error(f"Error: {e}")
    finally:
        if world:
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
        for s in sensor_actors: 
            if s.is_alive: s.destroy()
        if vehicle and vehicle.is_alive: vehicle.destroy()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--host', default='localhost')
    argparser.add_argument('--port', type=int, default=2000)
    argparser.add_argument('-f', '--file', default='lincoln.json', required=True)
    args = argparser.parse_args()
    logging.basicConfig(level=logging.INFO)
    main(args)