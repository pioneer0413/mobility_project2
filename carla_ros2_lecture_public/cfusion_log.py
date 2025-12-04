#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
import cv2
import numpy as np
import os
import sys
import time
import json
import argparse
import math
import csv
from std_msgs.msg import String
from sensor_msgs.msg import Image, PointCloud2, Imu
from cv_bridge import CvBridge
from ultralytics import YOLO

# -----------------------------
# LiDAR → NumPy 변환
# -----------------------------
def pointcloud2_to_array(cloud_msg):
    cloud_arr = np.frombuffer(cloud_msg.data, dtype=np.float32)
    num_points = cloud_msg.width * cloud_msg.height
    arr = cloud_arr.reshape(num_points, -1)
    return arr[:, :3]


# ==========================================================
# Sensor Fusion Node
# ==========================================================
class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_node')

        print("=== [Fusion + Performance Logging Enabled] ===")

        # -----------------------------------------------------
        # 1) 성능지표 로그 파일 초기화
        # -----------------------------------------------------
        self.log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "fusion_perf_log.csv")

        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "fps", "latency_ms", "tracking_stability"])

        # FPS 계산용
        self.prev_frame_time = time.time()

        # Tracking stability 계산용
        self.prev_track_ids = set()

        # -----------------------------------------------------
        # 모델 로드
        # -----------------------------------------------------
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path_m = os.path.join(current_dir, "model", "m.pt")
        path_original = os.path.join(current_dir, "model", "original.pt")

        try:
            self.model_vehicle = YOLO(path_m, task='detect')
            self.classes_vehicle = [0,1,2,3,4,5,6]

            self.model_traffic = YOLO(path_original, task='detect')
            self.classes_traffic = [3,4,5]
            print(">> YOLO 모델 로딩 성공")

        except Exception as e:
            print(f"[모델 로딩 실패] {e}")
            sys.exit(1)

        self.bridge = CvBridge()

        self.decision_pub = self.create_publisher(String, "/fusion/decision", 10)
        self.result_pub   = self.create_publisher(Image, "/fusion/result", 10)

        qos_profile = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)

        # -----------------------------------------------------
        # ROS 구독 설정
        # -----------------------------------------------------
        self.sub_img = self.create_subscription(
            Image, "/carla/hero/camera_front/image_color",
            self.image_callback, qos_profile)

        self.sub_view = self.create_subscription(
            Image, "/carla/hero/camera_view/image_color",
            self.view_callback, qos_profile)

        self.sub_lidar = self.create_subscription(
            PointCloud2, "/carla/hero/lidar/point_cloud",
            self.lidar_callback, qos_profile)

        self.sub_imu = self.create_subscription(
            Imu, "/carla/hero/imu", self.imu_callback, 10)

        # -----------------------------------------------------
        # 상태 변수
        # -----------------------------------------------------
        self.latest_img = None
        self.latest_view = None
        self.latest_lidar = None

        # LiDAR → Camera 변환 행렬
        self.R_lidar2cam = np.array([[0,-1,0],[0,0,-1],[1,0,0]])
        self.T_lidar2cam = np.array([0,-0.7,-1.6])

        self.K = None

        self.timer = self.create_timer(0.05, self.fusion_loop)

        # IMU 상태
        self.current_yaw = 0.0
        self.yaw_rate = 0.0
        self.lateral_accel = 0.0
        self.forward_accel = 0.0

    # ----------------------------------------------------------------------
    # 카메라 콜백
    # ----------------------------------------------------------------------
    def image_callback(self, msg):
        self.latest_img = msg
        if self.K is None:
            w, h = msg.width, msg.height
            fov = 110.0
            f = w / (2 * np.tan(np.deg2rad(fov/2)))
            self.K = np.array([[f,0,w/2],[0,f,h/2],[0,0,1]])

    def view_callback(self, msg): self.latest_view = msg
    def lidar_callback(self, msg): self.latest_lidar = msg

    # IMU 콜백
    def imu_callback(self, msg):
        x, y, z, w = msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w

        siny_cosp = 2*(w*z + x*y)
        cosy_cosp = 1 - 2*(y*y + z*z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

        self.yaw_rate = msg.angular_velocity.z
        self.forward_accel = msg.linear_acceleration.x
        self.lateral_accel = msg.linear_acceleration.y

    # ----------------------------------------------------------------------
    # 탐지 함수
    # ----------------------------------------------------------------------
    def process_detection(self, cv_img, model, classes, lidar_points, prefix, detect_traffic=False):

        if lidar_points is None or self.K is None:
            return [], cv_img

        # LiDAR Projection
        p_cam = np.dot(lidar_points, self.R_lidar2cam.T) + self.T_lidar2cam
        valid = (p_cam[:,2] > 0.5) & (p_cam[:,2] < 80)
        p_cam = p_cam[valid]

        p_2d = np.dot(p_cam, self.K.T)
        p_2d[:,0] /= p_2d[:,2]
        p_2d[:,1] /= p_2d[:,2]

        h, w = cv_img.shape[:2]

        u = p_2d[:,0].astype(int)
        v = p_2d[:,1].astype(int)
        d = p_cam[:,2]

        mask = (u>=0)&(u<w)&(v>=0)&(v<h)
        u, v, d = u[mask], v[mask], d[mask]

        # YOLO Tracking
        results = model.track(cv_img, persist=True, tracker="bytetrack.yaml",
                              conf=0.45, classes=classes, verbose=False)

        detected_objects = []
        fx = self.K[0,0]; cx = self.K[0,2]

        if results[0].boxes:
            for box in results[0].boxes:
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                track_id = int(box.id[0]) if box.id is not None else -1

                box_cx = (x1+x2)/2
                angle_deg = np.degrees(np.arctan((box_cx - cx)/fx))

                dist = 999
                roi = (u>=x1)&(u<=x2)&(v>=y1)&(v<=y2)
                depth_vals = d[roi]
                if len(depth_vals)>0:
                    valid_d = depth_vals[depth_vals>1.5]
                    if len(valid_d)>0:
                        dist = np.min(valid_d)

                detected_objects.append({
                    "id": f"{prefix}_{track_id}",
                    "angle": angle_deg,
                    "dist": dist,
                    "cls": cls_id,
                    "type": "traffic" if detect_traffic else "vehicle"
                })

        return detected_objects, cv_img

    # ----------------------------------------------------------------------
    # IMU Overlay 생략 (필요 시 유지)
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # 메인 Fusion Loop
    # ----------------------------------------------------------------------
    def fusion_loop(self):
        if self.latest_img is None or self.latest_lidar is None or self.K is None:
            return

        start_time = time.time()   # ⏱ Latency 측정 시작

        # FPS 계산
        now = time.time()
        fps = 1.0 / (now - self.prev_frame_time)
        self.prev_frame_time = now

        frame = self.bridge.imgmsg_to_cv2(self.latest_img, "bgr8")
        lidar_pts = pointcloud2_to_array(self.latest_lidar)

        # 탐지 수행
        objs_light, frame = self.process_detection(
            frame, self.model_traffic, self.classes_traffic, lidar_pts, "light", True)

        objs_car, frame = self.process_detection(
            frame, self.model_vehicle, self.classes_vehicle, lidar_pts, "car", False)

        all_objects = objs_light + objs_car

        # -----------------------------
        # Tracking Stability 계산
        # -----------------------------
        curr_ids = {obj["id"] for obj in all_objects}
        if len(self.prev_track_ids) > 0:
            tracking_stability = len(curr_ids & self.prev_track_ids) / len(self.prev_track_ids)
        else:
            tracking_stability = 1.0

        self.prev_track_ids = curr_ids

        # -----------------------------
        # Latency 계산
        # -----------------------------
        latency_ms = (time.time() - start_time) * 1000

        # -----------------------------
        # 성능 로그 저장
        # -----------------------------
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                time.time(),
                fps,
                latency_ms,
                tracking_stability
            ])

        # -----------------------------
        # 인지 결과를 판단 노드로 전달
        # -----------------------------
        min_light = 999
        light_state = "none"
        min_car = 999
        car_angle = 0

        for obj in all_objects:
            if obj["dist"] < 999:
                if obj["type"] == "traffic":
                    if obj["cls"] == 4: status = "traffic_red"
                    elif obj["cls"] == 5: status = "traffic_yellow"
                    else: status = "traffic_green"

                    if obj["dist"] < min_light:
                        min_light = obj["dist"]
                        light_state = status

                elif obj["type"] == "vehicle":
                    if obj["dist"] < min_car:
                        min_car = obj["dist"]
                        car_angle = obj["angle"]

        msg_data = {
            "light": light_state,
            "light_dist": min_light if min_light < 999 else -1.0,
            "vehicle_dist": min_car if min_car < 999 else -1.0,
            "vehicle_angle": car_angle
        }

        self.decision_pub.publish(String(data=json.dumps(msg_data)))

        # 출력 화면 표시
        out_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        self.result_pub.publish(out_msg)


# ================================================================
# MAIN
# ================================================================
def main(args=None):
    rclpy.init(args=args)
    node = SensorFusionNode()
    try:
        rclpy.spin(node)
    except:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
