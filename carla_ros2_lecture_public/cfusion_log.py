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
import csv
from std_msgs.msg import String
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
from ultralytics import YOLO


def pointcloud2_to_array(cloud_msg):
    cloud_arr = np.frombuffer(cloud_msg.data, dtype=np.float32)
    num_points = cloud_msg.width * cloud_msg.height
    arr = cloud_arr.reshape(num_points, -1)
    return arr[:, :3]


class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_node')

        print("=== [Fusion + Performance Logging Enabled] ===")

        # ------------------------------------------------------------------
        # ✔ 로그 디렉토리 및 파일 초기화
        # ------------------------------------------------------------------
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)

        self.log_perf = os.path.join(log_dir, "fusion_perf_log.csv")
        self.log_det = os.path.join(log_dir, "detection_log.csv")

        # 성능 로그 파일 생성
        if not os.path.exists(self.log_perf):
            with open(self.log_perf, "w") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "fps", "latency_ms",
                    "num_vehicle", "num_traffic",
                    "min_vehicle_dist", "min_traffic_dist"
                ])

        # Detection 로그 파일 생성
        if not os.path.exists(self.log_det):
            with open(self.log_det, "w") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "type", "cls", "track_id",
                    "conf", "bbox_w", "bbox_h", "bbox_area",
                    "depth", "angle"
                ])

        self.prev_time = time.time()

        # ------------------------------------------------------------------
        # YOLO 모델 로드
        # ------------------------------------------------------------------
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path_m = os.path.join(current_dir, "model", "m.pt")
        path_original = os.path.join(current_dir, "model", "original.pt")

        try:
            self.model_vehicle = YOLO(path_m, task='detect')
            self.classes_vehicle = [0, 1, 2, 3, 4, 5, 6]

            self.model_traffic = YOLO(path_original, task='detect')
            self.classes_traffic = [3, 4, 5]

            print(">> 모델 로딩 완료")
        except Exception as e:
            print(f"[오류] 모델 로딩 실패: {e}")
            sys.exit(1)

        self.bridge = CvBridge()
        self.decision_pub = self.create_publisher(String, "/fusion/decision", 10)
        self.result_pub = self.create_publisher(Image, "/fusion/result", 10)

        qos_profile = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.sub_img = self.create_subscription(Image, "/carla/hero/camera_front/image_color", self.image_callback, qos_profile)
        self.sub_view = self.create_subscription(Image, "/carla/hero/camera_view/image_color", self.view_callback, qos_profile)
        self.sub_lidar = self.create_subscription(PointCloud2, "/carla/hero/lidar/point_cloud", self.lidar_callback, qos_profile)

        self.latest_img = None
        self.latest_view = None
        self.latest_lidar = None
        self.memory_buffer = {}

        # LiDAR → Camera 변환
        self.R_lidar2cam = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
        self.T_lidar2cam = np.array([0, -0.7, -1.6])

        self.K = None
        self.timer = self.create_timer(0.05, self.fusion_loop)

    # ----------------------------------------------------------------------
    # 카메라 콜백
    # ----------------------------------------------------------------------
    def image_callback(self, msg):
        self.latest_img = msg
        if self.K is None:
            w, h = msg.width, msg.height
            fov = 110.0
            f = w / (2.0 * np.tan(np.deg2rad(fov / 2.0)))
            self.K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])

    def view_callback(self, msg):
        self.latest_view = msg

    def lidar_callback(self, msg):
        self.latest_lidar = msg

    # ----------------------------------------------------------------------
    # ✔ Detection 단위 성능 로그 저장 함수
    # ----------------------------------------------------------------------
    def write_detection_log(self, info):
        with open(self.log_det, "a") as f:
            writer = csv.writer(f)
            writer.writerow([
                time.time(),
                info["type"],
                info["cls"],
                info["track_id"],
                info["conf"],
                info["bbox_w"], info["bbox_h"], info["bbox_area"],
                info["depth"], info["angle"]
            ])

    # ----------------------------------------------------------------------
    # ✔ Frame-level 성능 로그 저장 함수
    # ----------------------------------------------------------------------
    def write_perf_log(self, fps, latency_ms, v_count, t_count, min_v, min_t):
        with open(self.log_perf, "a") as f:
            writer = csv.writer(f)
            writer.writerow([
                time.time(), fps, latency_ms,
                v_count, t_count, min_v, min_t
            ])

    # ----------------------------------------------------------------------
    # YOLO + LiDAR Fusion 처리
    # ----------------------------------------------------------------------
    def process_detection(self, cv_img, model, classes, lidar_points, prefix, detect_traffic=False):
        if lidar_points is None or self.K is None:
            return [], cv_img

        # LiDAR → Camera Projection
        p_cam = np.dot(lidar_points, self.R_lidar2cam.T) + self.T_lidar2cam
        valid = (p_cam[:, 2] > 0.5) & (p_cam[:, 2] < 100)
        p_cam = p_cam[valid]

        p_2d = np.dot(p_cam, self.K.T)
        p_2d[:, 0] /= p_2d[:, 2]
        p_2d[:, 1] /= p_2d[:, 2]

        u = p_2d[:, 0].astype(np.int32)
        v = p_2d[:, 1].astype(np.int32)
        d = p_cam[:, 2]

        h, w = cv_img.shape[:2]
        in_view = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u, v, d = u[in_view], v[in_view], d[in_view]

        # YOLO inference
        results = model.track(cv_img, persist=True, verbose=False,
                              tracker="bytetrack.yaml",
                              conf=0.45, classes=classes, device=0, imgsz=960)

        detected_objects = []
        fx = self.K[0, 0]
        cx = self.K[0, 2]

        if results[0].boxes:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                track_id = int(box.id[0]) if box.id is not None else -1

                # Angle
                box_cx = (x1 + x2) / 2
                angle_deg = np.degrees(np.arctan((box_cx - cx) / fx))

                # Depth estimation
                roi_mask = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
                roi_depths = d[roi_mask]
                dist = np.min(roi_depths) if len(roi_depths) else 999

                obj = {
                    "type": "traffic" if detect_traffic else "vehicle",
                    "cls": cls_id,
                    "track_id": track_id,
                    "conf": conf,
                    "bbox_w": x2 - x1,
                    "bbox_h": y2 - y1,
                    "bbox_area": (x2 - x1) * (y2 - y1),
                    "depth": float(dist),
                    "angle": float(angle_deg)
                }
                detected_objects.append(obj)

                # ✔ Detection 단위 로그 저장
                self.write_detection_log(obj)

                # Visualization
                color = (255,100,0) if not detect_traffic else (0,255,0)
                cv2.rectangle(cv_img, (x1,y1),(x2,y2), color, 2)

        return detected_objects, cv_img

    # ----------------------------------------------------------------------
    # Fusion Loop (메인)
    # ----------------------------------------------------------------------
    def fusion_loop(self):
        loop_start = time.time()

        if self.latest_view is not None:
            try:
                cv_view = self.bridge.imgmsg_to_cv2(self.latest_view, "bgr8")
                cv2.imshow("Spectator View", cv_view)
            except:
                pass

        if self.latest_img is None or self.latest_lidar is None or self.K is None:
            return

        frame = self.bridge.imgmsg_to_cv2(self.latest_img, "bgr8")
        lidar_pts = pointcloud2_to_array(self.latest_lidar)

        # --- Detection ---
        cars, frame = self.process_detection(frame, self.model_vehicle, self.classes_vehicle, lidar_pts, "car", False)
        lights, frame = self.process_detection(frame, self.model_traffic, self.classes_traffic, lidar_pts, "light", True)

        # 프레임 성능 지표 계산
        v_count = len(cars)
        t_count = len(lights)
        min_vehicle_dist = min([c["depth"] for c in cars], default=999)
        min_light_dist = min([l["depth"] for l in lights], default=999)

        # FPS/Latency 계산
        now = time.time()
        latency_ms = (now - loop_start) * 1000
        fps = 1.0 / (now - self.prev_time)
        self.prev_time = now

        # ✔ Frame-level 성능 로그 저장
        self.write_perf_log(fps, latency_ms, v_count, t_count, min_vehicle_dist, min_light_dist)

        # Visualization
        cv2.imshow("Detection Result", frame)
        cv2.waitKey(1)

        msg_out = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        self.result_pub.publish(msg_out)


def main(args=None):
    rclpy.init(args=args)
    node = SensorFusionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
