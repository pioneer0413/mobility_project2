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
from std_msgs.msg import String
from sensor_msgs.msg import Image, PointCloud2, Imu
from cv_bridge import CvBridge
from ultralytics import YOLO

def pointcloud2_to_array(cloud_msg):
    dtype_list = [('x', np.float32), ('y', np.float32), ('z', np.float32)]
    cloud_arr = np.frombuffer(cloud_msg.data, dtype=np.float32)
    num_points = cloud_msg.width * cloud_msg.height
    arr = cloud_arr.reshape(num_points, -1)
    return arr[:, :3]

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_node')
        
        print("=== [Fusion with Display] ===")
        print(">> Camera: FOV 110 (Detection) + FOV 90 (View)")
        
        cwd = os.getcwd()
        # 모델 경로 설정 (경로가 다를 경우 수정 필요)
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
        
        # [구독 1] 전방 카메라 (인식용)
        self.sub_img = self.create_subscription(
            Image, "/carla/hero/camera_front/image_color", self.image_callback, qos_profile)
        
        # [구독 2] 3인칭 카메라 (화면 표시용) - 추가됨
        self.sub_view = self.create_subscription(
            Image, "/carla/hero/camera_view/image_color", self.view_callback, qos_profile)

        # [구독 3] 라이다
        self.sub_lidar = self.create_subscription(
            PointCloud2, "/carla/hero/lidar/point_cloud", self.lidar_callback, qos_profile)

        # IMU
        self.sub_imu = self.create_subscription(
            Imu, "/carla/hero/imu", self.imu_callback, 10)

        self.latest_img = None
        self.latest_view = None  # 3인칭 이미지 저장용
        self.latest_lidar = None
        self.memory_buffer = {} 

        # IMU 데이터 저장
        self.current_yaw = 0.0
        self.current_pitch = 0.0
        self.current_roll = 0.0
        self.yaw_rate = 0.0
        self.lateral_accel = 0.0
        self.forward_accel = 0.0

        # 좌표 변환 행렬
        self.R_lidar2cam = np.array([[ 0, -1,  0], [ 0,  0, -1], [ 1,  0,  0]])
        self.T_lidar2cam = np.array([0, -0.7, -1.6]) 
        
        self.K = None
        self.timer = self.create_timer(0.05, self.fusion_loop)

    def image_callback(self, msg):
        self.latest_img = msg
        if self.K is None:
            w, h = msg.width, msg.height
            fov = 110.0
            f = w / (2.0 * np.tan(np.deg2rad(fov / 2.0)))
            self.K = np.array([[f, 0, w/2.0], [0, f, h/2.0], [0, 0, 1]])

    def view_callback(self, msg):
        self.latest_view = msg

    def lidar_callback(self, msg):
        self.latest_lidar = msg

    def imu_callback(self, msg: Imu):
        """IMU 콜백: 차량 자세 및 관성 정보"""
        # Quaternion -> Euler 각도 변환
        x, y, z, w = msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
        
        # Roll
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        self.current_roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch
        sinp = 2 * (w * y - z * x)
        self.current_pitch = math.asin(max(-1.0, min(1.0, sinp)))
        
        # Yaw
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)
        
        # 각속도
        self.yaw_rate = msg.angular_velocity.z
        
        # 가속도
        self.forward_accel = msg.linear_acceleration.x
        self.lateral_accel = msg.linear_acceleration.y

    def process_detection(self, cv_img, model, classes, lidar_points, prefix, detect_traffic=False):
        if lidar_points is None or self.K is None: return [], cv_img

        # 1. LiDAR Projection
        p_cam = np.dot(lidar_points, self.R_lidar2cam.T) + self.T_lidar2cam
        valid = (p_cam[:, 2] > 0.5) & (p_cam[:, 2] < 100.0)
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

        # 2. YOLO Inference
        results = model.track(cv_img, persist=True, verbose=False, tracker="bytetrack.yaml",
                              conf=0.45, classes=classes, device=0, imgsz=960)
        
        detected_objects = []
        fx = self.K[0, 0]; cx = self.K[0, 2]

        if results[0].boxes:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                track_id = int(box.id[0]) if box.id is not None else -1
                
                # 중심 각도 계산
                box_cx = (x1 + x2) / 2.0
                angle_deg = np.degrees(np.arctan((box_cx - cx) / fx))
                
                # 거리 측정
                roi_mask = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
                roi_depths = d[roi_mask]
                
                dist = 999.0
                if len(roi_depths) > 0:
                    valid_d = roi_depths[roi_depths > 1.5]
                    if len(valid_d) > 0:
                        dist = np.min(valid_d)
                
                obj_info = {
                    "type": "traffic" if detect_traffic else "vehicle",
                    "cls": cls_id,
                    "dist": dist,
                    "angle": angle_deg,
                    "id": f"{prefix}_{track_id}"
                }
                detected_objects.append(obj_info)

                # 시각화 (enable_viz가 True일 때만)
                color = (0, 255, 0)
                label_txt = ""
                if detect_traffic:
                    if cls_id == 4: color = (0, 0, 255); label_txt="Red"
                    elif cls_id == 5: color = (0, 255, 255); label_txt="Yellow"
                    else: label_txt="Green"
                else:
                    color = (255, 100, 0); label_txt="Car"

                cv2.rectangle(cv_img, (x1, y1), (x2, y2), color, 2)
                if dist < 100:
                    info = f"{dist:.1f}m {angle_deg:.0f}dg"
                    cv2.putText(cv_img, info, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return detected_objects, cv_img

    def draw_imu_overlay(self, frame):
        """IMU 정보를 프레임에 오버레이"""
        h, w = frame.shape[:2]
        
        # 배경 패널 (반투명)
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, h-150), (400, h-10), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        y_offset = h - 130
        line_height = 25
        
        # IMU 데이터 표시
        texts = [
            f"Yaw: {math.degrees(self.current_yaw):>6.1f}deg",
            f"YawRate: {self.yaw_rate:>5.2f}r/s",
            f"Pitch: {math.degrees(self.current_pitch):>6.1f}deg",
            f"Roll: {math.degrees(self.current_roll):>6.1f}deg",
            f"Accel(F/L): {self.forward_accel:>4.1f}/{self.lateral_accel:>4.1f}m/s2"
        ]
        
        for i, text in enumerate(texts):
            color = (0, 255, 255)  # 기본 노란색
            
            # 경고 색상 변경
            if "YawRate" in text and abs(self.yaw_rate) > 0.8:
                color = (0, 0, 255)  # 빨간색
            elif "Accel" in text and (abs(self.forward_accel) > 5 or abs(self.lateral_accel) > 5):
                color = (0, 165, 255)  # 주황색
            
            cv2.putText(frame, text, (20, y_offset + i * line_height),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 차량 자세 시각화 (간단한 화살표)
        center_x, center_y = w - 100, h - 75
        arrow_len = 50
        
        # Yaw 방향 화살표
        end_x = int(center_x + arrow_len * math.cos(self.current_yaw - math.pi/2))
        end_y = int(center_y + arrow_len * math.sin(self.current_yaw - math.pi/2))
        cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), (0, 255, 0), 3, tipLength=0.3)
        cv2.circle(frame, (center_x, center_y), 5, (255, 255, 255), -1)
        
        return frame

    def fusion_loop(self):
        # 3인칭 화면 출력 (데이터가 들어오면 바로 표시)
        if self.latest_view is not None:
            try:
                cv_view = self.bridge.imgmsg_to_cv2(self.latest_view, "bgr8")
                cv2.imshow("Spectator View", cv_view)
            except Exception: pass

        # 메인 로직 수행 조건
        if self.latest_img is None or self.latest_lidar is None or self.K is None: 
            cv2.waitKey(1)
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(self.latest_img, "bgr8")
            lidar_pts = pointcloud2_to_array(self.latest_lidar)
            current_time = time.time()
            all_objects = []

            # 1. 탐지 실행
            objs_light, frame = self.process_detection(
                frame, self.model_traffic, self.classes_traffic, 
                lidar_pts, "light", detect_traffic=True)
            all_objects.extend(objs_light)

            objs_car, frame = self.process_detection(
                frame, self.model_vehicle, self.classes_vehicle, 
                lidar_pts, "car", detect_traffic=False)
            all_objects.extend(objs_car)

            frame = self.draw_imu_overlay(frame)

            # 2. 판단 로직
            min_light_dist = 999.0
            closest_light = "none"
            min_vehicle_dist = 999.0
            closest_vehicle_angle = 0.0

            for obj in all_objects:
                d = obj["dist"]
                if d >= 999.0: continue

                # 스무딩
                key = obj["id"]
                if key in self.memory_buffer:
                    d = self.memory_buffer[key]["dist"] * 0.4 + d * 0.6
                self.memory_buffer[key] = {"dist": d, "time": current_time}

                if obj["type"] == "traffic":
                    status = "traffic_green"
                    if obj["cls"] == 4: status = "traffic_red"
                    elif obj["cls"] == 5: status = "traffic_yellow"
                    
                    if d < min_light_dist:
                        min_light_dist = d
                        closest_light = status
                        
                elif obj["type"] == "vehicle":
                    if d < min_vehicle_dist:
                        min_vehicle_dist = d
                        closest_vehicle_angle = obj["angle"]

            # 3. 발행
            msg_data = {
                "light": closest_light,
                "light_dist": min_light_dist if min_light_dist < 999 else -1.0,
                "vehicle_dist": min_vehicle_dist if min_vehicle_dist < 999 else -1.0,
                "vehicle_angle": float(closest_vehicle_angle)
            }
            self.decision_pub.publish(String(data=json.dumps(msg_data)))

            info_txt = f"Li:{closest_light} | Car:{msg_data['vehicle_dist']:.1f}m({closest_vehicle_angle:.0f}dg)"
            cv2.putText(frame, info_txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # 4. 결과 화면 텍스트 & 출력 (시각화 모드일 때만)
            if self.enable_viz:    
                cv2.imshow("Detection Result", frame)
                cv2.waitKey(1)
            
            out_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            self.result_pub.publish(out_msg)

        except Exception as e:
            # print(f"Error: {e}")
            pass

def main(args=None):
    rclpy.init(args=args)
    node = SensorFusionNode()
    try: rclpy.spin(node)
    except: pass
    finally: node.destroy_node(); rclpy.shutdown(); cv2.destroyAllWindows()

if __name__ == '__main__': main()