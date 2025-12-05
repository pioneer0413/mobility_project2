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
from std_msgs.msg import String
from sensor_msgs.msg import Image, PointCloud2
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
        
        print("=== [Fusion with Angle Info] ===")
        print(">> Camera: FOV 110, 1280x720")
        
        cwd = os.getcwd()
        path_m = os.path.join(cwd, "m.pt")         
        path_original = os.path.join(cwd, "original.pt") 

        try:
            self.model_vehicle = YOLO(path_m, task='detect')
            self.classes_vehicle = [0, 1, 2, 3, 4, 5, 6]
            self.model_traffic = YOLO(path_original, task='detect')
            self.classes_traffic = [3, 4, 5] 
            print(">> 모델 로딩 완료")
        except Exception as e:
            print(f"[오류] {e}")
            sys.exit(1)

        self.bridge = CvBridge()
        self.decision_pub = self.create_publisher(String, "/fusion/decision", 10)
        self.result_pub = self.create_publisher(Image, "/fusion/result", 10)
        
        qos_profile = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        
        self.sub_img = self.create_subscription(
            Image, "/carla/hero/camera_front/image_color", self.image_callback, qos_profile)
        self.sub_lidar = self.create_subscription(
            PointCloud2, "/carla/hero/lidar/point_cloud", self.lidar_callback, qos_profile)

        self.latest_img = None
        self.latest_lidar = None
        self.memory_buffer = {} 

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

    def lidar_callback(self, msg):
        self.latest_lidar = msg

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
        
        # 카메라 파라미터 (각도 계산용)
        fx = self.K[0, 0]
        cx = self.K[0, 2]

        if results[0].boxes:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                track_id = int(box.id[0]) if box.id is not None else -1
                
                # [NEW] 중심 각도(Angle) 계산
                # 화면 중앙(cx) 기준, 객체 중심(box_cx)이 얼마나 떨어져 있나?
                box_cx = (x1 + x2) / 2.0
                angle_deg = np.degrees(np.arctan((box_cx - cx) / fx))
                
                # ROI 거리 측정
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
                    "angle": angle_deg,  # 각도 정보 저장
                    "id": f"{prefix}_{track_id}"
                }
                detected_objects.append(obj_info)

                # 시각화
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
                    # 거리와 각도 함께 표시
                    info = f"{dist:.1f}m {angle_deg:.0f}dg"
                    cv2.putText(cv_img, info, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return detected_objects, cv_img

    def fusion_loop(self):
        if self.latest_img is None or self.latest_lidar is None or self.K is None: return

        try:
            frame = self.bridge.imgmsg_to_cv2(self.latest_img, "bgr8")
            lidar_pts = pointcloud2_to_array(self.latest_lidar)
            current_time = time.time()
            all_objects = []

            # 1. 탐지 실행 (전체 화면)
            objs_light, frame = self.process_detection(
                frame, self.model_traffic, self.classes_traffic, 
                lidar_pts, "light", detect_traffic=True)
            all_objects.extend(objs_light)

            objs_car, frame = self.process_detection(
                frame, self.model_vehicle, self.classes_vehicle, 
                lidar_pts, "car", detect_traffic=False)
            all_objects.extend(objs_car)

            # 2. 판단 로직
            min_light_dist = 999.0
            closest_light = "none"
            
            min_vehicle_dist = 999.0
            closest_vehicle_angle = 0.0  # 가장 가까운 차의 각도

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
                    # 가장 가까운 차량 선택
                    if d < min_vehicle_dist:
                        min_vehicle_dist = d
                        closest_vehicle_angle = obj["angle"]

            # 3. 발행 (각도 정보 추가됨)
            msg_data = {
                "light": closest_light,
                "light_dist": min_light_dist if min_light_dist < 999 else -1.0,
                "vehicle_dist": min_vehicle_dist if min_vehicle_dist < 999 else -1.0,
                "vehicle_angle": float(closest_vehicle_angle)  # [NEW] 각도 정보
            }
            self.decision_pub.publish(String(data=json.dumps(msg_data)))

            # 4. 결과 화면
            info_txt = f"L:{closest_light} | Car:{msg_data['vehicle_dist']:.1f}m({closest_vehicle_angle:.0f}dg)"
            cv2.putText(frame, info_txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            out_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            self.result_pub.publish(out_msg)

        except Exception as e:
            pass

def main(args=None):
    rclpy.init(args=args)
    node = SensorFusionNode()
    try: rclpy.spin(node)
    except: pass
    finally: node.destroy_node(); rclpy.shutdown(); cv2.destroyAllWindows()

if __name__ == '__main__': main()