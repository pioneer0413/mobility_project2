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
        
        print("=== [Fusion Node: ROI Extended (1x Height)] ===")
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
        
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        
        self.image_sub = self.create_subscription(
            Image, "/carla/hero/camera_front/image_color", self.image_callback, qos_profile)
        self.view_sub = self.create_subscription(
            Image, "/carla/hero/camera_view/image_color", self.view_callback, qos_profile)
        self.lidar_sub = self.create_subscription(
            PointCloud2, "/carla/hero/lidar", self.lidar_callback, qos_profile)

        self.latest_img = None
        self.latest_view = None
        self.latest_lidar = None
        self.memory_buffer = {} 

        self.R_lidar2cam = np.array([[ 0, -1,  0], [ 0,  0, -1], [ 1,  0,  0]])
        self.T_lidar2cam = np.array([0, -0.7, -1.6]) 
        self.K = None
        self.img_width = 800  
        self.img_height = 600
        self.timer = self.create_timer(0.05, self.fusion_loop)

    def image_callback(self, msg):
        self.latest_img = msg
        if self.img_width != msg.width or self.img_height != msg.height or self.K is None:
            self.img_width = msg.width
            self.img_height = msg.height
            f = self.img_width / 2.0
            cx = self.img_width / 2.0
            cy = self.img_height / 2.0
            self.K = np.array([[f, 0, cx], [0, f, cy], [0, 0,  1]])

    def view_callback(self, msg): self.latest_view = msg
    def lidar_callback(self, msg): self.latest_lidar = msg

    def fusion_loop(self):
        if self.latest_view is not None:
            try:
                cv_view = self.bridge.imgmsg_to_cv2(self.latest_view, "bgr8")
                cv2.imshow("Spectator View", cv_view)
                cv2.waitKey(1)
            except: pass

        if self.latest_img is None or self.latest_lidar is None or self.K is None: return

        try:
            cv_img = self.bridge.imgmsg_to_cv2(self.latest_img, "bgr8")
            lidar_points = pointcloud2_to_array(self.latest_lidar)
            current_time = time.time()

            p_cam = np.dot(lidar_points, self.R_lidar2cam.T) + self.T_lidar2cam
            valid_indices = (p_cam[:, 2] > 0.5) & (p_cam[:, 2] < 150.0)
            p_cam = p_cam[valid_indices]
            p_2d = np.dot(p_cam, self.K.T)
            p_2d[:, 2] = np.maximum(p_2d[:, 2], 0.001) 
            p_2d[:, 0] /= p_2d[:, 2]
            p_2d[:, 1] /= p_2d[:, 2]
            pixel_u = p_2d[:, 0].astype(np.int32)
            pixel_v = p_2d[:, 1].astype(np.int32)
            depths = p_cam[:, 2]

            in_img = (pixel_u >= 0) & (pixel_u < self.img_width) & \
                     (pixel_v >= 0) & (pixel_v < self.img_height)
            pixel_u = pixel_u[in_img]
            pixel_v = pixel_v[in_img]
            depths = depths[in_img]

            results_v = self.model_vehicle.track(
                cv_img, persist=True, verbose=False, tracker="bytetrack.yaml",
                conf=0.5, classes=self.classes_vehicle, device=0, imgsz=640)
            
            results_t = self.model_traffic.track(
                cv_img, persist=True, verbose=False, tracker="bytetrack.yaml",
                conf=0.5, classes=self.classes_traffic, device=0, imgsz=960)

            detection_groups = [(results_v[0], "v", "vehicle"), (results_t[0], "t", "traffic")]

            for result, prefix, model_type in detection_groups:
                if result.boxes is None: continue
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    raw_id = int(box.id[0]) if box.id is not None else -1
                    unique_track_key = f"{prefix}_{raw_id}"
                    label_name = result.names[cls_id]

                    if model_type == "traffic":
                        if cls_id == 4: traffic_status = "traffic_red"; color = (0, 0, 255)
                        elif cls_id == 5: traffic_status = "traffic_yellow"; color = (0, 255, 255)
                        else: traffic_status = "traffic_green"; color = (0, 255, 0)
                        
                        # [핵심 수정] 박스 높이만큼 아래로 더 검색 (100% 확장)
                        box_height = y2 - y1
                        search_y2 = min(self.img_height, y2 + box_height)
                    else:
                        traffic_status = "vehicle"; color = (255, 165, 0)
                        search_y2 = min(self.img_height, y2) # 차량은 확장 안 함

                    search_x1, search_x2 = max(0, x1), min(self.img_width, x2)
                    search_y1 = max(0, y1)

                    in_box_mask = (pixel_u >= search_x1) & (pixel_u <= search_x2) & \
                                  (pixel_v >= search_y1) & (pixel_v <= search_y2)
                    roi_depths = depths[in_box_mask]

                    final_dist_str = ""
                    if len(roi_depths) > 0:
                        valid_depths = roi_depths[roi_depths > 2.0]
                        if len(valid_depths) == 0: valid_depths = roi_depths
                        
                        raw_dist = np.min(valid_depths)

                        if unique_track_key in self.memory_buffer:
                            prev_dist = self.memory_buffer[unique_track_key]["dist"]
                            current_dist = prev_dist * 0.4 + raw_dist * 0.6
                        else:
                            current_dist = raw_dist

                        self.memory_buffer[unique_track_key] = {
                            "dist": current_dist, "time": current_time, "type": traffic_status 
                        }
                        final_dist_str = f"{current_dist:.1f}m"
                        
                        # [시각화]
                        cv2.rectangle(cv_img, (x1, y1), (x2, y2), color, 3)
                        
                        # [디버깅] 확장된 검색 영역 표시 (얇은 선)
                        if model_type == "traffic":
                            cv2.rectangle(cv_img, (x1, y2), (x2, search_y2), color, 1) 
                    else:
                        if unique_track_key in self.memory_buffer:
                            last_data = self.memory_buffer[unique_track_key]
                            if current_time - last_data["time"] < 1.0:
                                final_dist_str = f"({last_data['dist']:.1f}m)"
                        cv2.rectangle(cv_img, (x1, y1), (x2, y2), color, 1)

                    cv2.putText(cv_img, f"{label_name} {final_dist_str}", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            min_dist = 999.0
            detected_type = "none"
            
            for key, data in self.memory_buffer.items():
                if current_time - data["time"] < 0.5:
                    dist = data["dist"]
                    obj_type = data.get("type", "unknown")
                    if dist < min_dist:
                        min_dist = dist
                        detected_type = obj_type

            if min_dist == 999.0: min_dist = -1.0
            msg = String()
            msg.data = json.dumps({"obj": detected_type, "dist": min_dist})
            self.decision_pub.publish(msg)

            out_msg = self.bridge.cv2_to_imgmsg(cv_img, encoding="bgr8")
            self.result_pub.publish(out_msg)
            cv2.imshow("Fusion: Detection Result", cv_img)
            cv2.waitKey(1)

        except Exception as e: pass

def main(args=None):
    rclpy.init(args=args)
    node = SensorFusionNode()
    try: rclpy.spin(node)
    except: pass
    finally: node.destroy_node(); rclpy.shutdown(); cv2.destroyAllWindows()

if __name__ == '__main__': main()