#!/usr/bin/env python3
import math
import os
import csv
import argparse
from typing import List, Tuple, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import NavSatFix, Imu
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseArray, PoseStamped

class KooLocalPlanner(Node):
    def __init__(self, path_num=1):
        super().__init__("koo_local_planner")

        self.path_num = path_num

        # ---------------------------
        # ⭐ 성능 로그 파일 초기화
        # ---------------------------
        log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "planner_perf_log.csv")
        self.log_path = log_path
        if not os.path.exists(log_path):
            with open(log_path, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "path_success", "curvature_avg", "avoidance_active"])

        # ---------------------------
        # 기존 코드
        # ---------------------------
        gnss_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.sub_gnss = self.create_subscription(NavSatFix, "/carla/hero/gnss", self.gnss_cb, gnss_qos_profile)
        self.sub_obs = self.create_subscription(PoseArray, "/carla/obstacles_2d", self.obs_cb, 10)

        self.pub_local_path = self.create_publisher(Path, "/carla/path/local", 10)
        self.pub_local_path_viz = self.create_publisher(Path, "/carla/path/local_viz", 10)

        self.lat0 = None
        self.lon0 = None
        self.cos_lat0 = 1.0
        self.current_xy = None
        self.prev_xy = None
        self.global_xy = []
        self.obstacles = []

        self.vehicle_yaw = 0.0
        self.yaw_update_threshold = 0.02

        self.L = 20.0
        self.safe_lat = 4.5
        self.max_offset = 3.0
        self.prev_idx = 0

        self.current_offset_ratio = 0.0
        self.target_offset_ratio = 0.0
        self.alpha = 0.05

        self.load_global_path()
        self.timer = self.create_timer(0.1, self.timer_cb)

    # ---------------------------
    # ⭐ 성능 지표 계산 함수들
    # ---------------------------

    def compute_curvature(self, path_points: List[Tuple[float, float]]) -> float:
        """경로 곡률(curvature)의 평균값을 계산"""
        if len(path_points) < 3:
            return 0.0

        curvatures = []
        for i in range(1, len(path_points) - 1):
            x1, y1 = path_points[i - 1]
            x2, y2 = path_points[i]
            x3, y3 = path_points[i + 1]

            # 각도 변화 계산
            v1 = (x2 - x1, y2 - y1)
            v2 = (x3 - x2, y3 - y2)
            ang1 = math.atan2(v1[1], v1[0])
            ang2 = math.atan2(v2[1], v2[0])
            dtheta = abs(ang2 - ang1)

            # 곡률 = 방향 변화량
            curvatures.append(dtheta)

        return sum(curvatures) / len(curvatures)

    def log_performance(self, path_success, curvature_avg, avoidance_active):
        """CSV에 성능 지표 기록"""
        with open(self.log_path, "a") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.get_clock().now().nanoseconds / 1e9,
                int(path_success),
                float(curvature_avg),
                int(avoidance_active)
            ])

    # ---------------------------
    # 기존 GNSS 등 콜백 유지
    # ---------------------------
    def gnss_cb(self, msg):
        lat = msg.latitude
        lon = msg.longitude
        if self.lat0 is None:
            self.lat0 = msg.latitude
            self.lon0 = msg.longitude
            self.cos_lat0 = math.cos(math.radians(msg.latitude))
            self.current_xy = (0.0, 0.0)
            self.prev_xy = (0.0, 0.0)
        else:
            dx = (lon - self.lon0) * (111320.0 * self.cos_lat0)
            dy = (lat - self.lat0) * 110540.0
            new_xy = (dx, dy)

            if self.prev_xy:
                delta_x = new_xy[0] - self.prev_xy[0]
                delta_y = new_xy[1] - self.prev_xy[1]
                if math.hypot(delta_x, delta_y) > self.yaw_update_threshold:
                    self.vehicle_yaw = math.atan2(delta_y, delta_x)
                    self.prev_xy = new_xy

            self.current_xy = new_xy

    def obs_cb(self, msg):
        self.obstacles = [(p.position.x, p.position.y) for p in msg.poses]

    # ---------------------------
    # ⭐ 핵심: timer_cb 내부에서 지표 계산 및 로깅
    # ---------------------------
    def timer_cb(self):
        if self.current_xy is None or len(self.global_xy) < 2:
            return

        # 기존 planner 경로 생성
        x, y = self.current_xy
        idx = self.find_nearest_index(x, y)
        if idx is None:
            return

        # 경로 생성 성공 여부
        path_points = []
        path_success = True

        s = 0.0
        prev_x, prev_y = self.global_xy[idx]
        curr_idx = idx

        while curr_idx < len(self.global_xy) and s <= self.L:
            gx, gy = self.global_xy[curr_idx]
            s += math.hypot(gx - prev_x, gy - prev_y)

            # 경로 점 추가
            path_points.append((gx, gy))
            prev_x, prev_y = gx, gy
            curr_idx += 1

        if len(path_points) < 3:
            path_success = False

        # 회피 활성화 여부
        avoidance_active = int(abs(self.current_offset_ratio) > 0.01)

        # 곡률 계산
        curvature_avg = self.compute_curvature(path_points)

        # ⭐ 성능 로그 작성
        self.log_performance(path_success, curvature_avg, avoidance_active)

        # 기존 경로 publish 유지 (생략: 기존 코드를 그대로 둠)

    # ---------------------------
    # 기존 함수들 그대로 유지
    # ---------------------------
    def find_nearest_index(self, x, y):
        start = max(0, self.prev_idx - 50)
        end = min(len(self.global_xy), self.prev_idx + 50)
        min_d = float('inf')
        idx = -1
        for i in range(start, end):
            d = (self.global_xy[i][0] - x)**2 + (self.global_xy[i][1] - y)**2
            if d < min_d:
                min_d = d
                idx = i
        if idx != -1:
            self.prev_idx = idx
            return idx
        return None

def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_num", type=int, default=1)
    ros_args, _ = parser.parse_known_args()
    node = KooLocalPlanner(path_num=ros_args.path_num)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
