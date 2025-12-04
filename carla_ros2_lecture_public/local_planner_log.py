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
from rclpy.time import Time


class KooLocalPlanner(Node):
    def __init__(self, path_num=1):
        super().__init__("koo_local_planner")

        self.path_num = path_num

        # ---------------------------------------------------------
        # ⭐ 성능지표 로그 파일 생성
        # ---------------------------------------------------------
        self.log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "planner_perf_log.csv")
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "path_success",
                    "curvature_avg",
                    "avoidance_active"
                ])

        # ---------------------------------------------------------
        # ROS 구독/발행 설정
        # ---------------------------------------------------------
        gnss_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.sub_gnss = self.create_subscription(
            NavSatFix, "/carla/hero/gnss", self.gnss_cb, gnss_qos_profile)

        self.sub_obs = self.create_subscription(
            PoseArray, "/carla/obstacles_2d", self.obs_cb, 10)

        self.pub_local_path = self.create_publisher(
            Path, "/carla/path/local", 10)

        self.pub_local_path_viz = self.create_publisher(
            Path, "/carla/path/local_viz", 10)

        # ---------------------------------------------------------
        # 상태 변수 초기화
        # ---------------------------------------------------------
        self.lat0 = None
        self.lon0 = None
        self.cos_lat0 = 1.0

        self.current_xy = None
        self.prev_xy = None
        self.global_xy = []
        self.obstacles = []

        # 차량 yaw
        self.vehicle_yaw = 0.0
        self.yaw_update_threshold = 0.02

        # Path 생성 설정
        self.L = 20.0
        self.safe_lat = 4.5
        self.max_offset = 3.0
        self.prev_idx = 0

        # Debounce 로직 변수
        self.obs_start_time = None
        self.last_seen_time = None
        self.wait_time = 3.0
        self.reset_timeout = 1.0

        # 스무딩 변수
        self.current_offset_ratio = 0.0
        self.target_offset_ratio = 0.0
        self.alpha = 0.05

        self.load_global_path()

        # 주기 실행
        self.timer = self.create_timer(0.1, self.timer_cb)

        print(f"=== Koo Planner (Path: {self.path_num}) Started ===")

    # ----------------------------------------------------------------------
    # ⭐ 경로 곡률 계산 함수 (평균 curvature)
    # ----------------------------------------------------------------------
    def compute_curvature(self, pts: List[Tuple[float, float]]) -> float:
        if len(pts) < 3:
            return 0.0

        curvatures = []
        for i in range(1, len(pts) - 1):
            x1, y1 = pts[i - 1]
            x2, y2 = pts[i]
            x3, y3 = pts[i + 1]

            # 방향 벡터 계산
            v1 = (x2 - x1, y2 - y1)
            v2 = (x3 - x2, y3 - y2)

            ang1 = math.atan2(v1[1], v1[0])
            ang2 = math.atan2(v2[1], v2[0])

            dtheta = abs(ang2 - ang1)
            curvatures.append(dtheta)

        return sum(curvatures) / len(curvatures)

    # ----------------------------------------------------------------------
    # GNSS → (x, y), yaw 추정
    # ----------------------------------------------------------------------
    def gnss_cb(self, msg):
        lat = msg.latitude
        lon = msg.longitude

        if self.lat0 is None:
            self.lat0 = lat
            self.lon0 = lon
            self.cos_lat0 = math.cos(math.radians(lat))
            self.current_xy = (0.0, 0.0)
            self.prev_xy = (0.0, 0.0)
            return

        dx = (lon - self.lon0) * (111320.0 * self.cos_lat0)
        dy = (lat - self.lat0) * 110540.0
        new_xy = (dx, dy)

        if self.prev_xy is not None:
            delta_x = new_xy[0] - self.prev_xy[0]
            delta_y = new_xy[1] - self.prev_xy[1]
            dist_moved = math.hypot(delta_x, delta_y)

            if dist_moved > self.yaw_update_threshold:
                self.vehicle_yaw = math.atan2(delta_y, delta_x)
                self.prev_xy = new_xy

        self.current_xy = new_xy

    # ----------------------------------------------------------------------
    # 장애물 2D 클러스터 콜백
    # ----------------------------------------------------------------------
    def obs_cb(self, msg):
        self.obstacles = [(p.position.x, p.position.y) for p in msg.poses]

    # ----------------------------------------------------------------------
    # 타이머 콜백 (Planning 메인 루프)
    # ----------------------------------------------------------------------
    def timer_cb(self):
        if self.current_xy is None or len(self.global_xy) < 2:
            return

        x, y = self.current_xy
        idx = self.find_nearest_index(x, y)
        if idx is None:
            return

        # ⭐ 회피 방향 결정
        target_dir, debug_info = self.decide_target_direction(self.obstacles)

        now = self.get_clock().now()
        is_obstacle_present = (target_dir != 0.0)

        if is_obstacle_present:
            self.last_seen_time = now
            if self.obs_start_time is None:
                self.obs_start_time = now
        else:
            if self.last_seen_time is not None:
                elapsed = (now - self.last_seen_time).nanoseconds / 1e9
                if elapsed >= self.reset_timeout:
                    self.obs_start_time = None
                    self.last_seen_time = None
                    self.target_offset_ratio = 0.0

        # 대기/회피 상태
        if self.obs_start_time is not None:
            elapsed = (now - self.obs_start_time).nanoseconds / 1e9
            if elapsed < self.wait_time:
                self.target_offset_ratio = 0.0
            else:
                if is_obstacle_present:
                    self.target_offset_ratio = target_dir

        # 스무딩 적용
        self.current_offset_ratio += self.alpha * (self.target_offset_ratio - self.current_offset_ratio)

        # ----------------------------------------------------------------------
        # ⭐ Local Path 생성
        # ----------------------------------------------------------------------
        path_pts = []            # 곡률 계산용
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = now.to_msg()

        path_viz = Path()
        path_viz.header.frame_id = "hero"
        path_viz.header.stamp = now.to_msg()

        s = 0.0
        prev_px, prev_py = self.global_xy[idx]
        curr_idx = idx
        cos_yaw = math.cos(-self.vehicle_yaw)
        sin_yaw = math.sin(-self.vehicle_yaw)

        while curr_idx < len(self.global_xy) and s <= self.L:
            gx, gy = self.global_xy[curr_idx]
            s += math.hypot(gx - prev_px, gy - prev_py)

            px, py = gx, gy

            # 회피 offset 적용
            if abs(self.current_offset_ratio) > 0.01:
                t = min(max(s / self.L, 0.0), 1.0)
                bezier = 3.0 * t * (1 - t)
                offset = self.max_offset * bezier * self.current_offset_ratio

                if curr_idx + 1 < len(self.global_xy):
                    nx, ny = self.global_xy[curr_idx + 1]
                else:
                    nx, ny = gx, gy

                yaw_path = math.atan2(ny - gy, nx - gx)

                px += offset * -math.sin(yaw_path)
                py += offset * math.cos(yaw_path)

            path_pts.append((px, py))

            # Map frame path
            ps = PoseStamped()
            ps.pose.position.x = px
            ps.pose.position.y = py
            path_msg.poses.append(ps)

            # Hero frame path
            dx_map = px - x
            dy_map = py - y
            px_local = dx_map * cos_yaw - dy_map * sin_yaw
            py_local = dx_map * sin_yaw + dy_map * cos_yaw

            ps2 = PoseStamped()
            ps2.pose.position.x = px_local
            ps2.pose.position.y = -py_local
            path_viz.poses.append(ps2)

            prev_px, prev_py = gx, gy
            curr_idx += 1

        self.pub_local_path.publish(path_msg)
        self.pub_local_path_viz.publish(path_viz)

        # ----------------------------------------------------------------------
        # ⭐ 성능지표 계산
        # ----------------------------------------------------------------------
        # 1) Path Success
        path_success = 1 if len(path_pts) >= 3 else 0

        # 2) Curvature
        curvature_avg = self.compute_curvature(path_pts)

        # 3) Avoidance Active 여부
        avoidance_active = 1 if abs(self.current_offset_ratio) > 0.01 else 0

        # ----------------------------------------------------------------------
        # ⭐ CSV 로그 기록
        # ----------------------------------------------------------------------
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                time.time(),
                path_success,
                curvature_avg,
                avoidance_active
            ])

    # ----------------------------------------------------------------------
    def find_nearest_index(self, x, y):
        start = max(0, self.prev_idx - 50)
        end   = min(len(self.global_xy), self.prev_idx + 50)
        min_d = float('inf')
        idx   = -1

        for i in range(start, end):
            d = (self.global_xy[i][0] - x)**2 + (self.global_xy[i][1] - y)**2
            if d < min_d:
                min_d = d
                idx = i

        if idx != -1:
            self.prev_idx = idx
            return idx

        return None

    # ----------------------------------------------------------------------
    def decide_target_direction(self, obs_xy):
        if not obs_xy:
            return 0.0, "Clean"

        relevant = [o for o in obs_xy if 0.1 < o[0] < self.L and abs(o[1]) < self.safe_lat]
        if not relevant:
            return 0.0, "Clean"

        left = sum(1 for o in relevant if o[1] > 0)
        right = sum(1 for o in relevant if o[1] < 0)

        if any(o[1] > 0 and abs(o[1]) < 1.0 for o in relevant):
            left += 10
        if any(o[1] < 0 and abs(o[1]) < 1.0 for o in relevant):
            right += 10

        if right > left:
            return 1.0, "Go left"
        else:
            return -1.0, "Go right"


def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_num', type=int, default=1)
    ros_args, _ = parser.parse_known_args()

    node = KooLocalPlanner(path_num=ros_args.path_num)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
