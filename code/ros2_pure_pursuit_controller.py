#!/usr/bin/env python3
import math
from typing import List, Tuple, Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import Path
from geometry_msgs.msg import Twist

import argparse


class PurePursuitFromGNSS(Node):
    def __init__(self, args):
        super().__init__("pure_pursuit_controller")

        # 구독자
        self.sub_gnss = self.create_subscription(
            NavSatFix, "/carla/hero/gnss", self.gnss_cb, 10)
        self.sub_path = self.create_subscription(
            Path, "/carla/path/local", self.path_cb, 10)

        # 발행자
        self.pub_cmd = self.create_publisher(
            Twist, "/carla/hero/cmd_vel", 10)

        # 좌표 변환 변수
        self.lat0 = None
        self.lon0 = None
        self.cos_lat0 = 1.0
        # 상태 변수
        self.curr_xy: Optional[Tuple[float, float]] = None
        self.prev_xy: Optional[Tuple[float, float]] = None
        self.local_xy: List[Tuple[float, float]] = []
        
        # ⭐ Global Path 변수 (Local Planner 방식)
        self.global_xy: List[Tuple[float, float]] = []
        self.goal_xy: Optional[Tuple[float, float]] = None

        # 제어 파라미터
        self.lookahead = args.lookahead
        self.wheel_base = args.wheel_base
        self.target_speed = args.target_speed  # m/s
        
        # ⭐ 목적지 도달 파라미터
        self.goal_threshold = args.goal_threshold   # 정지 거리 (m)
        self.slow_zone = args.slow_zone          # 감속 시작 거리 (m)
        self.min_speed = args.min_speed           # 최소 속도 (m/s)
        self.is_goal_reached = False

        # ⭐ Global Path 로드 (Local Planner와 동일)
        self._load_global_path(f"../path/global_path_{args.path_num}.csv")
        if not self.global_xy:
            self.get_logger().warn("global_path.csv is empty or not found.")
        elif len(self.global_xy) > 0:
            # 마지막 점을 목적지로 설정
            self.goal_xy = self.global_xy[-1]
            self.get_logger().info(
                f"📍 Goal loaded from CSV: ({self.goal_xy[0]:.1f}, {self.goal_xy[1]:.1f}), "
                f"Path length: {len(self.global_xy)} points"
            )

        self.timer = self.create_timer(0.05, self.control_loop)

    # ⭐ Global Path 로드 (Local Planner에서 복사)
    def _load_global_path(self, filename: str):
        """Load global path from csv file: x,y per line."""
        try:
            with open(filename, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split(",")
                    if len(parts) < 2:
                        continue
                    x = float(parts[0])
                    y = float(parts[1])
                    self.global_xy.append((x, y))
            self.get_logger().info(
                f"✅ Loaded {len(self.global_xy)} points from {filename}"
            )
        except Exception as e:
            self.get_logger().error(f"❌ Failed to load {filename}: {e}")

    def gnss_cb(self, msg: NavSatFix):
        lat = msg.latitude
        lon = msg.longitude
        if self.lat0 is None:
            self.lat0 = lat
            self.lon0 = lon
            self.cos_lat0 = math.cos(math.radians(lat))
            xy = (0.0, 0.0)
        else:
            xy = self.latlon_to_xy(lat, lon)

        self.prev_xy = self.curr_xy
        self.curr_xy = xy

    def latlon_to_xy(self, lat: float, lon: float):
        dx = (lon - self.lon0) * (111320.0 * self.cos_lat0)
        dy = (lat - self.lat0) * 110540.0
        return dx, dy

    def path_cb(self, msg: Path):
        self.local_xy = [
            (p.pose.position.x, p.pose.position.y) for p in msg.poses
        ]

    # ⭐ 속도 계산 함수
    def _calculate_speed(self, dist_to_goal: float) -> float:
        """목적지까지 거리에 따른 속도 계산"""
        if dist_to_goal <= self.goal_threshold:
            return 0.0  # 정지
        
        elif dist_to_goal <= self.slow_zone:
            # 선형 감속
            # slow_zone(15m) → target_speed(5.0 m/s)
            # goal_threshold(2m) → min_speed(1.0 m/s)
            ratio = (dist_to_goal - self.goal_threshold) / (self.slow_zone - self.goal_threshold)
            speed = self.min_speed + (self.target_speed - self.min_speed) * ratio
            return max(self.min_speed, speed)
        
        else:
            return self.target_speed  # 정상 속도

    def control_loop(self):
        # 데이터 검증
        if self.curr_xy is None or self.prev_xy is None or len(self.local_xy) < 2:
            return

        # ⭐ 목적지 거리 계산 및 속도 결정
        current_speed = self.target_speed
        
        if self.goal_xy is not None and self.curr_xy is not None:
            # 목적지까지 거리
            dist_to_goal = math.hypot(
                self.goal_xy[0] - self.curr_xy[0],
                self.goal_xy[1] - self.curr_xy[1]
            )
            
            # 거리 기반 속도 계산
            current_speed = self._calculate_speed(dist_to_goal)
            
            # ⭐ 목적지 도달 시 정지
            if current_speed == 0.0:
                if not self.is_goal_reached:
                    self.get_logger().info(
                        f"🎯 Goal reached! Distance: {dist_to_goal:.2f}m"
                    )
                    self.is_goal_reached = True
                
                # 정지 명령
                cmd = Twist()
                cmd.linear.x = 0.0
                cmd.linear.y = 1.0  # 브레이크
                cmd.angular.z = 0.0
                self.pub_cmd.publish(cmd)
                return
            
            # ⭐ 감속 중 로그 (2m마다)
            elif current_speed < self.target_speed:
                if int(dist_to_goal * 10) % 20 == 0:  # 2m마다
                    self.get_logger().info(
                        f"🐌 Approaching goal: {dist_to_goal:.1f}m, "
                        f"speed={current_speed:.1f}m/s"
                    )

        # ⭐ Pure Pursuit 계산
        x, y = self.curr_xy
        px, py = self.prev_xy

        yaw = math.atan2(y - py, x - px)

        cos_y = math.cos(-yaw)
        sin_y = math.sin(-yaw)

        pts_local = []
        for gx, gy in self.local_xy:
            dx = gx - x
            dy = gy - y
            x_l = dx * cos_y - dy * sin_y
            y_l = dx * sin_y + dy * cos_y
            pts_local.append((x_l, y_l))

        target = None
        for x_l, y_l in pts_local:
            d = math.hypot(x_l, y_l)
            if x_l > 0.0 and d >= self.lookahead:
                target = (x_l, y_l)
                break
        
        if target is None:
            # ⭐ 목표 없으면 정지 (경로 끝 근처)
            self.get_logger().warn("⚠️ No lookahead target, stopping")
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.linear.y = 1.0
            cmd.angular.z = 0.0
            self.pub_cmd.publish(cmd)
            return

        xt, yt = target
        ld = math.hypot(xt, yt)
        alpha = math.atan2(yt, xt)
        delta = math.atan2(2.0 * self.wheel_base * math.sin(alpha), ld)
        steer_deg = math.degrees(delta)

        # ⭐ 제어 명령 발행 (계산된 속도 사용)
        cmd = Twist()
        cmd.linear.x = float(current_speed)  # 가변 속도
        cmd.linear.y = 0.0
        cmd.angular.z = float(steer_deg)
        self.pub_cmd.publish(cmd)


def main(args=None, cargs=None):
    rclpy.init(args=args)
    node = PurePursuitFromGNSS(args=cargs)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_num', type=int, default=1, choices=[1,2,3,4], help='Global path number to use (e.g., 1 or 2)')
    parser.add_argument('--lookahead', type=float, default=5.0, help='Lookahead distance in meters')
    parser.add_argument('--wheel_base', type=float, default=2.7, help='Wheel base of the vehicle in meters')
    parser.add_argument('--target_speed', type=float, default=5.0, help='Target speed in m/s')
    parser.add_argument('--goal_threshold', type=float, default=2.0, help='Distance to goal to consider as reached (m)')
    parser.add_argument('--slow_zone', type=float, default=15.0, help='Distance to start slowing down (m)')
    parser.add_argument('--min_speed', type=float, default=1.0, help='Minimum speed when approaching goal (m/s)')
    args = parser.parse_args()
    main(cargs=args)
