#!/usr/bin/env python3
import math
from typing import List, Tuple, Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseArray, PoseStamped


class LocalPathAvoid(Node):
    def __init__(self):
        super().__init__("local_path_avoid")

        self.sub_gnss = self.create_subscription(
            NavSatFix, "/carla/hero/gnss", self.gnss_cb, 10)
        self.sub_obs = self.create_subscription(
            PoseArray, "/carla/obstacles_2d", self.obs_cb, 10)

        self.pub_local = self.create_publisher(
            Path, "/carla/path/local", 10)

        self.lat0 = None
        self.lon0 = None
        self.cos_lat0 = 1.0

        self.current_xy: Optional[Tuple[float, float]] = None
        self.global_xy: List[Tuple[float, float]] = []   # loaded from csv
        self.obstacles: List[Tuple[float, float]] = []   # in vehicle frame (x,y)

        self.L = 20.0
        self.ds = 0.5
        self.safe_lat = 2.0      # 차량 y기준 좌우 2m 이내면 장애물
        self.max_offset = 3.0    # 최대 회피 lateral offset

        self.prev_idx: Optional[int] = None  # ★ 전역 경로 최근접 인덱스 캐시

        self.load_global_path("/home/itec/carla/PythonAPI/examples/ros2/mobility_project2/path/global_path_1.csv")  # 네가 쓰는 파일명으로 맞춰라
        if not self.global_xy:
            self.get_logger().warn("global path is empty or not found.")

        self.timer = self.create_timer(0.1, self.timer_cb)
    def load_global_path(self, filename: str): #csv파일을 통해 global_xy 상대거리값[m] 획득
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
            self.get_logger().info(f"Loaded {len(self.global_xy)} points from {filename}")
        except Exception as e:
            self.get_logger().error(f"Failed to load {filename}: {e}")

    def gnss_cb(self, msg: NavSatFix): 
        # gnns값을 받고, 위경도 근사값을 이용하여 차량 첫스폰 위치를 기준으로 얼마나 
        # 떨어져있는지 [m]단위로 반환
        lat = msg.latitude
        lon = msg.longitude
        if self.lat0 is None:
            self.lat0 = lat
            self.lon0 = lon
            self.cos_lat0 = math.cos(math.radians(lat))
            self.current_xy = (0.0, 0.0)
        else:
            self.current_xy = self.latlon_to_xy(lat, lon)

    def latlon_to_xy(self, lat: float, lon: float): # 3. 근사
        dx = (lon - self.lon0) * (111320.0 * self.cos_lat0)
        dy = (lat - self.lat0) * 110540.0
        return dx, dy

    def obs_cb(self, msg: PoseArray):
        self.obstacles = [(p.position.x, p.position.y) for p in msg.poses]
    # self.currnet_xy에 [m]단위 상대위치값
    # self.obstacles에 차량 기준 상대좌표 위치값
    def timer_cb(self):
        if self.current_xy is None or len(self.global_xy) < 2:
            return

        x, y = self.current_xy

        # ① 전역 경로에서 현재 위치와 가장 가까운 인덱스
        idx = self.find_nearest_index(x, y, self.global_xy)
        if idx is None:
            return

        # ② 장애물 기준 회피 방향 (연속적인 side 값: -1.0 ~ +1.0)
        side = self.decide_side(self.obstacles)

        # ③ path 메시지 준비 (map 좌표계 기준)
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = "map"

        # ④ local path 생성
        s = 0.0
        prev_px, prev_py = x, y
        i = idx

        n = len(self.global_xy)

        while i < n and s <= self.L:
            gx, gy = self.global_xy[i]

            # 현재 점까지의 누적 거리
            seg = math.hypot(gx - prev_px, gy - prev_py)
            s += seg
            prev_px, prev_py = gx, gy

            px, py = gx, gy

            if side != 0.0:
                # ---- ④-1. Bezier 기반 offset 프로파일 ----
                # t: 0 ~ 1, 3t(1-t) : 가운데에서 부드럽게 최대
                t = min(max(s / self.L, 0.0), 1.0)
                bezier = 3.0 * t * (1.0 - t)  # 0~0.75 범위
                offset = self.max_offset * bezier * side  # side까지 포함 (연속값)

                # ---- ④-2. 각 점에서의 local yaw ----
                if i + 1 < n:
                    gx2, gy2 = self.global_xy[i + 1]
                else:
                    gx2, gy2 = gx, gy

                yaw_i = math.atan2(gy2 - gy, gx2 - gx)

                # tangent 기준 법선벡터(nx, ny) : 왼쪽(+y)이 +방향
                nx = -math.sin(yaw_i)
                ny =  math.cos(yaw_i)

                px += offset * nx
                py += offset * ny

            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = float(px)
            ps.pose.position.y = float(py)
            ps.pose.position.z = 0.0
            path.poses.append(ps)

            i += 1

        self.pub_local.publish(path)
    def _compute_local_yaw(self, pts):
        yaws = []
        for i in range(len(pts)-1):
            x1, y1 = pts[i]
            x2, y2 = pts[i+1]
            yaws.append(math.atan2(y2 - y1, x2 - x1))
        yaws.append(yaws[-1])
        return yaws

    def find_nearest_index(self, x: float, y: float, pts: List[Tuple[float, float]]):
        if not pts:
            return None

        n = len(pts)

        # ★ 이전 인덱스 주변만 먼저 검색해서 연산량 감소
        if self.prev_idx is not None:
            radius = 50  # 전역 경로가 촘촘하면 50이면 충분
            start = max(0, self.prev_idx - radius)
            end = min(n, self.prev_idx + radius)
        else:
            start = 0
            end = n

        min_d = float("inf")
        idx = None

        for i in range(start, end):
            px, py = pts[i]
            d = (px - x) ** 2 + (py - y) ** 2
            if d < min_d:
                min_d = d
                idx = i

        # 혹시 초기 프레임 등에서 튀어버린 경우 전체 검색 한번 더
        if idx is None:
            for i, (px, py) in enumerate(pts):
                d = (px - x) ** 2 + (py - y) ** 2
                if d < min_d:
                    min_d = d
                    idx = i

        self.prev_idx = idx
        return idx

    def decide_side(self, obs_xy: List[Tuple[float, float]]) -> float:
        """
        장애물 여러 개를 고려해서 회피 방향 결정.
        반환값 side:
           > 0  → 경로 왼쪽으로 회피 (장애물이 오른쪽에 많다)
           < 0  → 경로 오른쪽으로 회피 (장애물이 왼쪽에 많다)
           = 0  → 회피 없음
        """
        if not obs_xy:
            return 0.0

        front = []
        for ox, oy in obs_xy:
            # 전방 L[m] 이내 & 좌우 safe_lat[m] 이내 장애물만 고려
            if 0.0 < ox < self.L and abs(oy) < self.safe_lat:
                front.append((ox, oy))

        if not front:
            return 0.0

        # 좌/우 쪽 "위험도" 계산 (가까울수록, 가운데에 있을수록 가중치 큼)
        left_cost = 0.0
        right_cost = 0.0

        for ox, oy in front:
            dist_forward = max(0.1, ox)
            lat_dist = max(0.1, abs(oy))
            w = 1.0 / (dist_forward * lat_dist)

            if oy > 0:      # 차량 기준 왼쪽 장애물
                left_cost += w
            elif oy < 0:    # 차량 기준 오른쪽 장애물
                right_cost += w

        # 위험도가 전혀 없으면 (전부 y≈0인 경우 등)
        if left_cost == 0.0 and right_cost == 0.0:
            # 가장 가까운 하나 기준으로 기존 방식 fallback
            front.sort(key=lambda p: p[0])  # x(전방거리) 기준
            _, y_local = front[0]
            return 1.0 if y_local < 0.0 else -1.0

        # 위험도 비율로 연속 side 계산
        # right_cost > left_cost → 장애물 오른쪽이 더 위험 → 왼쪽으로 회피 (side > 0)
        num = (right_cost - left_cost)
        den = (right_cost + left_cost)
        raw = num / den  # -1 ~ +1 근처
        side = max(-1.0, min(1.0, raw))

        return side

def main(args=None):
    rclpy.init(args=args)
    node = LocalPathAvoid()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
