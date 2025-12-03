#!/usr/bin/env python3
import math
from typing import List, Tuple, Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseArray, PoseStamped

import argparse


class LocalPathAvoid(Node):
    def __init__(self, 
                 path_num: int=1, 
                 L: float=20.0, 
                 ds: float=0.5, 
                 safe_lat: float=1.0, 
                 max_offset: float=3.5,
                 num_lattices: int=7,
                 lateral_offsets: List[float]=[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
                 planner: str='lattice'):
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

        self.L = L
        self.ds = ds
        self.safe_lat = safe_lat      # 차량 y기준 좌우 2m 이내면 장애물
        self.max_offset = max_offset    # 최대 회피 lateral offset

        self.prev_idx: Optional[int] = None  # ★ 전역 경로 최근접 인덱스 캐시

        # Lattice 파라미터
        self.num_lattices = num_lattices  # 후보 경로 개수
        self.lateral_offsets = lateral_offsets  # offset 후보

        self.planner = planner # 플래너 종류
    
        self._load_global_path(f"../path/global_path_{path_num}.csv")  # 네가 쓰는 파일명으로 맞춰라
        if not self.global_xy:
            self.get_logger().warn("global path is empty or not found.")

        self.timer = self.create_timer(0.1, self.timer_cb)

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

        if self.planner == 'lattice':
            path = self._lattice_planner(x, y, 
                                       self._find_nearest_index(x, y, self.global_xy), 
                                       self.obstacles)
        elif self.planner == 'heuristic':
            # ① 전역 경로에서 현재 위치와 가장 가까운 인덱스
            idx = self._find_nearest_index(x, y, self.global_xy)
            if idx is None:
                return

            # ② 장애물 기준 회피 방향 (연속적인 side 값: -1.0 ~ +1.0)
            side = self._decide_side(self.obstacles)

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
    
    def _lattice_planner(self, x, y, idx, obs_xy):
        """여러 후보 경로 생성 및 최적 경로 선택"""
        candidates = []
        
        # 1. 각 offset으로 후보 경로 생성
        for offset in self.lateral_offsets:
            path = self._generate_path_with_offset(x, y, idx, offset)
            cost = self._calculate_cost(path, obs_xy, offset)
            candidates.append((cost, path))
        
        # 2. Cost 최소 경로 선택
        candidates.sort(key=lambda c: c[0])
        best_cost, best_path = candidates[0]
        
        self.get_logger().info(f"Best path cost: {best_cost:.2f}, offset: {self.lateral_offsets[candidates.index((best_cost, best_path))]}")
        
        return best_path
    
    def _generate_path_with_offset(self, x, y, idx, lateral_offset):
        """특정 offset으로 경로 생성"""
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = "map"
        
        s = 0.0
        prev_px, prev_py = x, y
        i = idx
        n = len(self.global_xy)
        
        while i < n and s <= self.L:
            gx, gy = self.global_xy[i]
            seg = math.hypot(gx - prev_px, gy - prev_py)
            s += seg
            prev_px, prev_py = gx, gy
            
            # Bezier 프로파일 적용
            t = min(max(s / self.L, 0.0), 1.0)
            bezier = 3.0 * t * (1.0 - t)
            offset = lateral_offset * bezier
            
            # Local yaw 계산
            if i + 1 < n:
                gx2, gy2 = self.global_xy[i + 1]
            else:
                gx2, gy2 = gx, gy
            yaw_i = math.atan2(gy2 - gy, gx2 - gx)
            
            # 법선 방향 offset
            nx = -math.sin(yaw_i)
            ny = math.cos(yaw_i)
            px = gx + offset * nx
            py = gy + offset * ny
            
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = float(px)
            ps.pose.position.y = float(py)
            path.poses.append(ps)
            
            i += 1
        
        return path
    
    def _calculate_cost(self, path, obs_xy, offset):
        """경로 Cost 계산"""
        cost = 0.0
        
        # 1. 장애물 회피 Cost
        obstacle_cost = self._obstacle_cost(path, obs_xy)
        
        # 2. 경로 이탈 Cost (offset이 클수록 불이익)
        deviation_cost = abs(offset) * 5.0
        
        # 3. 부드러움 Cost (급격한 변화 불이익)
        smoothness_cost = self._smoothness_cost(path)
        
        # 가중치 적용
        cost = (
            obstacle_cost * 10.0 +     # 장애물 회피 최우선
            deviation_cost * 2.0 +     # 경로 이탈 페널티
            smoothness_cost * 1.0      # 부드러움
        )
        
        return cost
    
    def _obstacle_cost(self, path, obs_xy):
        """장애물 근접 Cost"""
        if not obs_xy:
            return 0.0
        
        cost = 0.0
        for pose in path.poses:
            px, py = pose.pose.position.x, pose.pose.position.y
            
            for ox, oy in obs_xy:
                # 차량 좌표 → 맵 좌표 변환 필요 (간단히 근사)
                dist = math.hypot(px - ox, py - oy)
                
                if dist < 0.5:  # 충돌 위험
                    cost += 100.0
                elif dist < 2.0:  # 근접
                    cost += 50.0 / dist
                elif dist < 5.0:  # 경계
                    cost += 10.0 / dist
        
        return cost
    
    def _smoothness_cost(self, path):
        """경로 부드러움 Cost (곡률 기반)"""
        if len(path.poses) < 3:
            return 0.0
        
        cost = 0.0
        for i in range(1, len(path.poses) - 1):
            p0 = path.poses[i - 1].pose.position
            p1 = path.poses[i].pose.position
            p2 = path.poses[i + 1].pose.position
            
            # 각도 변화량 계산
            dx1 = p1.x - p0.x
            dy1 = p1.y - p0.y
            dx2 = p2.x - p1.x
            dy2 = p2.y - p1.y
            
            angle1 = math.atan2(dy1, dx1)
            angle2 = math.atan2(dy2, dx2)
            angle_diff = abs(angle2 - angle1)
            
            # 각도 변화가 클수록 Cost 증가
            cost += angle_diff ** 2
        
        return cost

    def _compute_local_yaw(self, pts):
        yaws = []
        for i in range(len(pts)-1):
            x1, y1 = pts[i]
            x2, y2 = pts[i+1]
            yaws.append(math.atan2(y2 - y1, x2 - x1))
        yaws.append(yaws[-1])
        return yaws

    def _find_nearest_index(self, x: float, y: float, pts: List[Tuple[float, float]]):
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

    def _decide_side(self, obs_xy: List[Tuple[float, float]]) -> float:
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

def main(args=None, cargs=None):
    rclpy.init(args=args)
    node = LocalPathAvoid(path_num=cargs.path_num, 
                          L=cargs.L, 
                          ds=cargs.ds, 
                          safe_lat=cargs.safe_lat, 
                          max_offset=cargs.max_offset, 
                          num_lattices=cargs.num_lattices, 
                          lateral_offsets=cargs.lateral_offsets,
                          planner=cargs.planner)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_num', type=int, default=1, choices=[1,2,3,4], help='Global path number to use (e.g., 1 or 2)')
    parser.add_argument('--L', type=float, default=20.0, help='Lookahead distance in meters')
    parser.add_argument('--ds', type=float, default=0.5, help='Distance step in meters')
    parser.add_argument('--safe_lat', type=float, default=1.0, help='Safe lateral distance in meters')
    parser.add_argument('--max_offset', type=float, default=3.5, help='Maximum lateral offset in meters')
    parser.add_argument('--num_lattices', type=int, default=7, help='Number of lattice paths')
    parser.add_argument('--lateral_offsets', type=float, nargs='+', default=[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], help='Lateral offsets for lattice paths')
    parser.add_argument('--planner', type=str, default='lattice', choices=['lattice', 'heuristic'])
    args = parser.parse_args()
    main(cargs=args)