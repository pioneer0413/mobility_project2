#!/usr/bin/env python3
import math
import os
import csv
import argparse
from typing import List, Tuple, Optional
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseArray, PoseStamped

class KooLocalPlanner(Node):
    def __init__(self, path_num=1):
        super().__init__("koo_local_planner")
        
        self.path_num = path_num

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

        self.lat0 = None
        self.lon0 = None
        self.cos_lat0 = 1.0
        self.current_xy: Optional[Tuple[float, float]] = None
        self.prev_xy: Optional[Tuple[float, float]] = None

        # 경로 관련
        self.global_xy: List[Tuple[float, float]] = []   
        self.obstacles: List[Tuple[float, float]] = []   

        # ⭐ 추가: 차량 Yaw 관리
        self.vehicle_yaw = 0.0
        self.vehicle_speed = 0.0
        self.yaw_update_threshold = 0.02  # 2cm 이상 이동 시 업데이트
        
        # ⭐ 차량 중점 offset (GNSS가 차량 뒤쪽에 있다고 가정)
        self.vehicle_center_offset_x = 1.0  # 차량 앞쪽으로 1.5m (차량 길이의 절반)
        self.vehicle_center_offset_y = 0.0  # 좌우 중앙

        # ========== [개선 1] 동적 Look-ahead 거리 ==========
        self.L_base = 20.0           # 기본 look-ahead
        self.L_min = 10.0            # 최소
        self.L_max = 30.0            # 최대
        self.L_speed_gain = 1.5      # 속도에 따른 증가율

        # ========== [개선 2] 회피 파라미터 개선 ==========
        self.safe_lat = 5.0          # 탐지 범위
        self.max_offset = 2.5        # 최대 회피 거리 (3.0 -> 2.5로 감소)
        self.offset_speed_factor = 0.7  # 속도에 따른 회피 감소

        # ========== [개선 3] 다단계 스무딩 ==========
        self.current_offset_ratio = 0.0
        self.target_offset_ratio = 0.0
        self.alpha_fast = 0.15       # 빠른 반응 (장애물 발견 시)
        self.alpha_slow = 0.05       # 느린 반응 (복귀 시)

        # ========== [개선 4] 경로 평활화 ==========
        self.use_smoothing = True
        self.smoothing_sigma = 1.5   # Gaussian smoothing 파라미터

        # ========== [개선 5] 장애물 필터링 개선 ==========
        self.obstacle_history = []   # 최근 N프레임 장애물 저장
        self.history_size = 3
        self.min_obstacle_confidence = 2  # N회 이상 감지되어야 유효

        # 스마트 대기 로직 변수
        self.obs_start_time = None      
        self.last_seen_time = None      
        self.wait_time = 3.0            
        self.reset_timeout = 1.0 

        # 기타       
        self.prev_idx = 0

        self.load_global_path()
        self.timer = self.create_timer(0.1, self.timer_cb)
        
        print(f"=== Koo Planner (Path: {self.path_num}) Started ===")
        print(f">> Safe Range: {self.safe_lat}m | Smoothing Alpha Fast: {self.alpha_fast} | Smoothing Alpha Slow: {self.alpha_slow}")

    def load_global_path(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        filename = f"../path/global_path_{self.path_num}.csv"
        file_path = os.path.join(current_dir, filename)
        if not os.path.exists(file_path): return
        try:
            with open(file_path, "r") as f:
                for line in f:
                    parts = line.split(',')
                    if len(parts) < 2 or line.startswith("#"): continue
                    try: self.global_xy.append((float(parts[0]), float(parts[1])))
                    except: continue
            self.get_logger().info(f"Loaded {len(self.global_xy)} points")
        except: pass

    def gnss_cb(self, msg):
        lat = msg.latitude
        lon = msg.longitude
        if self.lat0 is None:
            self.lat0 = msg.latitude; self.lon0 = msg.longitude
            self.cos_lat0 = math.cos(math.radians(msg.latitude))
            self.current_xy = (0.0, 0.0)
            self.prev_xy = (0.0, 0.0)  # ⭐ 초기화
        else:
            dx = (lon - self.lon0) * (111320.0 * self.cos_lat0)
            dy = (lat - self.lat0) * 110540.0
            new_xy = (dx, dy)
            
            # ⭐ Yaw 계산: 이전 위치에서 현재 위치로의 방향
            if self.prev_xy is not None:
                delta_x = new_xy[0] - self.prev_xy[0]
                delta_y = new_xy[1] - self.prev_xy[1]
                distance_moved = math.hypot(delta_x, delta_y)
                
                # 충분히 이동했을 때만 yaw 업데이트 (노이즈 방지)
                if distance_moved > self.yaw_update_threshold:
                    self.vehicle_yaw = math.atan2(delta_y, delta_x)
                    # 속도 추정 (10Hz 콜백 가정)
                    self.vehicle_speed = distance_moved * 10.0  # m/s
                    self.prev_xy = new_xy
            
            self.current_xy = new_xy

    def obs_cb(self, msg): 
        current_obs = [(p.position.x, p.position.y) for p in msg.poses]
        
        # ========== [개선 5] 장애물 이력 관리 ==========
        self.obstacle_history.append(current_obs)
        if len(self.obstacle_history) > self.history_size:
            self.obstacle_history.pop(0)
        
        # 신뢰도 기반 필터링: N회 이상 감지된 장애물만 유효
        filtered_obs = []
        for obs in current_obs:
            count = sum(1 for hist in self.obstacle_history 
                       if any(math.hypot(obs[0]-o[0], obs[1]-o[1]) < 0.5 for o in hist))
            if count >= self.min_obstacle_confidence:
                filtered_obs.append(obs)
        
        self.obstacles = filtered_obs

    def timer_cb(self):
        if self.current_xy is None or len(self.global_xy) < 2: 
            return
        
        x, y = self.current_xy
        idx = self.find_nearest_index(x, y)
        if idx is None: 
            return

        # ========== [개선 1] 동적 Look-ahead 거리 계산 ==========
        L = self.L_base + self.vehicle_speed * self.L_speed_gain
        L = max(self.L_min, min(self.L_max, L))

        # 장애물 판단
        target_dir, debug_info = self.decide_target_direction(self.obstacles)
        
        now = self.get_clock().now()
        is_obstacle_present = (target_dir != 0.0)
        
        # Debouncing
        if is_obstacle_present:
            self.last_seen_time = now
            if self.obs_start_time is None:
                self.obs_start_time = now
        else:
            if self.last_seen_time is not None:
                elapsed_lost = (now - self.last_seen_time).nanoseconds / 1e9
                if elapsed_lost >= self.reset_timeout:
                    self.obs_start_time = None
                    self.last_seen_time = None
                    self.target_offset_ratio = 0.0

        # 대기 및 회피 로직
        if self.obs_start_time is not None:
            elapsed = (now - self.obs_start_time).nanoseconds / 1e9
            
            if elapsed < self.wait_time:
                self.target_offset_ratio = 0.0
            else:
                if is_obstacle_present:
                    self.target_offset_ratio = target_dir

        # ========== [개선 3] 적응형 스무딩 ==========
        # 회피 시작: 빠르게, 복귀 시: 천천히
        if abs(self.target_offset_ratio) > abs(self.current_offset_ratio):
            alpha = self.alpha_fast  # 회피 시작: 빠른 반응
        else:
            alpha = self.alpha_slow  # 복귀: 느린 반응
        
        self.current_offset_ratio += alpha * (self.target_offset_ratio - self.current_offset_ratio)

        # 경로 생성
        path, path_viz = self.generate_smooth_path(idx, L, x, y)
        
        self.pub_local_path.publish(path)
        self.pub_local_path_viz.publish(path_viz)

    def generate_smooth_path(self, idx, L, x, y):
        """개선된 경로 생성 함수"""
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = "map"

        path_viz = Path()
        path_viz.header.stamp = path.header.stamp
        path_viz.header.frame_id = "hero"

        # ========== 1. 원본 경로 샘플링 ==========
        s = 0.0
        prev_px, prev_py = self.global_xy[idx]
        curr_idx = idx
        path_len = len(self.global_xy)
        
        raw_points = []
        distances = []

        while curr_idx < path_len and s <= L:
            gx, gy = self.global_xy[curr_idx]
            seg_dist = math.hypot(gx - prev_px, gy - prev_py)
            s += seg_dist
            distances.append(s)
            
            # ========== [개선 2] 속도 기반 회피 거리 조정 ==========
            offset_scale = max(0.3, 1.0 - self.vehicle_speed * self.offset_speed_factor / 10.0)
            
            # 회피 적용
            px, py = gx, gy
            if abs(self.current_offset_ratio) > 0.01:
                t = min(max(s / L, 0.0), 1.0)
                # Cubic ease-in-out (Bezier보다 부드러움)
                if t < 0.5:
                    bezier = 4 * t * t * t
                else:
                    bezier = 1 - pow(-2 * t + 2, 3) / 2
                
                offset = self.max_offset * bezier * self.current_offset_ratio * offset_scale
                
                if curr_idx + 1 < path_len:
                    nx, ny = self.global_xy[curr_idx + 1]
                else:
                    nx, ny = gx, gy
                
                path_yaw = math.atan2(ny - gy, nx - gx)
                px += offset * -math.sin(path_yaw)
                py += offset * math.cos(path_yaw)
            
            raw_points.append((px, py))
            prev_px, prev_py = gx, gy
            curr_idx += 1

        if len(raw_points) < 3:
            # 포인트가 너무 적으면 원본 반환
            for px, py in raw_points:
                ps = PoseStamped()
                ps.header = path.header
                ps.pose.position.x = px
                ps.pose.position.y = py
                path.poses.append(ps)
            return path, path_viz

        # ========== [개선 4] Cubic Spline 보간 + Gaussian Smoothing ==========
        if self.use_smoothing and len(raw_points) >= 4:
            raw_array = np.array(raw_points)
            distances_array = np.array(distances)
            
            # Cubic Spline 보간 (더 촘촘하게)
            cs_x = CubicSpline(distances_array, raw_array[:, 0])
            cs_y = CubicSpline(distances_array, raw_array[:, 1])
            
            # 보간할 거리 배열 (0.2m 간격)
            s_interp = np.arange(0, distances_array[-1], 0.2)
            x_interp = cs_x(s_interp)
            y_interp = cs_y(s_interp)
            
            # Gaussian Smoothing
            x_smooth = gaussian_filter1d(x_interp, self.smoothing_sigma)
            y_smooth = gaussian_filter1d(y_interp, self.smoothing_sigma)
            
            smooth_points = list(zip(x_smooth, y_smooth))
        else:
            smooth_points = raw_points

        # ========== 2. 좌표 변환 ==========
        cos_yaw = math.cos(-self.vehicle_yaw)
        sin_yaw = math.sin(-self.vehicle_yaw)

        for px, py in smooth_points:
            # Map frame
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = px
            ps.pose.position.y = py
            ps.pose.position.z = 0.0
            path.poses.append(ps)

            # Hero frame
            dx_map = px - x
            dy_map = py - y
            px_local = dx_map * cos_yaw - dy_map * sin_yaw
            py_local = dx_map * sin_yaw + dy_map * cos_yaw
            px_centered = px_local - self.vehicle_center_offset_x
            py_centered = py_local - self.vehicle_center_offset_y
            
            ps_viz = PoseStamped()
            ps_viz.header = path_viz.header
            ps_viz.pose.position.x = px_centered
            ps_viz.pose.position.y = -py_centered
            ps_viz.pose.position.z = 0.0
            path_viz.poses.append(ps_viz)

        return path, path_viz

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
        return self.prev_idx

    def decide_target_direction(self, obs_xy) -> Tuple[float, str]:
        if not obs_xy: 
            return 0.0, "Clean"
        
        # 동적 Look-ahead 거리 사용
        L = self.L_base + self.vehicle_speed * self.L_speed_gain
        L = max(self.L_min, min(self.L_max, L))
        
        relevant = [o for o in obs_xy if 0.1 < o[0] < L and abs(o[1]) < self.safe_lat]
        
        if not relevant: 
            return 0.0, "Clean"

        # 거리 가중치 적용 (가까운 장애물에 더 높은 가중치)
        left_score = 0.0
        right_score = 0.0
        
        for ox, oy in relevant:
            weight = 1.0 / max(ox, 1.0)  # 거리 역수 (가까울수록 큼)
            
            if oy > 0:  # 왼쪽
                left_score += weight
                if abs(oy) < 1.5:  # 매우 가까움
                    left_score += 5.0 * weight
            else:  # 오른쪽
                right_score += weight
                if abs(oy) < 1.5:
                    right_score += 5.0 * weight

        if left_score < 0.1 and right_score < 0.1: 
            return 0.0, "Clean"

        # 명확한 방향 결정
        diff = abs(right_score - left_score)
        if diff < 0.5:  # 너무 애매하면 직진
            return 0.0, f"Ambiguous(L:{left_score:.1f}/R:{right_score:.1f})"
        
        if right_score > left_score:
            return 1.0, f"Obs Right({right_score:.1f})->Go Left"
        else:
            return -1.0, f"Obs Left({left_score:.1f})->Go Right"

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