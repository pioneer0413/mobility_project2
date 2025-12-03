#!/usr/bin/env python3
import math
import os
import csv
import argparse
from typing import List, Tuple, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseArray, PoseStamped
from rclpy.time import Time

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

        self.lat0 = None; self.lon0 = None; self.cos_lat0 = 1.0
        self.current_xy = None; self.global_xy = []; self.obstacles = []

        self.L = 20.0            
        self.safe_lat = 2.0      
        self.max_offset = 3.0    
        self.prev_idx = 0        

        # [NEW] 스마트 대기 로직 변수
        self.obs_start_time = None      # 최초 감지 시간
        self.last_seen_time = None      # 마지막으로 감지된 시간
        
        # [수정됨] 5.0초 -> 3.0초로 변경
        self.wait_time = 3.0            
        
        self.reset_timeout = 2.0        # 깜빡임 방지 시간

        self.load_global_path()
        self.timer = self.create_timer(0.1, self.timer_cb)
        self.get_logger().info(f">> Koo Planner (Path: {self.path_num}) Wait time set to 3.0s")

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
        if self.lat0 is None:
            self.lat0 = msg.latitude; self.lon0 = msg.longitude
            self.cos_lat0 = math.cos(math.radians(msg.latitude))
            self.current_xy = (0.0, 0.0)
        else:
            dx = (msg.longitude - self.lon0) * (111320.0 * self.cos_lat0)
            dy = (msg.latitude - self.lat0) * 110540.0
            self.current_xy = (dx, dy)

    def obs_cb(self, msg): self.obstacles = [(p.position.x, p.position.y) for p in msg.poses]

    def timer_cb(self):
        if self.current_xy is None or len(self.global_xy) < 2: return
        x, y = self.current_xy
        idx = self.find_nearest_index(x, y)
        if idx is None: return

        # 1. 장애물 감지 여부 확인 (Debouncing 적용)
        raw_side = self.decide_side(self.obstacles)
        now = self.get_clock().now()
        is_obstacle_present = False
        
        if raw_side != 0.0:
            self.last_seen_time = now
            if self.obs_start_time is None:
                self.obs_start_time = now
            is_obstacle_present = True
        else:
            if self.last_seen_time is not None:
                time_since_lost = (now - self.last_seen_time).nanoseconds / 1e9
                if time_since_lost < self.reset_timeout:
                    is_obstacle_present = True
                else:
                    self.obs_start_time = None
                    self.last_seen_time = None
                    is_obstacle_present = False

        # 2. 회피 결정 (3초 대기)
        final_side = 0.0
        if is_obstacle_present and self.obs_start_time is not None:
            elapsed = (now - self.obs_start_time).nanoseconds / 1e9
            
            if elapsed < self.wait_time:
                # 3초 미만: 대기 (직진 -> ACC 정지 유도)
                final_side = 0.0
                if int(elapsed * 10) % 10 == 0:
                    self.get_logger().info(f"🛑 Waiting... {elapsed:.1f}s / {self.wait_time:.1f}s")
            else:
                # 3초 경과: 회피 시작
                if raw_side != 0.0:
                    final_side = raw_side
                    self.get_logger().info(f"⚠️ Avoidance Active! (Side: {final_side})")
                else:
                    final_side = 0.0 

        path = Path(); path.header.stamp = now.to_msg(); path.header.frame_id = "map"
        s = 0.0; prev_px, prev_py = self.global_xy[idx]; curr_idx = idx; path_len = len(self.global_xy)

        while curr_idx < path_len and s <= self.L:
            gx, gy = self.global_xy[curr_idx]
            s += math.hypot(gx - prev_px, gy - prev_py)
            prev_px, prev_py = gx, gy; px, py = gx, gy

            if final_side != 0.0:
                t = min(max(s / self.L, 0.0), 1.0); bezier = 3.0 * t * (1.0 - t)
                offset = self.max_offset * bezier * final_side
                if curr_idx + 1 < path_len: nx, ny = self.global_xy[curr_idx+1]
                else: nx, ny = gx, gy
                path_yaw = math.atan2(ny - gy, nx - gx)
                px += offset * -math.sin(path_yaw); py += offset * math.cos(path_yaw)

            ps = PoseStamped(); ps.header = path.header
            ps.pose.position.x = px; ps.pose.position.y = py; path.poses.append(ps)
            curr_idx += 1

        self.pub_local_path.publish(path)

    def find_nearest_index(self, x, y):
        start = max(0, self.prev_idx - 50); end = min(len(self.global_xy), self.prev_idx + 50)
        min_d = float('inf'); idx = -1
        for i in range(start, end):
            d = (self.global_xy[i][0] - x)**2 + (self.global_xy[i][1] - y)**2
            if d < min_d: min_d = d; idx = i
        if idx != -1: self.prev_idx = idx; return idx
        return self.prev_idx

    def decide_side(self, obs_xy) -> float:
        if not obs_xy: return 0.0
        relevant = [o for o in obs_xy if 0.1 < o[0] < self.L and abs(o[1]) < self.safe_lat]
        if not relevant: return 0.0
        left = sum(1.0/(o[0]*max(0.5, abs(o[1]))) for o in relevant if o[1] > 0)
        right = sum(1.0/(o[0]*max(0.5, abs(o[1]))) for o in relevant if o[1] < 0)
        if left == 0 and right == 0: return 0.0
        return max(-1.0, min(1.0, (right - left) / (right + left)))

def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_num', type=int, default=1)
    ros_args, _ = parser.parse_known_args()
    node = KooLocalPlanner(path_num=ros_args.path_num)
    rclpy.spin(node); node.destroy_node(); rclpy.shutdown()

if __name__ == "__main__": main()