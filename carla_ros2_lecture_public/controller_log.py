#!/usr/bin/env python3
# koo_controller.py (with performance logging)

import math
import json
import time
import os
import csv

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import Twist, PointStamped
from std_msgs.msg import String
from nav_msgs.msg import Path

STATE_DRIVE = 0
STATE_STOP_WAIT = 1
STATE_FINISHED = 2

class KooController(Node):
    def __init__(self, path_num=1):
        super().__init__("koo_controller")
        self.path_num = path_num

        # ---------------------------------------------------------------
        # ⭐ 성능지표 로그 파일 생성
        # ---------------------------------------------------------------
        self.log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "controller_perf_log.csv")

        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "lateral_error",
                    "speed_error",
                    "steer_rate"
                ])

        # ---------------------------------------------------------------
        # ROS 통신 설정
        # ---------------------------------------------------------------
        self.sub_gnss = self.create_subscription(NavSatFix, "/carla/hero/gnss", self.gnss_cb, 10)
        self.sub_path = self.create_subscription(Path, "/carla/path/local", self.local_path_cb, 10)
        self.sub_decision = self.create_subscription(String, "/fusion/decision", self.decision_cb, 10)
        self.sub_lane = self.create_subscription(PointStamped, "/carla/lane/center", self.lane_cb, 10)
        self.pub_cmd = self.create_publisher(Twist, "/carla/hero/cmd_vel", 10)

        # ---------------------------------------------------------------
        # 제어 파라미터
        # ---------------------------------------------------------------
        self.wheel_base = 2.7
        self.max_steer = 35.0
        self.dt = 0.05        # 20Hz
        self.steer_alpha = 0.2

        self.target_speed = 5.0
        self.accel_limit = 2.0
        self.decel_limit = 4.0
        self.min_speed_guarantee = 2.0

        self.k_lat = 1.0
        self.k_ang = 1.5

        self.fov_main_deg = 12.0
        self.fov_buffer_deg = 30.0

        self.dist_acc_start = 20.0
        self.dist_stop_gap = 12.0
        self.dist_side_warn = 8.0
        self.dist_light_stop = 35.0

        self.factor_side_danger = 0.6

        # ---------------------------------------------------------------
        # 상태 변수
        # ---------------------------------------------------------------
        self.curr_xy = None
        self.yaw = 0.0
        self.local_path = []
        self.lat0 = None; self.lon0 = None; self.cos_lat0 = 1.0

        self.light_state = "none"
        self.light_dist = -1.0
        
        self.vehicle_dist = -1.0
        self.vehicle_angle = 0.0
        
        self.traffic_state = STATE_DRIVE 
        self.is_goal_reached = False
        self.current_speed_cmd = 0.0
        
        self.lane_offset = 0.0
        self.lane_angle = 0.0
        self.last_lane_time = 0.0
        
        # 조향 스무딩용 변수
        self.prev_steer = 0.0
        self.prev_steer_cmd = 0.0   # ⭐ 성능지표용 이전 스무딩 조향값

        self.log_counter = 0
        self.timer = self.create_timer(self.dt, self.control_loop)

        print(f"=== KooController Started (Path: {self.path_num}) ===")


    # ----------------------------------------------------------------------
    # GNSS Callback
    # ----------------------------------------------------------------------
    def gnss_cb(self, msg):
        if self.lat0 is None:
            self.lat0 = msg.latitude
            self.lon0 = msg.longitude
            self.cos_lat0 = math.cos(math.radians(self.lat0))

        x = (msg.longitude - self.lon0) * 111320.0 * self.cos_lat0
        y = (msg.latitude - self.lat0) * 111320.0
        
        if self.curr_xy is not None:
            dx = x - self.curr_xy[0]
            dy = y - self.curr_xy[1]
            if dx*dx + dy*dy > 0.0001:
                self.yaw = math.atan2(dy, dx)

        self.curr_xy = (x, y)

    # ----------------------------------------------------------------------
    def local_path_cb(self, msg):
        self.local_path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]

    # ----------------------------------------------------------------------
    def decision_cb(self, msg):
        try:
            data = json.loads(msg.data)
            self.light_state = data.get("light", "none")
            self.light_dist = float(data.get("light_dist", -1.0))
            self.vehicle_dist = float(data.get("vehicle_dist", -1.0))
            self.vehicle_angle = float(data.get("vehicle_angle", 0.0))
        except:
            pass

    # ----------------------------------------------------------------------
    def lane_cb(self, msg: PointStamped):
        self.lane_offset = msg.point.x
        self.lane_angle = msg.point.y
        self.last_lane_time = time.time()


    # ----------------------------------------------------------------------
    # Main Control Loop
    # ----------------------------------------------------------------------
    def control_loop(self):
        self.log_counter += 1
        if self.curr_xy is None:
            return

        # ---------------------------------------------------------------
        # 1) 도착 확인
        # ---------------------------------------------------------------
        dist_to_goal = 999.0
        if self.local_path:
            gx, gy = self.local_path[-1]
            dist_to_goal = math.hypot(gx - self.curr_xy[0], gy - self.curr_xy[1])

        if dist_to_goal < 2.0 and not self.is_goal_reached:
            self.is_goal_reached = True
            self.traffic_state = STATE_FINISHED
            print(f"[ARRIVED] 목적지 도착!")

        if self.is_goal_reached:
            stop_cmd = Twist()
            stop_cmd.linear.x = 0.0
            stop_cmd.linear.y = 1.0
            self.pub_cmd.publish(stop_cmd)
            return

        # ---------------------------------------------------------------
        # 2) 조향(Pure Pursuit + LKA)
        # ---------------------------------------------------------------
        steer_deg = 0.0
        lateral_error = 0.0   # ⭐ 성능지표 1번

        if self.local_path:
            lookahead = max(6.0, self.current_speed_cmd * 1.8)
            target = None
            best = 1e9

            for px, py in self.local_path:
                d = math.hypot(px - self.curr_xy[0], py - self.curr_xy[1])
                if d >= lookahead and d < best:
                    best = d
                    target = (px, py)

            if target is None:
                target = self.local_path[-1]

            dx = target[0] - self.curr_xy[0]
            dy = target[1] - self.curr_xy[1]

            local_x = dx * math.cos(-self.yaw) - dy * math.sin(-self.yaw)
            local_y = dx * math.sin(-self.yaw) + dy * math.cos(-self.yaw)

            lateral_error = local_y

            steer_rad = math.atan2(2*self.wheel_base*local_y, lookahead**2)
            steer_deg = math.degrees(steer_rad)

        # Lane Keeping override
        if time.time() - self.last_lane_time < 0.5:
            lane_steer_rad = self.k_lat*self.lane_offset + self.k_ang*self.lane_angle
            steer_deg = math.degrees(lane_steer_rad)
            lateral_error = self.lane_offset   # ⭐ 차선 기반 lateral error

        # ---------------------------------------------------------------
        # 3) 속도 제어 + 스무딩
        # ---------------------------------------------------------------
        desired_speed = self.target_speed
        speed_error = desired_speed - self.current_speed_cmd   # ⭐ 성능지표 2번

        # 속도 스무딩
        diff = desired_speed - self.current_speed_cmd
        if diff > 0:
            self.current_speed_cmd += min(diff, self.accel_limit * self.dt)
        else:
            self.current_speed_cmd -= min(-diff, self.decel_limit * self.dt)

        self.current_speed_cmd = max(0.0, self.current_speed_cmd)

        # ---------------------------------------------------------------
        # 4) 조향 스무딩 + steer_rate 계산
        # ---------------------------------------------------------------
        target_steer = max(-self.max_steer, min(self.max_steer, steer_deg))

        smoothed_steer = (self.prev_steer*(1-self.steer_alpha)
                          + target_steer*self.steer_alpha)

        steer_rate = (smoothed_steer - self.prev_steer_cmd) / self.dt   # ⭐ 성능지표 3번
        self.prev_steer_cmd = smoothed_steer

        self.prev_steer = smoothed_steer

        # ---------------------------------------------------------------
        # 5) 명령 발행
        # ---------------------------------------------------------------
        cmd = Twist()
        cmd.linear.x = float(self.current_speed_cmd)
        cmd.angular.z = float(smoothed_steer)

        if self.current_speed_cmd < 0.1 and self.traffic_state == STATE_STOP_WAIT:
            cmd.linear.x = 0.0
            cmd.linear.y = 1.0

        self.pub_cmd.publish(cmd)

        # ---------------------------------------------------------------
        # ⭐ 성능지표 CSV 저장
        # ---------------------------------------------------------------
        try:
            with open(self.log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    time.time(),
                    float(lateral_error),
                    float(speed_error),
                    float(steer_rate)
                ])
        except:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = KooController()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
