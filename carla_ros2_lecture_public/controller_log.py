#!/usr/bin/env python3
# koo_controller.py
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

# 상태 정의
STATE_DRIVE = 0
STATE_STOP_WAIT = 1
STATE_FINISHED = 2

class KooController(Node):
    def __init__(self, path_num=1):
        super().__init__("koo_controller")
        
        self.path_num = path_num

        # ================================
        # 🔹 성능 로그 파일 초기화
        # ================================
        log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "controller_perf_log.csv")
        self.log_path = log_path
        if not os.path.exists(log_path):
            with open(log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "lateral_error",
                    "speed_error",
                    "steer_rate"
                ])

        # ROS 통신 설정
        self.sub_gnss = self.create_subscription(NavSatFix, "/carla/hero/gnss", self.gnss_cb, 10)
        self.sub_path = self.create_subscription(Path, "/carla/path/local", self.local_path_cb, 10)
        self.sub_decision = self.create_subscription(String, "/fusion/decision", self.decision_cb, 10)
        self.sub_lane = self.create_subscription(PointStamped, "/carla/lane/center", self.lane_cb, 10)
        self.pub_cmd = self.create_publisher(Twist, "/carla/hero/cmd_vel", 10)

        # ===============================================================
        # [🛠️ 튜닝 파라미터 섹션 - 이 값들을 조절하세요]
        # ===============================================================
        
        # 1. 차량 물리 속성 & 기본 제어
        self.wheel_base = 2.7       # 휠베이스 (앞바퀴~뒷바퀴 거리) [m]
        self.max_steer = 35.0       # 최대 조향각 [도]
        self.dt = 0.05              # 제어 주기 [초] (20Hz)

        # [NEW] 조향 스무딩 계수 (0.0 ~ 1.0)
        self.steer_alpha = 0.2      

        # 2. 속도 설정 (Speed Profile)
        self.target_speed = 5.0     # 기본 주행 목표 속도 [m/s] (약 18km/h)
        self.accel_limit = 2.0      # 가속 제한 [m/s^2]
        self.decel_limit = 4.0      # 감속 제한 [m/s^2]
        self.min_speed_guarantee = 2.0 # 장애물이 없을 때 보장할 최저 속도 [m/s]

        # 3. 차선 유지 (Lane Keeping) 게인
        self.k_lat = 1.0  
        self.k_ang = 1.5  

        # 4. 장애물 인식 범위 (Field of View)
        self.fov_main_deg = 12.0    
        self.fov_buffer_deg = 30.0  

        # 5. 안전 거리 설정 (Distance Logic)
        self.dist_acc_start = 20.0
        self.dist_stop_gap = 12.0
        self.dist_side_warn = 8.0
        self.dist_light_stop = 35.0

        # 6. 감속 비율 (Factor)
        self.factor_side_danger = 0.6 
        # ===============================================================

        self.curr_xy = None
        self.yaw = 0.0
        self.local_path = []
        self.lat0 = None; self.lon0 = None; self.cos_lat0 = 1.0

        # 상태 변수
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
        
        # 스무딩용 이전 조향값 (명령 기준)
        self.prev_steer = 0.0            # 스무딩 내부 용
        self.prev_steer_cmd = 0.0        # 성능 로그용 (이전 스무딩 결과)

        self.log_counter = 0
        self.timer = self.create_timer(self.dt, self.control_loop)

        print(f"=== KooController Started (Path: {self.path_num}) ===")
        print(f">> Params Initialized. MaxSpeed: {self.target_speed}m/s")
        print(f">> Steering Alpha: {self.steer_alpha} (Smoothing ON)")

    # ---------------- GNSS ----------------
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

    # --------------- Local Path ---------------
    def local_path_cb(self, msg):
        self.local_path = []
        for pose in msg.poses:
            self.local_path.append((pose.pose.position.x, pose.pose.position.y))

    # --------------- Fusion Decision ---------------
    def decision_cb(self, msg):
        try:
            data = json.loads(msg.data)
            self.light_state = data.get("light", "none")
            self.light_dist = float(data.get("light_dist", -1.0))
            self.vehicle_dist = float(data.get("vehicle_dist", -1.0))
            self.vehicle_angle = float(data.get("vehicle_angle", 0.0))
        except:
            pass

    # --------------- Lane Center ---------------
    def lane_cb(self, msg: PointStamped):
        self.lane_offset = msg.point.x
        self.lane_angle = msg.point.y
        self.last_lane_time = time.time()

    # --------------- Control Loop ---------------
    def control_loop(self):
        self.log_counter += 1
        if self.curr_xy is None:
            return

        # === 1. 도착 확인 ===
        dist_to_goal = 999.0
        if self.local_path:
            goal_pt = self.local_path[-1]
            dist_to_goal = math.sqrt((goal_pt[0] - self.curr_xy[0])**2 +
                                     (goal_pt[1] - self.curr_xy[1])**2)
        
        if dist_to_goal < 2.0 and not self.is_goal_reached:
            self.is_goal_reached = True
            self.traffic_state = STATE_FINISHED
            print(f"\n🎉 [ARRIVED] 목적지 도착 (Dist: {dist_to_goal:.2f}m) 🎉\n")

        if self.is_goal_reached:
            stop_cmd = Twist()
            stop_cmd.linear.x = 0.0
            stop_cmd.linear.y = 1.0 
            self.pub_cmd.publish(stop_cmd)
            return

        # === 2. 조향 제어 (Hybrid) ===
        steer_deg = 0.0
        control_mode = "GNSS"
        lateral_error = 0.0   # 🔹 성능 로그용 초기값

        # (A) Pure Pursuit
        if self.local_path:
            lookahead_dist = max(6.0, self.current_speed_cmd * 1.8)
            
            target_pt = None
            min_diff = 1e9
            for px, py in self.local_path:
                dist = math.sqrt((px - self.curr_xy[0])**2 +
                                 (py - self.curr_xy[1])**2)
                if dist >= lookahead_dist and dist < min_diff:
                    min_diff = dist
                    target_pt = (px, py)
            if target_pt is None:
                target_pt = self.local_path[-1]

            # 차량 좌표계로 변환
            dx = target_pt[0] - self.curr_xy[0]
            dy = target_pt[1] - self.curr_xy[1]
            local_x = dx * math.cos(-self.yaw) - dy * math.sin(-self.yaw)
            local_y = dx * math.sin(-self.yaw) + dy * math.cos(-self.yaw)

            steer_rad = math.atan2(2.0 * self.wheel_base * local_y,
                                   lookahead_dist**2)
            steer_deg = math.degrees(steer_rad)

            # 🔹 Lateral error 후보 (Pure Pursuit 기준)
            lateral_error = local_y

        # (B) Lane Keeping (LKA) 덮어쓰기
        if time.time() - self.last_lane_time < 0.5:
            control_mode = "LANE"
            lane_steer_rad = self.k_lat * self.lane_offset + \
                             self.k_ang * self.lane_angle
            lane_steer_deg = math.degrees(lane_steer_rad)
            steer_deg = max(-self.max_steer,
                            min(self.max_steer, lane_steer_deg))
            # 🔹 Lane 기반일 때는 차선 offset을 lateral error로 사용
            lateral_error = self.lane_offset

        # === 3. 속도 제어 ===
        relevant_vehicle_dist = -1.0
        is_my_lane_vehicle = False
        is_side_danger = False
        
        if 0 < self.vehicle_dist < 100.0:
            abs_angle = abs(self.vehicle_angle)
            
            if abs_angle <= self.fov_main_deg:
                relevant_vehicle_dist = self.vehicle_dist
                is_my_lane_vehicle = True
            elif self.fov_main_deg < abs_angle <= self.fov_buffer_deg:
                if self.vehicle_dist < self.dist_side_warn:
                    is_side_danger = True

        is_light_safe = True
        if self.light_state in ["traffic_red", "traffic_yellow"]:
            if 0 < self.light_dist < self.dist_light_stop:
                is_light_safe = False

        is_vehicle_safe = True
        if is_my_lane_vehicle and relevant_vehicle_dist < self.dist_stop_gap:
            is_vehicle_safe = False

        desired_speed = 0.0

        if is_light_safe and is_vehicle_safe:
            self.traffic_state = STATE_DRIVE
            
            acc_factor = 1.0
            if is_my_lane_vehicle and relevant_vehicle_dist < self.dist_acc_start:
                acc_factor = (relevant_vehicle_dist - self.dist_stop_gap) / \
                             (self.dist_acc_start - self.dist_stop_gap)
                acc_factor = max(0.0, min(1.0, acc_factor))

            danger_factor = 1.0
            if is_side_danger:
                danger_factor = self.factor_side_danger

            arrival_factor = 1.0
            if dist_to_goal < 10.0:
                arrival_factor = dist_to_goal / 10.0
                arrival_factor = max(0.0, min(1.0, arrival_factor))
            
            steer_ratio = abs(steer_deg) / self.max_steer
            corner_factor = 1.0 - (steer_ratio * 0.5)

            desired_speed = self.target_speed * \
                            acc_factor * arrival_factor * \
                            corner_factor * danger_factor
            
            if (desired_speed < self.min_speed_guarantee and
                acc_factor > 0.1 and arrival_factor > 0.1 and
                not is_side_danger):
                desired_speed = self.min_speed_guarantee
        else:
            self.traffic_state = STATE_STOP_WAIT
            desired_speed = 0.0

        # === 4. 속도 스무딩 ===
        speed_diff = desired_speed - self.current_speed_cmd
        if speed_diff > 0:
            self.current_speed_cmd += min(speed_diff,
                                          self.accel_limit * self.dt)
        else:
            self.current_speed_cmd -= min(-speed_diff,
                                          self.decel_limit * self.dt)
        self.current_speed_cmd = max(0.0, self.current_speed_cmd)

        # 🔹 Speed Tracking Error
        speed_error = desired_speed - self.current_speed_cmd

        # === 5. 조향 스무딩 ===
        target_steer = max(-self.max_steer,
                           min(self.max_steer, steer_deg))
        
        smoothed_steer = (self.prev_steer * (1.0 - self.steer_alpha) +
                          target_steer * self.steer_alpha)
        self.prev_steer = smoothed_steer

        # 🔹 Steering rate (조향 변화율)
        steer_rate = (smoothed_steer - self.prev_steer_cmd) / self.dt
        self.prev_steer_cmd = smoothed_steer

        # === 6. 실제 명령 발행 ===
        cmd = Twist()
        cmd.linear.x = float(self.current_speed_cmd)
        cmd.angular.z = float(smoothed_steer)
        if self.current_speed_cmd < 0.1 and self.traffic_state == STATE_STOP_WAIT:
            cmd.linear.x = 0.0
            cmd.linear.y = 1.0
        self.pub_cmd.publish(cmd)

        # === 7. 성능 지표 CSV 로그 기록 ===
        try:
            with open(self.log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    time.time(),
                    float(lateral_error),
                    float(speed_error),
                    float(steer_rate)
                ])
        except Exception as e:
            # 로그 실패는 주행에 영향 없도록 무시
            pass

        # 디버깅 로그 (원래 있던 부분 유지)
        if self.log_counter % 20 == 0:
            state_str = ["DRIVE", "WAIT", "DONE"][self.traffic_state]
            
            if is_my_lane_vehicle:
                obs_info = f"ACC:{relevant_vehicle_dist:.1f}m"
            elif is_side_danger:
                obs_info = f"SIDE_WARN!({self.vehicle_dist:.1f}m,{self.vehicle_angle:.0f}dg)"
            else:
                if self.vehicle_dist > 0:
                    obs_info = f"Ignored({self.vehicle_dist:.1f}m,{self.vehicle_angle:.0f}dg)"
                else:
                    obs_info = "Clean"

            light_info = f"{self.light_state}"
            if self.light_dist > 0:
                light_info += f"({self.light_dist:.1f}m)"

            print(f"[{state_str}] [{control_mode}] "
                  f"Spd:{self.current_speed_cmd:.1f} | "
                  f"Steer:{smoothed_steer:.1f} | "
                  f"LatErr:{lateral_error:.2f} | "
                  f"SpdErr:{speed_error:.2f} | "
                  f"Light:{light_info} | Obs:{obs_info}")

def main(args=None):
    rclpy.init(args=args)
    node = KooController()
    try:
        rclpy.spin(node)
    except:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
