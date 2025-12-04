#!/usr/bin/env python3
import math
import json
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from nav_msgs.msg import Path

# 상태 정의
STATE_DRIVE = 0
STATE_STOP_WAIT = 1
STATE_FINISHED = 2  # [NEW] 도착 상태

class KooController(Node):
    def __init__(self, path_num=1):
        super().__init__("koo_controller")
        
        self.path_num = path_num
        self.sub_gnss = self.create_subscription(NavSatFix, "/carla/hero/gnss", self.gnss_cb, 10)
        self.sub_path = self.create_subscription(Path, "/carla/path/local", self.local_path_cb, 10)
        self.sub_decision = self.create_subscription(String, "/fusion/decision", self.decision_cb, 10)
        self.pub_cmd = self.create_publisher(Twist, "/carla/hero/cmd_vel", 10)

        self.curr_xy = None
        self.yaw = 0.0

        self.local_path = []
        self.lat0 = None; self.lon0 = None; self.cos_lat0 = 1.0

        # === [차량 파라미터] ===
        self.wheel_base = 2.7
        self.target_speed = 5.0  # Max Speed 10m/s
        self.max_steer = 35.0
        
        # 가감속 제한 (Smoothing)
        self.accel_limit = 2.0  
        self.decel_limit = 4.0  
        self.dt = 0.05          

        # 상태 변수
        self.light_state = "none"
        self.light_dist = -1.0
        self.vehicle_dist = -1.0
        self.traffic_state = STATE_DRIVE 
        self.is_goal_reached = False # [NEW] 도착 여부 체크
        
        self.current_speed_cmd = 0.0
        
        # 디버깅용
        self.prev_steer = 0.0
        self.prev_local_y = 0.0
        self.prev_target = (0.0, 0.0)
        self.log_counter = 0

        self.timer = self.create_timer(self.dt, self.control_loop)

        print(f"=== KooController Started (Path: {self.path_num}) ===")
        print(f">> MaxSpeed: {self.target_speed}m/s | Arrival Check: ON")

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

    def local_path_cb(self, msg):
        self.local_path = []
        for pose in msg.poses:
            self.local_path.append((pose.pose.position.x, pose.pose.position.y))

    def decision_cb(self, msg):
        try:
            data = json.loads(msg.data)
            self.light_state = data.get("light", "none")
            self.light_dist = float(data.get("light_dist", -1.0))
            self.vehicle_dist = float(data.get("vehicle_dist", -1.0))
        except:
            pass

    def control_loop(self):
        self.log_counter += 1
        if self.curr_xy is None or not self.local_path:
            return

        # === [NEW] 1. 최종 목적지 도착 확인 ===
        # local_path의 마지막 점 = 현재 경로의 목표점
        # 글로벌 경로의 끝에 다다르면 이 거리가 점점 줄어듦
        goal_pt = self.local_path[-1]
        dist_to_goal = math.sqrt((goal_pt[0] - self.curr_xy[0])**2 + (goal_pt[1] - self.curr_xy[1])**2)
        
        # 2.0m 이내 진입 시 도착으로 처리
        if dist_to_goal < 2.0 and not self.is_goal_reached:
            self.is_goal_reached = True
            self.traffic_state = STATE_FINISHED
            print(f"\n🎉🎉🎉 [ARRIVED] 최종 목적지에 도착했습니다! (Dist: {dist_to_goal:.2f}m) 🎉🎉🎉\n")

        # 도착 상태면 완전 정지 후 리턴
        if self.is_goal_reached:
            stop_cmd = Twist()
            stop_cmd.linear.x = 0.0
            stop_cmd.linear.y = 1.0 # Full Brake
            stop_cmd.angular.z = 0.0
            self.pub_cmd.publish(stop_cmd)
            return

        # === 2. Pure Pursuit (Lookahead) ===
        lookahead_dist = max(6.0, self.current_speed_cmd * 1.5)
        
        target_pt = None
        min_diff = 1e9
        
        for px, py in self.local_path:
            dist = math.sqrt((px - self.curr_xy[0])**2 + (py - self.curr_xy[1])**2)
            if dist >= lookahead_dist:
                if dist < min_diff:
                    min_diff = dist
                    target_pt = (px, py)
        
        if target_pt is None: target_pt = self.local_path[-1]

        local_x = (target_pt[0] - self.curr_xy[0]) * math.cos(-self.yaw) - \
                  (target_pt[1] - self.curr_xy[1]) * math.sin(-self.yaw)
        local_y = (target_pt[0] - self.curr_xy[0]) * math.sin(-self.yaw) + \
                  (target_pt[1] - self.curr_xy[1]) * math.cos(-self.yaw)
        
        steer_rad = math.atan2(2.0 * self.wheel_base * local_y, lookahead_dist**2)
        steer_deg = math.degrees(steer_rad)
        steer_deg = max(-self.max_steer, min(self.max_steer, steer_deg))

        # [디버깅] 조향 변경 분석
        if self.log_counter % 10 == 0:
            steer_diff = steer_deg - self.prev_steer
            if abs(steer_diff) > 2.0:
                 pass # 너무 자주 떠서 일단 생략 (필요하면 주석 해제)

        self.prev_steer = steer_deg
        self.prev_local_y = local_y
        self.prev_target = target_pt

        # === 3. 목표 속도 결정 ===
        is_light_safe = True
        if self.light_state in ["traffic_red", "traffic_yellow"]:
            if 0 < self.light_dist < 35.0:
                is_light_safe = False

        is_vehicle_safe = True
        if 0 < self.vehicle_dist < 12.0:
            is_vehicle_safe = False

        desired_speed = 0.0

        if is_light_safe and is_vehicle_safe:
            self.traffic_state = STATE_DRIVE
            
            # (1) ACC: 앞차 거리 비례 감속
            acc_factor = 1.0
            if 0 < self.vehicle_dist < 20.0:
                 acc_factor = (self.vehicle_dist - 12.0) / (20.0 - 12.0)
                 acc_factor = max(0.0, min(1.0, acc_factor))

            # (2) [NEW] Arrival: 목적지 거리 비례 감속 (15m 전부터)
            arrival_factor = 1.0
            if dist_to_goal < 10.0:
                 arrival_factor = dist_to_goal / 10.0
                 arrival_factor = max(0.0, min(1.0, arrival_factor))
            
            # 최종 속도 = 목표속도 * ACC비율 * 도착감속비율
            desired_speed = self.target_speed * acc_factor * arrival_factor

        else:
            self.traffic_state = STATE_STOP_WAIT
            desired_speed = 0.0

        # === 4. 속도 스무딩 (가감속 제한) ===
        speed_diff = desired_speed - self.current_speed_cmd
        max_accel = self.accel_limit * self.dt
        max_decel = self.decel_limit * self.dt
        
        if speed_diff > 0:
            self.current_speed_cmd += min(speed_diff, max_accel)
        else:
            self.current_speed_cmd -= min(-speed_diff, max_decel)
            
        self.current_speed_cmd = max(0.0, self.current_speed_cmd)

        # ROS 메시지 발행
        cmd = Twist()
        cmd.linear.x = float(self.current_speed_cmd)
        cmd.angular.z = float(steer_deg)
        
        if self.current_speed_cmd < 0.1 and self.traffic_state == STATE_STOP_WAIT:
             cmd.linear.x = 0.0
             cmd.linear.y = 1.0 # Brake
        
        self.pub_cmd.publish(cmd)

        # 로그 출력
        if self.log_counter % 20 == 0:
             status = "GO" if self.traffic_state == STATE_DRIVE else "STOP"
             # 도착 거리 정보 추가
             print(f"[{status}] Spd:{self.current_speed_cmd:.1f}m/s | Goal:{dist_to_goal:.1f}m | Car:{self.vehicle_dist:.1f}m")

def main(args=None):
    rclpy.init(args=args)
    node = KooController()
    try: rclpy.spin(node)
    except: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()