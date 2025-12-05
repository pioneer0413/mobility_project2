#!/usr/bin/env python3
# koo_controller.py
import math
import json
import time
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

        # 2. 속도 설정 (Speed Profile)
        self.target_speed = 5.0     # 기본 주행 목표 속도 [m/s] (약 18km/h)
        self.accel_limit = 2.0      # 가속 제한 (클수록 출발이 빠름) [m/s^2]
        self.decel_limit = 4.0      # 감속 제한 (클수록 브레이크가 강함) [m/s^2]
        self.min_speed_guarantee = 2.0 # 장애물이 없을 때 보장할 최저 속도 [m/s]

        # 3. 차선 유지 (Lane Keeping) 게인
        # - K_lat: 차선 중심에서 벗어난 거리(m)를 얼마나 강하게 보정할지
        # - K_ang: 차선 각도(deg)와 내 차의 틀어짐을 얼마나 강하게 보정할지
        self.k_lat = 0.6  
        self.k_ang = 1.5  

        # 4. 장애물 인식 범위 (Field of View)
        # 내 차선으로 인정하는 각도 범위 (중앙 기준 좌우 +/-)
        self.fov_main_deg = 12.0    
        # 측면 위험(방어운전) 감지 각도 범위 (Main 바깥 ~ 이 각도까지)
        self.fov_buffer_deg = 30.0  

        # 5. 안전 거리 설정 (Distance Logic)
        self.dist_acc_start = 20.0  # 앞차 감속 시작 거리 (이 거리부터 천천히 줄임) [m]
        self.dist_stop_gap = 12.0   # 앞차와 유지할 최소 정지 간격 [m]
        self.dist_side_warn = 8.0   # 측면/끼어들기 차량 위험 감지 거리 [m]
        self.dist_light_stop = 35.0 # 신호등 정지선 인식 거리 [m]

        # 6. 감속 비율 (Factor)
        # 측면 위험 감지 시 속도를 몇 %로 줄일지 (0.6 = 60% 속도)
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
        
        self.log_counter = 0
        self.timer = self.create_timer(self.dt, self.control_loop)

        print(f"=== KooController Started (Path: {self.path_num}) ===")
        print(f">> Params Initialized. MaxSpeed: {self.target_speed}m/s")
        print(f">> Main FOV: +/-{self.fov_main_deg} deg | Buffer: +/-{self.fov_buffer_deg} deg")

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
            self.vehicle_angle = float(data.get("vehicle_angle", 0.0))
        except: pass

    def lane_cb(self, msg: PointStamped):
        self.lane_offset = msg.point.x
        self.lane_angle = msg.point.y
        self.last_lane_time = time.time()

    def control_loop(self):
        self.log_counter += 1
        if self.curr_xy is None: return

        # === 1. 도착 확인 ===
        dist_to_goal = 999.0
        if self.local_path:
            goal_pt = self.local_path[-1]
            dist_to_goal = math.sqrt((goal_pt[0] - self.curr_xy[0])**2 + (goal_pt[1] - self.curr_xy[1])**2)
        
        if dist_to_goal < 2.0 and not self.is_goal_reached:
            self.is_goal_reached = True
            self.traffic_state = STATE_FINISHED
            print(f"\n🎉 [ARRIVED] 목적지 도착 (Dist: {dist_to_goal:.2f}m) 🎉\n")

        if self.is_goal_reached:
            stop_cmd = Twist()
            stop_cmd.linear.x = 0.0; stop_cmd.linear.y = 1.0 
            self.pub_cmd.publish(stop_cmd)
            return

        # === 2. 조향 제어 (Hybrid) ===
        steer_deg = 0.0
        control_mode = "GNSS"

        # (A) Pure Pursuit
        if self.local_path:
            lookahead_dist = max(3.5, self.current_speed_cmd * 1.2)
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

        # (B) Lane Keeping (LKA) 덮어쓰기
        if time.time() - self.last_lane_time < 0.5:
            control_mode = "LANE"
            # 튜닝 파라미터 적용
            lane_steer_rad = self.k_lat * self.lane_offset + self.k_ang * self.lane_angle
            lane_steer_deg = math.degrees(lane_steer_rad)
            steer_deg = max(-self.max_steer, min(self.max_steer, lane_steer_deg))

        steer_deg = max(-self.max_steer, min(self.max_steer, steer_deg))

        # === 3. 속도 제어 (Advanced Filtering) ===
        relevant_vehicle_dist = -1.0
        is_my_lane_vehicle = False
        is_side_danger = False
        
        if 0 < self.vehicle_dist < 100.0:
            abs_angle = abs(self.vehicle_angle)
            
            # (1) 내 차선 (Main Zone)
            if abs_angle <= self.fov_main_deg:
                relevant_vehicle_dist = self.vehicle_dist
                is_my_lane_vehicle = True
            
            # (2) 경계 구역 (Buffer Zone)
            elif self.fov_main_deg < abs_angle <= self.fov_buffer_deg:
                if self.vehicle_dist < self.dist_side_warn:
                    is_side_danger = True

        # 안전 체크
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
            
            # ACC (내 차선 차량)
            acc_factor = 1.0
            if is_my_lane_vehicle and relevant_vehicle_dist < self.dist_acc_start:
                 # 거리 비율 계산
                 acc_factor = (relevant_vehicle_dist - self.dist_stop_gap) / (self.dist_acc_start - self.dist_stop_gap)
                 acc_factor = max(0.0, min(1.0, acc_factor))

            # 방어 운전 (측면 위험)
            danger_factor = 1.0
            if is_side_danger:
                danger_factor = self.factor_side_danger

            # Arrival Factor
            arrival_factor = 1.0
            if dist_to_goal < 10.0:
                 arrival_factor = dist_to_goal / 10.0
                 arrival_factor = max(0.0, min(1.0, arrival_factor))
            
            # Cornering Factor
            steer_ratio = abs(steer_deg) / self.max_steer
            corner_factor = 1.0 - (steer_ratio * 0.5)

            # 최종 목표 속도 계산
            desired_speed = self.target_speed * acc_factor * arrival_factor * corner_factor * danger_factor
            
            # 최저 속도 보장
            if desired_speed < self.min_speed_guarantee and acc_factor > 0.1 and arrival_factor > 0.1 and not is_side_danger:
                desired_speed = self.min_speed_guarantee
        else:
            self.traffic_state = STATE_STOP_WAIT
            desired_speed = 0.0

        # === 4. 속도 스무딩 ===
        speed_diff = desired_speed - self.current_speed_cmd
        if speed_diff > 0:
            self.current_speed_cmd += min(speed_diff, self.accel_limit * self.dt)
        else:
            self.current_speed_cmd -= min(-speed_diff, self.decel_limit * self.dt)
        self.current_speed_cmd = max(0.0, self.current_speed_cmd)

        cmd = Twist()
        cmd.linear.x = float(self.current_speed_cmd)
        cmd.angular.z = float(steer_deg)
        if self.current_speed_cmd < 0.1 and self.traffic_state == STATE_STOP_WAIT:
             cmd.linear.x = 0.0; cmd.linear.y = 1.0
        self.pub_cmd.publish(cmd)

        # 디버깅 로그
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
            if self.light_dist > 0: light_info += f"({self.light_dist:.1f}m)"

            print(f"[{state_str}] [{control_mode}] Spd:{self.current_speed_cmd:.1f} | Steer:{steer_deg:.1f} | "
                  f"Light:{light_info} | Obs:{obs_info}")

def main(args=None):
    rclpy.init(args=args)
    node = KooController()
    try: rclpy.spin(node)
    except: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()