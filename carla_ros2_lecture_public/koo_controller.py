#!/usr/bin/env python3
import math
import csv
import os
import json
import argparse
import time
import datetime
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from nav_msgs.msg import Path

STATE_DRIVE = 0
STATE_STOP_WAIT = 1

class KooController(Node):
    def __init__(self, path_num=1):
        super().__init__("koo_controller")
        
        self.path_num = path_num
        self.sub_gnss = self.create_subscription(NavSatFix, "/carla/hero/gnss", self.gnss_cb, 10)
        self.sub_path = self.create_subscription(Path, "/carla/path/local", self.local_path_cb, 10)
        self.sub_decision = self.create_subscription(String, "/fusion/decision", self.decision_cb, 10)
        self.pub_cmd = self.create_publisher(Twist, "/carla/hero/cmd_vel", 10)

        self.curr_xy = None; self.prev_xy = None; self.yaw = 0.0
        self.last_loop_xy = None 

        self.local_path = []; self.global_goal = None
        self.lat0 = None; self.lon0 = None; self.cos_lat0 = 1.0

        # 파라미터
        self.wheel_base = 2.7
        self.max_speed = 5.0
        self.current_speed_cmd = 0.0
        self.accel_limit = 1.5; self.brake_limit = 3.0; self.dt = 0.05
        self.min_lookahead = 3.0; self.max_lookahead = 7.0; self.lookahead_gain = 0.7 
        self.acc_speed = self.max_speed
        
        # [정지 거리 설정]
        self.dist_stop_car = 30.0        # 앞차 정지
        self.dist_slow_car = 60.0       # [NEW] 앞차 감속 시작 (20m)
        
        self.dist_stop_traffic = 30.0   # 신호등 정지
        self.dist_slow_traffic = 60.0   # 신호등 감속
        
        self.is_recovering = False; self.recovery_timer = 0; self.stuck_counter = 0; self.actual_speed = 0.0          
        self.load_global_goal(); self.is_goal_reached = False; self.log_counter = 0; self.debug_counter = 0

        self.traffic_state = STATE_DRIVE 
        self.last_traffic_dist = 999.0 
        self.last_traffic_type = "none" 
        
        self.stop_signal_count = 0      
        self.stop_signal_thresh = 30

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = f"driving_log_{ts}.csv"
        self.log_file = open(self.log_filename, 'w', newline='')
        self.csv_writer = csv.writer(self.log_file)
        self.csv_writer.writerow(["Time", "State", "CmdSpeed", "ActSpeed", "AccTarget", "ObjType", "ObjDist", "Steer", "Recovering"])

        self.timer = self.create_timer(self.dt, self.control_loop)
        self.get_logger().info(f">> Koo Controller (Path: {self.path_num}) Vehicle Slow Down @ 20m")

    def load_global_goal(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(current_dir, f"../path/global_path_{self.path_num}.csv")
        try:
            points = []
            with open(path, 'r') as f:
                for line in f:
                    parts = line.split(',')
                    if len(parts) >= 2 and not line.startswith('#'):
                        try: points.append((float(parts[0]), float(parts[1])))
                        except: pass
            if points: self.global_goal = points[-1]
        except: pass

    def gnss_cb(self, msg):
        if self.lat0 is None:
            self.lat0 = msg.latitude; self.lon0 = msg.longitude
            self.cos_lat0 = math.cos(math.radians(msg.latitude))
            x, y = 0.0, 0.0
        else:
            x = (msg.longitude - self.lon0) * (111320.0 * self.cos_lat0)
            y = (msg.latitude - self.lat0) * 110540.0
        self.prev_xy = self.curr_xy; self.curr_xy = (x, y)

    def local_path_cb(self, msg): self.local_path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]

    def decision_cb(self, msg: String):
        try:
            data = json.loads(msg.data)
            obj_type = data.get("obj", "none") 
            dist = float(data.get("dist", -1.0))
            
            self.debug_counter += 1
            if dist > 0 and self.debug_counter % 5 == 0:
                self.get_logger().info(f"[RECV] Obj: {obj_type}, Dist: {dist:.1f}m")

            if obj_type == "traffic_green":
                self.traffic_state = STATE_DRIVE
                self.last_traffic_type = "traffic_green"
                self.stop_signal_count = 0 
                self.acc_speed = self.max_speed
                if self.debug_counter % 10 == 0: self.get_logger().info("-> [GREEN] GO!")
                return

            stop_signals = ["traffic_red", "traffic_stop", "traffic_yellow"]

            if self.traffic_state == STATE_DRIVE:
                if obj_type in stop_signals:
                    self.stop_signal_count += 1
                    if self.stop_signal_count >= self.stop_signal_thresh:
                        self.traffic_state = STATE_STOP_WAIT
                        if dist > 0: self.last_traffic_dist = dist
                        else: self.last_traffic_dist = self.dist_stop_traffic - 1.0
                        self.last_traffic_type = obj_type
                        self.get_logger().warn(f"!!! STOP SIGNAL VERIFIED ({obj_type}) -> STOP_WAIT")
                else:
                    self.stop_signal_count = 0

            elif self.traffic_state == STATE_STOP_WAIT:
                if obj_type in stop_signals:
                    self.last_traffic_type = obj_type
                    if dist > 0: self.last_traffic_dist = dist

            # [상태별 속도 결정]
            if self.traffic_state == STATE_STOP_WAIT:
                d = self.last_traffic_dist
                if d < self.dist_stop_traffic: self.acc_speed = 0.0
                elif d < self.dist_slow_traffic:
                    ratio = (d - self.dist_stop_traffic) / (self.dist_slow_traffic - self.dist_stop_traffic)
                    self.acc_speed = max(0.0, self.max_speed * ratio)
                else: self.acc_speed = self.max_speed

            else: 
                # [수정] 일반 차량(장애물) 처리 로직 강화
                if obj_type == "vehicle":
                    self.last_traffic_type = "vehicle"
                    
                    if dist < self.dist_stop_car: # 4m 미만 정지
                        self.acc_speed = 0.0
                    
                    elif dist < 8.0: # 4~8m 서행 (Creep)
                        self.acc_speed = 2.0 
                    
                    elif dist < self.dist_slow_car: # 8~20m 감속
                        # 20m에서 max_speed, 8m에서 2.0이 되도록 선형 보간
                        ratio = (dist - 8.0) / (self.dist_slow_car - 8.0)
                        # Speed = Min + (Max - Min) * ratio
                        self.acc_speed = 2.0 + (self.max_speed - 2.0) * ratio
                    
                    else:
                        self.acc_speed = self.max_speed
                else:
                    self.acc_speed = self.max_speed

        except: pass

    def control_loop(self):
        self.log_counter += 1
        if self.curr_xy is None: return
        x, y = self.curr_xy
        
        if self.prev_xy:
            px, py = self.prev_xy
            dist_moved = math.hypot(x-px, y-py)
            if dist_moved > 0.02: self.yaw = math.atan2(y-py, x-px)
        
        if self.last_loop_xy is not None:
            lx, ly = self.last_loop_xy
            dist_per_tick = math.hypot(x - lx, y - ly)
            self.actual_speed = dist_per_tick / self.dt 
        self.last_loop_xy = (x, y)

        if self.traffic_state == STATE_STOP_WAIT:
            dist_moved_now = self.actual_speed * self.dt
            self.last_traffic_dist -= dist_moved_now

        cmd = Twist()
        if self.is_recovering:
            self.recovery_timer += 1
            if self.recovery_timer < 50:
                cmd.linear.x = -2.0; self.pub_cmd.publish(cmd); 
                self.save_log(cmd.angular.z); return
            else:
                self.is_recovering = False; self.stuck_counter = 0; self.current_speed_cmd = 0.0

        if self.current_speed_cmd > 0.5 and self.actual_speed < 0.1: self.stuck_counter += 1
        else: self.stuck_counter = 0
        if self.stuck_counter > 60:
            self.is_recovering = True; self.recovery_timer = 0
            self.get_logger().error("💥 STUCK -> Recovery"); return

        if self.global_goal and math.hypot(self.global_goal[0]-x, self.global_goal[1]-y) < 2.0:
            if not self.is_goal_reached: self.is_goal_reached = True
            self.pub_cmd.publish(Twist()); return

        if not self.local_path: self.pub_cmd.publish(Twist()); return

        target = None
        cos_y = math.cos(-self.yaw); sin_y = math.sin(-self.yaw)
        current_lookahead = max(self.min_lookahead, min(self.min_lookahead + (self.current_speed_cmd * self.lookahead_gain), self.max_lookahead))

        for lx, ly in self.local_path:
            dx = lx - x; dy = ly - y
            xl = dx*cos_y - dy*sin_y; yl = dx*sin_y + dy*cos_y
            if xl > 0 and math.hypot(xl, yl) >= current_lookahead: target = (xl, yl); break
        
        if target is None:
             lx, ly = self.local_path[-1]
             dx = lx - x; dy = ly - y
             xl = dx*cos_y - dy*sin_y; yl = dx*sin_y + dy*cos_y
             target = (xl, yl)

        if target:
            tx, ty = target
            ld = math.hypot(tx, ty)
            alpha = math.atan2(ty, tx)
            steer = math.atan2(2.0 * self.wheel_base * math.sin(alpha), ld)
            steer_deg = math.degrees(steer)
            
            corner_limit = max(1.5, self.max_speed - (abs(steer) * 8.0))
            target_v = min(float(self.acc_speed), corner_limit)

            if abs(steer_deg) > 5.0 and self.current_speed_cmd < 0.1 and not self.is_goal_reached: target_v = 2.0
            
            diff = target_v - self.current_speed_cmd
            
            if diff > 0:
                self.current_speed_cmd = min(self.current_speed_cmd + (self.accel_limit * self.dt), target_v)
            else:
                brake_force = self.brake_limit 
                if self.traffic_state == STATE_STOP_WAIT:
                    dist_to_stop = self.last_traffic_dist - self.dist_stop_traffic
                    if dist_to_stop > 0.1:
                        required_decel = (self.current_speed_cmd ** 2) / (2 * dist_to_stop)
                        if required_decel > self.brake_limit:
                            brake_force = min(required_decel, 10.0) 
                    else:
                        brake_force = 10.0

                self.current_speed_cmd = max(self.current_speed_cmd - (brake_force * self.dt), target_v)

            cmd.linear.x = self.current_speed_cmd; cmd.angular.z = float(steer_deg)
            if self.log_counter % 20 == 0:
                est_dist = self.last_traffic_dist if self.traffic_state == 1 else 0.0
                self.get_logger().info(f"CMD Spd:{cmd.linear.x:.1f}, ACC:{self.acc_speed:.1f}, State:{'STOP' if self.traffic_state==1 else 'DRIVE'} (EstDist:{est_dist:.1f}m)")
        
        self.pub_cmd.publish(cmd)
        self.save_log(cmd.angular.z)

    def save_log(self, steer):
        try:
            state_str = "STOP_WAIT" if self.traffic_state == STATE_STOP_WAIT else "DRIVE"
            dist_val = self.last_traffic_dist if self.traffic_state == STATE_STOP_WAIT else 0.0
            self.csv_writer.writerow([
                f"{time.time():.2f}", state_str, f"{self.current_speed_cmd:.2f}",
                f"{self.actual_speed:.2f}", f"{self.acc_speed:.2f}",
                self.last_traffic_type, f"{dist_val:.1f}", f"{steer:.2f}", self.is_recovering
            ])
        except: pass

    def destroy_node(self):
        if self.log_file: self.log_file.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_num', type=int, default=1)
    ros_args, _ = parser.parse_known_args()
    node = KooController(path_num=ros_args.path_num)
    rclpy.spin(node)
    node.destroy_node(); rclpy.shutdown()

if __name__ == "__main__": main()