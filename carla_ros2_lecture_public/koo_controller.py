#!/usr/bin/env python3
import math
import csv
import os
import json
import argparse
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from nav_msgs.msg import Path

class KooController(Node):
    def __init__(self, path_num=1):
        super().__init__("koo_controller")
        
        self.path_num = path_num

        self.sub_gnss = self.create_subscription(
            NavSatFix, "/carla/hero/gnss", self.gnss_cb, 10)
        self.sub_path = self.create_subscription(
            Path, "/carla/path/local", self.local_path_cb, 10)
        self.sub_decision = self.create_subscription(
            String, "/fusion/decision", self.decision_cb, 10)

        self.pub_cmd = self.create_publisher(Twist, "/carla/hero/cmd_vel", 10)

        self.curr_xy = None
        self.prev_xy = None
        self.yaw = 0.0
        self.local_path = []
        self.global_goal = None
        self.lat0 = None
        self.lon0 = None
        self.cos_lat0 = 1.0

        # [튜닝 파라미터]
        self.wheel_base = 2.7
        self.max_speed = 5.0
        
        # [NEW] 물리 엔진 파라미터 (부드러운 주행 핵심)
        self.current_speed_cmd = 0.0  # 현재 내보내고 있는 속도 명령값
        self.accel_limit = 1.0        # 가속 제한 (m/s^2) - 낮을수록 부드럽게 출발
        self.brake_limit = 1.0        # 감속 제한 (m/s^2) - 높을수록 잘 섬
        self.dt = 0.05                # 제어 주기 (timer와 맞춰야 함)

        # 가변 룩어헤드
        self.min_lookahead = 3.0
        self.max_lookahead = 7.0
        self.lookahead_gain = 0.7 
        
        # ACC
        self.acc_speed = self.max_speed
        self.stop_dist = 4.0      
        self.creep_dist = 8.0     
        
        self.load_global_goal()
        self.is_goal_reached = False
        self.log_counter = 0

        self.timer = self.create_timer(self.dt, self.control_loop)
        self.get_logger().info(f">> Koo Controller (Path: {self.path_num}) Started with Smooth Physics")

    def load_global_goal(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        filename = f"../path/global_path_{self.path_num}.csv"
        path = os.path.join(current_dir, filename)
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

    def gnss_cb(self, msg: NavSatFix):
        lat = msg.latitude
        lon = msg.longitude
        if self.lat0 is None:
            self.lat0 = lat
            self.lon0 = lon
            self.cos_lat0 = math.cos(math.radians(lat))
            x, y = 0.0, 0.0
        else:
            x = (lon - self.lon0) * (111320.0 * self.cos_lat0)
            y = (lat - self.lat0) * 110540.0
        self.prev_xy = self.curr_xy
        self.curr_xy = (x, y)

    def local_path_cb(self, msg: Path):
        self.local_path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]

    def decision_cb(self, msg: String):
        try:
            data = json.loads(msg.data)
            dist = float(data.get("dist", -1.0))
            if dist < 0: self.acc_speed = self.max_speed
            elif dist < self.stop_dist: self.acc_speed = 0.0 
            elif dist < self.creep_dist: self.acc_speed = 2.0
            else: self.acc_speed = self.max_speed
        except: pass

    def control_loop(self):
        self.log_counter += 1
        if self.curr_xy is None: return

        x, y = self.curr_xy
        if self.prev_xy:
            px, py = self.prev_xy
            if math.hypot(x-px, y-py) > 0.02:
                self.yaw = math.atan2(y-py, x-px)
        
        if self.global_goal and math.hypot(self.global_goal[0]-x, self.global_goal[1]-y) < 2.0:
            if not self.is_goal_reached:
                self.get_logger().info("🎯 Goal Reached!")
                self.is_goal_reached = True
            self.pub_cmd.publish(Twist())
            return

        if not self.local_path:
            self.pub_cmd.publish(Twist())
            return

        target = None
        cos_y = math.cos(-self.yaw)
        sin_y = math.sin(-self.yaw)

        # 가변 룩어헤드 (현재 명령 속도 기반)
        current_lookahead = self.min_lookahead + (self.current_speed_cmd * self.lookahead_gain)
        current_lookahead = max(self.min_lookahead, min(current_lookahead, self.max_lookahead))

        for lx, ly in self.local_path:
            dx = lx - x
            dy = ly - y
            xl = dx*cos_y - dy*sin_y
            yl = dx*sin_y + dy*cos_y
            if xl > 0 and math.hypot(xl, yl) >= current_lookahead:
                target = (xl, yl)
                break
        
        if target is None and self.local_path:
            lx, ly = self.local_path[-1]
            dx = lx - x; dy = ly - y
            xl = dx*cos_y - dy*sin_y
            yl = dx*sin_y + dy*cos_y
            target = (xl, yl)

        cmd = Twist()
        if target:
            tx, ty = target
            ld = math.hypot(tx, ty)
            alpha = math.atan2(ty, tx)
            steer = math.atan2(2.0 * self.wheel_base * math.sin(alpha), ld)
            steer_deg = math.degrees(steer)
            
            # 1. 목표 속도 계산 (ACC vs 코너링)
            corner_limit = max(1.5, self.max_speed - (abs(steer) * 8.0))
            target_v = min(float(self.acc_speed), corner_limit)

            # 회피 기동 시 강제 속도
            if abs(steer_deg) > 5.0 and self.current_speed_cmd < 0.1 and not self.is_goal_reached:
                 target_v = 2.0
            
            # =========================================================
            # [NEW] 속도 스무딩 (Ramp Function)
            # 목표 속도로 한방에 점프하지 않고, 가속도 한계만큼만 변함
            # =========================================================
            diff = target_v - self.current_speed_cmd
            
            # 가속 상황인지 감속 상황인지 판단
            if diff > 0:
                # 가속: 초당 accel_limit 만큼만 증가
                step = self.accel_limit * self.dt
                self.current_speed_cmd = min(self.current_speed_cmd + step, target_v)
            else:
                # 감속: 초당 brake_limit 만큼만 감소 (브레이크는 좀 더 셈)
                step = self.brake_limit * self.dt
                self.current_speed_cmd = max(self.current_speed_cmd - step, target_v)

            cmd.linear.x = self.current_speed_cmd
            cmd.angular.z = float(steer_deg)
            
            if self.log_counter % 20 == 0:
                self.get_logger().info(
                    f"Spd: {cmd.linear.x:.2f} (Target: {target_v:.1f}), "
                    f"Str: {steer_deg:.1f}, LkHd: {current_lookahead:.1f}m"
                )
        
        self.pub_cmd.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_num', type=int, default=1, help='Path number (1, 2, ...)')
    ros_args, unknown_args = parser.parse_known_args()
    
    node = KooController(path_num=ros_args.path_num)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()