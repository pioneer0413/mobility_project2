#!/usr/bin/env python3
import math
import numpy as np
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path
from geometry_msgs.msg import Twist

# ========== DWA 구현 ==========
def motion(x, u, dt):
    x[2] += u[1] * dt
    x[0] += u[0] * math.cos(x[2]) * dt
    x[1] += u[0] * math.sin(x[2]) * dt
    x[3] = u[0]
    x[4] = u[1]
    return x

def calc_dynamic_window(x, cfg):
    Vs = [cfg.min_speed, cfg.max_speed,
          -cfg.max_yaw_rate, cfg.max_yaw_rate]

    Vd = [x[3] - cfg.max_accel * cfg.dt,
          x[3] + cfg.max_accel * cfg.dt,
          x[4] - cfg.max_delta_yaw_rate * cfg.dt,
          x[4] + cfg.max_delta_yaw_rate * cfg.dt]

    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
          max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

    return dw

def predict_trajectory(x_init, v, y, cfg):
    x = np.array(x_init)
    traj = np.array(x)
    time = 0
    while time <= cfg.predict_time:
        x = motion(x, [v, y], cfg.dt)
        traj = np.vstack((traj, x))
        time += cfg.dt
    return traj

def calc_obstacle_cost(traj, obstacles, cfg):
    if obstacles.size == 0:
        return 0.0

    ox = obstacles[:, 0]
    oy = obstacles[:, 1]

    dx = traj[:, 0] - ox[:, None]
    dy = traj[:, 1] - oy[:, None]
    r = np.hypot(dx, dy)

    if np.array(r <= cfg.robot_radius).any():
        return float("inf")

    return 1.0 / np.min(r)

def calc_to_goal_cost(traj, goal):
    dx = goal[0] - traj[-1, 0]
    dy = goal[1] - traj[-1, 1]
    error_angle = math.atan2(dy, dx)
    yaw_err = error_angle - traj[-1, 2]
    return abs(math.atan2(math.sin(yaw_err), math.cos(yaw_err)))

def dwa_control(x, cfg, goal, obstacles):
    dw = calc_dynamic_window(x, cfg)

    min_cost = float("inf")
    best_u = [0.0, 0.0]
    best_traj = x

    for v in np.arange(dw[0], dw[1], cfg.v_resolution):
        for y in np.arange(dw[2], dw[3], cfg.yaw_rate_resolution):
            traj = predict_trajectory(x, v, y, cfg)

            cost_goal = cfg.to_goal_cost_gain * calc_to_goal_cost(traj, goal)
            cost_speed = cfg.speed_cost_gain * (cfg.max_speed - traj[-1, 3])
            cost_obs = cfg.obstacle_cost_gain * calc_obstacle_cost(traj, obstacles, cfg)

            cost = cost_goal + cost_speed + cost_obs

            if cost < min_cost:
                min_cost = cost
                best_u = [v, y]
                best_traj = traj

    return best_u, best_traj


# ========== DWA Config ==========
class DWAConfig:
    max_speed = 6.0
    min_speed = -0.0
    max_yaw_rate = 40.0 * math.pi / 180.0
    max_accel = 1.2
    max_delta_yaw_rate = 40.0 * math.pi / 180.0
    v_resolution = 0.1
    yaw_rate_resolution = 0.1 * math.pi / 180.0
    dt = 0.1
    predict_time = 2.0
    to_goal_cost_gain = 1.0
    speed_cost_gain = 1.0
    obstacle_cost_gain = 1.5
    robot_radius = 1.2


# ============================================================


class LocalDWAController(Node):
    def __init__(self):
        super().__init__("local_dwa_controller")

        self.current_path = None
        self.obstacles = np.zeros((0, 2))

        self.config = DWAConfig()
        self.L = 2.5  # 차량 wheelbase (steering 변환용)

        self.sub_path = self.create_subscription(Path, "/carla/path/local", self.path_cb, 10)
        self.pub_cmd = self.create_publisher(Twist, "/carla/hero/cmd_vel", 10)

        self.timer = self.create_timer(0.05, self.timer_cb)

    def path_cb(self, msg: Path):
        self.current_path = msg

    def timer_cb(self):
        if self.current_path is None or len(self.current_path.poses) < 5:
            return

        # 목표점 = local path 중간 지점 (더 안정적)
        mid = int(len(self.current_path.poses) * 0.6)
        goal_x = self.current_path.poses[mid].pose.position.x
        goal_y = self.current_path.poses[mid].pose.position.y
        goal = np.array([goal_x, goal_y])

        # 현재 차량 상태 미니멀하게 추정 (GNSS 없이도 동작 가능)
        # 실제 yaw는 /carla/hero/gnss 대신 TF나 odometry로 받는 것이 이상적이지만
        # 여기서는 간단히 local path 시작의 기울기 사용
        x0 = self.current_path.poses[0].pose.position.x
        y0 = self.current_path.poses[0].pose.position.y
        x1 = self.current_path.poses[1].pose.position.x
        y1 = self.current_path.poses[1].pose.position.y
        yaw = math.atan2(y1 - y0, x1 - x0)

        x_state = np.array([x0, y0, yaw, 0.0, 0.0])

        # 장애물 (없으면 np.zeros)
        obstacles = self.obstacles

        # === DWA 컨트롤 계산 ===
        u, traj = dwa_control(x_state, self.config, goal, obstacles)
        v, yaw_rate = u

        # === yaw_rate → steering angle 변환 (Ackermann) ===
        steer = math.atan2(self.L * yaw_rate, max(v, 0.1))
        steer_deg = math.degrees(steer)

        # === /cmd_vel publish ===
        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(steer_deg)
        cmd.linear.y = 0.0  # brake 없음
        self.pub_cmd.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = LocalDWAController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
