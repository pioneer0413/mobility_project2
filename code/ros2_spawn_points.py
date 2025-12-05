#!/usr/bin/env python3
import sys
import os
import glob

# ==============================================================================
# CARLA 모듈 경로 설정
# ==============================================================================
try:
    sys.path.append(glob.glob('../../../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import rclpy
from rclpy.node import Node
import matplotlib.pyplot as plt

class SpawnPointViewer(Node):
    def __init__(self):
        super().__init__('spawn_point_viewer')
        
        try:
            self.host = '127.0.0.1'
            self.port = 2000
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(20.0)
            
            world = self.client.get_world()
            if "Town01" not in world.get_map().name:
                self.get_logger().info("Loading Town01 map...")
                self.client.load_world('Town01')
                world = self.client.get_world()
            
            self.world = world
            self.map = self.world.get_map()
            self.get_logger().info(f"Connected to CARLA. Map: {self.map.name}")

        except Exception as e:
            self.get_logger().error(f"Connection Failed: {e}")
            return

        self.spawn_points = self.map.get_spawn_points()
        self.plot_map_and_points()

    def plot_map_and_points(self):
        if os.environ.get('DISPLAY', '') == '':
            self.get_logger().warn("No display found.")
            return

        self.get_logger().info("Plotting map... Zoom in to check IDs.")

        # 1. 지도 데이터 준비
        topology = self.map.get_topology()
        ox, oy = [], []
        for wp1, wp2 in topology:
            l1, l2 = wp1.transform.location, wp2.transform.location
            ox.append(l1.x); oy.append(l1.y)
            ox.append(l2.x); oy.append(l2.y)
            ox.append(None); oy.append(None)

        # 2. 스폰 포인트 데이터 준비
        sx = [sp.location.x for sp in self.spawn_points]
        sy = [sp.location.y for sp in self.spawn_points]

        # 3. 그리기
        plt.figure(figsize=(16, 16))
        
        # 도로 그리기
        plt.plot(ox, oy, "k-", linewidth=0.5, alpha=0.3, label="Roads")
        
        # 스폰 포인트 점 찍기
        plt.scatter(sx, sy, c='blue', s=15, marker='o', alpha=0.6, label="Spawn Points")

        # [핵심 수정] 모든 텍스트를 하단(Bottom)에 배치 + 오프셋 15
        for i, (x, y) in enumerate(zip(sx, sy)):
            
            offset_y = -5.0  # y축 방향으로 15만큼 아래로 내림
            
            # 텍스트 표시 (가로 정렬: 중앙, 세로 정렬: 위쪽)
            plt.text(x, y + offset_y, str(i), fontsize=7, color='red', fontweight='bold', 
                     ha='center', va='top', clip_on=True)
            
            # 점과 텍스트를 잇는 세로선 추가
            plt.plot([x, x], [y, y + offset_y], 'r-', linewidth=0.3, alpha=0.5)

        # 시작점(0번) 강조
        if len(sx) > 0:
            plt.plot(sx[0], sy[0], "g*", markersize=15, label="Index 0")

        plt.gca().invert_xaxis() # 좌우 반전 유지
        plt.title(f"Spawn Points (All Labels at Bottom)")
        plt.grid(True)
        plt.legend()
        plt.axis("equal")
        
        print("맵 생성 완료. 확대해서 번호를 확인하세요.")
        plt.show()

def main():
    rclpy.init()
    node = SpawnPointViewer()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
