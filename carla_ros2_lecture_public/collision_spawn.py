#!/usr/bin/env python3
import glob
import os
import sys
import time
import random
import logging
import math
import carla
from carla.command import SpawnActor, SetAutopilot, FutureActor, DestroyActor

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

# ==============================================================================
# 설정된 거리, 차선 오프셋 및 차량 모델
# ==============================================================================
DIST_LIST = [10.0, 14.0, 18.0, 22.0]  # 장애물 생성 후보 거리 (미터)
LAT_FACTORS = [0.0, 0.3, -0.3]       # 차선 폭 대비 좌우 오프셋 (0.0=중앙)
MODEL = "vehicle.audi.tt"            # 장애물 차량 모델
TM_PORT = 8010                      # Traffic Manager 포트

def get_hero(world):
    """
    현재 월드에서 'hero' 역할을 가진 차량을 찾거나, 
    없으면 첫 번째 차량을 반환하는 함수
    """
    for v in world.get_actors().filter("vehicle.*"):
        if v.attributes.get("role_name") == "hero":
            return v
    vs = world.get_actors().filter("vehicle.*")
    return vs[0] if len(vs) > 0 else None

def set_autopilot_speed(traffic_manager, vehicle, speed_percentage):
    """
    자율주행 차량의 속도를 설정하는 함수.
    :param traffic_manager: Traffic Manager 객체
    :param vehicle: 차량 객체
    :param speed_percentage: 목표 속도 비율 (100 = 기본 속도)
    """
    # 속도를 설정 (기본 속도의 `speed_percentage` 비율로)
    traffic_manager.vehicle_percentage_speed_difference(vehicle, speed_percentage)
    print(f"Setting vehicle speed to {speed_percentage}% of the max speed.")

def main():
    # CARLA 클라이언트 연결
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    m = world.get_map()
    bp_lib = world.get_blueprint_library()

    # 1. 주인공 차량(Hero) 찾기
    hero = get_hero(world)
    if not hero:
        print("Error: No vehicle found. Please spawn a vehicle first.")
        return
    print(f"Found hero vehicle: {hero.id}")

    # 2. 장애물 차량 블루프린트 설정
    if bp_lib.find(MODEL):
        bp = bp_lib.find(MODEL)
    else:
        bp = bp_lib.filter("vehicle.*")[0]  # 모델 없으면 아무거나
    
    bp.set_attribute("role_name", "obstacle")
    bp.set_attribute("color", "255,0,0")  # 빨간색

    # 3. 스폰 위치 계산 (Hero 차량 앞쪽 차선)
    h_loc = hero.get_location()
    
    # Hero 차량 위치에서 가장 가까운 웨이포인트(차선 중심) 찾기
    wp0 = m.get_waypoint(h_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    lane_w = wp0.lane_width if wp0 else 3.5

    cands = []  # 생성 후보 위치 리스트

    for d in DIST_LIST:
        # 현재 위치에서 d 미터 앞의 웨이포인트 가져오기
        wps = wp0.next(d) if wp0 else []
        if not wps: 
            continue
        
        base_wp = wps[0]
        base_tf = base_wp.transform
        
        # 바닥 충돌 방지를 위해 z축을 살짝 띄움
        base_tf.location.z += 0.5 
        
        # 차선 오른쪽 벡터 (좌우 이동 계산용)
        right = base_tf.get_right_vector()

        # 좌우 오프셋 적용하여 후보 추가
        for lf in LAT_FACTORS:
            # 위치 계산: 기준점 + (오른쪽벡터 * 차선폭 * 계수)
            loc = carla.Location(
                base_tf.location.x + right.x * (lane_w * lf),
                base_tf.location.y + right.y * (lane_w * lf),
                base_tf.location.z
            )
            # 회전은 차선 방향 그대로
            rot = carla.Rotation(pitch=0.0, yaw=base_tf.rotation.yaw, roll=0.0)
            
            cands.append(carla.Transform(loc, rot))

    # 4. 첫 번째 후보 위치에서 장애물 1대만 스폰
    obstacle = None  # 생성된 장애물 차량을 저장할 변수

    if cands:
        # 첫 번째 후보 위치에서 장애물 차량 생성 시도
        obstacle = world.try_spawn_actor(bp, cands[0])  # 첫 번째 위치에서만 생성
        if obstacle:
            print(f"Spawned obstacle at {cands[0].location}")
        else:
            print("Failed to spawn obstacle.")

    # 장애물이 성공적으로 생성되었으면
    if obstacle:
        # 5. 장애물 고정 (브레이크 설정)
        obstacle.set_autopilot(True)  # 자율주행 모드
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = 1.0
        control.hand_brake = True
        obstacle.apply_control(control)

        # 6. Traffic Manager로 속도 설정
        traffic_manager = client.get_trafficmanager(TM_PORT)
        set_autopilot_speed(traffic_manager, obstacle, 20.0) # 20%
    
    print("Only one obstacle spawned ahead of hero.")

    # -----------------------------
    # Ctrl+C가 눌렸을 때 차량 삭제 및 종료
    # -----------------------------
    try:
        while True:
            time.sleep(1)  # 계속 시뮬레이션을 진행
    except KeyboardInterrupt:
        print('\nCtrl+C detected. Destroying actors...')
        # 차량 삭제
        if obstacle:
            obstacle.destroy()  # 생성된 차량 삭제
        print("Obstacle destroyed.")
        return

if __name__ == "__main__":
    main()
