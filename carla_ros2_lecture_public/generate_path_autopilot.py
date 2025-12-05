#!/usr/bin/env python3
import glob
import os
import sys
import math
import time

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

HOST, PORT = "127.0.0.1", 2000

def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(10.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()
    
    # 차량 모델 선택
    bp = bp_lib.find("vehicle.tesla.model3")
    
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        print("No spawn points found!")
        return

    # [설정] 시작 위치
    start_index = 0 
    spawn = spawn_points[start_index]
    
    vehicle = None
    recorded_path = [] 
    min_distance = 0.5 

    try:
        # 1. 차량 스폰
        vehicle = world.spawn_actor(bp, spawn)
        vehicle.set_autopilot(True)
        
        # Traffic Manager 설정
        tm = client.get_trafficmanager()
        tm.vehicle_percentage_speed_difference(vehicle, 40.0) 
        
        print(f"Vehicle Spawned at Index {start_index}")
        print("Recording RELATIVE path... Press Ctrl+C to save and exit.")

        # 2. 카메라 설정
        spec = world.get_spectator()
        
        # [핵심] 시작점 좌표 저장 (상대 좌표 기준점)
        # 차량이 스폰된 직후의 정확한 위치를 잡기 위해 잠시 대기 후 가져옴
        time.sleep(0.5) 
        start_tf = vehicle.get_transform()
        start_x = start_tf.location.x
        start_y = start_tf.location.y
        
        print(f"Origin set to: ({start_x:.2f}, {start_y:.2f}) -> (0.0, 0.0)")

        prev_loc = None

        while True:
            tf = vehicle.get_transform()
            loc = tf.location
            
            # --- [핵심] 상대 좌표 변환 ---
            # 현재 위치에서 시작점 위치를 뺍니다.
            rel_x = loc.x - start_x
            rel_y = loc.y - start_y

            # --- 기록 로직 ---
            if prev_loc is None:
                dist = min_distance + 1.0 
            else:
                dist = math.hypot(loc.x - prev_loc.x, loc.y - prev_loc.y)

            if dist >= min_distance:
                # 변환된 상대 좌표(rel_x, rel_y)를 저장
                recorded_path.append((rel_x, rel_y))
                prev_loc = loc
                
                # 현재 좌표 출력
                print(f"\rRec: {len(recorded_path)} pts | Rel Pos: ({rel_x:.1f}, {rel_y:.1f})", end="")

            # --- 카메라 추적 ---
            f = tf.get_forward_vector()
            cam_loc = loc - f * 6.0
            cam_loc.z += 2.5
            rot = carla.Rotation(pitch=-10.0, yaw=tf.rotation.yaw)
            spec.set_transform(carla.Transform(cam_loc, rot))
            
            time.sleep(0.03)

    except KeyboardInterrupt:
        print("\n\nStopping recording...")

    finally:
        # 3. 파일 저장
        if recorded_path:
            filename = "global_path.csv"
            print(f"Saving {len(recorded_path)} points to {filename}...")
            
            with open(filename, "w") as f:
                for x, y in recorded_path:
                    # 상대 좌표가 저장됩니다.
                    f.write(f"{x},{y}\n")
            
            print("Save Complete! (Relative Coordinates)")
        
        # 4. 차량 삭제
        if vehicle is not None:
            print("Destroying vehicle...")
            vehicle.destroy()
            print("Done.")

if __name__ == "__main__":
    main()
