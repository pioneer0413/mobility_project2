#!/usr/bin/env python3
import glob
import os
import sys
import carla
import random
import time
import math

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

def main():
    # ==============================================================================
    # [사용자 설정] 장애물 위치 미세 조정 (Offset)
    # 형식: { 인덱스: (앞뒤_이동, 좌우_이동) }  단위: 미터(m)
    # 
    # 앞뒤(Forward): +가 앞, -가 뒤
    # 좌우(Right):   +가 오른쪽, -가 왼쪽
    # ==============================================================================
    
    # 예시: 82번 위치에서...
    # (0.0, 0.0) -> 원래 위치 그대로
    # (2.0, 0.0) -> 앞으로 2m 이동
    # (0.0, -1.5) -> 왼쪽으로 1.5m 이동 (중앙선 침범 유도 등)
    
    # 1. 차량 (Vehicle)
    target_vehicle_indices = {
        #120: (0.0, 0.0),   # 원래 위치
        #200: (5.0, 0.5)    # 앞으로 5m, 오른쪽으로 0.5m 이동
        65: (0.0, 0.0)
    }
    
    # 2. 고깔 (Cone)
    target_cone_indices = {
        #50: (0.0, 0.0),
        #51: (0.0, 1.0),    # 오른쪽으로 1m 비켜서 설치
        #52: (0.0, -1.0)    # 왼쪽으로 1m 비켜서 설치
    }
    
    # 3. 안전 펜스 (Barrier)
    target_barrier_indices = {} # 비워둘 때는 빈 딕셔너리 {}
    
    # 4. 자전거 (Cyclist)
    target_cyclist_indices = {
                           
    }

    # 5. 기타 장애물들 
    target_box_indices = {
        65: (0.0, 0.0),
        # 64: (0.0, 0.0)
    }
    target_barrel_indices = {
        223: (0.0, 0.0)
    }

    target_trash_indices = {

    }

    target_tire_indices = {

    }

    target_sign_indices = {
        
    }

    # ==============================================================================

    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()

    actor_list = []

    # 오프셋 적용 함수
    def apply_offset(transform, forward_offset, right_offset):
        # 현재 위치의 회전각(Yaw)을 라디안으로 변환
        yaw_rad = math.radians(transform.rotation.yaw)
        
        # 전진 벡터 계산
        fw_x = math.cos(yaw_rad)
        fw_y = math.sin(yaw_rad)
        
        # 우측 벡터 계산 (전진 벡터에서 90도 회전)
        r_x = math.sin(yaw_rad)
        r_y = -math.cos(yaw_rad) # CARLA 좌표계(Left-handed) 고려
        
        # 좌표 이동
        transform.location.x += (fw_x * forward_offset) + (r_x * right_offset)
        transform.location.y += (fw_y * forward_offset) + (r_y * right_offset)
        
        return transform

    # 장애물 생성 함수
    def spawn_obstacles(targets_dict, filter_pattern, type_name):
        if not targets_dict:
            return

        print(f"Spawning {type_name} ({filter_pattern})...")
        
        # 딕셔너리 순회: idx(번호), (f_off, r_off)(이동량)
        for idx, (f_off, r_off) in targets_dict.items():
            if idx >= len(spawn_points):
                print(f"  [Warning] Index {idx} is out of range. Skipping.")
                continue
            
            # 원본 위치 복사 (참조가 아닌 값 복사)
            original_tf = spawn_points[idx]
            transform = carla.Transform(original_tf.location, original_tf.rotation)
            
            # [핵심] 오프셋 적용
            transform = apply_offset(transform, f_off, r_off)
            
            # 블루프린트 검색
            blueprints = bp_lib.filter(filter_pattern)
            if not blueprints: continue

            if type_name == "Vehicle":
                blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
            
            bp = random.choice(blueprints)

            if bp.has_attribute('color'):
                color = random.choice(bp.get_attribute('color').recommended_values)
                bp.set_attribute('color', color)

            # 스폰 시도
            actor = world.try_spawn_actor(bp, transform)
            
            if actor:
                if type_name in ["Vehicle", "Cyclist"]:
                    actor.set_autopilot(False, 8005)
                    actor.set_simulate_physics(True)
                    control = carla.VehicleControl()
                    control.hand_brake = True
                    actor.apply_control(control)
                else:
                    actor.set_simulate_physics(True)
                
                actor_list.append(actor)
                print(f"  -> Spawned {bp.id} at Index {idx} (Offset: F={f_off}, R={r_off})")
            else:
                print(f"  -> Failed at Index {idx} (Collision?)")

    try:
        spawn_obstacles(target_vehicle_indices, 'vehicle.*', "Vehicle")
        spawn_obstacles(target_cyclist_indices, 'vehicle.bh.crossbike', "Cyclist")
        
        spawn_obstacles(target_cone_indices, 'static.prop.constructioncone', "Cone")
        spawn_obstacles(target_barrier_indices, 'static.prop.streetbarrier', "Barrier")
        
        spawn_obstacles(target_box_indices, 'static.prop.box*', "Box")
        spawn_obstacles(target_barrel_indices, 'static.prop.barrel', "Barrel")
        spawn_obstacles(target_trash_indices, 'static.prop.trashcan*', "TrashCan")
        spawn_obstacles(target_tire_indices, 'static.prop.tire', "Tire")
        spawn_obstacles(target_sign_indices, 'static.prop.warning*', "Sign")

        print(f"\nSuccessfully spawned total {len(actor_list)} obstacles.")
        print("Press Ctrl+C to remove obstacles and exit.")

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nRemoving obstacles...")
    finally:
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        print("Done.")

if __name__ == '__main__':
    main()
