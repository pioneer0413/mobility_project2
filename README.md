# Mobility Team Project 2

## Objective
- 센서 융합의 객체 탐지 및 회피 기능을 가지는 자율 주행 차를 시뮬레이션 환경에서 구현

## Installation Guide
```
bash ros2_init.sh

source ~/.bashrc

# Download CARLA_0.9.16.tar.gz 
# at https://github.com/carla-simulator/carla/releases/tag/0.9.16/

bash carla_init.sh
```

## Usage

Note1: Make sure each script run on different terminal
<br>
Note2: Keep the order of script execution

```
# CARLA 실행 (Terminal 1)
bash carla/CarlaUE4.sh --ros2 -RenderOffScreen

# 맵/날씨 변경 (Terminal 2)
python3 carla_setup.py

# 시각화 프로그램 rviz 실행 (Terminal 3)
rviz2

# 자율 주행 차량 스폰 (Terminal 4) 
python3 koo.py -f lincoln.json

# 테스트용 차량/보행자 생성 (Terminal 4) 
python3 ros2_obstacles.py

# Lidar-Camera 데이터 처리 (Terminal 5) 
python3 cfusion.py

# Lidar 기반 장애물 감지 정보 전달 (Terminal 6)
python3 ros2_lidar_clustering.py

# 센서 데이터 기반 장애물 회피 지역 경로 생성 (Terminal 7) 
python3 local_planner.py

# 전역/지역 경로 수신 후 차량 제어 (Terminal 8) 
python3 controller.py
```