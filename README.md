# Mobility Team Project 2

## Objective
- 센서 융합의 객체 탐지 및 회피 기능을 가지는 자율 주행 차를 시뮬레이션 환경에서 구현

## Installation Guid
```
bash ros2_init.sh
source ~/.bashrc

# Download CARLA_0.9.16.tar.gz 
# at https://github.com/carla-simulator/carla/releases/tag/0.9.16/

bash carla_init.sh
```

## Usage
```
# CARLA 실행
bash carla/CarlaUE4.sh

# 맵/날씨 변경
python3 carla_setup.py

# rviz 실행
rviz2

# 자율 주행 세팅
python3 koo.py # Spawn Vehicle

python3 ros2_lidar_clustering.py # Perception

python3 cfusion.py # Perception

python3 koo_local_planner.py # Planning

python3 koo_controller.py # Control
```