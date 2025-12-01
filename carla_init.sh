#!/bin/bash

# Download and set up CARLA simulator
cd ~/mobility_project2/
mkdir carla
cd carla
#
# Download manually CARLA 0.9.16 (https://github.com/carla-simulator/carla/releases/tag/0.9.16/)
#
tar -xvzf carla-0-9-16-linux.tar.gz

# Set up CARLA Python API
sudo apt update 
sudo apt install -y build-essential g++-12 cmake ninja-build libvulkan1 python3 python3-dev python3-pip python3-venv autoconf wget curl rsync unzip git git-lfs libpng-dev libtiff5-dev libjpeg-dev

# CARLA 실행
#./CarlaUE4.sh --ros2 -RenderOffScreen