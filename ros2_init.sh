#!/bin/bash

# locale setup
locale
sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
locale

# ros2 apt repository setup
sudo apt install software-properties-common -y
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}')
curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb"
sudo dpkg -i /tmp/ros2-apt-source.deb

# ros2 package installation
sudo apt update && sudo apt upgrade -y
sudo apt install ros-humble-desktop -y
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc

# ros2 작업 폴더 생성 및 예제 실행
sudo apt install python3-colcon-common-extensions -y
cd ~/mobility_project2
mkdir -p lecture_ws/src
cd lecture_ws
colcon build
echo "source ~/mobility_project2/lecture_ws/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
cd ~/mobility_project2/lecture_ws/src

ros2 --help
# 만약 ros2 명령어를 찾을 수 없다는 에러 발생 시
# source ~/.bashrc