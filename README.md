Reactive Car
=============

This repository provides a package containing an adaptive and reactive neural controller for an Ackermann car system.

### Prerequisites
- ackermann_msgs
- mocap4r2_msgs
- pytorch
- casadi
- F1Tenth Gym: Follow instructions here: [https://github.com/f1tenth/f1tenth_gym ](https://github.com/f1tenth/f1tenth_gym_ros/tree/dev-dynamics)

```
sudo apt install ros-${ROS_DISTRO}-ackermann-msgs
git clone https://github.com/MOCAP4ROS2-Project/mocap4r2_msgs
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install casadi
```

### Install and Build Package:
```
git clone https://github.com/erisa-arise/MEAM6230_underactuated_car_go_vroom.git

colcon build
source install/setup.bash
```

### Notes on running the package:
To launch the neural controller, first run the neural controller node:

```
ros2 run reactive_car neural_controller.py
```

Initialize the car somewhere along the nominal trajectory, then generate the nominal trajectory:

```
ros2 service call /generate_nominal_trajectory reactive_car/srv/GenerateNominalTrajectory
```

The car will immediately begin to follow the generated nominal trajectory.


### Notes on data:
The data/ directory includes all the demonstrations collected as well as several ros2 bags. The /utils directory contains several processing scripts that were used to parse the ros2 bags as well as recreate the plots shown in the report.
