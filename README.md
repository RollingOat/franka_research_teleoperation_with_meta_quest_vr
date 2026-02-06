# Franka VR Robot Teleoperation

We modify the original code for it to work with VR controllers rather than hand tracking for better controllability and tracking accuracy.

Please refer to https://github.com/wengmister/franka-vr-teleop for teleoperation with hand tracking. The README is partly modified to include the original author's readme. Thanks the original authors for their work!


## Architecture

```
VR Headset → UDP → ROS1_work_space_in_docker -> UDP → ROS2 Workstation → UDP → VR Robot Client (Realtime PC) → libfranka → Franka Robot
(controller tracking)     (send controller pose data)                     (IK+Ruckig trajectory generation)  (joint velocity control)
```

## Tested Environment
1. OS: ubuntu24.04
2. ROS2 Jazzy
3. ROS1 Noetic in docker

## System Components

1. **VR Headset**: Streams hand tracking data via UDP (port 8888)
2. **ROS2 Node**: Wrist vector visualization, frame conversion, and input data smoothing.
3. **VR Robot Client**: Real-time system with kinematic motion control:
   - **Weighted IK**: Optimizes joint configurations for manipulability, smoothness, and base stability. This is an implementation adapted from [PC Lopez-Custodio et al.'s amazing work, GeoFIK](https://github.com/PabloLopezCustodio/GeoFIK).
   - **Ruckig Trajectory Generator**: Provides jerk-limited, time-optimal motion profiles 
   - **Joint-Space Velocity Control**: Direct velocity commands for responsive control
   - **Real-time Processing**: 1kHz control loop with <1ms trajectory calculations
4. **Franka Robot**: Motion profile output and controlled via `libfranka`

## Setup Instructions

### Prerequisites

- **libfranka**: For Franka robot control. You will need a version corresponding to your firmware version.
- **Ruckig**: For trajectory generation (`sudo apt install libruckig-dev` or build from [source](https://github.com/pantor/ruckig))
- **Eigen3**: For linear algebra (`sudo apt install libeigen3-dev`)
- **ROS2**: For frame conversion, data smoothing and wrist vector visualization.
- **ADB**: For deploying app .apk, and optionally stream data via USB through `adb reverse`.

### VR Robot Client Setup

#### Structure:
```
vr_robot_client/
├── CMakeLists.txt
├── include/
│   ├── examples_common.h
│   ├── geofik.h           # Geometric inverse kinematics
│   └── weighted_ik.h      # Weighted IK solver with optimization and stabilization
├── src/
│   ├── examples_common.cpp
│   ├── geofik.cpp         # Franka analytical IK
│   ├── weighted_ik.cpp    # Multi-criteria IK optimization
│   └── franka_vr_control_client.cpp  # Main VR control system
└── build/
```

#### Build:
```bash
mkdir vr_robot_client/build && cd vr_robot_client/build
cmake -DFRANKA_INSTALL_PATH=/your/path/to/libfranka/install ..
make -j4

# Run the VR control client on your realtime workstation
./franka_vr_control_client <robot-hostname>
```

#### Notes:
- **libfranka**: Real-time robot control interface
- **Ruckig**: Time-optimal trajectory generation with jerk constraints
- **geofik**: Custom geometric inverse kinematics library for Franka
- **weighted_ik**: Multi-objective IK solver optimizing manipulability, joint limits, and base stability

### ROS2 Node:

```bash

rosdep update && rosdep install --from-paths src --ignore-src -r -y

colcon build --packages-select franka_vr_teleop
```

### VR Headset Setup

TODO: to be explained

## Usage

### 1. Start VR Robot Client

```bash
cd vr_robot_client/build
./franka_vr_control_client <robot-hostname> [bidexhand]
```

When [bidexhand] is set to `true`, IK solver will limit the joint range of J7 to prevent damaging the servo sleeve attachment. Argument currently defaults to true.

### 2. Start VR Application

Put on your VR headset and start the application `arm tracker` that streams hand tracking data to the ROS2 node.

### 3. Launch ROS2 Node

```bash
# on ROS2 workstation
. install/setup.bash
ros2 launch franka_vr_teleop vr_control.launch.py
```

The robot teleop will be live now!
