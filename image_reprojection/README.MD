
## Steps To Run Code
1. roscore
2. roslaunch realsense2_camera rs_camera.launch     align_depth:=true     unite_imu_method:="linear_interpolation"     enable_gyro:=true      enable_accel:=true
3. rosrun imu_filter_madgwick imu_filter_node     _use_mag:=false     _publish_tf:=false     _world_frame:="enu"     /imu/data_raw:=/camera/imu     /imu/data:=/rtabmap/imu
4. roslaunch rtabmap_ros rtabmap.launch     rtabmap_args:="--delete_db_on_start --Optimizer/GravitySigma 0.3"     depth_topic:=/camera/aligned_depth_to_color/image_raw     rgb_topic:=/camera/color/image_raw     camera_info_topic:=/camera/color/camera_info     approx_sync:=false     wait_imu_to_init:=true     imu_topic:=/rtabmap/imu
5. rostopic echo /rtabmap/odom  > ./odom.txt
6. python3 hyspimManager.py
7. python3 odomProcessorQuat.py

## Requirements
1. Ubuntu 20.04
2. Ros noetic
3. rtab map ros version for noetic
4. ros realsense wrapper
5. ros madgwick imu filtering
6. opencv and basic preinstalled python packages (should come with ros)