cmake_minimum_required(VERSION 3.0.2)
project(perception)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  cv_bridge
  perception_msgs
)


catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES image_receiver
#  CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)



catkin_install_python(PROGRAMS
   scripts/perception_all_in_one.py
   scripts/airsim_ros_wrapper.py
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

catkin_python_setup()


