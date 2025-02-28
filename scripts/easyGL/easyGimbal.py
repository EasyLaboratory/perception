from enum import Enum,auto
import numpy as np
import rospy



class GimbalState(Enum):
    INITIAL = auto()    # 初始状态：回归初始位置
    SEARCH = auto()     # 搜索状态：旋转寻找目标
    TRACKING = auto()   # 跟踪状态：目标被检测后进入跟踪
    LOST = auto()       # 丢失状态：跟踪中目标突然丢失


class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp  # 比例系数
        self.ki = ki  # 积分系数
        self.kd = kd  # 微分系数
        self.prev_error = 0
        self.integral = 0

    def update(self, error):
        self.integral += error
        derivative = error - self.prev_error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output


class VisualServo:
    """
    Use airsim ENU coordinate,all the control points are under this coord
    """
    def __init__(self,image_width,image_height,desire_area_scale=0.5):
        # pid error
        self.desire_area = image_width*image_height*desire_area_scale
        self.error_x = 0
        self.error_y = 0
        self.error_z = 0

        # define pid controller for three axis
        self.pid_x = PIDController(kp=0.1, ki=0.01, kd=0.05)
        self.pid_y = PIDController(kp=0.1, ki=0.01, kd=0.05)
        self.pid_z = PIDController(kp=0.05, ki=0.005, kd=0.01)
        
        # drone current state
        self.current_position =[0,0,0]
        self.current_yaw = 0

        # time to calculate the dt
        self.last_time = 0
        self.current_drone_odom = None
        pass

    def control(self,odom_msg,gimbal_msg):
        current_time = rospy.Time.now()
        dt = current_time-self.last_time if self.last_time !=0 else 0.1
        
        gimbal_yaw_rad = gimbal_msg.data.data[2]
        gimbal_pitch_rad = gimbal_msg.data.data[0]
        rospy.loginfo("-----------------------------------")
        rospy.loginfo(gimbal_yaw_rad)
        rospy.loginfo(gimbal_pitch_rad)
        pass


