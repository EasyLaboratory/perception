import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from message_filters import Subscriber, ApproximateTimeSynchronizer
from perception.msg import GimbalControl
from pathlib import Path
from easyGL.airsim_gl import *
from model_loader import model
import socket
from easyGL.gimbalMessage import GimbalControlCode
from typing import *
import struct

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


class Gimbal:
    def __init__(self):
        # 初始化 ROS 节点
        rospy.init_node('Gimbal', anonymous=True)
        
        # 创建 CvBridge 对象
        self.bridge = CvBridge()
        
        # 创建两个话题的订阅者
        self.rgb_subscriber = Subscriber("camera/rgb/image", Image)
        self.gimbal_subsciber = Subscriber("/gimbal_control",GimbalControl)
        
        # 控制器相关参数设置,仅控制俯仰角和偏航角
        self.fov_horizontal = 90  # 水平视场角
        self.fov_vertical = 90    # 垂直视场角
        self.pid_yaw = PIDController(kp=0.5, ki=0.0, kd=0.5)
        self.pid_pitch = PIDController(kp=0.5, ki=0.0, kd=0.5)

        # UDPsender
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.target_ip = rospy.get_param("~target_ip","127.0.0.1")
        self.target_port = rospy.get_param("~target_port",50505)
        
        # 创建同步器，使用 ApproximateTimeSynchronizer 允许近似时间同步
        self.sync = ApproximateTimeSynchronizer([self.rgb_subscriber,self.gimbal_subsciber], 
                                                queue_size=10, slop=0.1)
        self.sync.registerCallback(self.synced_callback)
        rospy.on_shutdown(self.clean_up)

    def synced_callback(self, rgb_msg,gimbal_msg):
        try:
            # 转换 RGB 图像
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            results=track(model,rgb_image)
            # 如果找到目标进入跟踪模式，如果没找到目标进入云台漫游模式
            if len(results) == 1:
                cat2id2_xywhbox = get_target_category_box(results,[0])
                x,y,w,h = get_box(cat2id2_xywhbox,0,1)
                
                conf = get_conf(results,[0])
                if not conf or conf[0.0][1.0] < 0.4:
                    return
                else:
                    conf_label = conf[0.0][1.0]
                height = rgb_image.shape[0]
                width = rgb_image.shape[1]
                offset_x = x-width/2
                offset_y = -(y - height/2)
                yaw_error = offset_x/width*self.fov_horizontal
                pitch_error = offset_y/height*self.fov_vertical
                rospy.loginfo("--------------------------------")
                rospy.loginfo(f"yaw error: {yaw_error} pitch error: {pitch_error}")
                # 设置pid控制死区
                if yaw_error <2.0 and yaw_error>-2.0:
                    yaw_error = 0.0
                if pitch_error < 2.0 and pitch_error>-2.0:
                    pitch_error = 0.0
                yaw = self.pid_yaw.update(yaw_error)
                pitch = self.pid_pitch.update(pitch_error)
                
                self.send_gimbal_control((pitch,yaw),gimbal_msg)
            # else:
            #     self.send_gimbal_control((0,2),gimbal_msg)
            
        except Exception as e:
            rospy.logerr("Error processing synchronized images: %s", str(e))


    def send_gimbal_control(self,control:Tuple[float],gimbal_msg):
        pitch,yaw = control
        roll = 0.0
        gimbalControlArray = gimbal_msg.data
        speed = 25
        command = gimbalControlArray.data[GimbalControlCode.command.value]
        # rospy.loginfo("-----------------------------------")
        # rospy.loginfo(speed)
        # rospy.loginfo(command)
        # rospy.loginfo(pitch)
        # rospy.loginfo(yaw)
        # rospy.loginfo(f"{self.target_ip} {self.target_port}")
        message_content = struct.pack("!fffff", pitch, roll, yaw, speed, command)
        message_header = struct.pack("!I", len(message_content))
        udp_message = message_header + message_content
        self.sock.sendto(udp_message, (self.target_ip,self.target_port))
        rospy.loginfo("succ")
        
    
    def clean_up(self):
        self.sock.close()
        rospy.loginfo("the sock close")


    def run(self):
        # 保持 ROS 节点运行
        rospy.spin()

if __name__ == '__main__':
    receiver = Gimbal()
    receiver.run()