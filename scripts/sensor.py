import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from message_filters import Subscriber, ApproximateTimeSynchronizer
from geometry_msgs.msg import PointStamped
from perception.msg import GimbalControl
from easyGL.airsim_gl import *
from model_loader import model
import socket
from typing import *
import struct
from easyGL.easyGimbal import GimbalState
from easyGL.easyGimbal import PIDController
from easyGL.easyGimbal import VisualServo
from easyGL.airsim_gl import publish_point_msg
from easyGL.airsim_gl import publish_annotated_image
from easyGL.airsim_gl import get_detect_target






class DroneSensor:
    def __init__(self):
        # create ROS node and cv object
        rospy.init_node('Gimbal', anonymous=True)
        self.bridge = CvBridge()
        
        # subscribe the sensor message
        self.depth_subscriber = Subscriber("/camera/depth/image",Image)
        self.rgb_subscriber = Subscriber("camera/rgb/image", Image)
        self.gimbal_subscriber = Subscriber("/gimbal_control",GimbalControl)
        self.odemetry_subscriber = Subscriber("/airsim_node/drone_1/odom_local_enu",Odometry)

        # topic to analyse the error
        # self.target_position_truth = Subscriber("/easysim_ros_wrapper/player_odom",Odometry)
        
        # Gimbal PID Controller
        self.state = GimbalState.INITIAL
        self.fov_horizontal = 90  
        self.fov_vertical = 90    
        self.pid_yaw = PIDController(kp=0.1, ki=0.0, kd=0.05)
        self.pid_pitch = PIDController(kp=0.1, ki=0.0, kd=0.05)

        # target track data to instruct the search strategy
        self.yaw_error_history_val = 0.0
        self.pitch_error_histoty_val = 0.0

        # camera setting
        self.K = [640.0, 0.0, 640.0, 0.0, 640.0, 360.0, 0.0, 0.0, 1.0]
        self.camera_intrinsic_matrix = construct_inverse_intrinsic_with_k(self.K)
        self.camera_eular_angle = Eular_angle(pitch=0,roll=0,yaw=0)
        self.camera_translation = Translation(x=0,y=0,z=0)

        # UDPsender
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.target_ip = rospy.get_param("~target_ip","127.0.0.1")
        self.target_port = rospy.get_param("~target_port",50505)

        # topic to control the drone
        self.annotated_frame_publisher = rospy.Publisher("/annotated_image",Image,queue_size=9)
        self.odom_publisher = rospy.Publisher('/target/odom_airsim', Odometry, queue_size=10)
        self.point_publisher = rospy.Publisher('points', PointStamped, queue_size=10)

        # temp var
        self.track_count = 0
        self.lost_frame = 0

        # visual servo controller
        self.visual_servo = VisualServo(50,50)

        self.sync = ApproximateTimeSynchronizer([self.rgb_subscriber,self.depth_subscriber,self.gimbal_subscriber,
                                                 self.odemetry_subscriber], 
                                                queue_size=5, slop=0.05)
        self.sync.registerCallback(self.synced_callback)
        rospy.on_shutdown(self.clean_up)

    def synced_callback(self, rgb_msg,depth_msg,gimbal_msg,odemetry_msg):
        try:
            # 转换 RGB 图像
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            results=track(model,rgb_image)
            self.update_camera_eula_angle(gimbal_msg)

            if self.state == GimbalState.INITIAL:
                rospy.loginfo("------------in initial state-------------")
                if self.detect_target(results):
                    self.transition_to(GimbalState.TRACKING)
                self.reset_to_initial()
                self.transition_to(GimbalState.SEARCH)
                
            elif self.state == GimbalState.SEARCH:
                rospy.loginfo("------------in search state-------------")
                self.set_tracking_strategy()
                if self.detect_target(results):
                    self.transition_to(GimbalState.TRACKING)
            elif self.state == GimbalState.TRACKING:
                rospy.loginfo("------------in tracking state-------------")
                if self.detect_target(results):
                    self.visual_servo.control(odemetry_msg,gimbal_msg)
                    self.gimbal_track_target(results,rgb_image)
                    self.sensor_controller(results,depth_msg,odemetry_msg)
                    self.track_count += 1
                else:
                   self.transition_to(GimbalState.LOST)
            elif self.state == GimbalState.LOST:
                rospy.loginfo("------------in lost state-------------")
                self.track_count = 0
                if self.detect_target(results):
                    self.transition_to(GimbalState.TRACKING)
                self.transition_to(GimbalState.SEARCH)
        except Exception as e:
            pass
    
    def detect_target(self,results):
        if len(results) == 1:
            conf = get_conf(results,[0])
            if not conf or conf[0.0][1.0] < 0.4:
                return False
            else:
                return True
        return False
    
    def visual_servo_track_target(self,odo,gimbal_msg):
        VisualServo.control(gimbal_msg)
        


    def gimbal_track_target(self,results,rgb_image):
        cat2id2_xywhbox = get_target_category_box(results,[0])
        x,y,w,h = get_box(cat2id2_xywhbox,0,1)
        height = rgb_image.shape[0]
        width = rgb_image.shape[1]
        offset_x = x-width/2
        offset_y = -(y - height/2)
        yaw_error = offset_x/width*self.fov_horizontal
        pitch_error = offset_y/height*self.fov_vertical
        self.yaw_error_history_val = yaw_error
        self.pitch_error_histoty_val = pitch_error
        # 设置pid控制死区
        if yaw_error <2.0 and yaw_error>-2.0:
            yaw_error = 0.0
        if pitch_error < 2.0 and pitch_error>-2.0:
            pitch_error = 0.0
        yaw = self.pid_yaw.update(yaw_error)
        pitch = self.pid_pitch.update(pitch_error)
        self.send_gimbal_control((pitch,yaw))
    

    def send_gimbal_control(self,control:Tuple[float]):
        pitch,yaw = control
        roll = 0.0
        speed = 25
        command = 1.0
        message_content = struct.pack("!fffff", pitch, roll, yaw, speed, command)
        message_header = struct.pack("!I", len(message_content))
        udp_message = message_header + message_content
        self.sock.sendto(udp_message, (self.target_ip,self.target_port))
        rospy.loginfo("succ")  

    def reset_to_initial(self):
        message_content = struct.pack("!fffff", 0.0, 0.0, 0.0, 25, 0)
        message_header = struct.pack("!I", len(message_content))
        udp_message = message_header + message_content
        self.sock.sendto(udp_message, (self.target_ip,self.target_port))

    def transition_to(self,new_state):
        self.state = new_state
        
    def sensor_controller(self,results,depth_image,odometry_msg):
        if len(results) == 1:
            world_point_ENU,conf_label = get_detect_target(self.bridge,results,odometry_msg,
                                                           depth_image,self.camera_intrinsic_matrix,
                                                           self.camera_eular_angle,self.camera_translation)
            publish_annotated_image(self.bridge,self.annotated_frame_publisher,results,conf_label)
            # publish point msg for rviz debug
            publish_point_msg(self.point_publisher,world_point_ENU,odometry_msg)
            # publish odometry message for planner
            linear_velocity = self.get_linear_velocity(world_point_ENU,rospy.Time.now())
            publish_odometry_msg(self.odom_publisher,world_point_ENU,odometry_msg,linear_velocity,"drone_1")
    
    def get_linear_velocity(self,current_position,current_time:rospy.Time):
        if self.previous_position is None and current_position is not None:
            self.previous_position = current_position
            self.previous_time = rospy.Time.now()
            return np.full((3,),np.nan)
        elif self.previous_position is not None and self.previous_time is not None:
            time_diff = (current_time-self.previous_time).to_sec()
            if time_diff > 0:
                # 计算位置差
                position_diff = current_position-self.previous_position
                linear_velocity = position_diff/time_diff
                return linear_velocity
        else:
            return np.full((3,),np.nan)
        
    def analyse_error():
        pass


    def set_tracking_strategy(self):
        if self.yaw_error_history_val > 0:
            self.send_gimbal_control((0,4))
        else:
            self.send_gimbal_control((0,-4))
    
    def update_camera_eula_angle(self,gimbal_msg):
        self.camera_eular_angle.pitch = gimbal_msg.data.data[0]
        self.camera_eular_angle.roll = gimbal_msg.data.data[1]
        self.camera_eular_angle.yaw = gimbal_msg.data.data[2]

    def clean_up(self):
        self.sock.close()
        rospy.loginfo("the sock close")


    def run(self):
        # 保持 ROS 节点运行
        rospy.spin()

if __name__ == '__main__':
    sensor = DroneSensor()
    sensor.run()