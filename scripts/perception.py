#!/home/rich/easyLab/src/perception/yolo_venv/bin/python3.8


import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge,CvBridgeError
from easyGL.airsim_gl import *
from pathlib import Path
from ultralytics import YOLO
from message_filters import Subscriber, ApproximateTimeSynchronizer
from nav_msgs.msg import Odometry
from easyGL.transform import quaternion_from_euler,Eular_angle
from geometry_msgs.msg import PointStamped
from  perception_msgs.msg import Row,Matrix
import signal
import airsim


client = None
current_dir = Path(__file__).resolve()
project_root = current_dir.parent.parent
model_base_path = project_root/"models"
model_path = model_base_path/"yolov10s_v1.pt"

model:ultralytics.YOLO = YOLO(model_path)

bridge = CvBridge()
annotated_frame_publisher = rospy.Publisher("/annotated_image",Image,queue_size=9)
track_data_publisher = rospy.Publisher("/perception_data",Matrix,queue_size=10)

camera_eular_angle = Eular_angle(pitch=-30,roll=0,yaw=0)
camera_translation = Translation(x=0.5,y=0,z=0)


K=[320.0, 0.0, 320.0, 0.0, 320.0, 240.0, 0.0, 0.0, 1.0]

camera_intrinsic_matrix = construct_inverse_intrinsic_with_k(K)


def get_uv_depth(cv_depth:np.ndarray,u,v):
    return cv_depth[v][u]


def connect2client(ip):
    global client
    if client is not None:
        rospy.logwarn("the client already exists")
    client = airsim.MultirotorClient(ip=ip)
    client.confirmConnection()
    client.enableApiControl(True)
    rospy.loginfo("init the client")


def get_linear_velocity(current_position,current_time:rospy.Time):
    global previous_position
    global previous_time
    if previous_position is None and current_position is not None:
        previous_position = current_position
        previous_time = rospy.Time.now()
        return np.full((3,),np.nan)
    elif previous_position is not None and previous_time is not None:
        time_diff = (current_time-previous_time).to_sec()
        if time_diff > 0:
            # 计算位置差
            position_diff = current_position-previous_position
            linear_velocity = position_diff/time_diff
            return linear_velocity
    else:
        return np.full((3,),np.nan)
    

        



def perception_callback(rgb_msg:Image,depth_msg:Image,odemetry_msg:Odometry):
    global annotated_frame_publisher
    global bridge
    global model
    try:
        # Convert the ROS Image message to a format OpenCV can work with
        cv_image = bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='passthrough')
        results=track(model,cv_image)
        for result in results:
            result_tensor = result.boxes.data
            matrix = [Row(data=row_data) for row_data in result_tensor]
            perception_data = Matrix(matrix=matrix)
            track_data_publisher.publish(perception_data)
        
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))



def signal_handler(sig, frame):
    print('Shutting down gracefully')
    rospy.signal_shutdown('Signal received')

signal.signal(signal.SIGINT, signal_handler)
 


def sensor_perception():
   
    rospy.init_node('perception_node', anonymous=True)

    remote_ip = rospy.get_param("~remote_ip","127.0.0.1")
    connect2client(remote_ip)
    client.armDisarm(True)
    client.takeoffAsync().join()
    client.moveToPositionAsync(0, 0, -6, 3).join()
    client.hoverAsync().join()
    rospy.loginfo("take off succ")

    # rgb image in camera_1
    vehicle_name = rospy.get_param("/vehicle_name", "drone_1")
    rgb_camera = "camera_1"
    camera_type_scene = "Scene"
    rgb_topic = f"/airsim_node/{vehicle_name}/{rgb_camera}/{camera_type_scene}"

    # depth image in camera_2
    depth_camera = "camera_2"
    camera_type_depth = "DepthPlanar"
    depth_topic = f"/airsim_node/{vehicle_name}/{depth_camera}/{camera_type_depth}"

    # drone odemetry
    odom_local_enu = "odom_local_enu"
    odemetry_topic = f"/airsim_node/{vehicle_name}/{odom_local_enu}"

    rgb_sub = Subscriber(rgb_topic,Image)
    depth_sub = Subscriber(depth_topic,Image)
    odemetry_sub = Subscriber(odemetry_topic,Odometry)


    ats = ApproximateTimeSynchronizer([rgb_sub, depth_sub,odemetry_sub], queue_size=20, slop=0.1)
    ats.registerCallback(perception_callback)
    rospy.spin()




if __name__ == "__main__":
    sensor_perception()