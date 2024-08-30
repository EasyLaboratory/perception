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
import signal
import airsim
from perception_msgs.msg import SyncedImg


client = None
current_dir = Path(__file__).resolve()
project_root = current_dir.parent.parent
model_base_path = project_root/"models"
model_path = model_base_path/"yolov10s_v1.pt"

model:ultralytics.YOLO = YOLO(model_path)

bridge = CvBridge()
annotated_frame_publisher = rospy.Publisher("/annotated_image",Image,queue_size=9)
odom_publisher = rospy.Publisher('/target/odom_airsim', Odometry, queue_size=10)
point_publisher = rospy.Publisher('points', PointStamped, queue_size=10)

camera_eular_angle = Eular_angle(pitch=0,roll=0,yaw=0)
camera_translation = Translation(x=0,y=0,z=0)


K=[640.0, 0.0, 640.0, 0.0, 640.0, 360.0, 0.0, 0.0, 1.0]


camera_intrinsic_matrix = construct_inverse_intrinsic_with_k(K)

previous_position = None
previous_time:rospy.Time = None


target_x=0
target_y=0

drone_yaw = 0
current_yaw = 0



def calculate_yaw(drone_pos, target_pos):
    if len(drone_pos)<2 or len(target_pos)<2:
        return 0 
    
    dx = target_pos[0] - drone_pos[0]
    dy = target_pos[1] - drone_pos[1]

    yaw = math.atan2(dy, dx)

    return  90-math.degrees(yaw)


def pub_cmd(event):
    client.moveToPositionAsync(target_y,target_x, -6, 10, 5,yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=drone_yaw))


def get_uv_depth(cv_depth:np.ndarray,u,v):
    return cv_depth[v][u]

def construct_point_msg_array(point):
    point_msg = PointStamped(point[0],point[1],point[2])
    return point_msg


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
    

        



def perception_callback(synced_msg:SyncedImg,odemetry_msg:Odometry):
    global annotated_frame_publisher
    global bridge
    global model
    try:
        # Convert the ROS Image message to a format OpenCV can work with
        cv_image = bridge.imgmsg_to_cv2(synced_msg.rgb_image, desired_encoding='passthrough')
        results=track(model,cv_image)
        if len(results) == 1:
            cat2id2_xywhbox = get_target_category_box(results,[0])
            x,y,w,h = get_box(cat2id2_xywhbox,0,1)
            
            conf = get_conf(results,[0])
            if not conf or conf[0.0][1.0] < 0.8:
                return
            else:
                conf_label = conf[0.0][1.0]
            
            cat2id2_xyxybox = get_target_category_box(results,[0],box_type="xyxy")
            x1,y1,x2,y2 = get_box(cat2id2_xyxybox,0,1)
            annotated_image=get_annotated_image(results,"car",conf_label,x1,y1,x2,y2)
            if annotated_image:
                first_image = annotated_image[0]
                # Convert the processed image (result) back to a ROS Image message
                annotated_msg = bridge.cv2_to_imgmsg(first_image, encoding='bgr8')

                # Publish the annotaprint(a)ted target
                annotated_frame_publisher.publish(annotated_msg)
            

            if x!=-1 and y!=-1:
                cv_depth = bridge.imgmsg_to_cv2(synced_msg.depth_image,desired_encoding="passthrough")
                depth = get_uv_depth(cv_depth,x,y)
                t = odemetry_msg.pose.pose.position
                t_array = np.array([t.x,t.y,t.z])
                o = odemetry_msg.pose.pose.orientation
                o_array = np.array([o.w,o.x,o.y,o.z])
                extrinsic_matrix = construct_extrinsic_with_quaternion(o_array,t_array)
                world_point_ENU =unproject(x,y,depth,camera_intrinsic_matrix,camera_eular_angle,camera_translation,extrinsic_matrix)

                res_point = PointStamped()
                res_point.header.stamp = odemetry_msg.header.stamp
                res_point.header.frame_id = odemetry_msg.header.frame_id
                res_point.point.x = world_point_ENU[0]
                res_point.point.y = world_point_ENU[1]
                res_point.point.z = world_point_ENU[2]
                point_publisher.publish(res_point)

                # point_publisher.publish(res_point)
                odo_msg = Odometry()
                odo_msg.header.stamp = odemetry_msg.header.stamp
                odo_msg.header.frame_id = "drone_1"
                odo_msg.pose.pose.position.x = world_point_ENU[0]
                odo_msg.pose.pose.position.y = world_point_ENU[1]
                odo_msg.pose.pose.position.z = world_point_ENU[2]
                # 设置方向为默认值，因为没有方向信息
                odo_msg.pose.pose.orientation.x = 0.0
                odo_msg.pose.pose.orientation.y = 0.0
                odo_msg.pose.pose.orientation.z = 0.0
                odo_msg.pose.pose.orientation.w = 1.0
                linear_velocity = get_linear_velocity(world_point_ENU,rospy.Time.now())
                odo_msg.twist.twist.linear.x = linear_velocity[0]
                odo_msg.twist.twist.linear.y = linear_velocity[1]
                odo_msg.twist.twist.linear.z = linear_velocity[2]

                odom_publisher.publish(odo_msg)

                global target_x,target_y
                target_x=world_point_ENU[0]
                target_y=world_point_ENU[1]
                drone_pos = np.array([odemetry_msg.pose.pose.position.x,odemetry_msg.pose.pose.position.y])
               
                global drone_yaw
                drone_yaw = calculate_yaw(drone_pos,np.array([target_x,target_y]))
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
    rospy.loginfo("take off")
    
    # # rgb image in camera_1
    vehicle_name = rospy.get_param("/vehicle_name", "drone_1")
    # rgb_camera = "camera_1"
    # camera_type_scene = "Scene"
    # rgb_topic = f"/airsim_node/{vehicle_name}/{rgb_camera}/{camera_type_scene}"

    # # depth image in camera_2
    # depth_camera = "camera_2"
    # camera_type_depth = "DepthPlanar"
    # depth_topic = f"/airsim_node/{vehicle_name}/{depth_camera}/{camera_type_depth}"

    # camera_topic 
    camera_topic  = '/airsim/synced_image'

    # drone odemetry
    odom_local_enu = "odom_local_enu"
    odemetry_topic = f"/airsim_node/{vehicle_name}/{odom_local_enu}"

    camera_sub = Subscriber(camera_topic,SyncedImg)
    odemetry_sub = Subscriber(odemetry_topic,Odometry)

    rospy.Timer(rospy.Duration(0.1), pub_cmd)

    ats = ApproximateTimeSynchronizer([camera_sub,odemetry_sub], queue_size=20, slop=0.01)
    ats.registerCallback(perception_callback)
    rospy.spin()




if __name__ == "__main__":
    sensor_perception()