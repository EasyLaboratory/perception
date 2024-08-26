import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge,CvBridgeError
from easyGL.airsim_gl import *
from pathlib import Path
from ultralytics import YOLO
from message_filters import Subscriber, ApproximateTimeSynchronizer
from nav_msgs.msg import Odometry
from easyGL.transform import quaternion_from_euler,quaternion_to_yaw
from typing import Tuple
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import Quaternion
import signal
import airsim
from enum import Enum


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




K=[320.0, 0.0, 320.0, 0.0, 320.0, 240.0, 0.0, 0.0, 1.0]

intrinsic_matrix = construct_inverse_intrinsic_with_k(K)

previous_position = None
previous_velocity = None
previous_time:rospy.Time = None


target_x=0
target_y=0

drone_yaw = 0
current_yaw = 0

class SequnceState(Enum):
    FIRST_POSITION = 1
    FIRST_VELOCITY = 2
    OUTLIER = 3
    NORMAL = 4

def expire_previous_position_velocity(current_time):
    global previous_position
    global previous_velocity
    global previous_time
    if previous_time is None:
       return
    elif (current_time - previous_time).to_sec() > 2:
        previous_position = None
        previous_velocity = None
        previous_time = None
    


def calculate_yaw(drone_pos, target_pos):
    if len(drone_pos)<2 or len(target_pos)<2:
        return 0 
    
    dx = target_pos[0] - drone_pos[0]
    dy = target_pos[1] - drone_pos[1]
    yaw = math.atan2(dy, dx)
    return  90-math.degrees(yaw)


def pub_cmd(event):
    client.moveToPositionAsync( target_y,target_x, -6, 15, 5,yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=drone_yaw))


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


def is_outlier(current_position,current_time:rospy.Time,threshold):
    global previous_position
    global previous_velocity
    global previous_time
    if previous_position is None and current_position is not None:
        previous_position = current_position
        previous_time = current_time
        return SequnceState.FIRST_POSITION
    elif previous_position is not None and previous_velocity is None:
        time_diff = (current_time-previous_time).to_sec()
        if time_diff > 0:
            # 计算位置差
            position_diff = current_position-previous_position
            previous_velocity = position_diff/time_diff
            previous_position = current_position
            previous_time = current_time
        return SequnceState.FIRST_VELOCITY
    elif previous_position is not None and previous_velocity is not None:
         time_diff = (current_time-previous_time).to_sec()
         # calc the expected position
         expected_position = previous_position + previous_velocity*time_diff
         distance = np.linalg.norm(expected_position-current_position)
         if distance < threshold:
            return SequnceState.NORMAL
         else:
            return SequnceState.OUTLIER


def get_linear_velocity(current_position,current_time:rospy.Time):
    global previous_position
    global previous_time
    if previous_position is not None and previous_time is not None:
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
        if len(results) == 1:
            cat2id2_xywhbox = get_target_category_box(results,[0])
            x,y,w,h = get_box(cat2id2_xywhbox,0,1)
            # todo: use new uv to print
            cat2id2_xyxybox = get_target_category_box(results,[0],box_type="xyxy")
            x1,y1,x2,y2 = get_box(cat2id2_xyxybox,0,1)
            annotated_image=get_annotated_image(results,1,x1,y1,x2,y2)
            if annotated_image:
                first_image = annotated_image[0]
                # Convert the processed image (result) back to a ROS Image message
                annotated_msg = bridge.cv2_to_imgmsg(first_image, encoding='bgr8')

                # Publish the annotaprint(a)ted target
                annotated_frame_publisher.publish(annotated_msg)
            

            if x!=-1 and y!=-1:
                expire_previous_position_velocity(rospy.Time.now())
                cv_depth = bridge.imgmsg_to_cv2(depth_msg,desired_encoding="passthrough")
                depth = get_uv_depth(cv_depth,x,y)
                t = odemetry_msg.pose.pose.position
                t_array = np.array([t.x,t.y,t.z])
                o = odemetry_msg.pose.pose.orientation
                o_array = np.array([o.w,o.x,o.y,o.z])
                extrinsic_matrix = construct_inverse_extrinsic_with_quaternion(o_array,t_array)
                world_point_ENU =unproject(x,y,depth,intrinsic_matrix,extrinsic_matrix)
                
                # test if the point is outlier point 
                state = is_outlier(world_point_ENU,rospy.Time.now(),30)
                if state == SequnceState.FIRST_POSITION or state == SequnceState.FIRST_VELOCITY:
                   return
                elif state == SequnceState.OUTLIER:
                    world_point_ENU = previous_position
                    linear_velocity = previous_velocity
                
                elif state==SequnceState.NORMAL:
                    linear_velocity = get_linear_velocity(world_point_ENU,rospy.Time.now())
                res_point = PointStamped()
                res_point.header.stamp = odemetry_msg.header.stamp
                res_point.header.frame_id = odemetry_msg.header.frame_id
                res_point.point.x = world_point_ENU[0]
                res_point.point.y = world_point_ENU[1]
                res_point.point.z = world_point_ENU[2]

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
                # linear_velocity = get_linear_velocity(world_point_ENU,rospy.Time.now())
                odo_msg.twist.twist.linear.x = linear_velocity[0]
                odo_msg.twist.twist.linear.y = linear_velocity[1]
                odo_msg.twist.twist.linear.z = linear_velocity[2]

                odom_publisher.publish(odo_msg)
                
                rospy.loginfo("----------------------------------------------")
                rospy.loginfo(world_point_ENU)


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

    rospy.Timer(rospy.Duration(0.1), pub_cmd)

    ats = ApproximateTimeSynchronizer([rgb_sub, depth_sub,odemetry_sub], queue_size=10, slop=0.25)
    ats.registerCallback(perception_callback)
    rospy.spin()




if __name__ == "__main__":
    sensor_perception()