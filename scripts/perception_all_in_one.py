import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge,CvBridgeError
from easyGL.airsim_gl import *
from pathlib import Path
from ultralytics import YOLO
from message_filters import Subscriber, ApproximateTimeSynchronizer
from nav_msgs.msg import Odometry
from easyGL.transform import unproject_uv_list
from typing import Tuple
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import Quaternion
import signal
import airsim
import msgpackrpc.error
from airsim import DrivetrainType
import tf.transformations
from trajectory_msgs.msg import TrajectoryPoint


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


trajectory_point = TrajectoryPoint()


K=[320.0, 0.0, 320.0, 0.0, 320.0, 240.0, 0.0, 0.0, 1.0]

intrinsic_matrix = construct_inverse_intrinsic_with_k(K)

previous_position = None
previous_time:rospy.Time = None


target_x=0
target_y=0


def pub_cmd(event):
    client.moveToPositionAsync( target_y,target_x, -6, 4, 5 )


def quaternion_from_euler(roll, pitch, yaw):
    """
    将欧拉角 (roll, pitch, yaw) 转换为四元数 (qx, qy, qz, qw)
    """
    qx = math.sin(roll / 2) * math.cos(pitch / 2) * math.cos(yaw / 2) - math.cos(roll / 2) * math.sin(pitch / 2) * math.sin(yaw / 2)
    qy = math.cos(roll / 2) * math.sin(pitch / 2) * math.cos(yaw / 2) + math.sin(roll / 2) * math.cos(pitch / 2) * math.sin(yaw / 2)
    qz = math.cos(roll / 2) * math.cos(pitch / 2) * math.sin(yaw / 2) - math.sin(roll / 2) * math.sin(pitch / 2) * math.cos(yaw / 2)
    qw = math.cos(roll / 2) * math.cos(pitch / 2) * math.cos(yaw / 2) + math.sin(roll / 2) * math.sin(pitch / 2) * math.sin(yaw / 2)
    return [qx, qy, qz, qw]



def get_uv_depth(cv_depth:np.ndarray,u,v):
    return cv_depth[v][u]

def construct_point_msg_array(point):
    point_msg = PointStamped(point[0],point[1],point[2])
    return point_msg


def connect2client():
    global client
    if client is not None:
        rospy.logwarn("the client already exists")
    client = airsim.MultirotorClient()
    try:
        client.confirmConnection()
        client.enableApiControl(True)
    except msgpackrpc.error.TransportError:
        rospy.logerr("can not connect the client")
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
    
        
def simple_track(curr_x, curr_y, tar_x, tar_y, tar_z, v):
    global client

    rospy.loginfo("-------------------------------------------------")

    # 计算方向向量
    # direction_x = tar_x - curr_x
    # direction_y = tar_y - curr_y

    # # 计算方向向量的长度
    # length = (direction_x**2 + direction_y**2 )**0.5

    # # 归一化方向向量
    # unit_direction_x = direction_x / length
    # unit_direction_y = direction_y / length

    # # 计算新的目标位置
    # new_x = curr_x + unit_direction_x * 10
    # new_y = curr_y + unit_direction_y * 10

    # 移动到新的目标位置
    client.moveToPositionAsync( tar_y,tar_x,-tar_z, v,5).join()

        



def perception_callback(rgb_msg:Image,depth_msg:Image,odemetry_msg:Odometry):
    global annotated_frame_publisher
    global bridge
    global model
    try:
        # Convert the ROS Image message to a format OpenCV can work with
        cv_image = bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='passthrough')
        results=track(model,cv_image)
        rospy.loginfo(len(results))
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
            
            """
            unproject the uv_list
            1. get the extrinsic matrix

            2. unproject
            """
            if x!=-1 and y!=-1:
                cv_depth = bridge.imgmsg_to_cv2(depth_msg,desired_encoding="passthrough")
                depth = get_uv_depth(cv_depth,x,y)
                t = odemetry_msg.pose.pose.position
                t_array = np.array([t.x,t.y,t.z])
                o = odemetry_msg.pose.pose.orientation
                o_array = np.array([o.w,o.x,o.y,o.z])
                extrinsic_matrix = construct_inverse_extrinsic_with_quaternion(o_array,t_array)
                world_point_ENU =unproject(x,y,depth,intrinsic_matrix,extrinsic_matrix)




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
                linear_velocity = get_linear_velocity(world_point_ENU,rospy.Time.now())
                odo_msg.twist.twist.linear.x = linear_velocity[0]
                odo_msg.twist.twist.linear.y = linear_velocity[1]
                odo_msg.twist.twist.linear.z = linear_velocity[2]

                yaw = math.atan2(linear_velocity[1], linear_velocity[0])
                quat = quaternion_from_euler(0, 0, yaw)
                odo_msg.pose.pose.orientation.x = quat[0]
                odo_msg.pose.pose.orientation.y = quat[1]
                odo_msg.pose.pose.orientation.z = quat[2]
                odo_msg.pose.pose.orientation.w = quat[3]

                odom_publisher.publish(odo_msg)

                # simple_track(t_array[0],t_array[1],unproject_world_x,unproject_world_y,6,4)

                # global target_x,target_y
                # target_x=unproject_world_x
                # target_y=unproject_world_y
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))



def signal_handler(sig, frame):
    print('Shutting down gracefully')
    rospy.signal_shutdown('Signal received')

signal.signal(signal.SIGINT, signal_handler)
 


def sensor_perception():
    global client
    connect2client()
    client.armDisarm(True)
    client.takeoffAsync().join()
    client.moveToPositionAsync(0, 0, -6, 3).join()
    client.hoverAsync().join()

    rospy.init_node('airsim_image_subscriber', anonymous=True)
    
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