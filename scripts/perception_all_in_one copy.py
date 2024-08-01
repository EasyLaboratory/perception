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

import tf.transformations
from trajectory_msgs.msg import TrajectoryPoint


current_dir = Path(__file__).resolve()
project_root = current_dir.parent.parent
model_base_path = project_root/"models"
model_path = model_base_path/"yolov10s_v1.pt"

model:ultralytics.YOLO = YOLO(model_path)
rospy.loginfo("init the yolo model")

bridge = CvBridge()
annotated_frame_publisher = rospy.Publisher("/annotated_image",Image,queue_size=9)
odom_publisher = rospy.Publisher('/target/odom_airsim', Odometry, queue_size=10)
point_publisher = rospy.Publisher('points', PointStamped, queue_size=10)

target_pub = rospy.Publisher('command/trajectory', TrajectoryPoint, queue_size=10)

trajectory_point = TrajectoryPoint()



intrinsic_matrix = construct_inverse_intrinsic(90,1920,1080,True)


def pub_cmd(event):

    global trajectory_point
    target_pub.publish(trajectory_point)



def get_uv_depth_list(cv_depth:np.ndarray,uv_list:List[Tuple[int,int]]):
    print("cv_dpths shape",cv_depth.shape)
    depth_list = []
    for uv in uv_list:
        u = uv[0]
        v = uv[1]
        depth_list.append(cv_depth[v][u])
    return depth_list

def construct_point_msg_array(point):
    point_msg = PointStamped(point[0],point[1],point[2])
    return point_msg




def simple_track(my_odom_array,target_point):

    distance = np.sqrt((my_odom_array[0] - target_point.point.x) ** 2 + (my_odom_array[1] - target_point.point.y) ** 2)
    
    
    global trajectory_point
    if distance > 8:
            # 创建消息
        # 组装TrajectoryPoint消息
        trajectory_point.pose.position.x = target_point.point.x
        trajectory_point.pose.position.y = target_point.point.y
        trajectory_point.pose.position.z = my_odom_array[2]

        trajectory_point.velocity.linear.x = 7;
        trajectory_point.velocity.linear.y = 0;
        trajectory_point.velocity.linear.z = 0; 


        vx = target_point.point.x - my_odom_array[0]
        vy = target_point.point.y - my_odom_array[1]
        yaw = np.arctan2(vy, vx)
        # 将yaw角度转换为四元数
        q = tf.transformations.quaternion_from_euler(0, 0, yaw)
        trajectory_point.pose.orientation = Quaternion(*q)
        # 发布消息
        print("============track==========")

    else:
        print("============keep===========")

        # 如果距离小于8米，停留在当前位置
        trajectory_point.pose.position.x = my_odom_array[0]
        trajectory_point.pose.position.y = my_odom_array[1]
        trajectory_point.pose.position.z=5
        trajectory_point.velocity.linear.x = 0;
        trajectory_point.velocity.linear.y = 0;

        # 保持当前朝向
        yaw = 0  # 假设当前朝向是0，需要根据实际情况调整
        q = tf.transformations.quaternion_from_euler(0, 0, yaw)
        trajectory_point.pose.orientation = Quaternion(*q)

        



def perception_callback(rgb_msg:Image,depth_msg:Image,odemetry_msg:Odometry):
    global annotated_frame_publisher
    global bridge
    global model
    rospy.loginfo("jinru")
    # try:
    #     # Convert the ROS Image message to a format OpenCV can work with
    #     cv_image = bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='passthrough')
    #     results=track(model,cv_image)
    #     rospy.loginfo(results)
    #     if len(results) <= 0:
    #         return
    #     # boxes,category,id = get_target_box(result,[0,1,2,3,4,5,6,7,8,9,10,11],[1])
    #     cat2id2box = get_target_box(results,[0],[1])
    #     rospy.loginfo(cat2id2box)
    #     rospy.loginfo("----------------------------------------------")
    #     # uv_list = get_uv(boxes)
    #     # annotated_image=get_annotated_frame(result,boxes,uv_list) 
    #     # # Convert the processed image (result) back to a ROS Image message
    #     # annotated_msg = bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')

    #     # # Publish the annotaprint(a)ted target
    #     # annotated_frame_publisher.publish(annotated_msg)

    #     """
    #     unproject the uv_list
    #     1. get the extrinsic matrix

    #     2. unproject
    #     """
    #     # cv_depth = bridge.imgmsg_to_cv2(depth_msg,desired_encoding="passthrough")
    #     # depth_list = get_uv_depth_list(cv_depth,uv_list)
    #     # t = odemetry_msg.pose.pose.position
    #     # t_array = np.array([t.x,t.y,t.z])
    #     # o = odemetry_msg.pose.pose.orientation
    #     # o_array = np.array([o.w,o.x,o.y,o.z])
    #     # extrinsic_matrix = construct_inverse_extrinsic_with_quaternion(o_array,t_array)
    #     # unproject_world_points =unproject_uv_list(uv_list,depth_list,intrinsic_matrix,extrinsic_matrix)

    #     # point = unproject_world_points[0]
    #     # res_point = PointStamped()
    #     # # print("uv list",uv_list)
    #     # # print("t_array",t_array)
    #     # # print("depth_list",depth_list)
    #     # # print("world point",point)
    #     # # print("----------------------------------------------------------")

    #     # res_point.header.stamp = odemetry_msg.header.stamp
    #     # res_point.header.frame_id = "drone_1"
    #     # res_point.point.x = point[0]
    #     # res_point.point.y = point[1]
    #     # res_point.point.z = point[2]

    #     # point_publisher.publish(res_point)
    #     # odo_msg = Odometry()
    #     # odo_msg.header.stamp = odemetry_msg.header.stamp
    #     # odo_msg.header.frame_id = "drone_1"
    #     # odo_msg.pose.pose.x = 1



    #     # simple_track(t_array,res_point)



    # except CvBridgeError as e:
    #     rospy.logerr("CvBridge Error: {0}".format(e))

    # rospy.loginfo("get and publish rgb message")



 


def sensor_perception():
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
    odom_local_ned = "odom_local_ne"
    odemetry_topic = f"/airsim_node/{vehicle_name}/{odom_local_ned}"

    rgb_sub = Subscriber(rgb_topic,Image)
    depth_sub = Subscriber(depth_topic,Image)
    odemetry_sub = Subscriber(odemetry_topic,Odometry)

    rospy.Timer(rospy.Duration(0.1), pub_cmd)

    ats = ApproximateTimeSynchronizer([rgb_sub, depth_sub,odemetry_sub], queue_size=10, slop=1)
    ats.registerCallback(perception_callback)
    rospy.loginfo("初始化成功")
    rospy.spin()

if __name__ == "__main__":
    sensor_perception()