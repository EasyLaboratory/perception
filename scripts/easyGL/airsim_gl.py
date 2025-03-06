import ultralytics.engine
import ultralytics.engine.results
from easyGL.transform import *
import ultralytics
from easyGL import get_logger
import numpy as np
import cv2
from typing import List,Dict
import copy
from typing import Union
from typing import Tuple
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
import rospy

logger = get_logger(__name__)


        

def track(model:ultralytics.YOLO,image:np.ndarray)->ultralytics.engine.results:
    result = model.track(image)[0]
    return result


def get_target_category_box(results:List,target_category_list:List[int],box_type = "xywh")->Dict: 
    """Parse from the yolo model target tracking results in specific format.
    Args:
        results: yolo model target tracking results
        target_category_list: the target id
    Returns:
        Dict:{car:{1:[],2:[]}}
    """
    cat2id2box = {}
    for result in results:
        box:ultralytics.engine.results.Boxes = result.boxes
        if box.id is None:
            return {}
        category = box.cls # 对象类别
        track_id = box.id # 跟踪 ID
        bbox = box.xywh  # 边界框信息
        if track_id is None:
            return {}
        if box_type == "xywh":
            bbox = box.xywh  # 边界框信息
        elif box_type == "xyxy":
            bbox = box.xyxy
        else:
            bbox = box.xywh
        if category in target_category_list:
           cat2id2box[category.item()] = { entity.item():bbox.tolist() for entity,bbox in zip(track_id,bbox)}
    return cat2id2box

def get_conf(results:List,target_category_list:List[str]):
    cat2id2conf = {}
    for result in results:
        box:ultralytics.engine.results.Boxes = result.boxes
        if box.id is None:
            return {}
        category = box.cls # 对象类别
        track_id = box.id # 跟踪 ID
        conf = box.conf
        if category in target_category_list:
            cat2id2conf[category.item()] = {entity.item():bbox.tolist() for entity,bbox in zip(track_id,conf)}
    return cat2id2conf
    
def get_center(x,y,w,h):
    return (x+w)/2,(y+h)/2

def get_uv(cat2id2box,category,id)->List[int]:
    if cat2id2box:
       return int(cat2id2box[category][id][0]),int(cat2id2box[category][id][1])
    return []

def get_box(cat2id2box,category,id):
    """Find the category id in the cat2id2box dictionary
    Args:
        cat2id2box: dictionary
        category: category key value
        id: id key value
    Returns:
        boundary box tuple, if the cat2id2box is None return -1,-1,-1,-1 
    """
    if cat2id2box:
        return int(cat2id2box[category][id][0]),int(cat2id2box[category][id][1]),int(cat2id2box[category][id][2]),int(cat2id2box[category][id][3])
    return -1,-1,-1,-1

# def get_conf(cat2id2box,category,id):
#     if cat2id2box:
#         return cat2id2box[category][]

def get_annotated_image(results:List[ultralytics.engine.results.Results],label,conf,x1,y1,x2,y2):
    if x1==-1 and x2==-1 and y1==-1 and y2 == -1:
        return []
    annotated_image:List = []
    for result in results:
        center_point_x = int((x1+x2)/2)
        center_point_y = int((y1+y2)/2)
        temp = copy.deepcopy(result.orig_img)
        temp = cv2.rectangle(temp, (x1, y1), (x2, y2), (0, 255, 0), 2)
        temp = cv2.putText(temp, str(label), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        temp = cv2.putText(temp,str(conf),(x1+40,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.8, (0, 255, 0), 2)
        temp = cv2.circle(temp, (center_point_x, center_point_y), radius=8, color=(255, 0, 0), thickness=-1)  # 使用蓝色圆点标记
        annotated_image.append(temp)
    return annotated_image


def get_annotated_image_center(results:List[ultralytics.engine.results.Results],label,x,y):
    if x == -1 and y == -1:
        return []
    
    annotated_image:List = []
    for result in results:
        temp = copy.deepcopy(result.orig_img)
        temp = cv2.circle(temp, (x, y), radius=20, color=(255, 0, 0), thickness=-1)  # 使用蓝色圆点标记
        annotated_image.append(temp)
    return annotated_image


def add_noise(depth_map, theta_y):
    """
    根据已知噪声模型对超出理想范围的深度图进行去噪处理
    :param depth_map: 原始深度图，单位为米，numpy数组
    :param theta_y: 入射角度，单位为弧度，numpy数组，与depth_map形状相同
    :return: 添加噪声的深度图
    """
    # 噪声模型参数
    a0 = 0.001063
    a1 = 0.0007278
    a2 = 0.003949
    b = 0.022
    
    # 计算距离z
    z = depth_map
    
    # 计算轴向噪声 σz(z, θy)
    axial_noise_sigma = (a0 + a1*z + a2*z**2) + (b * z**(3/2)) / (theta_y * (np.pi/2 - theta_y)**2)

    noise = np.random.normal(0,axial_noise_sigma)
    return depth_map+noise



    

def get_uv_depth(cv_depth:np.ndarray,u,v,scale=1,max_distance=100):
    # 深度相机的分辨率是彩色相机的一半
    u = int(u/scale)
    v = int(v/scale)
    return cv_depth[v][u]/255*max_distance

def get_linear_velocity(current_position,current_time:rospy.Time,previous_position,previous_time):
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
    

def publish_point_msg(point_publisher,world_point_ENU,odemetry_msg):
    res_point = PointStamped()
    res_point.header.stamp = odemetry_msg.header.stamp
    res_point.header.frame_id = odemetry_msg.header.frame_id
    res_point.point.x = world_point_ENU[0]
    res_point.point.y = world_point_ENU[1]
    res_point.point.z = world_point_ENU[2]
    point_publisher.publish(res_point)

def publish_odometry_msg(odometry_publisher,world_point_ENU,odometry_msg,linear_velocity,frane_id):
    odo_msg = Odometry()
    odo_msg.header.stamp = odometry_msg.header.stamp
    odo_msg.header.frame_id = frane_id
    odo_msg.pose.pose.position.x = world_point_ENU[0]
    odo_msg.pose.pose.position.y = world_point_ENU[1]
    odo_msg.pose.pose.position.z = world_point_ENU[2]
    # 设置方向为默认值，因为没有方向信息
    odo_msg.pose.pose.orientation.x = 0.0
    odo_msg.pose.pose.orientation.y = 0.0
    odo_msg.pose.pose.orientation.z = 0.0
    odo_msg.pose.pose.orientation.w = 1.0
    odo_msg.twist.twist.linear.x = linear_velocity[0]
    odo_msg.twist.twist.linear.y = linear_velocity[1]
    odo_msg.twist.twist.linear.z = linear_velocity[2]
    odometry_publisher.publish(odo_msg)

def publish_annotated_image(bridge,annotated_frame_publisher,results,conf_label):
    cat2id2_xyxybox = get_target_category_box(results,[0],box_type="xyxy")
    x1,y1,x2,y2 = get_box(cat2id2_xyxybox,0,1)
    annotated_image=get_annotated_image(results,"ship",conf_label,x1,y1,x2,y2)
    if annotated_image:
        first_image = annotated_image[0]
        # Convert the processed image (result) back to a ROS Image message
        annotated_msg = bridge.cv2_to_imgmsg(first_image, encoding='bgr8')
        # Publish the annotaprint(a)ted target
        annotated_frame_publisher.publish(annotated_msg)

def get_detect_target(bridge,results,odometry_msg,depth_image,camera_intrinsic_matrix,
                      camera_eular_angle,camera_translation)->Tuple[Union[None,np.array],float]:
    """Get the 3D position of the target from target bounding box and center point depth.
    Args:
        bridge: ros image cv bridge to convert ros message to numpy array.
        results: YOLO target tracking results.
        odometry_msg: ego drone pose and position message
        depth_msg: depth value from the depth camera.
        camera_intrinsic_matrix: intrinx matrix of the camera.
        camera_eular_angle: gimbal rotation eular angle
        camera_translation: gimabal position relative to the dorne
    Returns:
        Tuple: target world ENU position and confidence, if not find the target return None
    """
    cat2id2_xywhbox = get_target_category_box(results,[0])
    x,y,w,h = get_box(cat2id2_xywhbox,0,1)       
    conf = get_conf(results,[0])
    if not conf or conf[0.0][1.0] < 0.4:
        return None,-1.0
    else:
        conf_label = conf[0.0][1.0]
    if x!=-1 and y!=-1:
                
        cv_depth = bridge.imgmsg_to_cv2(depth_image,desired_encoding="passthrough")
        cv_depth_r_channel = cv_depth[:,:,0]
        depth = get_uv_depth(cv_depth_r_channel,x,y)
        t = odometry_msg.pose.pose.position
        t_array = np.array([t.x,t.y,t.z])
        o_array = np.array([1,0,0,0])
        extrinsic_matrix = construct_extrinsic_with_quaternion(o_array,t_array)
        world_point_ENU =unproject(x,y,depth,camera_intrinsic_matrix,camera_eular_angle,camera_translation,extrinsic_matrix)
        return world_point_ENU,conf_label
    
def analyse_error(target_position_truth,position_3D,velocity_3D):
    position_truth = np.zeros(3)
    velocity_truth = np.zeros(3)
    # rospy.loginfo(target_position_truth)
    rospy.loginfo(target_position_truth.pose.pose.position.x)
    position_truth[0] = target_position_truth.pose.pose.position.x
    position_truth[1] = target_position_truth.pose.pose.position.y
    position_truth[2] = target_position_truth.pose.pose.position.z
    rospy.loginfo(position_truth)