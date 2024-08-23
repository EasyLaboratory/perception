import airsim
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
from perception_msgs.msg import SyncedImg


def process_rgb_msg2numpyarray(rgb_response):
    # Check if the RGB response is valid
    if rgb_response.width > 0 and rgb_response.height > 0:
        # Get numpy array from the RGB response
        img_rgb1d = np.frombuffer(rgb_response.image_data_uint8, dtype=np.uint8)
        # Ensure the array size matches the expected dimensions
        if img_rgb1d.size == rgb_response.width * rgb_response.height * 3:
            # Reshape array to 3 channel image array H X W X 3
            rgb_image = img_rgb1d.reshape(rgb_response.height, rgb_response.width, 3)
        else:
            rospy.logfatal("The size of the received RGB image data does not match the expected dimensions.")
    else:
        rospy.loginfo("Invalid RGB response received.")
    return rgb_image


def process_depth_msg2numpyarray(depth_response):
    # 处理深度图像
    if depth_response.width > 0 and depth_response.height > 0:
        # 将深度图像数据转换为 numpy 数组
        img_depth1d = np.array(depth_response.image_data_float, dtype=np.float32)

        if img_depth1d.size == depth_response.width * depth_response.height:
            img_depth = img_depth1d.reshape(depth_response.height, depth_response.width)
            depth_image = img_depth
            # 将深度图像转换为 8 位单通道图像以便显示
            # depth_image = cv2.normalize(img_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            rospy.logerr("dimension mismatch")
    else:
        rospy.loginfo("Invalid Depth response received.")
    return depth_image

class AirSimImagePublisher:
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('airsim_image_publisher', anonymous=True)

        self.synced_image_pub = rospy.Publisher('/airsim/synced_image',SyncedImg,queue_size=100)

        # 初始化cv_bridge
        self.bridge = CvBridge()

        # 连接到AirSim
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)

        # 设置Timer，周期性地调用回调函数
        self.timer = rospy.Timer(rospy.Duration(0.05), self.timer_callback)  # 20HZ

    def timer_callback(self, event):
        # 获取RGB图像和深度图像
        responses = self.client.simGetImages([
            airsim.ImageRequest("camera_1", airsim.ImageType.Scene, False, False),
            airsim.ImageRequest("camera_1", airsim.ImageType.DepthPlanar, True, False),
        ])

        rgb_response = responses[0]
        depth_response = responses[1]
        
        rgb_img = process_rgb_msg2numpyarray(rgb_response)
        depth_img = process_depth_msg2numpyarray(depth_response)
        

        # 将OpenCV图像转换为ROS图像
        ros_rgb_image = self.bridge.cv2_to_imgmsg(rgb_img, encoding="bgr8")
        ros_depth_image = self.bridge.cv2_to_imgmsg(depth_img, encoding="32FC1")
        
        ns = rgb_response.time_stamp
        secs = ns // 1_000_000_000  # 取整，得到秒部分
        nsecs = ns % 1_000_000_000  # 取余，得到纳秒部分
        synced_img_msg =  SyncedImg()
        synced_img_msg.header.stamp = rospy.Time(secs,nsecs)
        synced_img_msg.rgb_image = ros_rgb_image
        synced_img_msg.depth_image = ros_depth_image

        self.synced_image_pub.publish(synced_img_msg)

if __name__ == "__main__":
    airsim_publisher = AirSimImagePublisher()
    rospy.spin()  # 保持ROS节点运行


