import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from message_filters import Subscriber, ApproximateTimeSynchronizer
from std_msgs.msg import Float32MultiArray

class Gimbal:
    def __init__(self):
        # 初始化 ROS 节点
        rospy.init_node('Gimbal', anonymous=True)
        
        # 创建 CvBridge 对象
        self.bridge = CvBridge()
        
        # 创建两个话题的订阅者
        self.rgb_subscriber = Subscriber("camera/rgb/image", Image)
        self.depth_subscriber = Subscriber("camera/depth/image", Image)
        self.gimbal_subsciber = Subscriber("/gimbal_control",Float32MultiArray)
        
        # 创建同步器，使用 ApproximateTimeSynchronizer 允许近似时间同步
        self.sync = ApproximateTimeSynchronizer([self.rgb_subscriber, self.depth_subscriber,self.gimbal_subsciber], queue_size=10, slop=0.1)
        self.sync.registerCallback(self.synced_callback)

    def synced_callback(self, rgb_msg, depth_msg):
        try:
            # 转换 RGB 图像
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            # 转换 Depth 图像
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
            rospy.loginfo("成功转换")
            # 同时处理 RGB 和 Depth 图像
            # self.process_images(rgb_image, depth_image)
        except Exception as e:
            rospy.logerr("Error processing synchronized images: %s", str(e))
    
    def process_images(self, rgb_image, depth_image):
        # 示例处理逻辑：将 RGB 和 Depth 图像显示在同一个窗口中
        combined_image = cv2.hconcat([rgb_image, cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)])
        cv2.imshow("RGB and Depth Image", combined_image)
        cv2.waitKey(1)

    def run(self):
        # 保持 ROS 节点运行
        rospy.spin()

if __name__ == '__main__':
    receiver = Gimbal()
    receiver.run()