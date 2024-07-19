import rospy
from cv_bridge import CvBridge,CvBridgeError
import cv2
from sensor_msgs.msg import Image



def rgb_callback(msg):
    bridge = CvBridge()
    try:
        # Convert the ROS Image message to a format OpenCV can work with
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))

    # Display the image
    cv2.imshow("AirSim Camera Image", cv_image)
    cv2.waitKey(10)
    rospy.loginfo("get rgb")

def depth_callback(msg):
    bridge = CvBridge()
    try:
        # Convert the ROS Image message to a format OpenCV can work with
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))

    # Display the image
    cv2.imshow("AirSim Camera Image", cv_image)
    cv2.waitKey(10)
    rospy.loginfo("get depth")

def test_callback(msg):
    rospy.loginfo("hello")

def listener():
    rospy.init_node('airsim_image_subscriber', anonymous=True)
    
    # rgb image in camera_1
    vehicle_name = rospy.get_param("/vehicle_name", "drone_1")
    # rgb_camera = "camera_1"
    # camera_type_scene = "Scene"
    # rgb_topic = "/airsim_node/{}/{}/{}".format(vehicle_name, rgb_camera, camera_type_scene)
    
    # rospy.Subscriber(rgb_topic, Image, rgb_callback)
    # rospy.loginfo("Subscribed to topic: {}".format(rgb_topic))

    # depth image in camera_2
    depth_camera = "camera_2"
    camera_type_depth = "DepthVis"
    depth_topic = "/airsim_node/{}/{}/{}".format(vehicle_name, depth_camera, camera_type_depth)
    test_topic = "/airsim_node/drone_1/camera_2/DepthVis"
    rospy.Subscriber(test_topic,Image,test_callback)
    rospy.loginfo(f"Subscribed to topic:{depth_topic}")
  
    rospy.spin()

if __name__ == "__main__":
    listener()