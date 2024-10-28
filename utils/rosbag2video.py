import rosbag
from cv_bridge import CvBridge
import cv2


def rosbog2Video():
    bag = rosbag.Bag('/home/paw/display.bag', 'r')
    bridge = CvBridge()

    # 获取图像话题的第一个消息来确定宽度和高度
    # width, height = None, None
    # for topic, msg, t in bag.read_messages(topics=["/annotated_image"]):
    #     width = msg.width
    #     height = msg.height
    #     break  # 只需要读取第一个消息来获取分辨率
    

    out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'XVID'), 10, (1280, 720))


    for topic, msg, t in bag.read_messages(topics=['/annotated_image']):
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        out.write(cv_image)

    bag.close()
    out.release()


if __name__ == "__main__":
    rosbog2Video()