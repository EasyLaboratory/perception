import rospy
import json
import socket
from std_msgs.msg import Float32MultiArray

class UDP2ROS:
    def __init__(self):
        # 初始化 UDP 和 ROS 设置
        self.listen_ip = rospy.get_param("~listen_ip","0.0.0.0")
        self.listen_port = rospy.get_param("~listen_port",9000)

        # 初始化 UDP Socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.listen_ip, self.listen_port))

        # 初始化 ROS 节点
        rospy.init_node('udp2ros_node')

        # 创建发布器，发布云台控制数据
        self.pub = rospy.Publisher('gimbal_control', Float32MultiArray, queue_size=10)

        rospy.loginfo(f"Listening for status messages on {self.listen_ip}:{self.listen_port}...")

    def listen_and_publish(self):
        while not rospy.is_shutdown():
            # 接收 UDP 消息
            data, addr = self.sock.recvfrom(1024)  # 接收最多 1024 字节的数据
            message = data.decode("utf-8")

            try:
                # 解析 JSON 数据
                json_data = json.loads(message)

                # 打印收到的数据
                rospy.loginfo(f"Received message: {json_data}")

                # 将 JSON 数据转换为 Float32MultiArray 类型
                control_data = Float32MultiArray()
                control_data.data = [
                    json_data["control_speed"],
                    json_data["pitch"],
                    json_data["roll"],
                    json_data["yaw"],
                    json_data["control_command"]
                ]

                # 发布云台控制数据到 ROS 话题
                self.pub.publish(control_data)
                rospy.loginfo(f"Published control data: {control_data.data}")

            except json.JSONDecodeError:
                rospy.logerr("Received data is not valid JSON.")
            except KeyError as e:
                rospy.logerr(f"Missing key in received message: {e}")

    def start(self):
        # 开始监听并发布
        self.listen_and_publish()

if __name__ == "__main__":
    # 创建 UDPToROS 对象并启动
    udp_ros = UDP2ROS()
    udp_ros.start()

