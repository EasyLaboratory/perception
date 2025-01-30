import rospy
import socket
import struct
from std_msgs.msg import Float32MultiArray
from perception.msg import GimbalControl
from easyGL.gimbalMessage import GimbalControlCode





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
        self.pub = rospy.Publisher('gimbal_control', GimbalControl, queue_size=10)

        rospy.loginfo(f"Listening for status messages on {self.listen_ip}:{self.listen_port}...")

    def listen_and_publish(self):
        while not rospy.is_shutdown():
            # 接收 UDP 消息
            data, addr = self.sock.recvfrom(1024)  # 接收最多 1024 字节的数据
          
            if len(data) % 4 == 0:
                # 每 4 字节解码为一个浮点数
                num_floats = len(data) // 4
                gimbal_control_array = struct.unpack(f'!{num_floats}f', data)
                rospy.loginfo(f"Received Floats: {gimbal_control_array}")
            else:
                rospy.loginfo(f"Invalid Data Length {len(data)}: {data}")
            # 将 JSON 数据转换为 Float32MultiArray 
            control_data = GimbalControl()
            control_data_array = Float32MultiArray()
            control_data_array.data = [
                gimbal_control_array[GimbalControlCode.pitch.value],
                gimbal_control_array[GimbalControlCode.roll.value],
                gimbal_control_array[GimbalControlCode.yaw.value],
                gimbal_control_array[GimbalControlCode.speed.value],
                gimbal_control_array[GimbalControlCode.command.value]
            ]
            
            control_data.header.stamp = rospy.Time.now()
            control_data.data = control_data_array
            # 发布云台控制数据到 ROS 话题
            self.pub.publish(control_data)
            rospy.loginfo(f"Published control data: {control_data.data}  {control_data.header.stamp}")


    def start(self):
        # 开始监听并发布
        self.listen_and_publish()

if __name__ == "__main__":
    # 创建 UDPToROS 对象并启动
    udp_ros = UDP2ROS()
    udp_ros.start()

