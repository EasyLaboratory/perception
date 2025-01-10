import json
import socket

# 创建 UDP Socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", 9000))

print("Listening for status messages...")
while True:
    data, addr = sock.recvfrom(1024)  # 接收最多 1024 字节的数据
    message = data.decode("utf-8")
    json_data = json.loads(message)
    print(json_data)