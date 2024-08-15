import numpy as np
import math


class Eular_angle:
    def __init__(self,pitch,roll,yaw) -> None:
        """
        The given eular angle is under NED coords. 
        """
        self.pitch = pitch
        self.roll = roll
        self.yaw = yaw
    
    @property
    def ENU_pitch_rad(self):
        return np.deg2rad(self.pitch)
    @property
    def ENU_roll_rad(self):
        return np.deg2rad(-self.yaw)
    @property
    def ENU_yaw_rad(self):
        return np.deg2rad(self.roll)

class Translation:
    def __init__(self,x,y,z) -> None:
        self.x = x
        self.y = y
        self.z = z
    
    def get_xyz(self):
        return np.array([self.x,self.y,self.z])

def quaternion_to_yaw(q_x, q_y, q_z, q_w):
    """
    将四元数转换为航向角（yaw）。
    
    参数:
    q_x, q_y, q_z, q_w -- 四元数的四个分量
    
    返回值:
    yaw -- 以弧度为单位的航向角
    """
    # 四元数转换为欧拉角（roll, pitch, yaw）
    siny_cosp = 2 * (q_w * q_z + q_x * q_y)
    cosy_cosp = 1 - 2 * (q_y * q_y + q_z * q_z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    return yaw


def quaternion_from_euler(roll, pitch, yaw):
    """
    将欧拉角 (roll, pitch, yaw) 转换为四元数 (qx, qy, qz, qw)
    """
    qx = math.sin(roll / 2) * math.cos(pitch / 2) * math.cos(yaw / 2) - math.cos(roll / 2) * math.sin(pitch / 2) * math.sin(yaw / 2)
    qy = math.cos(roll / 2) * math.sin(pitch / 2) * math.cos(yaw / 2) + math.sin(roll / 2) * math.cos(pitch / 2) * math.sin(yaw / 2)
    qz = math.cos(roll / 2) * math.cos(pitch / 2) * math.sin(yaw / 2) - math.sin(roll / 2) * math.sin(pitch / 2) * math.cos(yaw / 2)
    qw = math.cos(roll / 2) * math.cos(pitch / 2) * math.cos(yaw / 2) + math.sin(roll / 2) * math.sin(pitch / 2) * math.sin(yaw / 2)
    return [qx, qy, qz, qw]

def construct_extrinsic(R, t):
    """
       构造变换矩阵 [R | t]。

       参数:
       R -- 3x3 旋转矩阵
       t -- 3x1 平移向量

       返回:
       T -- 4x4 变换矩阵
       """
    # 确保 R 是 3x3 矩阵，t 是长度为 3 的向量
    if R.shape != (3, 3) or t.shape != (3,):
        raise ValueError("旋转矩阵必须是 3x3，平移向量必须是长度为 3 的一维数组。")
    # 创建一个 4x4 的单位矩阵
    T = np.eye(4)

    # 将旋转矩阵 R 和平移向量 t 填充到变换矩阵 T 中
    T[:3, :3] = R
    T[:3, 3] = t

    return T


def construct_inverse_extrinsic(R, t):
    # 构造 [R|t] 矩阵
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    # 计算 [R|t] 的逆矩阵
    R_inv = R.T
    t_inv = -np.dot(R_inv, t)
    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv

    return T_inv


def construct_intrinsic(fov: float, width: int, height: int, is_degree=True):
    """
    @param fov: field of view
    @return:intrinsic matrix
    [[fx,0,cx]
    [0,fy,cy]
    [0,0,1]]
    """
    if is_degree:
        fov = fov * (math.pi / 180)
    

    fx = width / (2 * math.tan(fov / 2) + 0.1)
    fy = height / (2 * math.tan(fov / 2) + 0.1)
    cx = width / 2
    cy = height / 2
    intrinsic = np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]])
    return intrinsic


def construct_inverse_intrinsic(fov: float, width: int, height: int, is_degree=True):
    if is_degree:
        fov = fov * (math.pi / 180)

    fx = width / (2 * math.tan(fov / 2) + 0.1)
    fy = height / (2 * math.tan(fov / 2) + 0.1)
    cx = width / 2
    cy = height / 2
    k_inv = np.array([[1 / fx, 0, -cx / fx],
                      [0, 1 / fy, -cy / fy],
                      [0, 0, 1]])
    return k_inv

def construct_inverse_intrinsic_with_k(K):
    fx = K[0]
    fy = K[4]
    cx = K[2]
    cy = K[5]
    
    k_inv = np.array([
        [1 / fx, 0, -cx / fx],
        [0, 1 / fy, -cy / fy],
        [0, 0, 1]
    ])
    return k_inv


def rotation_x(theta):
    """ 绕x轴旋转的旋转矩阵 """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    return np.array([
        [1, 0, 0],
        [0, cos_theta, -sin_theta],
        [0, sin_theta, cos_theta]
    ])


def rotation_y(theta):
    """ 绕y轴旋转的旋转矩阵 """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    return np.array([
        [cos_theta, 0, sin_theta],
        [0, 1, 0],
        [-sin_theta, 0, cos_theta]
    ])


def rotation_z(theta):
    """ 绕z轴旋转的旋转矩阵 """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    return np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])


# def unproject(u, v, depth,inverse_K, inverse_E,camera_eular:Eular_angle,
#               camera_translation:Translation)->np.ndarray:
#     """
#     unproject the u,v point 
#     1. camera uv to camera xyz
#     2. camera xyz to NED
#     3. NED xyz to world xyz
#     """

#     # step1
#     homo_uv = np.array([u, v, 1])
#     camera_xyz = inverse_K @ homo_uv
#     camera_xyz = camera_xyz*depth

#     # step2
#     mav_xyz = camera2mav(camera_xyz,camera_eular,camera_translation)
#     homo_mav_xyz = np.append(mav_xyz, 1)
#     homo_word_xyz = inverse_E @ homo_mav_xyz
#     return homo_word_xyz[0:3]

def unproject(u, v, depth,inverse_camera_K,camera_eular:Eular_angle,
              camera_translation:Translation,inverse_air_E)->np.ndarray:
    camera_xyz = camera2D2camera3D(u,v,depth,inverse_camera_K)
    mav_xyz = camera2mavENU(camera_xyz,camera_eular,camera_translation)
    return mavENU2worldENU(mav_xyz,inverse_air_E)


def camera2D2camera3D(u,v,depth,inverse_camera_K):
    homo_uv = np.array([u, v, 1])
    camera_xyz = inverse_camera_K @ homo_uv
    camera_xyz = camera_xyz*depth
    return camera_xyz

def camera2mavENU(camera_xyz,camera_eular:Eular_angle,
              camera_translation:Translation):
    """
    Convert from the camera coordinate system to ENU coordsystem 
    """
    camera_coord2camera_ENU = np.array([[1, 0, 0],
                            [0, 0, 1],
                            [0, -1, 0]])
    camera_xyz = camera_coord2camera_ENU@camera_xyz
    r_yaw = np.array([
        [np.cos(camera_eular.ENU_yaw_rad), -np.sin(camera_eular.ENU_yaw_rad), 0],
        [np.sin(camera_eular.ENU_yaw_rad), np.cos(camera_eular.ENU_yaw_rad), 0],
        [0, 0, 1]
    ])

    r_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(camera_eular.ENU_pitch_rad), -np.sin(camera_eular.ENU_pitch_rad)],
        [0, np.sin(camera_eular.ENU_pitch_rad), np.cos(camera_eular.ENU_pitch_rad)]
    ])
    
    r_roll =   np.array([
        [np.cos(camera_eular.ENU_roll_rad), 0, np.sin(camera_eular.ENU_roll_rad)],
        [0, 1, 0],
        [-np.sin(camera_eular.ENU_roll_rad), 0, np.cos(camera_eular.ENU_roll_rad)]
    ])
    
    NED2ENU = np.array([
        [1,0,0],
        [0,1,0],
        [0,0,-1]])
    
    ENU_translation = NED2ENU@camera_translation.get_xyz()

    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :3] = r_roll@r_pitch@r_yaw
    extrinsic_matrix[:3, 3] = ENU_translation

    homo_camera_xyz = np.append(camera_xyz,1)
    mav_point = extrinsic_matrix@homo_camera_xyz
    return mav_point[:3]

def mavENU2worldENU(mav_xyz,inverse_air_E):
    homo_mav_xyz = np.append(mav_xyz, 1)
    homo_word_xyz = inverse_air_E @ homo_mav_xyz
    return homo_word_xyz[0:3]

def quaternion_to_rotation_matrix(quat):
    w, x, y, z = quat
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])

def construct_extrinsic_with_quaternion(quat,translation):
    w, x, y, z = quat
    
    rotation = np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])
    
    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :3] = rotation
    extrinsic_matrix[:3, 3] = translation

    return extrinsic_matrix
    
