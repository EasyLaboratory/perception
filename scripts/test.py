from easyGL.transform import *

# inverse_e = construct_inverse_extrinsic_with_quaternion(np.array([1,2,3,4]),np.array([1,2,3]))
# print(inverse_e)
intr = construct_inverse_intrinsic(90,1920,1080)
# a = unproject(10,10,intr,inverse_e)
# uv_list = [(1,2),(3,4)]
# res = unproject_uv_list(uv_list,intr,inverse_e)
# print(res)

print(np.full((3,),np.nan))