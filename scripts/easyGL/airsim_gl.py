import ultralytics.engine
import ultralytics.engine.results
from easyGL.transform import *
from easyGL.utils import *
import ultralytics
from easyGL import get_logger
import numpy as np
import cv2
from typing import List,Dict
import rospy

logger = get_logger(__name__)

def reproduce_3D():
    u, v = get_uv()
    rotation_mat = get_rotation()
    translation_mat = get_translation()
    inverse_E = construct_inverse_extrinsic(rotation_mat, translation_mat)
    inverse_K = construct_inverse_intrinsic()
    unproject(u, v, inverse_K, inverse_E)


        

def track(model:ultralytics.YOLO,image:np.ndarray)->ultralytics.engine.results:
    result = model.track(image)[0]
    return result


def get_target_category_box(results:List,target_category_list:List[str],box_type = "xywh")->Dict: 
    # {car:{1:[],2:[]}}
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


# def get_uv(boxes:List[ultralytics.engine.results.Boxes]):
#     uv_list = []
#     for box in boxes:
#         fisrt_box = box[0]
#         x = fisrt_box[0]
#         y = fisrt_box[1]
#         uv_list.append((int(x),int(y)))
#         return  uv_list

def get_uv(cat2id2box,category,id)->List[int]:
    if cat2id2box:
       return int(cat2id2box[category][id][0]),int(cat2id2box[category][id][1])
    return []

def get_box(cat2id2box,category,id):
    if cat2id2box:
        return int(cat2id2box[category][id][0]),int(cat2id2box[category][id][1]),int(cat2id2box[category][id][2]),int(cat2id2box[category][id][3])
    return -1,-1,-1,-1
def get_annotated_image(results:List[ultralytics.engine.results.Results],label,x1,y1,x2,y2):
    if x1==-1 and x2==-1 and y1==-1 and y2 == -1:
        return []
    annotated_image:List = []
    for result in results:
        center_point_x = int((x1+x2)/2)
        center_point_y = int((y1+y2)/2)
        temp = copy.deepcopy(result.orig_img)
        temp = cv2.rectangle(temp, (x1, y1), (x2, y2), (0, 255, 0), 2)
        temp = cv2.putText(temp, str(label), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        temp = cv2.circle(temp, (center_point_x, center_point_y), radius=20, color=(255, 0, 0), thickness=-1)  # 使用蓝色圆点标记
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




    
