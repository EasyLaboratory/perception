from easyGL.airsim_gl import *
from pathlib import Path
from ultralytics import YOLO

def getModel():
    current_dir = Path(__file__).resolve()
    project_root = current_dir.parent.parent
    model_base_path = project_root/"models"
    model_path = model_base_path/"yolov10n_v3.pt"
    return YOLO(model_path)

model:ultralytics.YOLO = getModel()