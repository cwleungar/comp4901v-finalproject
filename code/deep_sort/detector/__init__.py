from detector.YOLOv3.utils.torch_utils import smart_inference_mode
from .YOLOv3.models.yolo import DetectMultiBackend as YOLOv3
from .YOLOv3.utils.general import intersect_dicts
from .YOLOv4.models.models import Darknet as YOLOv4
from .YOLOv5.models.common import DetectMultiBackend as YOLOv5
from .YOLOv4 import models as models
import torch
import yaml

__all__ = ['build_detector']
def getname(path):
    with open(path, 'r') as f:
        name = f.readline().strip()
    return name
@smart_inference_mode()
def build_detector(cfg, use_cuda):
    if 'YOLOV3' in cfg:
        #ckpt = torch.load(cfg.YOLOV3.WEIGHT, map_location='cpu')
        #cfgr=cfg.YOLOV3.CFG
        #hyp=cfg.YOLOV3.HYP
        
        #if isinstance(hyp, str):
        #    with open(hyp, errors='ignore') as f:
        #        hyp = yaml.safe_load(f) 
        device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

        model = YOLOv3(cfg.YOLOV3.WEIGHT, dnn=False,device=device,data="/data/cwleungar/comp4901v-finalproject/code/yolov3/data/dataset.yaml",fp16=False)
        
        return model,getname(cfg.YOLOV3.CLASS_NAMES)
    elif 'YOLOV4' in cfg:
        device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

        cfgr=cfg.YOLOV4.CFG
        with torch.no_grad():
            model = YOLOv4(cfgr,640).to(device)  # create
            ckpt = torch.load(cfg.YOLOV4.WEIGHT, map_location=device)

            state_dict=ckpt['state_dict']
            model=model.load_state_dict(state_dict, strict=False)
            model=model.eval()

            return model,getname(cfg.YOLOV4.CLASS_NAMES)
    elif 'YOLOV5' in cfg:
        device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

        model = YOLOv5(cfg.YOLOV5.WEIGHT, dnn=False,device=device,data="/data/cwleungar/comp4901v-finalproject/code/yolov3/data/dataset.yaml",fp16=False)

        return model,getname(cfg.YOLOV5.CLASS_NAMES)