from detector.YOLOv3.utils.torch_utils import smart_inference_mode
from .YOLOv3.models.yolo import DetectMultiBackend as YOLOv3
from .YOLOv3.utils.general import intersect_dicts
from .YOLOv4.models.models import Darknet as YOLOv4
from .YOLOv5.models.yolo import Model as YOLOv5
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

        model = YOLOv3(cfg.YOLOV3.WEIGHT, dnn=False,device=device,data="/data/cwleungar/comp4901v-finalproject/code/yolov3/data/dataset.yaml")
        
        return model,getname(cfg.YOLOV3.CLASS_NAMES)
    elif 'YOLOV4' in cfg:
        device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
        ckpt = torch.load(cfg.YOLOV4.WEIGHT, map_location=device)  # load checkpoint
        cfgr=cfg.YOLOV4.CFG
        hyp=cfg.YOLOV4.HYP
        model = YOLOv4(cfgr).to(device)  # create
        state_dict=ckpt['model'].state_dict()
        model.load_state_dict(state_dict, strict=False)
        model=model.eval()
        return model,getname(cfg.YOLOV4.CLASS_NAMES)
    elif 'YOLOV5' in cfg:
        ckpt = torch.load(cfg.YOLOV5.WEIGHT, map_location='cpu')
        cfgr=cfg.YOLOV5.CFG
        hyp=cfg.YOLOV5.HYP
        if isinstance(hyp, str):
            with open(hyp, errors='ignore') as f:
                hyp = yaml.safe_load(f) 
        device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
        model = YOLOv5(cfgr or ckpt['model'].yaml, ch=3, nc=9, anchors=hyp.get('anchors')).to(device)  # create
        exclude = ['anchor'] if (cfgr or hyp.get('anchors')) else []  # exclude keys
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(csd, strict=False)  # load
        model=model.eval()
        return model,getname(cfg.YOLOV5.CLASS_NAMES)