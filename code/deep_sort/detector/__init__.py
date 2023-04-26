from .YOLOv3.models.yolo import Model as YOLOv3
from .YOLOv3.utils.general import intersect_dicts
from .MMDet import MMDet
from .YOLOv4.models.models import Darknet as YOLOv4
from .YOLOv5.models.yolo import Model as YOLOv5
import torch
__all__ = ['build_detector']

def build_detector(cfg, use_cuda):
    if cfg.MODEL == 'MMDet':
        return MMDet(cfg.MMDET.CFG, cfg.MMDET.CHECKPOINT,
                    score_thresh=cfg.MMDET.SCORE_THRESH,
                    is_xywh=True, use_cuda=use_cuda)
    elif cfg.MODEL == 'yolov3':
        ckpt = torch.load(cfg.YOLOV3.WEIGHT, map_location='cpu')
        cfgr=cfg.YOLOV3.CFG
        hyp=cfg.YOLOV3.HYP
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = YOLOv3(cfgr or ckpt['model'].yaml, ch=3, nc=8, anchors=hyp.get('anchors')).to(device)  # create
        exclude = ['anchor'] if (cfgr or hyp.get('anchors')) else []  # exclude keys
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(csd, strict=False)  # load
        return model
    elif cfg.MODEL == 'yovov4':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt = torch.load(cfg.YOLOV4.WEIGHT, map_location=device)  # load checkpoint
        cfgr=cfg.YOLOV4.CFG
        hyp=cfg.YOLOV4.HYP
        model = YOLOv4(cfgr).to(device)  # create
        state_dict=ckpt['model'].state_dict()
        model.load_state_dict(state_dict, strict=False)
        return model
    elif cfg.MODEL == 'yolov5':
        ckpt = torch.load(cfg.YOLOV5.WEIGHT, map_location='cpu')
        cfgr=cfg.YOLOV5.CFG
        hyp=cfg.YOLOV5.HYP
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = YOLOv5(cfgr or ckpt['model'].yaml, ch=3, nc=8, anchors=hyp.get('anchors')).to(device)  # create
        exclude = ['anchor'] if (cfgr or hyp.get('anchors')) else []  # exclude keys
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(csd, strict=False)  # load
        return model