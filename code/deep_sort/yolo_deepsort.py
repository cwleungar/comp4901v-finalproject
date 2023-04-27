import os
import cv2
import time
import argparse
import torch
import warnings
import numpy as np
import io

from detector import build_detector
from deep_sort import build_tracker
from detector.YOLOv3.utils.dataloaders import LoadImages
from detector.YOLOv4.utils.datasets import LoadImages as LoadImagesv4

from detector.YOLOv3.utils.general import Profile, check_img_size, non_max_suppression, scale_boxes, xyxy2xywh
from detector.YOLOv3.utils.torch_utils import profile
from utils.draw import draw_boxes
from utils.parser import get_config
from utils.log import get_logger
from utils.io import write_results

class VideoTracker(object):
    def __init__(self, cfg, args, video_path):
        self.cfg = cfg
        self.args = args
        self.video_path = video_path
        self.logger = get_logger("root")

        self.use_cuda = args.use_cuda and torch.cuda.is_available()
        if not self.use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        if args.cam != -1:
            print("Using webcam " + str(args.cam))
            self.vdo = cv2.VideoCapture(args.cam)
        else:
            self.vdo = cv2.VideoCapture()
        self.detector,self.class_names = build_detector(cfg, use_cuda=self.use_cuda)
        self.deepsort = build_tracker(cfg, use_cuda=self.use_cuda)

    def __enter__(self):
        if self.args.cam != -1:
            ret, frame = self.vdo.read()
            assert ret, "Error: Camera error"
            self.im_width = frame.shape[0]
            self.im_height = frame.shape[1]

        else:
            assert os.path.isfile(self.video_path) or os.path.isdir(self.video_path), "Path error"
            if os.path.isdir(self.video_path):
                self.video_path = os.path.join(self.video_path, '%6d.png')
                
            self.vdo.open(self.video_path)
            self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
            assert self.vdo.isOpened()

        if self.args.save_path:
            os.makedirs(self.args.save_path, exist_ok=True)

            # path of saved video and results
            self.save_video_path = os.path.join(self.args.save_path, "results.avi")
            self.save_results_path = os.path.join(self.args.save_path, "results.txt")

            # create video writer
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(self.save_video_path, fourcc, 20, (self.im_width, self.im_height))

            # logging
            self.logger.info("Save results to {}".format(self.args.save_path))

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def run(self):
        results = []
        idx_frame = 0
        with torch.no_grad():
            while self.vdo.grab():
                idx_frame += 1
                if idx_frame % self.args.frame_interval:
                    continue

                start = time.time()
                _, ori_im = self.vdo.retrieve()
                im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
                cv2.imwrite("temp.png", im)
                # do detection
                im0=im.copy()
                device = torch.device("cuda:2" if self.use_cuda else "cpu")

                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imgsz=(640, 640)

                screen=0
                model=self.detector
                stride=0
                pt=0 
                if 'YOLOV3' in cfg or 'YOLOV5' in cfg :
                    stride,pt= self.detector.stride , self.detector.pt
                    imgsz = check_img_size(imgsz, s=stride) 
                    dataset = LoadImages('./temp.png' , img_size=imgsz, stride=stride, auto=pt, vid_stride=1)
                    bs = 1  # batch_size
                    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
                    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
                else:
                    dataset = LoadImagesv4('./temp.png', img_size=imgsz, auto_size=64)
                    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
                    _ = model(img.half())

                for path, im, im0s, vid_cap, s in dataset:

                    with dt[0]:
                        im = torch.from_numpy(im).to(model.device)
                        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                        im /= 255  # 0 - 255 to 0.0 - 1.0
                        if len(im.shape) == 3:
                            im = im[None]  # expand for batch dim

                    # Inference
                    with dt[1]:
                        pred = model(im)

                    # NMS
                    with dt[2]:
                        pred = non_max_suppression(pred, 0.6, 0.45, None, False, max_det=1000)
                    det=pred[0]
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0s.shape).round()

                    #cls_ids = []
                    #bbox_xywh = [np.empty((0, 4), dtype=np.float32) for _ in range(9)]
                    #cls_conf = [np.empty(0, dtype=np.float32) for _ in range(9)]
                    bbox_xywh=[]
                    bbox_xyxy=[]
                    cls_conf=[]
                    cls_ids=[]
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        print("xyxy ",xyxy)
                        conf=conf.cpu().detach().numpy()
                        cls=cls.cpu().detach().numpy()
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) ).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) 

                        #bbox = np.array(xywh, dtype=np.float32).reshape(1, 4)
                        #bbox_xywh[int(cls)] = np.concatenate([bbox_xywh[int(cls)], bbox], axis=0)
                        #cls_conf[int(cls)] = np.concatenate([cls_conf[int(cls)], np.array([conf], dtype=np.float32)], axis=0)
                        print(xywh)
                        bbox_xywh.append(xywh)
                        cls_conf.append(conf)
                        cls_ids.append(cls)
                        x1,y1,x2,y2=xyxy
                        bbox_xyxy.append([int(x1),int(y1),int(x2),int(y2)])
                        #cv2.rectangle(im0,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),2)
                bbox_xywh, cls_conf, cls_ids = np.array(bbox_xywh), np.array(cls_conf), np.array(cls_ids)
                bbox_xyxy=np.array(bbox_xyxy)
                #bbox_xywh, cls_conf, cls_ids = self.detector(im)
                # select person class
                
                mask = np.ones_like(cls_ids, dtype=bool)
                bbox_xywh = bbox_xywh[mask]
                #bbox_xyxy = bbox_xyxy[mask]
                # bbox dilation just in case bbox too small, delete this line if using a better pedestrian detector
                #bbox_xywh[:, 3:] *= 1.2
                cls_conf = cls_conf[mask]

            
                # do tracking
                outputs = self.deepsort.update(bbox_xywh, cls_conf, im0)
                # draw boxes for visualization

                if len(outputs) > 0:
                    bbox_tlwh = []
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    ori_im = draw_boxes(ori_im, bbox_xyxy, identities)
                    for bb_xyxy in bbox_xyxy:
                        bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))
                    results.append((idx_frame - 1, bbox_tlwh, identities))
                end = time.time()
                if self.args.display:
                    cv2.imshow("test", ori_im)
                    #cv2.imshow("test", im0)
                    cv2.waitKey(1)

                if self.args.save_path:
                    self.writer.write(ori_im)

                # save results
                write_results(self.save_results_path, results, 'mot')

                # logging
                #self.logger.info("time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}" \
                #                 .format(end - start, 1 / (end - start), bbox_xywh.shape[0], len(outputs)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    # parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="./output/")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)

    with VideoTracker(cfg, args, video_path=args.VIDEO_PATH) as vdo_trk:
        vdo_trk.run()