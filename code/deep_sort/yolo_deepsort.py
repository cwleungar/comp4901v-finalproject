import os
import cv2
import time
import argparse
import torch
import warnings
import numpy as np

from detector import build_detector
from deep_sort import build_tracker
from detector.YOLOv3.utils.general import non_max_suppression, scale_boxes, xyxy2xywh
from utils.draw import draw_boxes
from utils.parser import get_config
from utils.log import get_logger
from utils.io import write_results

def extract_from_pred(pred):
    # Define the number of classes
    num_classes = 9

    # Define the detection threshold
    conf_threshold = 0.4

    # Define the list of detected objects
    detections = []
    bbox=[]
    conf=[]
    ids=[]
    # Iterate over the detection scales
    for i, pred_i in enumerate(pred):
        # Get the grid size for the current scale
        grid_size = pred_i.shape[2]

        # Compute the number of anchor boxes for the current scale
        num_anchors = pred_i.shape[1] // (5 + num_classes)

        # Reshape the output tensor to (batch_size, num_anchors, 5 + num_classes, grid_size, grid_size)
        pred_i = pred_i.view(1, num_anchors, 5 + num_classes, grid_size, grid_size)

        # Extract the bounding box coordinates (x, y, w, h) from the output tensor
        bbox_xywh = pred_i[..., :4].sigmoid()

        # Extract the objectness scores from the output tensor
        obj_conf = pred_i[..., 4].sigmoid()

        # Extract the class probabilities from the output tensor
        class_conf = pred_i[..., 5:].sigmoid() * obj_conf.unsqueeze(-1)

        # Apply the detection threshold
        mask = class_conf > conf_threshold

        # Iterate over the anchor boxes
        for j in range(num_anchors):
            # Get the mask for the current anchor box
            mask_j = mask[0, j, ...]

            # Get the bounding box coordinates for the current anchor box
            bbox_xywh_j = bbox_xywh[0, j, ...][mask_j]

            # Get the class confidence scores for the current anchor box
            class_conf_j = class_conf[0, j, ...][mask_j]

            # Get the class IDs for the current anchor box
            class_ids_j = class_conf_j.argmax(-1)

            # Add the detected objects to the list
            bbox.append(bbox_xywh_j)
            conf.append(class_conf_j)
            ids.append(class_ids_j)
            #detections.append((bbox_xywh_j, class_conf_j, class_ids_j))

    # Print the list of detected objects
    return torch.tensor(bbox),torch.tensor(conf),torch.tensor(ids)
    #return detections

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
        while self.vdo.grab():
            idx_frame += 1
            if idx_frame % self.args.frame_interval:
                continue

            start = time.time()
            _, ori_im = self.vdo.retrieve()
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

            # do detection
            im0=im.copy()
            im=cv2.resize(im,(640,640))
            device = torch.device("cuda:2" if self.use_cuda else "cpu")
            im = torch.from_numpy(im).to(device).permute(2,0, 1).float()
            
            im=im/255
            im=im.unsqueeze(0)
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            screen=0
            pred= self.detector(im)
            pred = non_max_suppression(pred, 0.4, 0.2, None, False, max_det=1000)
            det=pred[0]
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

            #cls_ids = []
            bbox_xywh = [np.empty((0, 4), dtype=np.float32) for _ in range(9)]
            cls_conf = [np.empty(0, dtype=np.float32) for _ in range(9)]

            # Write results
            for *xyxy, conf, cls in reversed(det):
                conf=conf.cpu().detach().numpy()
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) ).view(-1).tolist()  # normalized xywh
                bbox = np.array(xywh, dtype=np.float32).reshape(1, 4)
                bbox_xywh[int(cls)] = np.concatenate([bbox_xywh[int(cls)], bbox], axis=0)
                cls_conf[int(cls)] = np.concatenate([cls_conf[int(cls)], np.array([conf], dtype=np.float32)], axis=0)
            #bbox_xywh, cls_conf, cls_ids = self.detector(im)

            # select person class
            for i in range(9):
                mask = i
                
                bbox_xywhh = bbox_xywh[mask]
                # bbox dilation just in case bbox too small, delete this line if using a better pedestrian detector
                #bbox_xywh[:, 3:] *= 1.2
                cls_conff = cls_conf[mask]
                
                # do tracking
                outputs = self.deepsort.update(bbox_xywhh, cls_conff, im0)

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
            raise Exception
            if self.args.display:
                cv2.imshow("test", ori_im)
                cv2.waitKey(1)

            if self.args.save_path:
                self.writer.write(ori_im)

            # save results
            write_results(self.save_results_path, results, 'mot')

            # logging
            self.logger.info("time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}" \
                             .format(end - start, 1 / (end - start), bbox_xywh.shape[0], len(outputs)))


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