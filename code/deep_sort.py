import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import torch
import torch.nn as nn
import torchvision.models as models
from yolo import YOLO
max_age = 30

def iou(bb_test, bb_gt):
    """
    Computes the intersection over union (IOU) between two bounding boxes.
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[0]+bb_test[2], bb_gt[0]+bb_gt[2])
    yy2 = np.minimum(bb_test[1]+bb_test[3], bb_gt[1]+bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] * bb_test[3]) + (bb_gt[2] * bb_gt[3]) - wh)
    return o


def associate_detections_to_tracks(tracks, detections, scores, iou_threshold=0.5):
    """
    Associates detections to tracks using the Hungarian algorithm.
    """
    if len(tracks) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, )), 
    ious = np.zeros((len(detections), len(tracks)))
    for d, det in enumerate(detections):
        for t, trk in enumerate(tracks):
            ious[d, t] = iou(det, trk.to_tlbr())
    matched_indices = linear_sum_assignment(-ious)
    matched_indices = np.asarray(matched_indices)
    matched_indices = np.transpose(matched_indices)
    unmatched_detections = np.delete(np.arange(len(detections)), matched_indices[:, 0])
    unmatched_tracks = np.delete(np.arange(len(tracks)), matched_indices[:, 1])
    matches = []
    for m in matched_indices:
        if ious[m[0], m[1]] < iou_threshold:
            unmatched_detections = np.append(unmatched_detections, m[0])
            unmatched_tracks = np.append(unmatched_tracks, m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
    return matches, unmatched_detections, unmatched_tracks
import numpy as np


class Track:
    def __init__(self, image, detection, motion_model):
        # Initialize the track ID
        self.id = None

        # Initialize the Kalman filter state
        self.kf_state = motion_model.initiate(detection)

        # Initialize the track attributes
        self.age = 0
        self.consecutive_invisible_count = 0
        self.image = image
        self.bbox = detection

    def predict(self):
        # Predict the next state of the track using the Kalman filter
        self.kf_state = self.kf_state.predict()

        # Update the track attributes
        self.age += 1
        self.consecutive_invisible_count += 1

    def update(self, image, detection):
        # Update the Kalman filter state using the measurement
        self.kf_state = self.kf_state.update(detection)

        # Update the track attributes
        self.age = 0
        self.consecutive_invisible_count = 0
        self.image = image
        self.bbox = detection

    def get_state(self):
        # Get the current state of the track
        state = self.kf_state.get_state()
        return state.reshape(-1)

    def to_tlbr(self):
        # Convert bounding box coordinates to [left, top, right, bottom]
        tlbr = np.zeros(4)
        tlbr[:2] = self.bbox[:2]
        tlbr[2:] = self.bbox[:2] + self.bbox[2:]
        return tlbr

    def get_centroid(self):
        # Get the centroid of the bounding box
        centroid = np.zeros(2)
        centroid[0] = (self.bbox[0] + self.bbox[2] / 2)
        centroid[1] = (self.bbox[1] + self.bbox[3] / 2)
        return centroid

class DeepSORT(nn.Module):
    def __init__(self):
        super(DeepSORT, self).__init__()

        # Initialize the feature extractor
        self.feature_extractor = models.resnet50(pretrained=True)
        self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
        self.feature_extractor.fc = nn.Linear(2048, 128)

        # Initialize the Kalman filter parameters
        self.motion_model = KalmanFilter()

        # Initialize the object detector
        self.object_detector = YOLO()

        # Initialize the track objects
        self.tracks = []

    def forward(self, x):
        # Extract features from the input image
        features = self.feature_extractor(x)

        # Return the features
        return features

    def update(self, image):
        # Detect objects in the input image using YOLOv4
        detections = self.object_detector.detect(image)

        # Convert detections to PyTorch tensors
        boxes = torch.from_numpy(detections[:, :4])
        scores = torch.from_numpy(detections[:, 4])

        # Predict the next state of each track using the Kalman filter
        for track in self.tracks:
            track.predict()

        # Assign detections to tracks using the Hungarian algorithm
        matched, unmatched_dets, unmatched_tracks = associate_detections_to_tracks(
            self.tracks, boxes, scores
        )

        # Update the state of each matched track using the associated detection
        for track_idx, detection_idx in matched:
            self.tracks[track_idx].update(image, boxes[detection_idx])

        # Create new tracks for unmatched detections
        for detection_idx in unmatched_dets:
            track = Track(image, boxes[detection_idx], self.motion_model)
            self.tracks.append(track)

        # Remove tracks that have been unmatched for too many consecutive frames
        for track_idx in unmatched_tracks:
            if self.tracks[track_idx].consecutive_invisible_count >= max_age:
                self.tracks.pop(track_idx)

        # Return the updated track objects
        return self.tracks

    def train(self, tracks):
        # Update the Kalman filter parameters using the track objects
        self.motion_model.update(tracks)