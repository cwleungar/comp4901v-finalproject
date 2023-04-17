import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
class DeepSort(object):
    def __init__(self, model_path, max_dist=0.2, min_confidence=0.3, nms_max_overlap=1.0, max_iou_distance=0.7, max_age=70, n_init=3):
        self.max_dist = max_dist
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],
                              [0,1,0,0,0,1,0],
                              [0,0,1,0,0,0,1],
                              [0,0,0,1,0,0,0],
                              [0,0,0,0,1,0,0],
                              [0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],
                              [0,1,0,0,0,0,0],
                              [0,0,1,0,0,0,0],
                              [0,0,0,1,0,0,0]])
        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000. # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        self.kf.x[:4] = 0.
        self.means = []
        self.covariances = []
        self.labels = []
        self.track_id = 0
        self.model = torch.load(model_path)
        self.model = self.model.eval()

    def update(self, xywhs, confs, labels, img):
        # Filter out low-confidence detections and reformat to format expected by the deepsort model
        mask = confs > self.min_confidence
        xywhs = xywhs[mask]
        confs = confs[mask]
        labels = labels[mask]
        if xywhs.shape[0] == 0:
            self.track_id += 1
            return []

        # Run the deepsort model to obtain features for each detection
        features = self.model(img, xywhs).cpu().numpy()

        # Initialize new trackers for any detections that are not associated with an existing tracker
        if len(self.means) == 0:
            for i in range(xywhs.shape[0]):
                mean = np.concatenate((xywhs[i], [features[i][0], features[i][1], features[i][2], features[i][3]]))
                covariance = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
                self.means.append(mean)
                self.covariances.append(covariance)
                self.labels.append(self.track_id)
                self.track_id += 1
            return [(xywhs[i], self.labels[i]) for i in range(xywhs.shape[0])]

        # Predict tracks using Kalman filter
        self.kf.predict()
        self.means, self.covariances = [], []
        for i in range(len(self.labels)):
            self.means.append(self.kf.x[:7])
            self.covariances.append(self.kf.P)

        # Associate detections with existing trackers using the Hungarian algorithm
        cost_matrix = []
        for i in range(len(self.means)):
            cost_row = []
            for j in range(xywhs.shape[0]):
                mean_diff = self.means[i][:4] - xywhs[j]
                mean_diff = np.concatenate((mean_diff, self.means[i][4:]-features[j]))
                distance = np.sqrt(np.dot(mean_diff, np.dot(self.covariances[i], mean_diff.T)))
                cost_row.append(distance)
            cost_matrix.append(cost_row)
        cost_matrix = np.array(cost_matrix)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matches, unmatched_tracks, unmatched_detections = [], [], []
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] > self.max_dist:
                unmatched_tracks.append(i)
                unmatched_detections.append(j)
            else:
                matches.append((i, j))
        for i in range(len(self.means)):
            if i not in matches[:,0]:
                unmatched_tracks.append(i)
        for j in range(xywhs.shape[0]):
            if j not in matches[:,1]:
                unmatched_detections.append(j)

        # Update Kalman filter states and track labels for matched detections
        for i, j in matches:
            mean = self.means[i]
            covariance = self.covariances[i]
            z = np.concatenate((xywhs[j], [features[j][0], features[j][1], features[j][2], features[j][3]]))
            self.kf.update(z)
            self.means[i] = self.kf.x[:7]
            self.covariances[i] = self.kf.P
            self.labels[i] = self.labels[i]
        for i in unmatched_tracks:
            mean = self.means[i]
            covariance = self.covariances[i]
            mean[:4] += self.kf.x[4:7]
            self.kf.x[:7] = mean
            self.kf.P = covariance
            self.kf.predict()
            self.means[i] = self.kf.x[:7]
            self.covariances[i] = self.kf.P
            self.labels[i] = self.labels[i]
        for j in unmatched_detections:
            mean = np.concatenate((xywhs[j], [features[j][0], features[j][1], features[j][2], features[j][3]]))
            covariance = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            self.means.append(mean)
            self.covariances.append(covariance)
            self.labels.append(self.track_id)
            self.track_id += 1

        # Filter out old tracks and perform non-maximum suppression
        good_tracks = []
        for i in range(len(self.means)):
            if self.labels[i] >= self.n_init and np.trace(self.covariances[i]) < 100:
                good_tracks.append(i)
        self.means = [self.means[i] for i in good_tracks]
        self.covariances = [self.covariances[i] for i in good_tracks]
        self.labels = [self.labels[i] for i in good_tracks]
        filtered_boxes = [self.means[i][:4] for i in range(len(self.means))]
        filtered_labels = [self.labels[i] for i in range(len(self.means))]
        indices = np.argsort(filtered_labels)
        filtered_boxes = [filtered_boxes[i] for i in indices]
        filtered_labels = [filtered_labels[i] for i in indices]
        keep = nms(np.array(filtered_boxes), np.array(filtered_labels), self.nms_max_overlap)
        filtered_boxes = [filtered_boxes[i] for i in keep]
        filtered_labels = [filtered_labels[i] for i in keep]
        return [(filtered_boxes[i], filtered_labels[i]) for i in range(len(filtered_boxes))]

def nms(boxes, labels, overlap_threshold):
    if len(boxes) == 0:
        return []
    boxes = boxes.astype(np.float32)
    areas = (boxes[:,2]-boxes[:,0]+1) * (boxes[:,3]-boxes[:,1]+1)
    order = np.argsort(-boxes[:,3])
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i,0], boxes[order[1:],0])
        yy1 = np.maximum(boxes[i,1], boxes[order[1:],1])
        xx2 = np.minimum(boxes[i,2], boxes[order[1:],2])
        yy2 = np.minimum(boxes[i,3], boxes[order[1:],3])
        w = np.maximum(0.0, xx2-xx1+1)
        h = np.maximum(0.0, yy2-yy1+1)
        overlap = w * h / (areas[i] + areas[order[1:]] - w * h)
        inds = np.where(overlap <= overlap_threshold)[0]
        order = order[inds+1]
    return keep