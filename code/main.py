import torch
import cv2
import numpy as np
from deep_sort import DeepSort
from yolov5 import YOLOv5

# Load YOLOv5 model
model = YOLOv5(weights='yolov5s.pt')

# Load DeepSORT model
deepsort = DeepSort("ckpt.t7")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load video
cap = cv2.VideoCapture('video.mp4')

# Define colors for bounding boxes
colors = np.random.randint(0, 255, size=(model.num_classes, 3), dtype=np.uint8)

# Loop through frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection with YOLOv5
    detections = model(frame)

    # Convert detections to DeepSORT format
    bbox_xywh = []
    confs = []
    classes = []
    for x1, y1, x2, y2, conf, cls in detections:
        bbox_xywh.append([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1])
        confs.append(conf)
        classes.append(cls)
    bbox_xywh = torch.Tensor(bbox_xywh)
    confs = torch.Tensor(confs)
    classes = torch.Tensor(classes)

    # Perform tracking with DeepSORT
    outputs = deepsort.update(bbox_xywh, confs, classes, frame)

    # Draw bounding boxes and labels
    for (x1, y1, x2, y2), track_id in outputs:
        color = colors[int(track_id) % model.num_classes]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, str(int(track_id)), (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the resulting image
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and destroy windows
cap.release()
cv2.destroyAllWindows()