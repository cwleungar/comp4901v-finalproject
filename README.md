This is the code for YOLOv3 v4 v5 with DeepSORT
In this code, I used the repo of 
https://github.com/ultralytics/yolov3
https://github.com/ultralytics/yolov5
And modified version with 
https://github.com/WongKinYiu/PyTorch_YOLOv4
https://github.com/ZQPei/deep_sort_pytorch

download the model:

move ckpt.t7 to code/deep_sort/deep_sort/deep/checkpoint/ckpt.t7
move yolov3_best.pt to code/yolov3/runs/train/yolov3_saved/weights/best.pt
move yolov4_me.pt to code/yolov4/runs/train/yolov4_saved/weights/me.pt
move yolov5_best.pt to code/yolov5/runs/train/yolov5_saved/weights/best.pt


run yolov3 detect:
cd to the folder of yolov3
 python detect.py --weight "runs/train/yolov3_saved/weights/best.pt" --source "image path" --data "data/dataset.yaml" --device 0


run yolov4 detect:
cd to code/yolov4
 python detect.py --weight "runs/train/yolov4_saved/weights/me.pt" --cfg "models/yolov4-csp-x-mish.cfg" --source "image path" --names "data/dataset.name" --device 0
 

run yolov5 detect:
cd to code/yolov5
 python detect.py --weight "runs/train/yolov5_saved/weights/best.pt" --source "image path" --data "data/dataset.yaml" --device 0
 
run DeepSort with yolo
cd to code/deepsort
 python yolo_deepsort.py "video_path" --config_detection "./configs/yolov{3,4,5}.yaml" 