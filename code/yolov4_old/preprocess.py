import os
import random
import shutil
#
#
#classmap={
#    0: 'Car',
#    1: 'Pedestrian', 
#    2: 'Cyclist', 
#    3: 'Van', 
#    4: 'Person_sitting', 
#    5: 'Tram', 
#    6: 'Truck', 
#    7: 'Misc', 
#    8: 'DontCare', 
#    'Car': 0, 
#    'Pedestrian': 1, 
#    'Cyclist': 2, 
#    'Van': 3, 
#    'Person_sitting': 4,
#    'Tram': 5, 
#    'Truck': 6, 
#    'Misc': 7, 
#    'DontCare': 8
#}
#label_dir='D:/Users/samle/Documents/GitHub/comp4901v-finalproject/dataset/label' 
####'/content/drive/MyDrive/comp4901v-final/label' #'/content/drive/MyDrive/comp4901v-finalproject/dataset/label'
img_dir='D:/Users/samle/Documents/GitHub/comp4901v-finalproject/dataset/data' 
####'/content/drive/MyDrive/comp4901v-final/data' #'/content/drive/MyDrive/comp4901v-finalproject/dataset/image'
kk='D:/Users/samle/Documents/GitHub/comp4901v-finalproject/dataset/small_image'
split="training"
label_dir = os.path.join(label_dir, split, 'label_2')
filenames=os.listdir(label_dir)
buffer=[]
for file in filenames:
    with open(os.path.join(label_dir, file)) as f:
        lines=f.readlines()
        temp=file.split(".")[0]+".png"+" "
        for line in lines:
            line=line.split(" ")
            temp=temp+str(line[4])+","+str(line[5])+","+str(line[6])+","+str(line[7])+","+str(classmap[line[0]])+" "
        buffer.append(temp+'\n')
random.seed(42)
random.shuffle(buffer)
val_size = int(0.05 * len(buffer))
with open("code/yolov4/data/train.txt","w") as f:
    f.write("".join(buffer[0:int(len(buffer)*0.1)]))

with open("code/yolov4/data/val.txt","w") as f:
    f.write("".join(buffer[-int(0.01 * len(buffer)):]))
# Path: yolov4\preprocess.py
