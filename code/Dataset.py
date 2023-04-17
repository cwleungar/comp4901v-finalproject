from enum import Enum
import os
import random
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
classmap={
    0: 'Car',
    1: 'Pedestrian', 
    2: 'Cyclist', 
    3: 'Van', 
    4: 'Person_sitting', 
    5: 'Tram', 
    6: 'Truck', 
    7: 'Misc', 
    8: 'DontCare', 
    'Car': 0, 
    'Pedestrian': 1, 
    'Cyclist': 2, 
    'Van': 3, 
    'Person_sitting': 4,
    'Tram': 5, 
    'Truck': 6, 
    'Misc': 7, 
    'DontCare': 8
}

def resize_image_with_boxes(image, boxes, new_size):
    # Resize the image while maintaining aspect ratio
    old_size = image.size
    if old_size[0] > old_size[1]:
        new_width = new_size
        new_height = int(new_size * old_size[1] / old_size[0])
    else:
        new_height = new_size
        new_width = int(new_size * old_size[0] / old_size[1])
    image = image.resize((new_width, new_height))
    
    # Calculate the scaling factor for the bounding boxes
    width_scale = new_width / old_size[0]
    height_scale = new_height / old_size[1]
    
    # Adjust the bounding box locations based on the resized image
    new_boxes = []
    for box in boxes:
        x, y, w, h = box
        x = int(x * width_scale)
        y = int(y * height_scale)
        w = int(w * width_scale)
        h = int(h * height_scale)
        new_boxes.append([x, y, w, h])
    
    return image, new_boxes
def resize_image_with_boxes_to_square(image, boxes, new_size):
    width, height = image.size

    # Compute the scaling factor to resize the image to the new size
    scale = min(new_size/width, new_size/height)
    new_width = int(scale * width)
    new_height = int(scale * height)
    resized_image = F.resize(image, (new_height, new_width))

    # Compute the padding needed to make the image square
    padding_left = (new_size - new_width) // 2
    padding_top = (new_size - new_height) // 2

    # Pad the image and the boxes with zeros to make them square
    padded_image = F.pad(resized_image, (padding_left, padding_top, new_size - new_width - padding_left, new_size - new_height - padding_top), fill=0)
    padded_boxes = boxes * scale
    padded_boxes[:, [0, 2]] += padding_left
    padded_boxes[:, [1, 3]] += padding_top

    return padded_image, padded_boxes

class ObjectDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir,label_dir, split='training', val_split=0.1, transform=None):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.split = 'training' if (split == 'train' or split=='val') else 'testing'
        self.transform = transform
        self.image_dir = os.path.join(self.data_dir, self.split, 'image_2')
        self.label_dir = os.path.join(self.label_dir, self.split, 'label_2')
        self.filenames = os.listdir(self.image_dir)
        random.seed(42)
        random.shuffle(self.filenames)
        val_size = int(val_split * len(self.filenames))
        if split == 'train':
            self.filenames = self.filenames[val_size:]
        elif split == 'val':
            self.filenames = self.filenames[:val_size]
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        img_filename = os.path.join(self.image_dir, self.filenames[idx])
        img = Image.open(img_filename).convert('RGB')
        
        label_filename = os.path.join(self.label_dir, self.filenames[idx][:-4] + '.txt')
        with open(label_filename, "r") as f:
            labels_str = f.readlines()
        labels = []
        for word in labels_str:
            labels_str = word.split()
            labels.append([classmap[labels_str[0]]]+labels_str[1:])
        labels = np.array(labels, dtype=np.float32)        
        boxes = labels[:, 4:8]
        class_labels = labels[:, 0]
        img,boxes=resize_image_with_boxes_to_square(img, boxes, 416)
        if self.transform is not None:
            img, boxes = self.transform(img, boxes)
        
        target = torch.zeros((80, 5 + len(classmap)//2))
        for i in range(len(boxes)):
            target[i, :4] = boxes[i].float() / 416
            target[i, 4] = 1.0  # objectness
            target[i, 5:] =torch.tensor(class_labels[i]).float()
        return img, target
    


