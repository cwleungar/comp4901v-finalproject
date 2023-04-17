from enum import Enum
import os
import random
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
classmap={
    0: 'Car',
    1: 'Pedestrian', 
    2: 'Cyclist', 
    3: 'Van', 
    4: 'Person_sitting', 
    5: 'Tram', 
    6: 'Truck', 
    7: 'Misc', 
    -1: 'DontCare', 
    'Car': 0, 
    'Pedestrian': 1, 
    'Cyclist': 2, 
    'Van': 3, 
    'Person_sitting': 4,
    'Tram': 5, 
    'Truck': 6, 
    'Misc': 7, 
    'DontCare': -1
}


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
        
        if self.transform is not None:
            img, boxes = self.transform(img, boxes)
        boxes = torch.from_numpy(boxes)
        
        return img, boxes, class_labels
    