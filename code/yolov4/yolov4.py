import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2
import os
import time
import argparse
import pickle
import torch.nn as nn
from ..yolov3.utils.metrics import bbox_iou

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.leaky_relu(x)
        return x

class Darknet53(nn.Module):
    def __init__(self):
        super(Darknet53, self).__init__()
        
        self.conv1 = ConvBlock(3, 32, 3, padding=1)
        self.conv2 = ConvBlock(32, 64, 3, stride=2, padding=1)
        self.resblock1 = nn.Sequential(ConvBlock(64, 32, 1), ConvBlock(32, 64, 3, padding=1))
        self.conv3 = ConvBlock(64, 128, 3, stride=2, padding=1)
        self.resblock2 = nn.Sequential(ConvBlock(128, 64, 1), ConvBlock(64, 128, 3, padding=1))
        self.conv4 = ConvBlock(128, 256, 3, stride=2, padding=1)
        self.resblock3 = nn.Sequential(ConvBlock(256, 128, 1), ConvBlock(128, 256, 3, padding=1))
        self.conv5 = ConvBlock(256, 512, 3, stride=2, padding=1)
        self.resblock4 = nn.Sequential(ConvBlock(512, 256, 1), ConvBlock(256, 512, 3, padding=1))
        self.conv6 = ConvBlock(512, 1024, 3, stride=2, padding=1)
        self.resblock5 = nn.Sequential(ConvBlock(1024, 512, 1), ConvBlock(512, 1024, 3, padding=1))
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.resblock1(x) + x
        x = self.conv3(x)
        x = self.resblock2(x) + x
        x = self.conv4(x)
        x = self.resblock3(x) + x
        x = self.conv5(x)
        x = self.resblock4(x) + x
        x = self.conv6(x)
        x = self.resblock5(x) + x
        return x
class ConcatBlock(nn.Module):
    def __init__(self):
        super(ConcatBlock, self).__init__()

    def forward(self, x, y):
        # Upsample x to the size of y
        x = nn.functional.interpolate(x, size=y.size()[2:], mode='nearest')
        
        # Concatenate x and y along the channel dimension
        out = torch.cat([y, x], dim=1)
        return out
    
class YOLOv4(nn.Module):
    def __init__(self, num_classes=80):
        super(YOLOv4, self).__init__()
        self.num_classes = num_classes
        # Define the backbone network
        self.backbone = Darknet53()

        # Define the feature pyramid network
        self.fpn = nn.Sequential(
            ConvBlock(512, 256, 1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConcatBlock(),
            ConvBlock(768, 256, 1),
            ConvBlock(256, 512, 3, padding=1),
            ConvBlock(512, 256, 1),
            ConvBlock(256, 512, 3, padding=1),
            ConvBlock(512, 256, 1),
            ConvBlock(256, 512, 3, padding=1),
            ConvBlock(512, 256, 1),
            ConvBlock(256, 512, 3, padding=1)
        )
        
        # Define the detection layers
        self.head = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, (num_classes + 5) * 3, kernel_size=1, stride=1, padding=0),
        )
        
    def forward(self, x):
        # Pass the input through the backbone network
        print(x.shape)
        x = self.backbone(x)
        print(x.shape)
        # Pass the features through the feature pyramid network
        split_sizes = [int(x.shape[1] * 0.125), int(x.shape[1] * 0.25), int(x.shape[1] * 0.5)]
        total_split_size = sum(split_sizes)
        if total_split_size != x.shape[1]:
            diff = x.shape[1] - total_split_size
            split_sizes[1] += diff
        x0, x1, x2 = torch.split(x, split_sizes, dim=1)
        
        x = self.fpn(x2,x2)

        # Pass the features through the detection layers
        x = torch.cat([x, x1], dim=1)

        x = self.head(x)

        # Reshape the output to the YOLOv4 format
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        x = x.view(B, H*W, (self.num_classes + 5) * 3)

        return x
    
class YOLOv4Loss(nn.Module):
    def __init__(self, ignore_thresh=0.5):
        super(YOLOv4Loss, self).__init__()
        self.ignore_thresh = ignore_thresh
        self.mse_loss = nn.MSELoss()
        self.cn_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.obj_scale = 1
        self.noobj_scale = 100
        self.reg_scale = 0.1
        self.class_scale = 10
    def forward(self, pred, target):
        # Split the output tensor into bounding box coordinates, objectness scores, and class probabilities
        pred_boxes = pred[..., :4]
        pred_obj = pred[..., 4:5]
        pred_class = pred[..., 5:]
        
        # Split the target tensor into bounding box coordinates, objectness masks, and class masks
        target_boxes = target[..., :4]
        target_obj = target[..., 4:5]
        target_class = target[..., 5:]
        
        # Calculate the loss for the bounding box coordinates
        loss_x = self.mse_loss(pred_boxes[..., 0:1], target_boxes[..., 0:1])
        loss_y = self.mse_loss(pred_boxes[..., 1:2], target_boxes[..., 1:2])
        loss_w = self.mse_loss(torch.sqrt(pred_boxes[..., 2:3]), torch.sqrt(target_boxes[..., 2:3]))
        loss_h = self.mse_loss(torch.sqrt(pred_boxes[..., 3:4]), torch.sqrt(target_boxes[..., 3:4]))
        loss_coord = loss_x + loss_y + loss_w + loss_h
        
        # Calculate the loss for the objectness scores
        loss_obj = self.cn_loss(pred_obj, target_obj)
        
        # Calculate the loss for the class probabilities
        loss_class = self.cn_loss(pred_class, target_class)
        iou = bbox_iou(pred_boxes, target_boxes)
        obj_mask = target[..., 5].view(-1)
        noobj_mask = (iou < self.ignore_thresh).float() * (1 - obj_mask)
        noobj_loss = F.binary_cross_entropy_with_logits(pred_obj, noobj_mask, reduction='none')
        noobj_loss = noobj_loss.view(self.batch_size, -1).sum(dim=1).mean()
        # Calculate the total loss
        loss = self.obj_scale * loss_obj + self.reg_scale * loss_coord + loss_class*self.class_scale + self.noobj_scale * noobj_loss
        
        return loss
def get_anchors():
    anchors = np.array([[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]])
    anchors = anchors / 416
    anchors = anchors.astype(np.float32)
    return anchors
