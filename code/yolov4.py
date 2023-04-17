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
        x = self.backbone(x)
        
        # Pass the features through the feature pyramid network
        x = self.fpn(x)
        
        # Pass the features through the detection layers
        x = self.head(x)
        
        # Reshape the output to the YOLOv4 format
        x = x.permute(0, 2, 3, 1)
        x = x.view(x.shape[0], -1, (self.num_classes + 5))
        
        return x