import numpy as np
from PIL import Image
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class RandomHorizontalFlip(object):
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, target,depth=None):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
            if type(depth) != type(None):
                depth=F.hflip(depth)
        if type(depth) != type(None):
            return image, target,depth
        return image, target

class RandomVerticalFlip(object):
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, target,depth):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            target = F.vflip(target)
            if type(depth) != type(None):
                depth=F.vflip(depth)
        if type(depth) != type(None):
            return image, target,depth    
        return image, target
class RandomRotation(object):
    def __init__(self, degrees=(0,90)):
        self.degrees = degrees

    def __call__(self, image, target):
        d=random.randint(self.degrees[0],self.degrees[1])
        image = F.rotate(image,float(d))
        target=torch.unsqueeze(target,0)
        target = F.rotate(target,float(d),fill =[-1])
        target=target.squeeze()
        return image, target
    
class Normalize(T.Normalize):
    def __call__(self, image, target):
        return super().__call__(image), target


class Normalize3(T.Normalize):
    def __call__(self, image, target, target2):
        return super().__call__(image), target, target2


class ColorJitter(T.ColorJitter):
    def __call__(self, image, target):
        return super().__call__(image), target

class ColorJitter3(T.ColorJitter):
    def __call__(self, image, target, target2):
        return super().__call__(image), target, target2