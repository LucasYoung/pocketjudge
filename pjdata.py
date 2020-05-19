import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch

from utils import calc_dataset_stats
import pjdataset

class StandardizeSizeTransform():
    def __init__(self):
        pass
    """
    def __call__(self, image):
        if image.shape == (3, 640, 640):
            return image
        elif image.shape == (3, 360, 640):
            black = torch.zeros([3, 140, 640], dtype=torch.float)
            image = torch.cat([black, image, black], 1) # pad zeros along H axis
            return image
        else:
            return None
    """
    def __call__(self, image):
        if image.shape == (3, 640, 640):
            return image
        else: # Image Shape is (3, x, y)
            width = image.shape[1]
            height = image.shape[2]
            if height > 640:
                image = self._crop_height(image, width, height)
            if height < 640:
                image = self._extend_height(image, width, height)
            if width > 640:
                image = self._crop_width(image, width, height)
            if width < 640:
                image = self._extend_width(image, width, height)
            return image
            
    def _extend_width(self, image, width, height):
        extension = int((640 - width)/2)
        black = torch.zeros([3, extension, height], dtype=torch.float)
        image = torch.cat([black, image, black], 1)
        return image
    
    def _extend_height(self, image, width, height):
        extension = int((640 - height)/2)
        black = torch.zeros([3, width, extension], dtype=torch.float)
        image = torch.cat([black, image, black], 0)
        return image
        pass

    def _crop_width(self, image, width, height):
        crop = (width - 640)/2
        return image[:,crop:width-crop,:]

    def _crop_height(self, image, width, height):
        crop = (height - 640)/2
        return image[:,:,crop:height-crop]

class BenchPressData:
    def __init__(self, args):
        stats_transform = transforms.Compose(
            [transforms.ToTensor(),
            StandardizeSizeTransform()]
        )
        mean, std = calc_dataset_stats(
            pjdataset.BenchPressDataset(im_dir="./pj_dataset_train_mini", transform=stats_transform), 
            axis=(0, 1, 2, 3)
        )
        print(mean)

        train_transform = transforms.Compose(
            [#transforms.RandomCrop(args.img_height),
             transforms.RandomHorizontalFlip(),
             transforms.ColorJitter(0.3, 0.3, 0.3),
             transforms.ToTensor(),
             StandardizeSizeTransform(),
             transforms.Normalize(mean=mean, std=std)])

        test_transform = transforms.Compose(
            [transforms.ToTensor(),
            StandardizeSizeTransform(),
            transforms.Normalize(mean=mean, std=std)])

        self.trainloader = DataLoader(pjdataset.BenchPressDataset(im_dir="./data/pj_dataset_train", transform=train_transform),
                                      batch_size=args.batch_size,
                                      shuffle=args.shuffle, num_workers=args.dataloader_workers,
                                      pin_memory=args.pin_memory)

        self.testloader = DataLoader(pjdataset.BenchPressDataset(im_dir="./data/pj_dataset_test", transform=test_transform),
                                     batch_size=args.batch_size,
                                     shuffle=False, num_workers=args.dataloader_workers,
                                     pin_memory=args.pin_memory)


PJ_LABELS_LIST = [
    'bad'
    'good'
]
