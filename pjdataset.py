import os
from os import listdir
from os.path import isfile, join
import numpy as np
import torch
from torch.utils.data import Dataset 
import cv2
from PIL import Image

def _read_image(path):
    return Image.open(path)

class BenchPressDataset(Dataset):
    def __init__(self, im_dir, transform):
        """
        im_dir should contain only folders with the data in it
        Structure should look like this:
        
        --im_dir
            --1
                --good
                --bad
            --2
                --good
                --bad

        """
        self.transform = transform
        self.im_dir = im_dir
        self.paths = []
        self.labels = []
        for d in listdir(im_dir):
            #load good images
            good_dir = join(im_dir, d, "good")
            good_image_paths = [join(good_dir, f) for f in listdir(good_dir) if isfile(join(good_dir, f)) and '.jpg' in f]
            self.labels += [1 for _ in range(len(good_image_paths))]
            self.paths += good_image_paths
            
            #load bad images
            bad_dir = join(im_dir, d, "bad")
            bad_image_paths = [join(bad_dir, f) for f in listdir(bad_dir) if isfile(join(bad_dir, f)) and '.jpg' in f]
            self.labels += [0 for _ in range(len(bad_image_paths))]
            self.paths += bad_image_paths
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = _read_image(self.paths[idx])
        if image is None:
            return self.__getitem__((idx + 1) % self.__len__())
        image = self.transform(image)

        return (image, self.labels[idx])
