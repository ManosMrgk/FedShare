import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VisionDataset

import PIL
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

class UTKFaceDataset(VisionDataset):

    def __init__(self, root, train=True, transform=None, download=False):
        super(UTKFaceDataset, self).__init__(root, transform=transform)
        self.root = root
        self.transform = transform

        # Load UTKFace dataset
        data_folder = root
        self.image_paths = [os.path.join(data_folder, img) for img in os.listdir(data_folder)]

        if download:
            # Implement any downloading logic if needed
            raise NotImplementedError("Downloading not supported for UTKFace")

        if train:
            train_size = 0.8  # Adjust as needed
            train_paths, val_paths = train_test_split(self.image_paths, train_size=train_size, shuffle=True)
            self.image_paths = train_paths if train else val_paths

        self.targets = []
        data_list = []
        for img in self.image_paths:
            gender = int(img.split("_")[1])
            self.targets.append(gender)
            image_name = img
            image = cv2.imread(image_name)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            # image=cv2.resize(image,(32,32))
            data_list.append(image)
        self.data = np.array(data_list)

    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        image_name = self.image_paths[index]
        image = cv2.imread(image_name)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        # image=cv2.resize(image,(32,32))
        image=transforms.ToTensor()(image)
        gender = int(self.image_paths[index].split("_")[1])
        if self.transform:
            image = self.transform(image)
        return (image, gender)