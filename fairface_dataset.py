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

class FairFaceDataset(VisionDataset):

    def __init__(self, root, train=True, transform=None, download=False):
        super(FairFaceDataset, self).__init__(root, transform=transform)
        self.root = root
        self.transform = transform
        split = 'train' if train else 'val'
        # Load FairFace dataset
        data_folder = os.path.join(root, split)

        if download:
            # Implement any downloading logic if needed
            raise NotImplementedError("Downloading not supported for FairFace")

        self.image_paths = [os.path.join(data_folder, img) for img in os.listdir(data_folder)]
        
        self.targets = {'race': [], 'gender': [], 'age': []}
        data_list = []

        labels = pd.read_csv(os.path.join(root, f'fairface_label_{split}.csv'))

        for img_path in self.image_paths:
            # Extract attributes from the filename
            _, filename = os.path.split(img_path)
            attributes = labels[labels['file'] == split+'/'+filename]

            if not attributes.empty:
                gender = attributes.iloc[0]['gender']
                race = attributes.iloc[0]['race']
                age = attributes.iloc[0]['age']

                # Append labels
                self.targets['gender'].append(gender)
                self.targets['race'].append(race)
                self.targets['age'].append(age)

                # Load and preprocess the image
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = transforms.ToTensor()(image)
                data_list.append(image)

        self.data = np.array(data_list)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image = self.data[index]
        gender = self.targets['gender'][index]
        race = self.targets['race'][index]
        age = self.targets['age'][index]

        if self.transform:
            image = self.transform(image)
        return (image, gender)
        # return (image, {'gender': gender, 'race': race, 'age': age})