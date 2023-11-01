import os
import cv2
import pandas as pd
from torchvision import transforms
from torchvision.datasets import VisionDataset
from torchvision.io import read_image
import torch

class FairFaceDataset(VisionDataset):
    def __init__(self, root, train=True, transform=None, download=False):
        super(FairFaceDataset, self).__init__(root, transform=transform)
        self.root = root
        self.transform = transform
        self.split = 'train' if train else 'val'
        self.data = None
        # Load FairFace dataset
        data_folder = root #os.path.join(root, self.split)

        if download:
            raise NotImplementedError("Downloading not supported for FairFace")

        # Load label data
        self.data_info = pd.read_csv(os.path.join(root, f'fairface_label_{self.split}.csv'))
        self.targets = {'race': [], 'gender': [], 'age': []}
        self.image_paths = [os.path.join(data_folder, img) for img in self.data_info['file']]
        for img_path in self.image_paths:
            # Extract attributes from the filename
            _, filename = os.path.split(img_path)
            attributes = self.data_info[self.data_info['file'] == self.split+'/'+filename]

            if not attributes.empty:
                gender = attributes.iloc[0]['gender']
                race = attributes.iloc[0]['race']
                age = attributes.iloc[0]['age']

                # Append labels
                self.targets['gender'].append(0 if gender == 'Male' else 1)
                self.targets['race'].append(race)
                self.targets['age'].append(age)

    def __len__(self):
        if self.data:
            return len(self.data)
        return len(self.data_info)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        # image = read_image(image_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        image=transforms.ToTensor()(image)
        gender = self.targets['gender'][index]
        # race = self.targets['race'][index]
        # age = self.targets['age'][index]

        if self.transform:
            image = self.transform(image)

        return (image,gender)
