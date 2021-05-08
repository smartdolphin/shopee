import os
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


class ShopeeDataset(torch.utils.data.Dataset):
    def __init__(self, df, data_dir, transforms= None):
        self.df = df 
        self.root_dir = data_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.root_dir, row.image)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = row.label_group

        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented['image']

        return {
            'image' : image,
            'label' : torch.tensor(label).long()
        }

