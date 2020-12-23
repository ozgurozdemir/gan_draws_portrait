import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import PIL
import time
import random
import os


# Src: https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset
class WikiartDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
      self.images = images
      self.labels = labels
      if transform:
        self.transform = transform
      else:
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0), std=(1))]
            )

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
      if self.transform:
        return self.transform(self.images[idx]), int(self.labels[idx])
      else:
        return self.images[idx], int(self.labels[idx])
        
        
# Src: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html      
class CelebaDataset(torch.utils.data.Dataset):
    def __init__(self, file_list, root_path, transform=None):
      self.file_list = file_list
      self.root_path = root_path
      if transform:
        self.transform = transform
      else:
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((128,128)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0), std=(1))]
            )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
      image = PIL.Image.open(f"{self.root_path}/{self.file_list[idx]}")
      if self.transform:
        return self.transform(image), 0
      else:
        return image, 0