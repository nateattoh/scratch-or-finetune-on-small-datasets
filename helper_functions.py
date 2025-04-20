import torch
import numpy as np
import matplotlib.pyplot as plt

import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

import pathlib
from pathlib import Path
import os
import PIL.Image

import random

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, directory: str, labels: np.array, transforms: bool):
        self.paths = sorted(list(pathlib.Path(directory).glob("*.jpg"))) # gets the path of all jpg images in the directory
        self.transforms = transforms
        self.labels = labels # the labels array provided to us where each label at a particular index corresponds to an image at that order in the sorted folder names 
        self.classes = torch.unique(torch.tensor([int(x) for x in image_labels])) # all the unique class names in the dataset
    
    def load_image(self, index: int) -> PIL.Image.Image:
        image_path = self.paths[index] # gets the particular image at the specified index 
        return PIL.Image.open(image_path)

    # overriding the __len__() method
    def __len__(self) -> int:
        return len(self.paths)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        image = self.load_image(index=index)
        class_name = int(self.labels[index])

        if self.transforms:

            # to_tensor = transforms.ToTensor()
            # image_tensor = to_tensor(image)

            # mean = torch.mean(image_tensor, dim=(1, 2))
            # std = torch.std(image_tensor, dim=(1, 2))

            data_transforms = transforms.Compose([
                transforms.Resize(size=(299, 299)), # resizing images to 256x256
                transforms.CenterCrop(size=(224, 224)), #new
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5), #new
                transforms.ColorJitter(brightness=0.5, hue=0.2), #new
                transforms.RandomPosterize(bits=2),
                transforms.ToTensor()
                # transforms.Normalize(mean=mean, std=std)
            ])
        

            return data_transforms(image), class_name
        else:
            return image, class_name


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images: list[torch.Tensor], labels: list[int]):
        self.images = images
        self.labels = labels

    def __len__(self):
        if len(self.images) == len(self.labels):
            return len(self.images)
        else:
            raise ValueError("Lengths of image and label list are not equal")
    
    def __getitem__(self, index):
        return self.images[index], self.labels[index]




