from torch.utils.data import Dataset
import unidecode
import torch
import string
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, utils


class imgDataset(Dataset):
    """Our dataset."""

    def __init__(self, filedir):
        """
        Args:
            filedir (string): Path to the images (class is in filename).
        """
        self.filedir = os.path.join(os.getcwd(), filedir)
        self.raw_data = os.listdir(filedir)

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.filedir, self.raw_data[idx])
        image = io.imread(img_name)
        if 'Cat' in self.raw_data[idx]:
            target = 0
        else:
            target = 1
        return {'input': image, 'target': target}


def show_image(image):
    plt.figure()
    plt.imshow(image)
    plt.show()
