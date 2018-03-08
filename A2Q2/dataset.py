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

all_characters = string.printable
n_characters = len(all_characters)

class imgDataset(Dataset):
    """Our dataset."""

    def __init__(self, filename):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.filename = filename
        self.raw_data = unidecode.unidecode(open(filename).read())

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        return {'input': [], 'target': []}