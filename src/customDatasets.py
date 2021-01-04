import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io

class PeriocularDataSet(Dataset):
    """extends torch Dataset for UBIRIS Periocular dataset"""
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(path)
        center_x = torch.tensor(int(self.annotations.iloc[index, 1]))
        center_y = torch.tensor(int(self.annotations.iloc[index, 2]))

        if self.transform:
            image = self.transform(image)

        return (image, center_x, center_y)
