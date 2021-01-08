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
        targets = []
#        for i in range(self.__len__()):
        d = {}
        d['boxes'] = torch.FloatTensor(self.annotations.iloc[index, 5:9].astype('float32'))
        d['labels'] = 1 #torch.tensor([[1]]).to(torch.int64)
        targets.append(d)
        path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(path)

        if self.transform:
            image = self.transform(image)

        return (image, targets)
