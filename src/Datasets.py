import os
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    """Extension of torch Dataset for training on UBIRISPR images."""
    def __init__(self, csv_file):
        super().__init__()

        self.annotations = pd.read_csv(csv_file)
        self.imgs = self.annotations['FileName']

    def __getitem__(self, index: int):
        # maybe issue with float32
        transforms = T.Compose([T.Resize(800),
                                T.CenterCrop(800),
                                T.ToTensor(),
                                T.ConvertImageDtype(torch.float32)])
        img = Image.open(os.path.join(r'../data/UBIRISPr',self.imgs[index]))
        img = transforms(img)
        img /= 255.0

        boxes = self.annotations[['X1', 'Y1', 'X2', 'Y2']].values
        area = (boxes[:,3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # Convert to Tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area  = torch.as_tensor(area, dtype=torch.float32) # COCO
        labels = torch.ones((boxes.shape[0],), dtype=torch.int16)
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int16)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])
        target['area'] = area # COCO
        target['iscrowd'] = iscrowd
        return img, target

    def __len__(self) -> int:
        return self.imgs.shape[0]
