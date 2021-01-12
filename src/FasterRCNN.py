from argparse import ArgumentParser
import numpy as np
import os
import pandas as pd
from PIL import Image
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import pkbar
from Datasets import TrainDataset, TestDataset

# Hyper Parameters
seed = 29
num_classes = 2
num_epochs = 10
batch_size = 32
num_workers = 8 # 4 * GPUcount
learning_rate = 0.01
weight_decay = 0.005
gamma = 0.1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Replace head on pretrained Faster R-CNN ResNet and send to device.
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model = model.to(device)

# Load Data
def collate_fn(batch):
    return tuple(zip(*batch))
                 
train_dataset = TrainDataset(r'../data/Train_Set.csv')
#test_dataset = TestDataset(r'../darta/Test_Set.csv')
train_load = DataLoader(train_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=num_workers,
                        pin_memory=True,
                        collate_fn=collate_fn)

#params = [p for p in model.parameters() if p.requires_grad]
                 
optimizer = torch.optim.SGD(model.roi_heads.box_predictor.parameters(),
                            lr=learning_rate,
                            momentum=0.9,
                            weight_decay=weight_decay)
                 
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=gamma)

itr = 1

for epoch in range(num_epochs):
    for images, targets in train_load:

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        #loss_value = losses.item()
        loss_value = losses.detach()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        if itr % batch_size == 0:
            print(f"Iteration #{itr} loss: {loss_value}")

        itr += 1

        lr_scheduler.step()

    print(f"Epoch #{epoch} loss: {loss_value}")


torch.save(model.state_dict(), 'model.pth')
torch.save({'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'opitimizer_state_dict': optimizer.state_dict(),
            }, 'ckpt.pth')
    