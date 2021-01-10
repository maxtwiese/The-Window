import numpy as np
import os
import pandas as pd
from PIL import Image
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
from Datasets import TrainDataset

# Initialize Dataset
train_dataset = TrainDataset(r'../data/UBIRISPr_labels.csv')

def collate_fn(batch):
    return tuple(zip(*batch))

train_load = DataLoader(train_dataset,
                        batch_size=2, # arbitrary
                        shuffle=True,
                        num_workers=2, #arbitrary
                        collate_fn=collate_fn)

device = \
    torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#import matplotlib.pyplot as plt

#images, targets = next(iter(train_load))
#images = list(image.to(device) for image in images)
#targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

#boxes = targets[0]['boxes'].cpu().numpy().astype(np.int32)
#img = images[0].permute(1, 2, 0).cpu().numpy()
#fig, ax = plt.subplots(1, 1, figsize=(12, 6))

#for box in boxes:
#    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 1)
#ax.set_axis_off()
#ax.imshow(img)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace head
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9,
                            weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3,
                                               gamma=0.1)

num_epochs = 1 #40

itr = 1

for epoch in range(num_epochs):
    for images, targets in train_load:

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if itr % 50 == 0:
            print(f"Iteration #{itr} loss: {loss_value}")
        
        itr += 1

        lr_scheduler.step()

    print(f"Epoch #{epoch} loss: {loss_value}")

torch.save(model.state_dict(), 'model.pth')
torch.save({'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'opitimizer_state_dict': optimizer.state_dict(),
            }, 'ckpt.pth')
