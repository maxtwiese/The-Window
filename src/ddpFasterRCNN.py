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
from Datasets import TrainDataset

# Hyper Parameters
seed = 29
num_classes = 2
num_epochs = 1 #40
batch_size = 2 #16
num_workers = 8 # 4 * GPUcount
learning_rate = 0.5 # 0.001
weight_decay = 0.05 # 0.0005
gamma = 0.1

def main():
    # DDP adapted from @ sgraaf, to solve first GPU problem.
    parser = ArgumentParser('The Window DDP')
    parser.add_argument('--local_rank', type=int, default=-1, metavar='N',
                        help='Local process rank.')
    args = parser.parse_args()
    args.is_master = args.local_rank == 0 # Is 'master'
    args.device = torch.cuda.device(args.local_rank) # Set device
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(args.local_rank)
    torch.cuda.manual_seed_all(seed)
    
    # Replace head on pretrained Faster R-CNN ResNet and send to device.
    model = \
        torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features,
                                                      num_classes)
    model = model.to(args.local_rank)

    # Initialize Distributed Data Parallel
    ddp_model = DDP(model,
                device_ids=[args.local_rank],
                output_device=args.local_rank)

    # Load Data    
    def collate_fn(batch):
        return tuple(zip(*batch))

    train_dataset = TrainDataset(r'../data/Train_Set.csv')
    sampler = DistributedSampler(train_dataset)
    
    train_load = DataLoader(train_dataset,
                            batch_size=batch_size,
                            #shuffle=True,
                            num_workers=num_workers,
                            pin_memory=True,
                            collate_fn=collate_fn,
                            sampler=sampler)
    
    #params = [p for p in ddp_model.parameters() if p.requires_grad]
    
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
            
            # Where I assume my memory issue is.
            images = list(image.to(args.local_rank) for image in images)
            targets = [{k: v.to(args.local_rank) for k, v in t.items()} \
                       for t in targets]

            loss_dict = ddp_model(images, targets)
            losses = sum(loss for loss in loss_dict.values()) # Here too.
            loss_value = losses.detach()

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            lr_scheduler.step()
            
            if itr % batch_size == 0:
                print(f"Iteration #{itr} loss: {loss_value}")
            itr += 1
            
        print(f"Epoch #{epoch} loss: {loss_value}")

    torch.save(ddp_model.state_dict(), 'model.pth')
    torch.save({'epoch': epoch,
                'model_state_dict': ddp_model.state_dict(),
                'opitimizer_state_dict': optimizer.state_dict(),
                }, 'ckpt.pth')

if __name__ == '__main__':
    main()
