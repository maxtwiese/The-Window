from argparse import ArgumentParser
import numpy as np
import os
import pandas as pd
from PIL import Image
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision
from torch.utils.data import DataLoader, Subset
from torch.utils.data import DistributedSampler
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from engine import train_one_epoch, evaluate
from Datasets import UBIRISPrDataset

#from clearml import Task
#task = Task.init(project_name='The Window',
#                 task_name='DDP Faster R-CNN on ResNet50 FPN')

# Hyper Parameters
seed = 29
num_classes = 2
num_epochs = 10 #40
batch_size = 4 #16
num_workers = 8 # 4 * GPUcount
learning_rate = 0.1
weight_decay = 0.01
gamma = 0.1

def main():
    
    # Distributed Data Parellel (DDP) wrapping adapted from @ sgraaf, to
    # solve first GPU problem. Also in `ddp_agent.sh`.
    parser = ArgumentParser('The Window DDP')
    parser.add_argument('--local_rank', type=int, default=-1, metavar='N',
                        help='Local process rank.')
    args = parser.parse_args()
    args.is_master = args.local_rank == 0 # Is 'master'
    args.device = torch.cuda.device(args.local_rank) # Set device
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(args.local_rank)
    torch.cuda.manual_seed_all(seed)
    
    # Freeze weights on all layers of pretrained Faster R-CNN ResNet-50
    # by removing gradient requirement then attach a fresh head  and send
    # to device.
    model = \
        torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features,
                                                      num_classes)
    model = model.to(args.local_rank)

    # Initialize Distributed Data Parallel model.
    ddp_model = DDP(model,
                device_ids=[args.local_rank],
                output_device=args.local_rank)

    # Load Data    
    def collate_fn(batch):
        return tuple(zip(*batch))

    train_set = UBIRISPrDataset(r'../data/Train_Set_small.csv')
    #dataset = UBIRISPrDataset(r'../data/UBIRISPr_Labels_small.csv')
    test_set = UBIRISPrDataset(r'../data/Test_Set_small.csv')
    # Based on last error message these both need targets.
    #test_dataset = TrainDataset(r'../data/Test_Set.csv')

    # split the dataset in train and test set
    #train_set = Subset(dataset, np.arange(80))
    #test_set = Subset(dataset, np.arange(80,100))
    
    #sampler = DistributedSampler(dataset)
    
    train_sampler = DistributedSampler(train_set)
    test_sampler = DistributedSampler(test_set)
    #train_set, test_set = random_split(dataset, [2000, 500])
    
    train_load = DataLoader(train_set,
                            batch_size=batch_size,
                            #shuffle=True,
                            num_workers=num_workers,
                            pin_memory=True,
                            collate_fn=collate_fn,
                            sampler=train_sampler)
    test_load = DataLoader(test_set,
                           batch_size=batch_size,
                           #shuffle=True,
                           num_workers=num_workers,
                           pin_memory=True,
                           collate_fn=collate_fn,
                           sampler=test_sampler)
    
    # Optimizer: Stochastic Gradient Descent
    params = [p for p in ddp_model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params,
                                lr=learning_rate,
                                momentum=0.9,
                                weight_decay=weight_decay)
    
    # Learning Rate Scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=gamma)

    #itr = 1
    
    for epoch in range(num_epochs):
        # Train for one epoch, printing every 10 iterations.
        train_one_epoch(ddp_model, optimizer, train_load,
                        args.local_rank, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(ddp_model, test_load, device=[args.local_rank])
        
        #ddp_model.train()
        #dist.barrier()
        #
        #for images, targets in train_load:
        #    
        #    # Where I assume my memory issue is.
        #    images = list(image.to(args.local_rank) for image in images)
        #    targets = [{k: v.to(args.local_rank) for k, v in t.items()} \
        #               for t in targets]
        #
        #    loss_dict = ddp_model(images, targets) # Returns loss and detections.
        #    losses = sum(loss for loss in loss_dict.values()) # Here too.
        #    loss_value = losses.detach()
        #
        #    optimizer.zero_grad()
        #    losses.backward()
        #    optimizer.step()
        #    lr_scheduler.step()
            
            #if itr % batch_size == 0:
            #    print(f"Iteration #{itr} loss: {loss_value}")
            #itr += 1
            
        #print(f"Epoch #{epoch} loss: {loss_value}")

    torch.save(ddp_model.state_dict(), 'model.pth')
    torch.save({'epoch': epoch,
                'model_state_dict': ddp_model.state_dict(),
                'opitimizer_state_dict': optimizer.state_dict(),
                }, 'ckpt.pth')

if __name__ == '__main__':
    main()
