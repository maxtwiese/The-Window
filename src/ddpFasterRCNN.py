"""An implementation of Faster R-CNN for eye detection in PyTorch w/CUDA

With images from the UBIRISPr Database I am training a Faster Regional
Convolutional Neural Network (R-CNN) to detect eyes as the first step in
iris and pupil segmentation. This model is wrapped in a Data Distributed
Parallel (DDP) model that distributes the model in parallel across 4
GPUs training on their mean loss. The code is executed through a script
saved as 'ddp_agent.sh'. That file and the logic to wrap my code in a
DDP is based on the solution by @sgraaf who published a it as an intro
to wrapping in DDP (to which, this was my intro). The Faster R-CNN is
written for PyTorch and executed on their engine from the vision
repository as is it is written now (01-12-2021), but I have maintained
the original training below commented out. On an p3.8xlarge EC2 instance
(36 vCPU, 4 GPU) it is limping to train on 1800 images.

Next steps: 
- I am going to train using built in Datasets to debug where my load
pipeline is causing backup.
- Migrate to ClearML for better RCA on GPU cost issues
from clearml import Task
task = Task.init(project_name='The Window',
                 task_name='DDP Faster R-CNN on ResNet50 FPN')
"""



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



# Hyper Parameters
seed = 29
num_classes = 2
num_epochs = 10 
batch_size = 4 # Need more knowledge on batch_size % GPUCount != 0 situations.
num_workers = 8 # Have seen to use 4 * GPUcount, but need to investigate.
learning_rate = 0.1
weight_decay = 0.01
gamma = 0.1

def main():
    """"""
    
    # DDP wrapping
    parser = ArgumentParser('The Window DDP')
    parser.add_argument('--local_rank', type=int, default=-1, metavar='N',
                        help='Local process rank.')
    args = parser.parse_args()
    args.is_master = args.local_rank == 0
    args.device = torch.cuda.device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(args.local_rank)
    torch.cuda.manual_seed_all(seed)

    # Import a pretrained Faster R-CNN with a ResNet-50 backbone.
    model = \
        torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # Freeze weights
    for param in model.parameters():
        param.requires_grad = False
    # Attach a fresh head to train on UBIRISPr
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes)
    # Send to device.
    model = model.to(args.local_rank)
    # Wrap the Faster R-CNN in the DDP.
    ddp_model = DDP(model,
                device_ids=[args.local_rank],
                output_device=args.local_rank)

    # Build Datasets using the helper document build by csvBuilder
    train_set = UBIRISPrDataset(r'../data/Train_Set_small.csv')
    test_set = UBIRISPrDataset(r'../data/Test_Set_small.csv')
    #dataset = UBIRISPrDataset(r'../data/UBIRISPr_Labels_small.csv')`
    #train_set, test_set = random_split(dataset, [2000, 500])


    
    # Define collate function and samplers for DDP.
    def collate_fn(batch):
        return tuple(zip(*batch))

    train_sampler = DistributedSampler(train_set)
    test_sampler = DistributedSampler(test_set)
    #sampler = DistributedSampler(dataset)   

    # Build Train and Test DataLoaders.
    train_load = DataLoader(train_set,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=True,
                            collate_fn=collate_fn,
                            sampler=train_sampler)
    test_load = DataLoader(test_set,
                           batch_size=batch_size,
                           num_workers=num_workers,
                           pin_memory=True,
                           collate_fn=collate_fn,
                           sampler=test_sampler)
    
    # Optimizer: Stochastic Gradient Descent only optimize the head.
    params = [p for p in ddp_model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, # Need more efficient access method.
                                lr=learning_rate,
                                momentum=0.9,
                                weight_decay=weight_decay)
    
    # Learning Rate Scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=gamma)

    # Trained using the PyTorch Vision framework.
    for epoch in range(num_epochs):
        # Train for one epoch, printing every 10 iterations.
        train_one_epoch(ddp_model, optimizer, train_load,
                        args.local_rank, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset.
        evaluate(ddp_model, test_load, device=[args.local_rank])
    # Save the model
    torch.save(ddp_model.state_dict(), 'model.pth')
    torch.save({'epoch': epoch,
                'model_state_dict': ddp_model.state_dict(),
                'opitimizer_state_dict': optimizer.state_dict(),
                }, 'ckpt.pth')
    # Below is the original framework of training outside of the Vision
    # framework. I am still having memory issues so this will stay in
    # reserve for now.    
        #ddp_model.train()
        #dist.barrier()
        
        #for images, targets in train_load:
   
        #    # Where I assume my memory issue is.
        #    images = list(image.to(args.local_rank) for image in images)
        #    targets = [{k: v.to(args.local_rank) for k, v in t.items()} \
        #               for t in targets]
        
        #    loss_dict = ddp_model(images, targets)
        #    losses = sum(loss for loss in loss_dict.values())
        #    loss_value = losses.detach() # Drops graphical map.
        
        #    optimizer.zero_grad()
        #    losses.backward()
        #    optimizer.step()
        #    lr_scheduler.step()
            
        #print(f"Epoch #{epoch} loss: {loss_value}")

if __name__ == '__main__':
    main()
