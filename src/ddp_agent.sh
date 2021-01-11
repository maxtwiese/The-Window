#!/bin/bash

# Single node with 2 GPUs
export NUM_NODES=1
export NUM_GPUS_PER_NODE=4
export NODE_RANK=0
export WORLD_SIZE=$(($NUM_NODES * $NUM_GPUS_PER_NODE))

# launch your script w/ `torch.distributed.launch`
python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS_PER_NODE \
    --nnodes=$NUM_NODES \
    --node_rank $NODE_RANK \
    ddpFasterRCNN.py \
    # include any arguments to your script, e.g:
    #    --seed 42
    #    etc
