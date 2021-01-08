import numpy as np
import torch
from torch import nn
import torchvision

image = torch.zeros((1, 3, 800, 800)).float()

bbox = torch.FloatTensor([[20, 30, 400, 500], [300, 400, 500, 600]]) # [y1, x1, y2, x2] format
labels = torch.LongTensor([6, 8]) # 0 represents background
sub_sample = 16 # 800x800 -> 50x50

# Feature Extraction
dummy_img = torch.zeros((1, 3, 800, 800)).float()

model = torchvision.models.vgg16(pretrained=True)
fe = list(model.features)

req_features = []
k = dummy_img.clone()
for i in fe:
    k  = i(k)
    if k.size()[2] < 800//16:
        break
    req_features.append(i)
    out_channels = k.size()[1]

faster_rcnn_fe_extractor = nn.Sequential(*req_features)

out_map = faster_rcnn_fe_extractor(dummy_img)
print(out_map.size())

# Anchor Boxes.
ratios  = [0.5, 1, 2]
scales = [8, 16, 32]

anchor_base = np.zeros((len(ratios) * len(scales), 4), dtype=np.float32)
#anchor_base = torch.zeros((len(ratios) * len(scales), 4), dtype=torch.float32)

ctr_y = sub_sample / 2.
ctr_x = sub_sample / 2.

print(ctr_y, ctr_x)

for i in range(len(ratios)):
    for j in range(len(scales)):
        h = sub_sample * scales[j] * np.sqrt(ratios[i])
        w = sub_sample * scales[j] * np.sqrt(1./ratios[i])

        index = i * len(scales) + j
        anchor_base[index, 0] = ctr_y - h / 2
        anchor_base[index, 1] = ctr_x - w / 2
        anchor_base[index, 2] = ctr_y + h / 2
        anchor_base[index, 3] = ctr_x + w / 2

fe_size = (800//16)
shift_x = np.arange(16, (fe_size+1) * 16, 16)
shift_y = np.arange(16, (fe_size+1) * 16, 16)
ctr = np.zeros((len(shift_x) * len(shift_y), 2), dtype=np.float32)
#print(shift_x)
#print(shift_y)
index = 0
for x in shift_x:
    index = 0
    for y in shift_y:
        ctr[index, 1] = x - 8
        ctr[index, 0] = y - 8
        index += 1
print(ctr)

anchors = np.zeros((fe_size * fe_size * 9, 4))

index = 0
for c in ctr:
    ctr_y, ctr_x = c
    for i in range(len(ratios)):
        for j in range(len(scales)):
            h = sub_sample * scales[j] * np.sqrt(ratios[i])
            w = sub_sample * scales[j] * np.sqrt(1./ratios[i])

            index = i * len(scales) + j
            anchor_base[index, 0] = ctr_y - h / 2.
            anchor_base[index, 1] = ctr_x - w / 2.
            anchor_base[index, 2] = ctr_y + h / 2.
            anchor_base[index, 3] = ctr_x + w / 2.
            index += 1
print(anchors.shape)