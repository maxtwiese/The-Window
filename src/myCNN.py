# iris segmentation using Canny edge detection and a circular hough transform
# citation: [23] Sangwan, S. and R. Rani, A Review on: Iris Recognition.
#           (IJCSIT) International Journal of Computer Scienceand Information
#           Technologies, 2015. 6(4): p. 3871-3873

# iris normalization using 
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from customDatasets import PeriocularDataSet

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
#in_channel = 3
num_classes = 2
learning_rate = 1e-3
batch_size = 1
num_epochs = 1
#normalize = T.Normalize(mean=[0.485, 0.456, 0.496], std=[0.229, 0.224, 0.225])
normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
transform = T.Compose([T.ToTensor(),
                       T.Resize(256),
                       T.CenterCrop(224),
                       normalize])

# Load Data
dataset = PeriocularDataSet(csv_file=r'../data/UBIRISPr_Labels.csv',
                            root_dir=r'../data/UBIRISPr',
                            transform=transform)

train_set, test_set = random_split(dataset, [2, 10197])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

# model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.to(device)

# Loss and optimizer
#criterion = F.softmax(data, dim=0)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    losses = []
    for batch_idx, (data, targets) in enumerate(train_loader):
        print(train_loader.dataset.__getitem__(batch_idx)[1])
        # Get data to cuda if possible
        data = data.to(device=device)
        #targets = targets.to(device=device)

        # Forwards
        output = model(data, targets)
        print(output)
        loss = criterion(output['loss_classifier'], targets)
        #loss = criterion(output, targets)

        losses.append(loss.item())

        # Backwards
        optimizer.zero_grad()
        loss.backward()

        # Gradient Decent or Adam step
        optimizer.step()

    print(f'Cost at epoch {epoch} is {sum(losses)/len(losses)}')

# Check Accuracy on training to see how good our model is
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            output = model(x)
            _, predictions = output.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Accuracy: {float(num_correct)/float(num_samples)*100}%.')

    model.train()

print("Checking accuracy on Training Set.")
check_accuracy(train_loader, model)

print("Checking accuracy on Test Set.")
check_accuracy(test_loader, model)