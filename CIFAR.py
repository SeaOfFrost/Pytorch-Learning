#Initialization
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#Parameters
num_epochs = 10
num_classes = 10
batch_size = 100
learning_rate = 0.001

#Download Dataset
train_set = torchvision.datasets.CIFAR10(root='../../data/',
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=False)

test_set = torchvision.datasets.CIFAR10(root='../../data/',
                                          train=False, 
                                          transform=transforms.ToTensor())

#Load Dataset
train_loader = torch.utils.data.DataLoader(dataset=train_set,batch_size= batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set,batch_size= batch_size, shuffle=True)

from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

class SimpleCNN(torch.nn.Module):
    def  __init__(self, num_classes=10):
        super(SimpleCNN,self).__init__()
        self.conv1 = nn.Conv2d(3,10,kernel_size=5,stride=1,padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        self.conv2 = nn.Conv2d(10,12,kernel_size=5,stride=1,padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        self.fc1 = nn.Linear(12*22*22,1024)
        self.fc2 = nn.Linear(1024,num_classes)
        
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        #print(x.shape)
        x = x.reshape(-1,12*22*22)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return(x)

import torch.optim as optim
model = SimpleCNN(num_classes).to(device)

#Loss and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

#Training
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        #Forwardprop
        outputs = model(images)
        loss = loss_function(outputs,labels)
        #backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #Print data
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

#Testing
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted==labels).sum().item()
    print('Test Accuracy = {} %' .format(100*correct/total))
