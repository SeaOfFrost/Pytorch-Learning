#Initializations
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import numpy as np
import pdb
torch.set_printoptions(linewidth=120)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#Parameters
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001

#Download Dataset
train_set = torchvision.datasets.FashionMNIST(root='../../data/', train=True, transform=transforms.ToTensor())
#train_set = torchvision.datasets.FashionMNIST(
 #   root='./data/FashionMNIST'
  #  ,train=True
   # ,download=True
    #,transform=transforms.Compose([transforms.ToTensor()])
#) */

test_set = torchvision.datasets.FashionMNIST(root='../../data/', train=False, transform=transforms.ToTensor())

#Load Dataset
train_loader = torch.utils.data.DataLoader(dataset = train_set,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset = test_set,batch_size=batch_size,shuffle=True)

class network(nn.Module):
    def __init__(self, num_classes=10):
        super(network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        
        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)       
    
    def forward(self,t):
        #input layer
        t = t
        
        #hidden conv layer 1
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t,kernel_size=2, stride=2)
        
        #hidden conv layer 2
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t,kernel_size=2, stride=2)
        
        #First FC layer
        t = t.reshape(-1,12*4*4)
        t = self.fc1(t)
        t = F.relu(t)
        
        #Second FC layer
        t = self.fc2(t)
        t = F.relu(t)
        
        #Output layer
        t = self.out(t)
        
        return t

model = network(num_classes).to(device)

#Loss and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

#Training
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images,labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        #Forward Prop
        outputs = model(images)
        loss = loss_function(outputs,labels)
        #Backprop
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

