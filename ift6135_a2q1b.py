import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

# Hyper Parameters 
input_size = 784
hidden_size1 = 800
hidden_size2 = 800
num_classes = 10
num_epochs = 100
batch_size = 64
learning_rate = 0.02

# MNIST Dataset 
train_dataset = dsets.MNIST(root='./data', 
                            train=True, 
                            transform=transforms.ToTensor(),  
                            download=True)

test_dataset = dsets.MNIST(root='./data', 
                           train=False, 
                           transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

# Neural Network Model (2 hidden layer)
class Net(nn.Module):
    def __init__(self, input_size, hidden_size1,hidden_size2, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, num_classes)  
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.drop = nn.Dropout(p=0.5)
        
        nn.init.xavier_uniform(self.fc1.weight)
        nn.init.xavier_uniform(self.fc2.weight)
        nn.init.xavier_uniform(self.fc3.weight)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.drop(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.softmax(out)
        return out
    #1 Multiply 0.5 to the hidden layer before prediction
    def forward1(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = 0.5*out
        out = self.fc3(out)
        out = self.softmax(out)
        return out
    #2 Sample N dropout masks, and average the pre-softmax values  
    def forward2(self, x, N):
        out = net.fc1(x)
        out = net.relu(out)
        out = net.fc2(out)
        total=0
        for i in range(N):
            out1 = net.drop(out)
            out1 = net.relu(out1)
            out1 = net.fc3(out1)
            total+=out1
        out=total/N
        out = self.softmax(out)
        return out


    #3 Sample N dropout masks, and average the predictions  
    def forward3(self, x, N):
        out = net.fc1(x)
        out = net.relu(out)
        out = net.fc2(out)
        total=0
        for i in range(N):
            out1 = net.drop(out)
            out1 = net.relu(out1)
            out1 = net.fc3(out1)
            out1 = self.softmax(out1)
            total+=out1
        out=total/N
        return out    


net = Net(input_size, hidden_size1, hidden_size2, num_classes)
net.cuda()   
    
# Loss and Optimizer
criterion = nn.CrossEntropyLoss()  
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)  






# Train the Model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        # Convert torch tensor to Variable
        images = Variable(images.view(-1, 28*28).cuda())
        labels = Variable(labels.cuda())
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        


NN=[10,20,30,40,50,60,70,80,90,100]

accuracy1=np.zeros(len(NN))
accuracy2=np.zeros(len(NN))
accuracy3=np.zeros(len(NN))

# Test the Model
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images.view(-1, 28*28)).cuda()
    outputs = net.forward2(images,10)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()

accuracy1 += correct / total 



for i in range(len(NN)):
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images.view(-1, 28*28)).cuda()
        outputs = net.forward2(images,NN[i])
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum()
    accuracy2[i]=correct / total

    for images, labels in test_loader:
        images = Variable(images.view(-1, 28*28)).cuda()
        outputs = net.forward3(images,NN[i])
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum()
    accuracy3[i]=correct / total
    
    
plt.figure()
plt.plot(NN,accuracy1,'-^',label=u'part1')
plt.plot(NN,accuracy2,'-*',label=u'part2')
plt.plot(NN,accuracy3,'-.',label=u'part3')
plt.xlabel('N dropout masks')
plt.ylabel('Accuracy')
plt.grid(color='b' , linewidth='0.3' ,linestyle = "-.")
plt.legend(loc='best')
plt.show()