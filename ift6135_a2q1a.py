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
        
        nn.init.xavier_uniform(self.fc1.weight)
        nn.init.xavier_uniform(self.fc2.weight)
        nn.init.xavier_uniform(self.fc3.weight)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out
    
class Net1(nn.Module):
    def __init__(self, input_size, hidden_size1,hidden_size2, num_classes):
        super(Net1, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, num_classes)  
        self.relu = nn.ReLU()
        
        nn.init.xavier_uniform(self.fc1.weight)
        nn.init.xavier_uniform(self.fc2.weight)
        nn.init.xavier_uniform(self.fc3.weight)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out
    
net = Net(input_size, hidden_size1, hidden_size2, num_classes)
net1 = Net1(input_size, hidden_size1, hidden_size2, num_classes)
net.cuda()   
net1.cuda()
    
# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)  
optimizer_l2 = torch.optim.SGD(net1.parameters(), lr=learning_rate, 
                            weight_decay=2.5*(batch_size/len(train_dataset)))

loss0 = np.zeros(num_epochs)
loss1 = np.zeros(num_epochs)
norm = np.zeros(num_epochs)
l2norm = np.zeros(num_epochs)


#No regularisation
for epoch in range(num_epochs):
    losses=[]
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
        losses.append(loss.data[0])
        
    loss0[epoch] = np.mean(losses)
    norm[epoch] = torch.norm(net.fc1.weight) + torch.norm(net.fc2.weight) + torch.norm(net.fc3.weight)


# L2 regularisation
for epoch in range(num_epochs):
    losses_l2=[]
    for i, (images, labels) in enumerate(train_loader):  
        # Convert torch tensor to Variable
        images = Variable(images.view(-1, 28*28).cuda())
        labels = Variable(labels.cuda())
        
        # Forward + Backward + Optimize
        optimizer_l2.zero_grad()  # zero the gradient buffer
        outputs = net1(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_l2.step()
        losses_l2.append(loss.data[0])
        
    loss1[epoch] = np.mean(losses_l2)
    l2norm[epoch] = torch.norm(net1.fc1.weight) + torch.norm(net1.fc2.weight) + torch.norm(net1.fc3.weight)


plt.figure()
plt.plot(np.linspace(1,num_epochs,num_epochs),loss0,label=u'No regularisation')
plt.plot(np.linspace(1,num_epochs,num_epochs),loss1,label=u'L2 regularisation')
plt.xlabel('Epcoh')
plt.ylabel('Training Error')
plt.grid(color='b' , linewidth='0.3' ,linestyle = "-.")
plt.legend(loc='best')
plt.show()

plt.figure()
plt.plot(np.linspace(1,num_epochs,num_epochs),norm,label=u'No regularisation')
plt.plot(np.linspace(1,num_epochs,num_epochs),l2norm,label=u'L2 regularisation')
plt.xlabel('Epcoh')
plt.ylabel('Parameters L2 Norm')
plt.grid(color='b' , linewidth='0.3' ,linestyle = "-.")
plt.legend(loc='best')
plt.show()